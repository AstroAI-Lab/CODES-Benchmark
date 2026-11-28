import argparse
import os
import queue
import sys
import time
from pathlib import Path

import optuna
import yaml
from optuna.trial import TrialState
from tqdm import tqdm

from codes.benchmark import get_model_config
from codes.tune import (
    MaxValidTrialsCallback,
    _count_valid_trials,
    apply_tuning_defaults,
    build_fine_optuna_params,
    create_objective,
    ensure_study_timing,
    load_yaml_config,
    maybe_set_runtime_threshold,
    prepare_workspace,
)
from codes.tune.storage import create_optuna_storage
from codes.tune.tune_utils import (
    SAMPLER_ATTR,
    count_tuning_trials,
    create_sampler,
    is_pareto_pruned,
    is_valid_complete,
    sampler_name,
)
from codes.utils import download_data, nice_print


def resolve_storage_backend(
    config: dict, tuning_id: str
) -> tuple[optuna.storages.BaseStorage | str, bool]:
    """Return configured Optuna storage and whether it is SQLite."""
    storage, backend = create_optuna_storage(config, tuning_id)
    return storage, backend == "sqlite"


def run_single_study(
    config: dict,
    study_name: str,
    storage: optuna.storages.BaseStorage | str,
    sqlite_backend: bool,
):
    if not config.get("optuna_logging", False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    multi_objective = config.get("multi_objective", False)
    sampler = create_sampler(config, config["seed"])

    if multi_objective:
        pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(
            study_name=study_name,
            directions=["minimize", "minimize"],
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
    else:
        pruner = (
            optuna.pruners.HyperbandPruner(
                min_resource=max(1, config["epochs"] // 8),
                max_resource=config["epochs"],
                reduction_factor=2,
            )
            if config.get("prune", False)
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    selected_sampler = sampler_name(config)
    stored_sampler = study.user_attrs.get(SAMPLER_ATTR)
    if stored_sampler is not None and stored_sampler != selected_sampler:
        raise ValueError(
            f"Study {study_name!r} was created with sampler {stored_sampler!r}; "
            "use a new study name to change samplers"
        )
    historical_sampler = "nsgaii" if multi_objective else "tpe"
    if (
        stored_sampler is None
        and study.get_trials(deepcopy=False)
        and selected_sampler != historical_sampler
    ):
        raise ValueError(
            f"Legacy study {study_name!r} has trials using the historical "
            f"{historical_sampler!r} sampler; use a new study name to change samplers"
        )
    if multi_objective:
        ensure_study_timing(study, config)

    if stored_sampler is None:
        study.set_user_attr(SAMPLER_ATTR, selected_sampler)

    pareto_policy = config.get("multi_objective_pruning")
    have = count_tuning_trials(study) if pareto_policy else _count_valid_trials(study)
    if have >= config["n_trials"]:
        print(
            f"[skip] {study_name}: already reached its trial target {config['n_trials']} ({have} counted)."
        )
        return

    device_queue = queue.Queue()
    devices = config.get("devices", ["cpu"])
    if sqlite_backend and len(devices) > 1:
        print(
            "⚠️ SQLite storage does not handle concurrent writers well. "
            "Continuing with multiple devices may trigger 'database is locked' errors."
        )

    for slot_id, dev in enumerate(devices):
        device_queue.put((dev, slot_id))

    objective_fn = create_objective(config, study_name, device_queue)
    n_trials = config["n_trials"]
    n_jobs = len(devices)
    warmup_target = max(10, int(n_trials * 0.10))

    all_durations: list[float] = []

    def trial_complete_callback(study_: optuna.Study, trial_: optuna.trial.FrozenTrial):
        # progress bar update
        trial_oom = "exception" in trial_.user_attrs  # do not count OOM trials
        trial_timepruned = (
            "prune_reason" in trial_.user_attrs
        )  # do not count time-pruned trials
        counted = (
            (is_valid_complete(trial_) or is_pareto_pruned(trial_))
            if pareto_policy
            else trial_.state in (TrialState.COMPLETE, TrialState.PRUNED)
            and not (trial_oom or trial_timepruned)
        )
        if counted:
            trial_pbar.update(1)

        # duration/eta
        if trial_.datetime_start:
            dur = time.time() - trial_.datetime_start.timestamp()
            all_durations.append(dur)
            avg = sum(all_durations) / len(all_durations)
            eta = avg * max(n_trials - trial_pbar.n, 0) / n_jobs
            trial_pbar.set_postfix_str(
                f"ETA: {eta / 60:.1f}m, Avg: {avg:.1f}s, Last: {dur:.1f}s"
            )

        # try to set threshold (no-op if not enough data or already set)
        if config.get("time_pruning", not pareto_policy):
            maybe_set_runtime_threshold(study_, warmup_target, include_pruned=True)

    dataset_cfg = config["dataset"]
    download_data(dataset_cfg["name"])

    with tqdm(
        total=n_trials,
        desc=f"Tuning {study_name}",
        position=1,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}{postfix}]",
    ) as trial_pbar:
        try:
            if pareto_policy:
                trial_pbar.update(min(have, n_trials))
            if not pareto_policy:
                study.optimize(
                    objective_fn,
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    callbacks=[
                        MaxValidTrialsCallback(n_trials),
                        trial_complete_callback,
                    ],
                )
            else:
                max_attempts = config.get("max_attempts")
                if max_attempts is None:
                    max_attempts = 2 * n_trials
                if (
                    isinstance(max_attempts, bool)
                    or not isinstance(max_attempts, int)
                    or max_attempts < n_trials
                ):
                    raise ValueError(
                        "max_attempts must be an integer at least n_trials"
                    )
                while have < n_trials:
                    attempts = sum(
                        t.state != TrialState.WAITING
                        for t in study.get_trials(deepcopy=False)
                    )
                    remaining = max_attempts - attempts
                    if remaining <= 0:
                        break
                    study.optimize(
                        objective_fn,
                        n_trials=min(remaining, n_trials - have),
                        n_jobs=n_jobs,
                        callbacks=[trial_complete_callback],
                    )
                    have = count_tuning_trials(study)
                counts = study.get_trials(deepcopy=False)
                valid_complete = sum(is_valid_complete(trial) for trial in counts)
                mo_pruned = sum(is_pareto_pruned(trial) for trial in counts)
                attempts = sum(trial.state != TrialState.WAITING for trial in counts)
                print(
                    f"[status] {study_name}: valid={valid_complete}, "
                    f"pareto_pruned={mo_pruned}, target_count={have}/{n_trials}, "
                    f"attempts={attempts}/{max_attempts}"
                )
        except Exception as exc:  # pragma: no cover - guidance-oriented handling
            if sqlite_backend and "database is locked" in str(exc).lower():
                raise RuntimeError(
                    "SQLite storage encountered a concurrency lock. "
                    "Use one device, or use journal or PostgreSQL storage for a new run."
                ) from exc
            raise


def run_all_studies(
    config: dict,
    main_study_name: str,
    storage: optuna.storages.BaseStorage | str,
    sqlite_backend: bool,
):
    surrogates = config["surrogates"]
    dataset_cfg = config["dataset"]
    global_params = (
        {} if config.get("fine", False) else config.get("global_optuna_params", {})
    )

    fine_report: dict[str, dict] = {}

    total_sub_studies = len(surrogates)
    with tqdm(
        total=total_sub_studies, desc="Overall Surrogates", position=0, leave=True
    ) as arch_pbar:
        if config.get("multi_objective", False):
            policy = config.get("multi_objective_pruning")
            policy_mode = policy["mode"] if policy else "off"
            print(
                f"⚠️ Multi-objective mode enabled: using {sampler_name(config)} sampler; "
                f"checkpoint pruning mode is {policy_mode}."
            )

        for surr in surrogates:
            arch_name = surr["name"]
            n_trials_override = None
            if config.get("fine", False):
                # ignore manual search spaces
                surr["optuna_params"] = {}

                # derive fine space from previously best config
                base_cfg = get_model_config(arch_name, config)
                fine_space = build_fine_optuna_params(base_cfg)
                n_fine = len(fine_space)
                n_trials_override = max(10 * n_fine, 10)

                # CLI confirmation
                print(
                    f"[fine] {arch_name}: found fine-tunable parameters: {list(fine_space.keys()) or 'none'}"
                )
                for k, spec in fine_space.items():
                    print(f"  - {k}: [{spec['low']:.3g}, {spec['high']:.3g}] (log)")
                print(f"  -> running for {n_trials_override} trials\n")

                # stash for YAML and pass along to run_single_study
                fine_report[arch_name] = {
                    "trials": int(n_trials_override),
                    "params": {
                        k: {
                            "low": float(v["low"]),
                            "high": float(v["high"]),
                            "log": bool(v.get("log", False)),
                        }
                        for k, v in fine_space.items()
                    },
                }
            else:
                local = surr.get("optuna_params", {})
                for name, opts in global_params.items():
                    if name in local:
                        print(
                            f"⚠️ Hyperparameter '{name}' defined globally and locally for {surr['name']}; using local."
                        )
                    else:
                        local[name] = opts
                surr["optuna_params"] = local

            arch_name = surr["name"]
            study_name = f"{main_study_name}_{arch_name.lower()}"
            arch_pbar.set_postfix({"study": study_name})

            trials = surr.get("trials", config.get("trials"))
            sub_config = {
                "batch_size": surr["batch_size"],
                "dataset": dataset_cfg.copy(),
                "devices": list(config.get("devices", ["cpu"])),
                "epochs": surr["epochs"],
                "n_trials": n_trials_override if n_trials_override else trials,
                "seed": config.get("seed", 42),
                "surrogate": {"name": arch_name},
                "optuna_params": surr.get("optuna_params", {}),
                "prune": config.get("prune", True),
                "optuna_logging": config.get("optuna_logging", False),
                "use_optimal_params": config.get("use_optimal_params", True),
                "multi_objective": config.get("multi_objective", False),
                **({"sampler": config["sampler"]} if "sampler" in config else {}),
                "population_size": config.get("population_size", 50),
                "target_percentile": config.get("target_percentile", 0.99),
                "fine": config.get("fine", False),  # pass through
                "loss_cap": config.get("loss_cap", 20),
                "time_pruning": config.get("time_pruning", True),
                "multi_objective_pruning": config.get("multi_objective_pruning"),
                "max_attempts": config.get("max_attempts"),
            }
            if sub_config["multi_objective"]:
                sub_config["inference_timing"] = config.get(
                    "inference_timing", {"batch_size": 512}
                )

            if sub_config["n_trials"] is None:
                raise ValueError(
                    f"No trial count specified for surrogate '{arch_name}'. "
                    "Add 'trials' either globally or per surrogate."
                )

            if config.get("fine", False):
                sub_config["fine_space"] = fine_space

            run_single_study(sub_config, study_name, storage, sqlite_backend)
            arch_pbar.update(1)
            arch_pbar.set_postfix({"done": study_name})

    # Write YAML summary once per main study (only in fine mode)
    if config.get("fine", False):
        out_dir = os.path.join("tuned", main_study_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "fine_summary.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(fine_report, f, sort_keys=True, default_flow_style=False)
        print(f"[fine] Wrote summary: {out_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Optuna tuning studies.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tuning/config_smoke.yaml",
        help="Path to tuning config YAML.",
    )
    parser.add_argument("--slurm", choices=("prepare", "worker", "status", "submit"))
    parser.add_argument("--manifest", type=Path, help="Frozen Slurm study manifest")
    parser.add_argument(
        "--study", help="Surrogate to select from a multi-model tuning config"
    )
    parser.add_argument(
        "--max-attempts", type=int, help="Maximum training attempts for the study"
    )
    parser.add_argument("--device", default="cpu", help="Device visible to this worker")
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


def main():
    nice_print("Starting Optuna tuning")
    args = parse_arguments()

    if args.slurm:
        from codes.tune import slurm

        if args.slurm == "submit":
            from codes.slurm import SlurmOptions, submit_array

            config = apply_tuning_defaults(load_yaml_config(args.config))
            options = SlurmOptions.from_config(config.get("slurm", {}))
            data_path = (
                Path("datasets") / config["dataset"]["name"].lower() / "data.hdf5"
            )
            if not data_path.is_file():
                raise FileNotFoundError(
                    f"Dataset must be staged before Slurm submission: {data_path}"
                )
            selected, study_name = slurm._study_config(config, args.study)
            manifest = (
                args.manifest
                or Path("tuned") / config["tuning_id"] / f"{study_name}.slurm.json"
            )
            manifest = Path(manifest).resolve()
            slurm.prepare(config, manifest, args.study, args.max_attempts)
            counts = slurm.status(manifest)
            if counts["target_count"] >= selected["n_trials"]:
                print(f"Study {study_name} is complete; no array submitted")
                return
            worker_args = [
                str(Path(__file__).resolve()),
                "--slurm",
                "worker",
                "--manifest",
                str(manifest),
                "--device",
                options.device,
                "--cpu-threads",
                str(options.cpus_per_task),
            ]
            job_id = submit_array(
                manifest,
                list(range(options.workers)),
                options,
                worker_args,
                default_job_name=f"codes-tune-{study_name}",
            )
            print(f"Submitted Slurm job {job_id} for manifest {manifest}")
            return
        if args.manifest is None and args.slurm == "status":
            config = apply_tuning_defaults(load_yaml_config(args.config))
            _, study_name = slurm._study_config(config, args.study)
            args.manifest = (
                Path("tuned") / config["tuning_id"] / f"{study_name}.slurm.json"
            )
        if args.manifest is None:
            raise ValueError("--manifest is required for Slurm tuning")
        if args.slurm == "prepare":
            slurm.prepare(
                load_yaml_config(args.config),
                args.manifest,
                args.study,
                args.max_attempts,
            )
        elif args.slurm == "worker":
            slurm.worker(args.manifest, args.device, args.cpu_threads)
        else:
            slurm.status(args.manifest)
        return

    master_cfg_path = Path(args.config).resolve()
    if not master_cfg_path.exists():
        print(f"Config file not found: {master_cfg_path}")
        sys.exit(1)

    config = load_yaml_config(str(master_cfg_path))
    config = prepare_workspace(master_cfg_path, config)
    config = apply_tuning_defaults(config)
    tuning_id = config["tuning_id"]

    storage, sqlite_backend = resolve_storage_backend(config, tuning_id)

    # Run
    if "surrogates" in config:
        run_all_studies(config, tuning_id, storage, sqlite_backend)
    else:
        run_single_study(config, tuning_id, storage, sqlite_backend)

    nice_print("Optuna tuning completed!")


if __name__ == "__main__":
    main()
