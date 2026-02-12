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
    initialize_optuna_database,
    load_yaml_config,
    maybe_set_runtime_threshold,
    prepare_workspace,
)
from codes.utils import download_data, nice_print


def resolve_storage_backend(config: dict, tuning_id: str) -> tuple[str, bool]:
    """
    Return (storage_url, is_sqlite).
    Defaults to Postgres for backward compatibility.
    """
    storage_cfg = config.get("storage", {})
    backend = storage_cfg.get("backend", "postgres").lower()

    if backend == "sqlite":
        sqlite_path = storage_cfg.get("path")
        if not sqlite_path:
            raise ValueError(
                "SQLite storage requires `storage.path` in optuna_config.yaml."
            )
        sqlite_path = Path(sqlite_path).expanduser().resolve()
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{sqlite_path}", True

    if backend == "postgres":
        storage_url = initialize_optuna_database(config, study_folder_name=tuning_id)
        return storage_url, False

    raise ValueError(
        f"Unknown storage backend '{backend}'. Use 'sqlite' or 'postgres'."
    )


def run_single_study(config: dict, study_name: str, db_url: str, sqlite_backend: bool):
    if not config.get("optuna_logging", False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    multi_objective = config.get("multi_objective", False)

    if multi_objective:
        sampler = optuna.samplers.NSGAIISampler(
            seed=config["seed"], population_size=config["population_size"]
        )
        pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(
            study_name=study_name,
            directions=["minimize", "minimize"],
            storage=db_url,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=config["seed"])
        pruner = (
            optuna.pruners.HyperbandPruner(
                min_resource=config["epochs"] // 8,
                max_resource=config["epochs"],
                reduction_factor=2,
            )
            if config.get("prune", False)
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=db_url,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    have = _count_valid_trials(study)
    if have >= config["n_trials"]:
        print(
            f"[skip] {study_name}: already has {have} valid trials (target {config['n_trials']}). Skipping optimize()."
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
        wanted_states = (TrialState.COMPLETE, TrialState.PRUNED)
        if trial_.state in wanted_states and not (trial_oom or trial_timepruned):
            trial_pbar.update(1)

        # duration/eta
        if trial_.datetime_start:
            dur = time.time() - trial_.datetime_start.timestamp()
            all_durations.append(dur)
            avg = sum(all_durations) / len(all_durations)
            eta = (avg * (n_trials - len(all_durations))) / n_jobs
            trial_pbar.set_postfix_str(
                f"ETA: {eta / 60:.1f}m, Avg: {avg:.1f}s, Last: {dur:.1f}s"
            )

        # try to set threshold (no-op if not enough data or already set)
        if config.get("time_pruning", True):
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
            study.optimize(
                objective_fn,
                n_trials=n_trials,
                n_jobs=n_jobs,
                callbacks=[
                    MaxValidTrialsCallback(n_trials),
                    trial_complete_callback,
                ],
            )
        except Exception as exc:  # pragma: no cover - guidance-oriented handling
            if sqlite_backend and "database is locked" in str(exc).lower():
                raise RuntimeError(
                    "SQLite storage encountered a concurrency lock. "
                    "Please rerun with fewer devices or switch to Postgres."
                ) from exc
            raise


def run_all_studies(
    config: dict, main_study_name: str, db_url: str, sqlite_backend: bool
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
            print(
                "⚠️ Multi-objective mode enabled: using NSGA-II sampler and disabling pruning."
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
                "n_trials": trials if not n_trials_override else n_trials_override,
                "seed": config.get("seed", 42),
                "surrogate": {"name": arch_name},
                "optuna_params": surr.get("optuna_params", {}),
                "prune": config.get("prune", True),
                "optuna_logging": config.get("optuna_logging", False),
                "use_optimal_params": config.get("use_optimal_params", True),
                "multi_objective": config.get("multi_objective", False),
                "population_size": config.get("population_size", 50),
                "target_percentile": config.get("target_percentile", 0.99),
                "fine": config.get("fine", False),  # pass through
                "loss_cap": config.get("loss_cap", 20),
                "time_pruning": config.get("time_pruning", True),
            }

            if sub_config["n_trials"] is None:
                raise ValueError(
                    f"No trial count specified for surrogate '{arch_name}'. "
                    "Add 'trials' either globally or per surrogate."
                )

            if config.get("fine", False):
                sub_config["fine_space"] = fine_space

            run_single_study(sub_config, study_name, db_url, sqlite_backend)
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
        default="configs/tuning/sqlite_quickstart.yaml",
        help="Path to tuning config YAML.",
    )
    return parser.parse_args()


def main():
    nice_print("Starting Optuna tuning")
    args = parse_arguments()

    master_cfg_path = Path(args.config).resolve()
    if not master_cfg_path.exists():
        print(f"Config file not found: {master_cfg_path}")
        sys.exit(1)

    config = load_yaml_config(str(master_cfg_path))
    config = prepare_workspace(master_cfg_path, config)
    config = apply_tuning_defaults(config)
    tuning_id = config["tuning_id"]

    # Initialize DB (remote/local)
    db_url, sqlite_backend = resolve_storage_backend(config, tuning_id)

    # Run
    if "surrogates" in config:
        run_all_studies(config, tuning_id, db_url, sqlite_backend)
    else:
        run_single_study(config, tuning_id, db_url, sqlite_backend)

    nice_print("Optuna tuning completed!")


if __name__ == "__main__":
    main()
