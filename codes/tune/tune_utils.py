import copy
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import optuna
from optuna.trial import FrozenTrial, TrialState

from codes.utils.utils import read_yaml_config

from .optuna_fcts import endpoint_timing_protocol
from .pareto_pruning import PARETO_PRUNED_ATTR, parse_pareto_pruning

SAMPLER_ATTR = "codes_sampler"


def sampler_name(config: dict) -> str:
    """Resolve configured sampler, retaining the historical objective-based default."""
    name = config.get("sampler")
    if name is None:
        name = "nsgaii" if config.get("multi_objective", False) else "tpe"
    name = str(name).lower()
    if name not in {"nsgaii", "tpe"}:
        raise ValueError("sampler must be 'nsgaii' or 'tpe'")
    if name == "nsgaii" and not config.get("multi_objective", False):
        raise ValueError("NSGA-II requires multi_objective: true")
    return name


def create_sampler(config: dict, seed: int) -> optuna.samplers.BaseSampler:
    """Build the selected sampler for either objective type."""
    if sampler_name(config) == "nsgaii":
        return optuna.samplers.NSGAIISampler(
            seed=seed, population_size=config.get("population_size", 50)
        )
    return optuna.samplers.TPESampler(seed=seed)


def yes_no(prompt: str, default: bool = False) -> bool:
    """Simple Y/N prompt. default=False -> [y/N], True -> [Y/n]."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        ans = input(prompt + suffix).strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def copy_config(src: Path, dst_folder: Path) -> None:
    dst = dst_folder / "optuna_config.yaml"
    shutil.copy2(src, dst)


def build_study_names(config: dict, main_study_name: str) -> list[str]:
    if "surrogates" in config:
        return [f"{main_study_name}_{s['name'].lower()}" for s in config["surrogates"]]
    return [main_study_name]


def _prompt_config_decision(stored: Path, incoming: Path) -> str:
    prompt = (
        f"A stored config exists at '{stored}'.\n"
        f"The provided config '{incoming}' differs.\n"
        "Use stored version [U], overwrite with new version [O], or abort [A]? "
    )
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("u", "use"):
            return "use"
        if ans in ("o", "overwrite"):
            return "overwrite"
        if ans in ("a", "abort"):
            return "abort"
        print("Please answer with U, O, or A.")


def prepare_workspace(master_cfg_path: Path, config: dict) -> dict:
    tuning_id = config["tuning_id"]
    run_dir = Path("tuned") / tuning_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dst_cfg = run_dir / "optuna_config.yaml"

    config["_overwrite_run"] = False

    if dst_cfg.exists():
        stored_config = read_yaml_config(dst_cfg)
        # Compare both files parsed the same way; `config` already has extra keys.
        if stored_config != read_yaml_config(master_cfg_path):
            decision = _prompt_config_decision(dst_cfg, master_cfg_path)
            if decision == "use":
                print(
                    f"Using stored config for tuning_id '{tuning_id}'. "
                    "The provided config was ignored."
                )
                return stored_config
            if decision == "overwrite":
                backup = dst_cfg.with_name(
                    f"optuna_config.yaml.bak-{datetime.now():%Y%m%d-%H%M%S}"
                )
                shutil.copy2(dst_cfg, backup)
                copy_config(master_cfg_path, run_dir)
                config["_overwrite_run"] = True
                return config
            print("Aborting tuning run.")
            sys.exit(1)
        return stored_config

    copy_config(master_cfg_path, run_dir)
    config["_overwrite_run"] = True
    return config


def apply_tuning_defaults(config: dict) -> dict:
    cfg = copy.deepcopy(config)

    dataset = cfg.get("dataset", {})
    if "name" not in dataset:
        raise ValueError("Tuning config must specify dataset.name")
    dataset.setdefault("log10_transform", True)
    dataset.setdefault("log10_transform_params", True)
    dataset.setdefault("normalise", "minmax")
    dataset.setdefault("normalise_per_species", False)
    dataset.setdefault("tolerance", None)
    dataset.setdefault("subset_factor", 1)
    dataset.setdefault("log_timesteps", False)
    cfg["dataset"] = dataset

    cfg.setdefault("devices", ["cpu"])
    cfg.setdefault("seed", 42)
    cfg.setdefault("multi_objective", False)
    cfg.setdefault("target_percentile", 0.99)
    cfg.setdefault("loss_cap", 20)
    time_pruning_explicit = "time_pruning" in cfg
    cfg.setdefault("time_pruning", True)
    cfg.setdefault("population_size", 50)
    cfg.setdefault("optuna_logging", False)
    cfg.setdefault("use_optimal_params", True)
    cfg.setdefault("prune", True)
    cfg.setdefault("trials", 50)
    cfg["multi_objective_pruning"] = parse_pareto_pruning(cfg)
    if cfg["multi_objective_pruning"] is not None and not time_pruning_explicit:
        cfg["time_pruning"] = False
    if cfg.get("multi_objective", False):
        timing = endpoint_timing_protocol(cfg)
        timing.pop("protocol")
        cfg["inference_timing"] = timing

    return cfg


def is_valid_complete(trial: FrozenTrial) -> bool:
    return (
        trial.state == TrialState.COMPLETE
        and "exception" not in trial.user_attrs
        and trial.values is not None
        and all(math.isfinite(value) for value in trial.values)
    )


def is_pareto_pruned(trial: FrozenTrial) -> bool:
    return (
        trial.state == TrialState.PRUNED
        and trial.user_attrs.get(PARETO_PRUNED_ATTR) is True
    )


def count_tuning_trials(study: optuna.Study) -> int:
    return sum(
        is_valid_complete(trial) or is_pareto_pruned(trial)
        for trial in study.get_trials(deepcopy=False)
    )


def delete_studies_if_requested(
    config: dict, main_study_name: str, db_url: str
) -> None:
    """Delete studies if user overwrote run folder."""
    if not config.get("_overwrite_run", False):
        return
    mode = config.get("postgres_config", {}).get("mode", "local").lower()
    if mode == "local":
        return  # no need to delete studies in local mode, DB is handled later in this mode (overwrite or use existing)
    names = build_study_names(config, main_study_name)
    for name in names:
        try:
            optuna.delete_study(study_name=name, storage=db_url)
            print(f"Deleted existing study '{name}'.")
        except (KeyError, TypeError):
            # study didn't exist
            pass
