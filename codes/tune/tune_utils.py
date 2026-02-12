import copy
import shutil
import sys
from datetime import datetime
from pathlib import Path

import optuna

from codes.utils.utils import read_yaml_config


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
        if stored_config != config:
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
    dataset.setdefault("tolerance", 1e-25)
    dataset.setdefault("subset_factor", 1)
    dataset.setdefault("log_timesteps", False)
    cfg["dataset"] = dataset

    cfg.setdefault("devices", ["cpu"])
    cfg.setdefault("seed", 42)
    cfg.setdefault("multi_objective", False)
    cfg.setdefault("target_percentile", 0.99)
    cfg.setdefault("loss_cap", 20)
    cfg.setdefault("time_pruning", True)
    cfg.setdefault("population_size", 50)
    cfg.setdefault("optuna_logging", False)
    cfg.setdefault("use_optimal_params", True)
    cfg.setdefault("prune", True)
    cfg.setdefault("trials", 50)

    return cfg


def delete_studies_if_requested(config: dict, main_study_name: str, db_url: str):
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
        except KeyError or TypeError:
            # study didn't exist
            pass
