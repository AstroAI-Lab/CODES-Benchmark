"""Slurm workers for one persistent Optuna study.

The manifest is prepared once and read by independent array tasks. Connection
credentials stay in CODES_OPTUNA_STORAGE_URL and are never written to the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import tempfile
from pathlib import Path
from typing import Any

import optuna
import torch
import yaml
from optuna.trial import TrialState

from codes.benchmark import get_model_config
from codes.train.tasks import validate_execution_settings
from codes.tune.optuna_fcts import (
    build_fine_optuna_params,
    create_objective,
    ensure_study_timing,
    maybe_set_runtime_threshold,
)
from codes.tune.storage import (
    create_optuna_storage,
    postgres_environment_url,
    storage_metadata,
)
from codes.tune.tune_utils import (
    SAMPLER_ATTR,
    apply_tuning_defaults,
    create_sampler,
    is_pareto_pruned,
    is_valid_complete,
    sampler_name,
)

MANIFEST_VERSION = 1
STUDY_FINGERPRINT = "codes_slurm_fingerprint"


def _fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _storage(metadata: dict[str, str] | None = None, *, create: bool = True) -> Any:
    # Manifests predating journal support always used PostgreSQL.
    metadata = metadata or {"backend": "postgres"}
    if metadata["backend"] == "journal":
        return create_optuna_storage({"storage": metadata}, "unused", create=create)[0]
    if metadata["backend"] != "postgres":
        raise ValueError("Slurm tuning supports journal or PostgreSQL storage")
    url = postgres_environment_url()
    if not url:
        raise ValueError("Set CODES_OPTUNA_STORAGE_URL to a PostgreSQL database URL")
    # Preserve heartbeat recovery for the explicit PostgreSQL backend.
    return optuna.storages.RDBStorage(url, heartbeat_interval=60, grace_period=180)


def _study_config(
    config: dict[str, Any], surrogate: str | None
) -> tuple[dict[str, Any], str]:
    config = apply_tuning_defaults(config)
    tuning_id = config["tuning_id"]
    if (
        not isinstance(tuning_id, str)
        or not tuning_id
        or tuning_id in (".", "..")
        or Path(tuning_id).name != tuning_id
    ):
        raise ValueError("tuning_id must be a non-empty directory name")
    if "surrogates" not in config:
        if surrogate and surrogate.lower() != config["surrogate"]["name"].lower():
            raise ValueError("Selected study does not match the configured surrogate")
        model_name = config["surrogate"]["name"]
        result = {
            key: value
            for key, value in config.items()
            if key not in ("storage", "postgres_config", "devices", "slurm")
        }
        selected_sampler = sampler_name(result)
        default_sampler = sampler_name(
            {"multi_objective": result.get("multi_objective", False)}
        )
        if selected_sampler == default_sampler:
            result.pop("sampler", None)
        result["n_trials"] = config.get("n_trials", config["trials"])
        if result.get("multi_objective_pruning") is None:
            result.pop("multi_objective_pruning", None)
        return result, f"{tuning_id}_{model_name.lower()}"

    models = config["surrogates"]
    if surrogate is None and len(models) != 1:
        raise ValueError("Select exactly one surrogate with --study")
    selected = [
        item
        for item in models
        if surrogate is None or item["name"].lower() == surrogate.lower()
    ]
    if len(selected) != 1:
        raise ValueError("--study must match exactly one configured surrogate")
    model = selected[0]
    name = model["name"]
    fine = config.get("fine", False)
    params = {
        **config.get("global_optuna_params", {}),
        **model.get("optuna_params", {}),
    }
    result = {
        "batch_size": model["batch_size"],
        "dataset": config["dataset"],
        "epochs": model["epochs"],
        "n_trials": model.get("trials", config["trials"]),
        "seed": config["seed"],
        "surrogate": {"name": name},
        "optuna_params": params,
        "prune": config["prune"],
        "optuna_logging": config["optuna_logging"],
        "use_optimal_params": config["use_optimal_params"],
        "multi_objective": config["multi_objective"],
        "population_size": config["population_size"],
        "target_percentile": config["target_percentile"],
        "fine": fine,
        "loss_cap": config["loss_cap"],
        "time_pruning": config["time_pruning"],
    }
    selected_sampler = sampler_name(config)
    default_sampler = sampler_name({"multi_objective": config["multi_objective"]})
    if selected_sampler != default_sampler:
        result["sampler"] = selected_sampler
    if config.get("multi_objective_pruning") is not None:
        result["multi_objective_pruning"] = config["multi_objective_pruning"]
    if result.get("multi_objective", False):
        result["inference_timing"] = config.get("inference_timing", {"batch_size": 512})
    if fine:
        space = build_fine_optuna_params(get_model_config(name, config))
        result["fine_space"] = space
        result["n_trials"] = max(10 * len(space), 10)
    return result, f"{tuning_id}_{name.lower()}"


def _sampler(config: dict[str, Any], seed: int) -> optuna.samplers.BaseSampler:
    return create_sampler(config, seed)


def _pruner(config: dict[str, Any]) -> optuna.pruners.BasePruner:
    if not config.get("multi_objective", False) and config.get("prune", False):
        return optuna.pruners.HyperbandPruner(
            min_resource=max(1, config["epochs"] // 8),
            max_resource=config["epochs"],
            reduction_factor=2,
        )
    return optuna.pruners.NopPruner()


def _load_study(manifest: dict[str, Any], storage: Any, seed: int) -> optuna.Study:
    config = manifest["config"]
    return optuna.load_study(
        study_name=manifest["study_name"],
        storage=storage,
        sampler=_sampler(config, seed),
        pruner=_pruner(config),
    )


def _check_study(
    study: optuna.Study, manifest: dict[str, Any], *, check_timing: bool = True
) -> None:
    if study.user_attrs.get(STUDY_FINGERPRINT) != _fingerprint(manifest):
        raise ValueError("Study does not match the frozen Slurm manifest")
    stored_sampler = study.user_attrs.get(SAMPLER_ATTR)
    if stored_sampler is not None and stored_sampler != sampler_name(
        manifest["config"]
    ):
        raise ValueError("Study sampler does not match the frozen Slurm manifest")
    if check_timing and manifest["config"].get("multi_objective", False):
        ensure_study_timing(study, manifest["config"])


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("Manifest already exists with different contents")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise ValueError(
                    "Manifest already exists with different contents"
                ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _model_config_digest(config: dict[str, Any]) -> str | None:
    if not config["dataset"].get("use_optimal_params", True):
        return None
    path = Path("datasets") / config["dataset"]["name"].lower() / "surrogates_config.py"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _public_config(config: dict[str, Any], storage: dict[str, str]) -> dict[str, Any]:
    """Retain the scientific config while excluding connection details."""
    public = {
        key: value
        for key, value in config.items()
        if key
        not in {"postgres_config", "storage", "devices", "slurm", "_overwrite_run"}
    }
    configured_sampler = sampler_name(public)
    default_sampler = sampler_name(
        {"multi_objective": public.get("multi_objective", False)}
    )
    if configured_sampler == default_sampler:
        public.pop("sampler", None)
    elif public.get("sampler") is not None:
        public["sampler"] = configured_sampler
    public["storage"] = storage
    if public.get("multi_objective_pruning") is None:
        public.pop("multi_objective_pruning", None)
    return public


def _check_workspace(
    config: dict[str, Any], storage: dict[str, str]
) -> tuple[Path, dict[str, Any]]:
    path = Path("tuned") / config["tuning_id"] / "optuna_config.yaml"
    public = _public_config(config, storage)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = yaml.safe_load(handle)
        if not _workspace_matches(existing, public):
            raise ValueError(
                f"Existing tuning workspace {path.parent} has a different configuration"
            )
    elif path.parent.exists() and any(path.parent.iterdir()):
        raise ValueError(
            f"Existing tuning workspace {path.parent} has no configuration"
        )
    return path, public


def _workspace_matches(existing: Any, public: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    if existing == public:
        return True
    # Before storage metadata was recorded, Slurm workspaces were PostgreSQL-only.
    return (
        "storage" not in existing
        and public.get("storage", {}).get("backend") == "postgres"
        and existing
        == {key: value for key, value in public.items() if key != "storage"}
    )


def _write_workspace(path: Path, public: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            yaml.safe_dump(public, handle, sort_keys=False)
        try:
            os.link(temporary, path)
        except FileExistsError:
            with path.open(encoding="utf-8") as handle:
                existing = yaml.safe_load(handle)
            if not _workspace_matches(existing, public):
                raise ValueError(
                    f"Existing tuning workspace {path.parent} has a different configuration"
                ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(
    config: dict[str, Any], path: Path, surrogate: str | None, max_attempts: int | None
) -> None:
    selected, name = _study_config(config, surrogate)
    existing_manifest = _read_manifest(path) if path.exists() else None
    storage_info = storage_metadata(config, config["tuning_id"])
    if (
        existing_manifest is not None
        and "storage" not in existing_manifest
        and not config.get("storage")
    ):
        storage_info = {"backend": "postgres"}
    workspace_path, public_config = _check_workspace(config, storage_info)
    target = selected["n_trials"]
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("Study trial target must be a positive integer")
    if max_attempts is None:
        max_attempts = 2 * target
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < target
    ):
        raise ValueError("--max-attempts must be at least the trial target")
    manifest = {
        "version": MANIFEST_VERSION,
        "working_directory": str(Path.cwd().resolve()),
        "study_name": name,
        "config": selected,
        "model_config_sha256": _model_config_digest(selected),
        "max_attempts": max_attempts,
        "storage": storage_info,
    }
    if existing_manifest is not None:
        # Keep the pre-journal manifest format reusable; it unambiguously meant PG.
        if "storage" not in existing_manifest and storage_info["backend"] == "postgres":
            manifest.pop("storage")
        if existing_manifest != manifest:
            raise ValueError("Manifest already exists with different contents")
    storage = _storage(storage_info)
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        directions=["minimize", "minimize"]
        if selected.get("multi_objective", False)
        else ["minimize"],
        sampler=_sampler(selected, selected["seed"]),
        pruner=_pruner(selected),
    )
    expected_directions = [optuna.study.StudyDirection.MINIMIZE] * (
        2 if selected.get("multi_objective", False) else 1
    )
    if study.directions != expected_directions:
        raise ValueError("Existing study has incompatible objective directions")
    selected_sampler = sampler_name(selected)
    stored_sampler = study.user_attrs.get(SAMPLER_ATTR)
    if stored_sampler is not None and stored_sampler != selected_sampler:
        raise ValueError("Existing study was created with a different sampler")
    marker = study.user_attrs.get(STUDY_FINGERPRINT)
    if marker is None:
        checkpoint_dir = (
            workspace_path.parent / "models" / selected["surrogate"]["name"]
        )
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            raise ValueError(
                f"Unmarked study would overwrite existing checkpoints in {checkpoint_dir}"
            )
        if study.get_trials(deepcopy=False):
            raise ValueError(
                "Existing study contains trials but has no Slurm manifest fingerprint"
            )
        study.set_user_attr(STUDY_FINGERPRINT, _fingerprint(manifest))
    _check_study(study, manifest)
    if stored_sampler is None:
        historical_sampler = sampler_name(
            {"multi_objective": selected.get("multi_objective", False)}
        )
        if study.get_trials(deepcopy=False) and selected_sampler != historical_sampler:
            raise ValueError(
                "Legacy study uses its historical sampler; use a new study name"
            )
        study.set_user_attr(SAMPLER_ATTR, selected_sampler)
    _write_workspace(workspace_path, public_config)
    _write_manifest(path, manifest)
    print(
        f"Prepared study {name!r} with target {target} and maximum {max_attempts} attempts"
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("Unsupported Slurm tuning manifest version")
    if manifest.get("working_directory") != str(Path.cwd().resolve()):
        raise ValueError(
            "Run Slurm tuning from the directory where the manifest was prepared"
        )
    return manifest


def _counts(study: optuna.Study) -> dict[str, int]:
    trials = study.get_trials(deepcopy=False)
    counts = {state.name.lower(): 0 for state in TrialState}
    counts["valid"] = 0
    counts["pareto_pruned"] = 0
    for trial in trials:
        counts[trial.state.name.lower()] += 1
        if is_valid_complete(trial):
            counts["valid"] += 1
        if is_pareto_pruned(trial):
            counts["pareto_pruned"] += 1
    counts["target_count"] = counts["valid"] + counts["pareto_pruned"]
    return counts


def status(path: Path) -> dict[str, int]:
    manifest = _read_manifest(path)
    study = _load_study(
        manifest,
        _storage(manifest.get("storage"), create=False),
        manifest["config"]["seed"],
    )
    _check_study(study, manifest, check_timing=False)
    counts = _counts(study)
    print(
        json.dumps(
            {
                "study": study.study_name,
                "target": manifest["config"]["n_trials"],
                "max_attempts": manifest["max_attempts"],
                "budget_exhausted": len(study.get_trials(deepcopy=False))
                >= manifest["max_attempts"]
                and counts["target_count"] < manifest["config"]["n_trials"],
                **counts,
            },
            sort_keys=True,
        )
    )
    return counts


def worker(path: Path, device: str, cpu_threads: int) -> None:
    manifest = _read_manifest(path)
    try:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if index < 0:
            raise ValueError
    except (KeyError, ValueError):
        raise ValueError("SLURM_ARRAY_TASK_ID must be a non-negative integer") from None
    config = manifest["config"]
    data_path = Path("datasets") / config["dataset"]["name"].lower() / "data.hdf5"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset is not staged at {data_path}; stage it before submitting workers"
        )
    selected_device = validate_execution_settings(device, cpu_threads)
    torch.set_num_threads(cpu_threads)
    if selected_device.type == "cuda":
        torch.cuda.set_device(selected_device)
    if _model_config_digest(config) != manifest["model_config_sha256"]:
        raise ValueError(
            "Dataset surrogate configuration changed since the study was prepared"
        )
    seed_material = (
        f"{config['seed']}:{os.environ.get('SLURM_ARRAY_JOB_ID', '')}:{index}"
    )
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "big")
    study = _load_study(manifest, _storage(manifest.get("storage"), create=False), seed)
    _check_study(study, manifest)
    devices: queue.Queue = queue.Queue()
    devices.put((device, 0))
    objective = create_objective(config, manifest["study_name"], devices)
    target, max_attempts = config["n_trials"], manifest["max_attempts"]
    warmup_target = max(10, int(target * 0.10))

    def run_one(trial: optuna.Trial) -> float | tuple[float, float]:
        if trial.number >= max_attempts or _counts(study)["target_count"] >= target:
            raise optuna.TrialPruned("Slurm study budget reached")
        if _model_config_digest(config) != manifest["model_config_sha256"]:
            raise ValueError(
                "Dataset surrogate configuration changed since the study was prepared"
            )
        return objective(trial)

    def record_finished(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if config.get("time_pruning", True):
            maybe_set_runtime_threshold(study, warmup_target, include_pruned=True)

    while True:
        optuna.storages.fail_stale_trials(study)
        trials = study.get_trials(deepcopy=False)
        if _counts(study)["target_count"] >= target or any(
            t.number >= max_attempts - 1 for t in trials
        ):
            break
        study.optimize(run_one, n_trials=1, callbacks=[record_finished])
    print(f"Worker {index} finished study {study.study_name!r}")
    status(path)
