"""Prepare and execute fixed training assignments for scheduler arrays."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from codes.train.tasks import TrainingTask, execute_training_task
from codes.train.train_fcts import create_task_list_for_surrogate
from codes.utils import determine_batch_size, read_yaml_config


def _run_dir(training_id: str) -> Path:
    path = Path(training_id)
    if (
        not training_id
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != training_id
        or str(path) == "."
    ):
        raise ValueError("training_id must be a normalized, non-empty relative path")
    return Path("trained") / path


@contextlib.contextmanager
def training_lock(training_id: str, *, shared: bool = False) -> Iterator[None]:
    """Coordinate local runs, preparation, and external workers for one run."""
    if os.name != "posix":
        if shared:
            raise RuntimeError("Slurm workers require POSIX file locking")
        yield
        return

    import fcntl

    _run_dir(training_id)
    lock_dir = Path("trained") / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(training_id.encode()).hexdigest() + ".lock"
    with (lock_dir / lock_name).open("a+") as lock:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(lock, mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Training run {training_id!r} is already active"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if exclusive:
            os.link(temp_name, path)
        else:
            os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _model_config_digest(config: dict) -> str | None:
    if not config["dataset"].get("use_optimal_params", True):
        return None
    path = Path("datasets") / config["dataset"]["name"].lower() / "surrogates_config.py"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _checkpoint(config: dict, task: TrainingTask) -> Path:
    name, mode, metric, training_id, _, _ = task
    if mode == "batchsize":
        metric = determine_batch_size(
            config, config["surrogates"].index(name), mode, metric
        )
    model_name = f"{name.lower()}_{mode}_{metric}".strip("_").replace("__", "_")
    return _run_dir(training_id) / name / f"{model_name}.pth"


def _load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("version") != 1 or manifest.get("root") != str(
        Path.cwd().resolve()
    ):
        raise ValueError("Invalid manifest or different working directory")
    training_id = manifest["config"]["training_id"]
    run_dir = _run_dir(training_id)
    owner_path = run_dir / "slurm_manifest.json"
    if not owner_path.is_file():
        raise ValueError("Training run is missing its Slurm manifest registration")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if owner != {
        "manifest": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }:
        raise ValueError("Training manifest has changed or belongs to another run")
    if read_yaml_config(str(run_dir / "config.yaml")) != manifest["config"]:
        raise ValueError("Saved training config differs from the frozen manifest")
    return manifest


def prepare_training(config_path: str, manifest_path: str) -> None:
    """Freeze configuration and deterministic task assignments for one run."""
    config = read_yaml_config(config_path)
    training_id = config["training_id"]
    path = Path(manifest_path).resolve()
    run_dir = _run_dir(training_id)
    with training_lock(training_id):
        tasks = [
            list(task)
            for name in config["surrogates"]
            for task in create_task_list_for_surrogate(config, name)
        ]
        checkpoints = [_checkpoint(config, task) for task in tasks]
        if len(set(checkpoints)) != len(checkpoints):
            raise ValueError("Training tasks would write the same checkpoint")
        if not tasks:
            raise ValueError("Training manifest must contain at least one task")
        if path in {
            (run_dir / "config.yaml").resolve(),
            (run_dir / "slurm_manifest.json").resolve(),
            (run_dir / "slurm_initializing.json").resolve(),
        }:
            raise ValueError("Manifest path conflicts with training metadata")
        manifest = {
            "version": 1,
            "root": str(Path.cwd().resolve()),
            "config": config,
            "model_config_sha256": _model_config_digest(config),
            "tasks": tasks,
        }
        owner_path = run_dir / "slurm_manifest.json"
        initializing_path = run_dir / "slurm_initializing.json"
        initial_state = {
            "manifest": str(path),
            "config": config,
            "model_config_sha256": manifest["model_config_sha256"],
        }
        if owner_path.exists():
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if owner.get("manifest") != str(path):
                raise FileExistsError(
                    f"Training directory already belongs to {owner.get('manifest')}"
                )
            existing = _load_manifest(path)
            if existing["config"] != config:
                raise ValueError("Existing manifest uses a different training config")
            if existing["model_config_sha256"] != manifest["model_config_sha256"]:
                raise ValueError(
                    "Dataset surrogate configuration changed since preparation"
                )
            print(f"Existing manifest {path} contains {len(existing['tasks'])} tasks")
            return
        created_run = False
        if run_dir.exists():
            if not initializing_path.is_file():
                raise FileExistsError(f"Training directory already exists: {run_dir}")
            initial = json.loads(initializing_path.read_text(encoding="utf-8"))
            if initial != initial_state:
                raise ValueError(
                    "Incomplete preparation uses a different config or manifest"
                )
        else:
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=".slurm-prepare-", dir=run_dir.parent)
            )
            try:
                _atomic_json(staging / "slurm_initializing.json", initial_state)
                (staging / "config.yaml").write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                )
                os.rename(staging, run_dir)
                created_run = True
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        published = False
        try:
            if read_yaml_config(str(run_dir / "config.yaml")) != config:
                raise ValueError("Saved config differs from incomplete preparation")
            try:
                _atomic_json(path, manifest, exclusive=True)
                published = True
            except FileExistsError:
                if created_run:
                    raise FileExistsError(
                        f"Manifest path already exists: {path}"
                    ) from None
                with path.open(encoding="utf-8") as file:
                    existing = json.load(file)
                if existing != manifest:
                    raise FileExistsError(
                        f"Manifest path already exists: {path}"
                    ) from None
            owner = {
                "manifest": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            _atomic_json(owner_path, owner)
            initializing_path.unlink(missing_ok=True)
        except Exception:
            if published:
                path.unlink(missing_ok=True)
            if created_run:
                shutil.rmtree(run_dir)
            raise
        print(f"Prepared {len(tasks)} training tasks in {path}")
        print(f"Array indices: 0-{len(tasks) - 1}")


def run_worker(
    manifest_path: str, device: str = "cpu", index: int | None = None
) -> None:
    """Run one array task and atomically record completion after checkpointing."""
    import fcntl

    path = Path(manifest_path).resolve()
    manifest = _load_manifest(path)
    if index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("SLURM_ARRAY_TASK_ID is required")
        try:
            index = int(value)
        except ValueError as exc:
            raise ValueError("SLURM_ARRAY_TASK_ID must be an integer") from exc
    tasks = manifest["tasks"]
    if index < 0 or index >= len(tasks):
        raise ValueError(f"Task index {index} outside 0-{len(tasks) - 1}")
    config = manifest["config"]
    task = tasks[index]
    run_dir = _run_dir(config["training_id"])
    state_dir = run_dir / "slurm_tasks"
    state_dir.mkdir(parents=True, exist_ok=True)
    with training_lock(config["training_id"], shared=True):
        _load_manifest(path)
        with (state_dir / f"{index}.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Task {index} is already running") from exc
            done = state_dir / f"{index}.json"
            checkpoint = _checkpoint(config, task)
            if (
                done.exists()
                and checkpoint.is_file()
                and checkpoint.with_suffix(".yaml").is_file()
            ):
                print(f"Task {index} already complete")
                return
            if done.exists():
                done.unlink()
            dataset = Path("datasets") / config["dataset"]["name"].lower() / "data.hdf5"
            if not dataset.is_file():
                raise FileNotFoundError(
                    f"Dataset {dataset} must be available before running array workers"
                )
            if _model_config_digest(config) != manifest["model_config_sha256"]:
                raise ValueError(
                    "Dataset surrogate configuration changed since preparation"
                )
            execute_training_task(
                task, device=device, cpu_threads=config.get("cpu_threads", 1)
            )
            if (
                not checkpoint.is_file()
                or not checkpoint.with_suffix(".yaml").is_file()
            ):
                raise RuntimeError(
                    f"Task {index} did not produce checkpoint and metadata"
                )
            _atomic_json(done, {"index": index, "checkpoint": str(checkpoint)})
            print(f"Task {index} completed")


def training_status(manifest_path: str) -> None:
    """Report completed and retryable assignment indices."""
    manifest = _load_manifest(Path(manifest_path).resolve())
    config = manifest["config"]
    state_dir = _run_dir(config["training_id"]) / "slurm_tasks"
    done = []
    pending = []
    for index, task in enumerate(manifest["tasks"]):
        checkpoint = _checkpoint(config, task)
        if (
            (state_dir / f"{index}.json").is_file()
            and checkpoint.is_file()
            and checkpoint.with_suffix(".yaml").is_file()
        ):
            done.append(index)
        else:
            pending.append(index)
    print(f"Completed: {len(done)}/{len(manifest['tasks'])}")
    print("Retry indices: " + (",".join(map(str, pending)) if pending else "none"))
