"""Shared Slurm submission options and safe array submission helper."""

from __future__ import annotations

import getpass
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SlurmOptions:
    device: str = "cpu"
    cpus_per_task: int = 1
    workers: int = 1
    qos: str | None = None
    partition: str | None = None
    account: str | None = None
    time: str | None = None
    mem: str | None = None
    gres: str | None = None
    constraint: str | None = None
    nodelist: str | None = None
    reservation: str | None = None
    job_name: str | None = None
    log_dir: Path | None = None
    max_parallel: int | None = None
    launcher: Path | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> SlurmOptions:
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError("slurm configuration must be a mapping")
        allowed = {field for field in cls.__dataclass_fields__}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"Unknown slurm settings: {', '.join(sorted(unknown))}")
        values = dict(config)
        for key in ("cpus_per_task", "workers", "max_parallel"):
            value = values.get(key)
            if key in values and (type(value) is not int or value < 1):
                raise ValueError(f"slurm.{key} must be a positive integer")
        for key in (
            "device",
            "qos",
            "partition",
            "account",
            "time",
            "mem",
            "gres",
            "constraint",
            "nodelist",
            "reservation",
            "job_name",
        ):
            value = values.get(key)
            if value is not None and (
                not isinstance(value, str) or not value.strip() or "\x00" in value
            ):
                raise ValueError(f"slurm.{key} must be a non-empty string")
        for key in ("log_dir", "launcher"):
            value = values.get(key)
            if value is not None and not isinstance(value, (str, os.PathLike)):
                raise ValueError(f"slurm.{key} must be a path")
            if value is not None:
                values[key] = Path(value).expanduser()
        if "device" in values and (
            not isinstance(values["device"], str) or not values["device"].strip()
        ):
            raise ValueError("slurm.device must be a non-empty string")
        return cls(**values)


def _atomic_receipt(path: Path, job_id: str) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps({"job_id": job_id}) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def submit_array(
    manifest: Path,
    indices: Sequence[int],
    options: SlurmOptions,
    worker_args: Sequence[str],
) -> str:
    """Submit array indices and return the parsable Slurm job id."""
    manifest = Path(manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not indices or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("indices must contain non-negative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must be unique")
    if not worker_args or any(not isinstance(arg, str) for arg in worker_args):
        raise ValueError("worker_args must be a non-empty sequence of strings")
    ordered = sorted(indices)
    if ordered[-1] - ordered[0] + 1 == len(ordered):
        array_spec = (
            f"{ordered[0]}-{ordered[-1]}" if len(ordered) > 1 else str(ordered[0])
        )
    else:
        array_spec = ",".join(map(str, ordered))
    if options.max_parallel is not None:
        array_spec += f"%{options.max_parallel}"
    logs = (options.log_dir or manifest.parent / "slurm-logs").resolve()
    logs.mkdir(parents=True, exist_ok=True)
    receipt = manifest.with_suffix(manifest.suffix + ".submit.json")
    lock_path = manifest.with_suffix(manifest.suffix + ".submit.lock")
    import fcntl

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.is_file():
            try:
                previous = json.loads(receipt.read_text(encoding="utf-8"))["job_id"]
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    "Submission receipt is invalid; inspect it before resubmitting"
                ) from exc
            if not isinstance(previous, str) or not previous.isdigit():
                raise RuntimeError(
                    "Submission receipt has an invalid job id; inspect it before resubmitting"
                )
            try:
                active = subprocess.run(
                    ["squeue", "-h", "-u", getpass.getuser(), "-o", "%F"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(
                    "Could not check Slurm queue; verify the prior job before resubmitting"
                ) from exc
            if active.returncode != 0:
                raise RuntimeError(
                    "Could not check Slurm queue; verify the prior job before resubmitting"
                )
            if previous in {line.strip() for line in active.stdout.splitlines()}:
                raise RuntimeError(f"Manifest already has active Slurm job {previous}")
        command = [
            "sbatch",
            "--parsable",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={options.cpus_per_task}",
            f"--chdir={Path.cwd().resolve()}",
            f"--array={array_spec}",
            f"--output={logs}/%x-%A_%a.out",
            "--export=ALL",
        ]
        for name in (
            "qos",
            "partition",
            "account",
            "time",
            "mem",
            "gres",
            "constraint",
            "nodelist",
            "reservation",
            "job_name",
        ):
            value = getattr(options, name)
            if value is not None:
                command.append(f"--{name.replace('_', '-')}={value}")
        if options.launcher is not None:
            launcher = options.launcher.expanduser().resolve()
            if not launcher.is_file():
                raise FileNotFoundError(f"Slurm launcher not found: {launcher}")
            command.extend([str(launcher), *worker_args])
        else:
            python = os.environ.get("CODES_PYTHON") or sys.executable
            command.extend(["--wrap", shlex.join([python, *worker_args])])
        try:
            environment = os.environ.copy()
            environment["CODES_PYTHON"] = environment.get("CODES_PYTHON") or str(
                Path(sys.executable).absolute()
            )
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Slurm submission timed out; check the queue for an accepted job before retrying"
            ) from exc
        except OSError as exc:
            raise RuntimeError("Slurm submission could not be started") from exc
        if result.returncode != 0:
            diagnostic = result.stderr.strip()
            detail = f": {diagnostic}" if diagnostic else ""
            raise RuntimeError(
                f"Slurm submission failed (exit {result.returncode}){detail}"
            )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError("Slurm returned an invalid job id")
        _atomic_receipt(receipt, job_id)
        return job_id
