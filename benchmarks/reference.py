"""Prepare, run and estimate isolated reference benchmark experiments."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

MODELS = ("MultiONet", "FullyConnected", "LatentNeuralODE", "LatentPoly")
DATASETS = ("primordial", "primordial_parametric", "cloud", "cloud_parametric")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        gitdir = repo / ".git"
        if gitdir.is_file():
            gitdir = (
                repo / gitdir.read_text().strip().removeprefix("gitdir: ")
            ).resolve()
        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        loose = gitdir / ref
        if loose.is_file():
            return loose.read_text().strip()
        common = gitdir
        if (gitdir / "commondir").is_file():
            common = (gitdir / (gitdir / "commondir").read_text().strip()).resolve()
            if (common / ref).is_file():
                return (common / ref).read_text().strip()
        for line in (common / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
        raise ValueError(f"Cannot resolve git reference {ref}")


def code_identity(repo, require_revision=True):
    # Hash the executable source as well as the revision so dirty checkouts cannot
    # silently resume under different code. No git operation changes the checkout.
    files = [repo / "run_training.py", repo / "run_eval.py"]
    files += sorted((repo / "codes").rglob("*.py"))
    files += sorted((repo / "benchmarks").glob("*.py"))
    files += [
        repo / "pyproject.toml",
        repo / "uv.lock",
        repo / "requirements.txt",
    ]
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(repo)).encode())
        h.update(bytes.fromhex(sha(path)))
    try:
        rev = git_revision(repo)
    except (OSError, ValueError):
        raise ValueError("Cannot determine repository revision") from None
    return rev, h.hexdigest()


def ensure_clean_code(repo):
    """Require the executable sources to be committed before preparing a run."""
    try:
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Preparing a run requires git and a clean source checkout"
        ) from exc
    if status.strip():
        raise ValueError(
            "Commit or otherwise clean benchmark source changes before preparing a run"
        )


def software_identity():
    gpu = None
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                visible,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    versions = {}
    from importlib.metadata import PackageNotFoundError, version

    for name in (
        "torch",
        "torchode",
        "schedulefree",
        "numpy",
        "scipy",
        "h5py",
        "PyYAML",
        "codes",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    try:
        import torch

        cuda_runtime = torch.version.cuda
    except ImportError:
        cuda_runtime = None
    return {
        "python": platform.python_version(),
        "packages": versions,
        "gpu": gpu,
        "cuda_runtime": cuda_runtime,
    }


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate(protocol, dataset, phase="reference"):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    if protocol.get("models") != list(MODELS):
        raise ValueError(f"models must be {list(MODELS)!r}")
    seeds = protocol.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds)
    ):
        raise ValueError("seeds must contain three non-negative integers")
    settings = protocol.get("datasets", {}).get(dataset)
    if not isinstance(settings, dict):
        raise ValueError(f"Missing dataset settings for {dataset}")
    ensemble = positive(settings.get("ensemble_size"), "ensemble_size")
    if ensemble < 2:
        raise ValueError("ensemble_size must include at least two members")
    blocks = [set(range(s, s + ensemble)) for s in seeds]
    if any(s + ensemble - 1 > 2**32 - 1 for s in seeds):
        raise ValueError("ensemble seeds must fit in 32 bits")
    if any(blocks[i] & blocks[j] for i in range(3) for j in range(i)):
        raise ValueError("ensemble seed blocks overlap")
    for key in ("epochs", "batch_size"):
        vals = settings.get(key)
        if not isinstance(vals, dict) or set(vals) != set(MODELS):
            raise ValueError(f"{dataset}.{key} must specify all four models")
        for model in MODELS:
            if key == "epochs" and phase == "pilot" and vals[model] is None:
                continue
            positive(vals[model], f"{dataset}.{key}.{model}")
    if not isinstance(settings.get("checkpoint"), bool):
        raise ValueError(f"{dataset}.checkpoint must be explicitly true or false")
    ds = settings.get("dataset")
    if not isinstance(ds, dict) or ds.get("name") != dataset:
        raise ValueError(f"{dataset}.dataset.name must be {dataset}")
    for key in (
        "log10_transform",
        "log10_transform_params",
        "normalise",
        "normalise_per_species",
        "tolerance",
        "log_timesteps",
        "use_optimal_params",
        "subset_factor",
    ):
        if ds.get(key) is None:
            raise ValueError(f"{dataset}.dataset.{key} must be explicitly supplied")
    for key in (
        "log10_transform",
        "log10_transform_params",
        "normalise_per_species",
        "log_timesteps",
        "use_optimal_params",
    ):
        if not isinstance(ds[key], bool):
            raise ValueError(f"{dataset}.dataset.{key} must be a boolean")
    if ds["use_optimal_params"] is not True:
        raise ValueError("use_optimal_params must be true")
    if ds["normalise"] not in ("standardise", "minmax", "disable"):
        raise ValueError("Invalid normalise value")
    if (
        isinstance(ds["tolerance"], bool)
        or not isinstance(ds["tolerance"], (int, float))
        or ds["tolerance"] <= 0
    ):
        raise ValueError("tolerance must be positive")
    positive(ds["subset_factor"], f"{dataset}.dataset.subset_factor")
    pilot = protocol.get("pilot_epochs")
    if not isinstance(pilot, list) or len(pilot) < 2 or len(set(pilot)) != len(pilot):
        raise ValueError("pilot_epochs must have at least two distinct budgets")
    for x in pilot:
        positive(x, "pilot_epochs")
    if (
        not isinstance(protocol.get("devices"), list)
        or len(protocol["devices"]) != 1
        or not all(isinstance(x, str) and x for x in protocol["devices"])
    ):
        raise ValueError("devices must contain exactly one device")
    return settings


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def prepare(args):
    repo, work = Path(args.repo).resolve(), Path(args.workdir).resolve()
    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text())
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    settings = validate(protocol, args.dataset, args.phase)
    data_dir = (
        Path(args.data_root).resolve() if args.data_root else repo / "datasets"
    ) / args.dataset
    data = data_dir / "data.hdf5"
    hyper = repo / "datasets" / args.dataset / "surrogates_config.py"
    sources = repo / "datasets" / "data_sources.yaml"
    for path in (data, hyper, sources):
        if not path.is_file():
            raise ValueError(f"Required input file missing: {path}")
    ensure_clean_code(repo)
    revision, code_hash = code_identity(repo)
    units = []
    seeds = protocol["seeds"] if args.phase == "reference" else protocol["seeds"][:1]
    for model in MODELS:
        for seed in seeds:
            budgets = (
                [settings["epochs"][model]]
                if args.phase == "reference"
                else protocol["pilot_epochs"]
            )
            for epochs in budgets:
                training_id = f"{args.dataset}_{model}_{seed}" + (
                    f"_pilot_{epochs}" if args.phase == "pilot" else ""
                )
                config = {
                    "training_id": training_id,
                    "surrogates": [model],
                    "seed": seed,
                    "devices": protocol["devices"],
                    "epochs": [epochs],
                    "batch_size": [settings["batch_size"][model]],
                    "dataset": settings["dataset"],
                    "checkpoint": settings["checkpoint"],
                    "interpolation": {"enabled": False},
                    "extrapolation": {"enabled": False},
                    "sparse": {"enabled": False},
                    "batch_scaling": {"enabled": False},
                    "uncertainty": {
                        "enabled": args.phase == "reference",
                        "ensemble_size": settings["ensemble_size"],
                    },
                    "iterative": {"mode": "grid", "interval": 10},
                    "synchronize_timing": True,
                    "error_percentile": 99,
                    "relative_error_threshold": 0.0,
                    "losses": True,
                    "gradients": False,
                    "timing": True,
                    "compute": True,
                    "compare": False,
                }
                config_text = yaml.safe_dump(config, sort_keys=True)
                units.append(
                    {
                        "model": model,
                        "seed": seed,
                        "epochs": epochs,
                        "training_id": training_id,
                        "config": f"configs/{training_id}.yaml",
                        "config_sha256": hashlib.sha256(
                            config_text.encode()
                        ).hexdigest(),
                        "_text": config_text,
                    }
                )
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "phase": args.phase,
        "repo": str(repo),
        "workdir": str(work),
        "data_path": str(data.resolve()),
        "dataset_sha256": sha(data),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha(protocol_path),
        "hyperparameters_sha256": sha(hyper),
        "data_sources_sha256": sha(sources),
        "code_revision": revision,
        "code_sha256": code_hash,
        "software": software_identity(),
        "ensemble_size": settings["ensemble_size"],
        "units": units,
    }
    # Compare before writing anything: changing protocol, code, data, or software
    # must never silently reuse results from an earlier experiment.
    texts = [unit.pop("_text") for unit in units]
    target = work / "manifest.json"
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a fresh workdir")
        verify(work, manifest)
        return
    if work.exists() and any(work.iterdir()):
        raise ValueError("Workdir is not empty and has no manifest")
    (work / "configs").mkdir(parents=True, exist_ok=True)
    local_data = work / "datasets" / args.dataset
    local_data.mkdir(parents=True)
    (local_data / "data.hdf5").symlink_to(data.resolve())
    shutil.copy2(hyper, local_data / "surrogates_config.py")
    shutil.copy2(sources, work / "datasets" / "data_sources.yaml")
    shutil.copy2(protocol_path, work / "protocol.yaml")
    for unit, text in zip(units, texts):
        (work / unit["config"]).write_text(text)
    write_json(target, manifest)


def verify(work, manifest):
    repo = Path(manifest["repo"])
    rev, digest = code_identity(repo, require_revision=False)
    checks = [
        (
            (rev is None or rev == manifest["code_revision"])
            and digest == manifest["code_sha256"],
            "code",
        ),
        (str(work.resolve()) == manifest["workdir"], "workdir"),
        (sha(work / "protocol.yaml") == manifest["protocol_sha256"], "protocol"),
        (
            sha(work / "datasets" / manifest["dataset"] / "data.hdf5")
            == manifest["dataset_sha256"],
            "dataset",
        ),
        (
            sha(work / "datasets" / manifest["dataset"] / "surrogates_config.py")
            == manifest["hyperparameters_sha256"],
            "hyperparameters",
        ),
        (
            sha(work / "datasets" / "data_sources.yaml")
            == manifest["data_sources_sha256"],
            "data sources",
        ),
    ]
    for unit in manifest["units"]:
        checks.append(
            (sha(work / unit["config"]) == unit["config_sha256"], unit["config"])
        )
    for ok, name in checks:
        if not ok:
            raise ValueError(f"Experiment identity changed: {name}")


def run(args):
    work = Path(args.workdir).resolve()
    with (work / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Another benchmark process holds the workdir lock"
            ) from None
        run_locked(work)


def run_locked(work):
    manifest = json.loads((work / "manifest.json").read_text())
    verify(work, manifest)
    state_path = work / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    runtime = software_identity()
    if any(
        runtime.get(key) != manifest["software"].get(key)
        for key in ("python", "packages", "cuda_runtime")
    ):
        raise ValueError("Runtime software differs from preparation environment")
    if state.get("_software") and state["_software"] != runtime:
        raise ValueError("Runtime software or hardware differs from the previous run")
    state["_software"] = runtime
    write_json(state_path, state)
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTHONHASHSEED", "0")
    env["PYTHONPATH"] = manifest["repo"] + os.pathsep + env.get("PYTHONPATH", "")
    for unit in manifest["units"]:
        uid = unit["training_id"]
        item = state.setdefault(uid, {})
        for stage, script in [("train", "run_training.py"), ("eval", "run_eval.py")]:
            model_dir = work / "trained" / uid / unit["model"]
            expected = (
                (
                    [model_dir / f"{unit['model'].lower()}_main.pth"]
                    + [
                        model_dir / f"{unit['model'].lower()}_UQ_{i}.pth"
                        for i in range(1, manifest["ensemble_size"])
                    ]
                    if manifest["phase"] == "reference"
                    else [model_dir / f"{unit['model'].lower()}_main.pth"]
                )
                if stage == "train"
                else [work / "results" / uid / f"{unit['model'].lower()}_metrics.yaml"]
            )
            if item.get(stage, {}).get("success"):
                if not all(path.is_file() for path in expected):
                    raise ValueError(
                        f"Missing output for completed {uid} {stage}; inspect the workdir"
                    )
                continue
            print(f"{uid}: starting {stage}", flush=True)
            started = time.monotonic()
            with (logs / f"{uid}_{stage}.log").open("a") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(manifest["repo"]) / script),
                        "--config",
                        str(work / unit["config"]),
                    ],
                    cwd=work,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    check=False,
                )
            complete = result.returncode == 0 and all(
                path.is_file() for path in expected
            )
            item[stage] = {
                "success": complete,
                "seconds": time.monotonic() - started,
                "returncode": result.returncode,
            }
            write_json(state_path, state)
            if not complete:
                raise RuntimeError(
                    f"{uid} {stage} failed; see {logs / (uid + '_' + stage + '.log')}"
                )


def estimate(args):
    work = Path(args.workdir).resolve()
    manifest = json.loads((work / "manifest.json").read_text())
    if manifest["phase"] != "pilot":
        raise ValueError("estimate requires a pilot workdir")
    verify(work, manifest)
    state = json.loads((work / "state.json").read_text())
    protocol = yaml.safe_load((work / "protocol.yaml").read_text())
    if args.protocol:
        reference_protocol = yaml.safe_load(Path(args.protocol).read_text())
        settings = validate(reference_protocol, manifest["dataset"])
        # Only final epoch budgets may differ from the measured pilot protocol.
        pilot_comparable = json.loads(json.dumps(protocol))
        reference_comparable = json.loads(json.dumps(reference_protocol))
        pilot_comparable["datasets"][manifest["dataset"]]["epochs"] = (
            reference_comparable["datasets"][manifest["dataset"]]["epochs"]
        )
        if pilot_comparable != reference_comparable:
            raise ValueError(
                "Reference protocol may differ from the pilot only in final epoch budgets"
            )
    else:
        settings = validate(protocol, manifest["dataset"])
    estimates = {}
    for model in MODELS:
        units = sorted(
            (x for x in manifest["units"] if x["model"] == model),
            key=lambda x: x["epochs"],
        )
        if any(
            not state.get(u["training_id"], {}).get(stage, {}).get("success")
            for u in units
            for stage in ("train", "eval")
        ):
            raise ValueError(f"Complete all pilot units for {model} first")
        first, last = units[0], units[-1]
        lo = state[first["training_id"]]["train"]["seconds"]
        hi = state[last["training_id"]]["train"]["seconds"]
        slope = max(0, (hi - lo) / (last["epochs"] - first["epochs"]))
        overhead = max(0, lo - slope * first["epochs"])
        # Use the larger observed per-epoch rate as a conservative alternative.
        one = max(
            overhead + slope * settings["epochs"][model],
            hi / last["epochs"] * settings["epochs"][model],
        )
        estimates[model] = {
            "training_seconds": one
            * settings["ensemble_size"]
            * len(protocol["seeds"]),
            "evaluation_seconds_lower_bound": max(
                state[u["training_id"]]["eval"]["seconds"] for u in units
            )
            * len(protocol["seeds"]),
            "pilot_fixed_overhead_seconds": overhead,
            "pilot_seconds_per_epoch": slope,
        }
        estimates[model]["evaluation_budget_seconds"] = (
            estimates[model]["evaluation_seconds_lower_bound"]
            * settings["ensemble_size"]
        )
    print(
        json.dumps(
            {
                "warning": "Uncertain extrapolation: startup and validation may dominate short pilots. Ensemble evaluation is budgeted by scaling the single-model measurement by ensemble size; this is a heuristic, not a measured upper bound. Add a walltime margin and use the same GPU type.",
                "models": estimates,
                "total_budget_seconds": sum(
                    v["training_seconds"] + v["evaluation_budget_seconds"]
                    for v in estimates.values()
                ),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--protocol", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument(
        "--data-root",
        help="Directory containing dataset-name/data.hdf5 (defaults to repo/datasets)",
    )
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--phase", choices=("pilot", "reference"), required=True)
    p.set_defaults(func=prepare)
    for name, func in [("run", run), ("estimate", estimate)]:
        p = sub.add_parser(name)
        p.add_argument("--workdir", required=True)
        if name == "estimate":
            p.add_argument(
                "--protocol", help="Final protocol with confirmed epoch budgets"
            )
        p.set_defaults(func=func)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
