import fcntl
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks import reference


@pytest.mark.parametrize("packed", [False, True])
def test_revision_without_executable_git(tmp_path, monkeypatch, packed):
    gitdir = tmp_path / ".git"
    gitdir.mkdir()
    (gitdir / "HEAD").write_text("ref: refs/heads/reference\n")
    revision = "a" * 40
    if packed:
        (gitdir / "packed-refs").write_text(f"{revision} refs/heads/reference\n")
    else:
        (gitdir / "refs/heads").mkdir(parents=True)
        (gitdir / "refs/heads/reference").write_text(revision + "\n")

    def unavailable(*args, **kwargs):
        raise PermissionError("git unavailable on compute node")

    monkeypatch.setattr(reference.subprocess, "check_output", unavailable)
    assert reference.git_revision(tmp_path) == revision


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "datasets" / "primordial").mkdir(parents=True)
    (repo / "datasets" / "primordial" / "data.hdf5").write_bytes(b"toy data")
    (repo / "datasets" / "primordial" / "surrogates_config.py").write_text("# toy\n")
    (repo / "datasets" / "data_sources.yaml").write_text("{}\n")
    monkeypatch.setattr(reference, "ensure_clean_code", lambda repo: None)
    monkeypatch.setattr(
        reference, "code_identity", lambda repo, require_revision=True: ("abc", "123")
    )
    monkeypatch.setattr(reference, "software_identity", lambda: {"gpu": "toy"})
    settings = {
        "dataset": {
            "name": "primordial",
            "log10_transform": True,
            "log10_transform_params": False,
            "normalise": "minmax",
            "normalise_per_species": False,
            "tolerance": 1e-25,
            "log_timesteps": False,
            "use_optimal_params": True,
            "subset_factor": 1,
        },
        "epochs": {m: 8 for m in reference.MODELS},
        "batch_size": {m: 2 for m in reference.MODELS},
        "ensemble_size": 5,
        "checkpoint": False,
    }
    protocol = {
        "models": list(reference.MODELS),
        "seeds": [42, 142, 242],
        "pilot_epochs": [2, 4],
        "devices": ["cpu"],
        "datasets": {"primordial": settings},
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    args = Namespace(
        repo=str(repo),
        workdir=str(tmp_path / "work"),
        protocol=str(path),
        dataset="primordial",
        phase="reference",
        data_root=None,
    )
    return args, protocol


def test_manifest_immutable_and_disjoint(experiment):
    args, protocol = experiment
    reference.prepare(args)
    work = Path(args.workdir)
    manifest = json.loads((work / "manifest.json").read_text())
    assert len(manifest["units"]) == 12
    assert (work / "datasets/primordial/data.hdf5").is_symlink()
    config = yaml.safe_load((work / manifest["units"][0]["config"]).read_text())
    assert config["uncertainty"] == {"enabled": True, "ensemble_size": 5}
    assert config["iterative"]["mode"] == "grid"
    reference.prepare(args)
    (work / manifest["units"][0]["config"]).write_text("changed")
    with pytest.raises(ValueError, match="identity changed"):
        reference.prepare(args)
    protocol["seeds"] = [42, 44, 242]
    with pytest.raises(ValueError, match="overlap"):
        reference.validate(protocol, "primordial")


def test_failure_resume_skips_completed_steps(experiment, monkeypatch):
    args, _ = experiment
    reference.prepare(args)
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        if len(calls) != 2:
            fake_outputs(command, kwargs["cwd"])
        return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)

    monkeypatch.setattr(reference.subprocess, "run", invoke)
    with pytest.raises(RuntimeError, match="eval failed"):
        reference.run(args)
    state = json.loads((Path(args.workdir) / "state.json").read_text())
    uid = next(key for key in state if key != "_software")
    assert state[uid]["train"]["success"]
    assert not state[uid]["eval"]["success"]
    calls.clear()

    def success(command, **kwargs):
        calls.append(command)
        fake_outputs(command, kwargs["cwd"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reference.subprocess, "run", success)
    reference.run(args)
    assert len(calls) == 23
    assert calls[0][1].endswith("run_eval.py")


def test_pilot_allows_unset_reference_epochs(experiment):
    args, protocol = experiment
    args.phase = "pilot"
    protocol["datasets"]["primordial"]["epochs"] = dict.fromkeys(reference.MODELS)
    Path(args.protocol).write_text(yaml.safe_dump(protocol))
    reference.prepare(args)
    manifest = json.loads((Path(args.workdir) / "manifest.json").read_text())
    assert len(manifest["units"]) == 8
    assert {u["seed"] for u in manifest["units"]} == {42}
    config = yaml.safe_load(
        (Path(args.workdir) / manifest["units"][0]["config"]).read_text()
    )
    assert config["uncertainty"]["enabled"] is False
    with pytest.raises(ValueError, match="epochs"):
        reference.validate(protocol, "primordial")


def test_concurrent_runs_rejected(experiment):
    args, _ = experiment
    reference.prepare(args)
    with (Path(args.workdir) / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another benchmark process"):
            reference.run(args)


@pytest.mark.parametrize("input_name", ["data.hdf5", "surrogates_config.py"])
def test_resume_rejects_changed_scientific_inputs(experiment, input_name):
    args, _ = experiment
    reference.prepare(args)
    path = Path(args.workdir) / "datasets/primordial" / input_name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        reference.run(args)


def test_pilot_estimate_counts_all_replicas_and_ensemble_members(experiment, capsys):
    args, _ = experiment
    args.phase = "pilot"
    reference.prepare(args)
    work = Path(args.workdir)
    manifest = json.loads((work / "manifest.json").read_text())
    state = {
        unit["training_id"]: {
            "train": {"success": True, "seconds": 10 + unit["epochs"]},
            "eval": {"success": True, "seconds": 3},
        }
        for unit in manifest["units"]
    }
    (work / "state.json").write_text(json.dumps(state))
    reference.estimate(Namespace(workdir=str(work), protocol=None))
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["total_budget_seconds"] == 1860
    for values in estimate["models"].values():
        assert values["training_seconds"] == 420
        assert values["evaluation_seconds_lower_bound"] == 9
        assert values["evaluation_budget_seconds"] == 45


def test_supplied_paper_protocol_is_ready_and_preserves_cloud_subset():
    protocol = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/benchmark/reference.yaml").read_text()
    )
    for dataset in reference.DATASETS:
        reference.validate(protocol, dataset)
    assert protocol["datasets"]["cloud_parametric"]["dataset"]["subset_factor"] == 2
    assert protocol["datasets"]["primordial"]["epochs"]["MultiONet"] == 12000
    assert protocol["datasets"]["primordial"]["dataset"]["tolerance"] == 1e-15


def fake_outputs(command, work):
    config = yaml.safe_load(Path(command[-1]).read_text())
    uid, model = config["training_id"], config["surrogates"][0]
    if command[1].endswith("run_training.py"):
        folder = work / "trained" / uid / model
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{model.lower()}_main.pth").touch()
        if config["uncertainty"]["enabled"]:
            for i in range(1, config["uncertainty"]["ensemble_size"]):
                (folder / f"{model.lower()}_UQ_{i}.pth").touch()
    else:
        folder = work / "results" / uid
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{model.lower()}_metrics.yaml").touch()
