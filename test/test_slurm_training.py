"""Array training preparation, isolation and restart behavior."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import run_training
from codes.train import slurm


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "training_id": "array-run",
        "surrogates": ["FullyConnected"],
        "batch_size": 8,
        "epochs": 2,
        "devices": ["cpu"],
        "dataset": {"name": "test"},
        "uncertainty": {"enabled": True, "ensemble_size": 3},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    data = tmp_path / "datasets/test/data.hdf5"
    data.parent.mkdir(parents=True)
    data.touch()
    return config, config_path, tmp_path / "manifest.json"


def test_prepare_and_retry_completed_tasks(setup_run, monkeypatch, capsys):
    config, config_path, manifest_path = setup_run
    slurm.prepare_training(str(config_path), str(manifest_path))
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    assert [task[4] for task in manifest["tasks"]] == [42, 43, 44]
    slurm.prepare_training(str(config_path), str(manifest_path))
    assert manifest_path.read_bytes() == original
    calls = []

    def fake_execute(task, *, device, cpu_threads):
        calls.append((task, device, cpu_threads))
        checkpoint = slurm._checkpoint(config, task)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
        checkpoint.with_suffix(".yaml").touch()

    monkeypatch.setattr(slurm, "execute_training_task", fake_execute)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    slurm.run_worker(str(manifest_path))
    slurm.run_worker(str(manifest_path))
    assert len(calls) == 1
    assert calls[0][1:] == ("cpu", 1)
    slurm.training_status(str(manifest_path))
    assert "Completed: 1/3" in capsys.readouterr().out
    checkpoint = slurm._checkpoint(config, calls[0][0])
    checkpoint.unlink()
    slurm.run_worker(str(manifest_path))
    assert len(calls) == 2


def test_failed_work_can_retry_and_conflicting_runs_rejected(setup_run, monkeypatch):
    config, config_path, manifest_path = setup_run
    slurm.prepare_training(str(config_path), str(manifest_path))
    with pytest.raises(RuntimeError, match="Slurm manifest"):
        run_training.main(SimpleNamespace(config=str(config_path), slurm=None))
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(slurm, "execute_training_task", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="did not produce"):
        slurm.run_worker(str(manifest_path))
    assert not (Path("trained/array-run/slurm_tasks") / "0.json").exists()
    config["epochs"] = 3
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="different training config"):
        slurm.prepare_training(str(config_path), str(manifest_path))
    with pytest.raises(FileExistsError):
        slurm.prepare_training(str(config_path), "other.json")


def test_manifest_tampering_and_bad_indices(setup_run, monkeypatch):
    _, config_path, manifest_path = setup_run
    slurm.prepare_training(str(config_path), str(manifest_path))
    for value in ("-1", "3", "invalid"):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", value)
        with pytest.raises(ValueError):
            slurm.run_worker(str(manifest_path))
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    with pytest.raises(ValueError, match="required"):
        slurm.run_worker(str(manifest_path))
    manifest_path.write_text(manifest_path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        slurm.training_status(str(manifest_path))


def test_collision_is_rejected(setup_run):
    config, config_path, manifest_path = setup_run
    config["batch_scaling"] = {"enabled": True, "sizes": [0.5, 0.51]}
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="same checkpoint"):
        slurm.prepare_training(str(config_path), str(manifest_path))
    assert not Path("trained/array-run").exists()


def test_training_locks_block_concurrent_local_and_duplicate_workers(setup_run):
    _, config_path, manifest_path = setup_run
    with (
        slurm.training_lock("array-run"),
        pytest.raises(RuntimeError, match="already active"),
    ):
        slurm.prepare_training(str(config_path), str(manifest_path))
    slurm.prepare_training(str(config_path), str(manifest_path))
    with (
        slurm.training_lock("array-run", shared=True),
        pytest.raises(RuntimeError, match="already active"),
        slurm.training_lock("array-run"),
    ):
        pass


def test_worker_executes_real_training(setup_run, monkeypatch):
    import h5py
    import numpy as np

    config, config_path, manifest_path = setup_run
    config["epochs"] = 1
    config["uncertainty"]["enabled"] = False
    config["dataset"]["use_optimal_params"] = False
    config_path.write_text(yaml.safe_dump(config))
    data_path = Path("datasets/test/data.hdf5")
    with h5py.File(data_path, "w") as data:
        rng = np.random.default_rng(1)
        for split in ("train", "test", "val"):
            data.create_dataset(split, data=rng.uniform(0.1, 1.0, size=(4, 4, 2)))
        data.create_dataset("timesteps", data=np.linspace(0.0, 1.0, 4))
    slurm.prepare_training(str(config_path), str(manifest_path))
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    slurm.run_worker(str(manifest_path))
    checkpoint = Path("trained/array-run/FullyConnected/fullyconnected_main.pth")
    assert checkpoint.is_file()
    assert checkpoint.with_suffix(".yaml").is_file()
    assert Path("trained/array-run/slurm_tasks/0.json").is_file()


def test_duplicate_worker_is_excluded(setup_run, monkeypatch):
    import fcntl

    _, config_path, manifest_path = setup_run
    slurm.prepare_training(str(config_path), str(manifest_path))
    state_dir = Path("trained/array-run/slurm_tasks")
    state_dir.mkdir()
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    with (state_dir / "0.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="already running"):
            slurm.run_worker(str(manifest_path))


def test_competing_manifest_path_cannot_be_overwritten(setup_run):
    config, config_path, manifest_path = setup_run
    slurm.prepare_training(str(config_path), str(manifest_path))
    before = manifest_path.read_bytes()
    config["training_id"] = "other-run"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(FileExistsError, match="Manifest path already exists"):
        slurm.prepare_training(str(config_path), str(manifest_path))
    assert manifest_path.read_bytes() == before
    assert not Path("trained/other-run").exists()


def test_prepare_recovers_after_interruption(setup_run, monkeypatch):
    _, config_path, manifest_path = setup_run
    write_json = slurm._atomic_json

    def interrupt_owner(path, value, *, exclusive=False):
        if path.name == "slurm_manifest.json":
            raise KeyboardInterrupt()
        return write_json(path, value, exclusive=exclusive)

    monkeypatch.setattr(slurm, "_atomic_json", interrupt_owner)
    with pytest.raises(KeyboardInterrupt):
        slurm.prepare_training(str(config_path), str(manifest_path))
    before = manifest_path.read_bytes()
    with pytest.raises(RuntimeError, match="Slurm manifest"):
        run_training.main(SimpleNamespace(config=str(config_path), slurm=None))
    monkeypatch.setattr(slurm, "_atomic_json", write_json)
    slurm.prepare_training(str(config_path), str(manifest_path))
    assert manifest_path.read_bytes() == before
    assert Path("trained/array-run/slurm_manifest.json").is_file()
    assert not Path("trained/array-run/slurm_initializing.json").exists()


@pytest.mark.parametrize(
    ("initial", "updated"),
    [
        (None, "# version 2\n"),
        ("# version 1\n", None),
        ("# version 1\n", "# version 2\n"),
    ],
)
def test_model_config_changes_are_rejected_before_training(
    setup_run, monkeypatch, initial, updated
):
    _, config_path, manifest_path = setup_run
    model_config = Path("datasets/test/surrogates_config.py")
    if initial is not None:
        model_config.write_text(initial)
    slurm.prepare_training(str(config_path), str(manifest_path))
    if updated is None:
        model_config.unlink()
    else:
        model_config.write_text(updated)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(
        slurm, "execute_training_task", lambda *a, **kw: pytest.fail("must not train")
    )
    with pytest.raises(ValueError, match="surrogate configuration changed"):
        slurm.run_worker(str(manifest_path))
    with pytest.raises(ValueError, match="surrogate configuration changed"):
        slurm.prepare_training(str(config_path), str(manifest_path))


def test_disabled_dataset_config_does_not_affect_workers(setup_run, monkeypatch):
    config, config_path, manifest_path = setup_run
    config["dataset"]["use_optimal_params"] = False
    config_path.write_text(yaml.safe_dump(config))
    slurm.prepare_training(str(config_path), str(manifest_path))
    Path("datasets/test/surrogates_config.py").write_text("# irrelevant\n")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    called = []
    monkeypatch.setattr(
        slurm, "execute_training_task", lambda *a, **kw: called.append(True)
    )
    with pytest.raises(RuntimeError, match="did not produce"):
        slurm.run_worker(str(manifest_path))
    assert called == [True]
