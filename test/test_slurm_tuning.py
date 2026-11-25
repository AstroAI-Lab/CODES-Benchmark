import json
from pathlib import Path

import optuna
import pytest
from optuna.trial import TrialState

from codes.tune import slurm


@pytest.fixture
def config():
    return {
        "tuning_id": "slurm_example",
        "seed": 42,
        "dataset": {"name": "dummy", "use_optimal_params": False},
        "time_pruning": False,
        "prune": False,
        "surrogates": [
            {
                "name": "FullyConnected",
                "epochs": 2,
                "batch_size": 8,
                "trials": 2,
                "optuna_params": {"hidden_size": {"type": "int", "low": 2, "high": 4}},
            },
            {"name": "MultiONet", "epochs": 2, "batch_size": 8, "trials": 2},
        ],
        "postgres_config": {"password": "top-secret-token-409"},
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "datasets" / "dummy").mkdir(parents=True)
    (tmp_path / "datasets" / "dummy" / "data.hdf5").touch()
    storage = optuna.storages.InMemoryStorage()
    monkeypatch.setattr(slurm, "_storage", lambda: storage)
    monkeypatch.setattr(
        slurm,
        "validate_execution_settings",
        lambda device, threads: __import__("torch").device(device),
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    return tmp_path / "study.json", storage


def test_prepare_freezes_config_and_rejects_changes(config, setup):
    path, storage = setup
    with pytest.raises(ValueError, match="Select exactly one"):
        slurm.prepare(config, path, None, None)
    slurm.prepare(config, path, "FullyConnected", None)
    data = json.loads(path.read_text())
    assert data["study_name"] == "slurm_example_fullyconnected"
    assert data["max_attempts"] == 4
    assert "top-secret-token-409" not in path.read_text()
    slurm.prepare(config, path, "FullyConnected", None)
    config["surrogates"][0]["epochs"] = 3
    with pytest.raises(ValueError, match="different configuration"):
        slurm.prepare(config, path, "FullyConnected", None)
    study = optuna.load_study(study_name=data["study_name"], storage=storage)
    assert study.user_attrs[slurm.STUDY_FINGERPRINT]


def test_submit_prepares_selected_study_and_uses_worker_count(
    config, setup, monkeypatch
):
    from types import SimpleNamespace

    import run_tuning

    path, _ = setup
    config["slurm"] = {"workers": 3, "cpus_per_task": 2}
    submitted = []
    monkeypatch.setattr(
        "codes.slurm.submit_array",
        lambda manifest, indices, options, worker_args: (
            submitted.append((manifest, indices, options, worker_args)) or "7654"
        ),
    )
    monkeypatch.setattr(
        run_tuning,
        "parse_arguments",
        lambda: SimpleNamespace(
            config="optuna.yaml",
            slurm="submit",
            manifest=None,
            study="FullyConnected",
            max_attempts=None,
            device="cpu",
            cpu_threads=1,
        ),
    )
    monkeypatch.setattr(run_tuning, "load_yaml_config", lambda _: config)
    run_tuning.main()
    manifest, indices, options, worker_args = submitted[0]
    assert (
        manifest
        == Path("tuned/slurm_example/slurm_example_fullyconnected.slurm.json").resolve()
    )
    assert indices == [0, 1, 2]
    assert options.cpus_per_task == 2
    assert worker_args[worker_args.index("--cpu-threads") + 1] == "2"
    assert "slurm" not in json.loads(manifest.read_text())["config"]


def test_singular_config_uses_model_suffixed_study_name(config, setup):
    path, storage = setup
    config.pop("surrogates")
    config["surrogate"] = {"name": "FullyConnected"}
    slurm.prepare(config, path, None, None)
    data = json.loads(path.read_text())
    assert data["study_name"] == "slurm_example_fullyconnected"
    assert optuna.load_study(study_name="slurm_example_fullyconnected", storage=storage)


def test_worker_completes_target_and_resumes(config, setup, monkeypatch):
    path, storage = setup
    slurm.prepare(config, path, "FullyConnected", 4)
    observed = []

    def make_objective(config, study, devices):
        def objective(trial):
            observed.append(trial.number)
            return 0.5

        return objective

    monkeypatch.setattr(slurm, "create_objective", make_objective)
    slurm.worker(path, "cpu", 1)
    slurm.worker(path, "cpu", 1)
    study = optuna.load_study(
        study_name="slurm_example_fullyconnected", storage=storage
    )
    assert observed == [0, 1]
    assert slurm._counts(study)["valid"] == 2
    assert len(study.trials) == 2
    assert slurm.status(path)["valid"] == 2


def test_worker_limits_failed_attempts(config, setup, monkeypatch):
    path, storage = setup
    slurm.prepare(config, path, "FullyConnected", 2)
    observed = []

    def make_objective(config, study, devices):
        def objective(trial):
            observed.append(trial.number)
            trial.set_user_attr("exception", "simulated OOM")
            return 20.0

        return objective

    monkeypatch.setattr(slurm, "create_objective", make_objective)
    slurm.worker(path, "cpu", 1)
    slurm.worker(path, "cpu", 1)
    study = optuna.load_study(
        study_name="slurm_example_fullyconnected", storage=storage
    )
    assert observed == [0, 1]
    assert slurm._counts(study)["valid"] == 0
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


def test_preexisting_unmarked_study_and_modified_manifest_rejected(config, setup):
    path, storage = setup
    existing = optuna.create_study(
        study_name="slurm_example_fullyconnected", storage=storage
    )
    existing.optimize(lambda trial: 1.0, n_trials=1)
    with pytest.raises(ValueError, match="no Slurm manifest fingerprint"):
        slurm.prepare(config, path, "FullyConnected", None)
    assert not path.exists()


def test_unmarked_study_rejects_existing_surrogate_checkpoints(config, setup):
    path, storage = setup
    slurm.prepare(config, path.parent / "first.json", "FullyConnected", None)
    checkpoints = path.parent / "tuned" / "slurm_example" / "models" / "MultiONet"
    checkpoints.mkdir(parents=True)
    (checkpoints / "multionet_0").write_text("existing checkpoint")
    optuna.create_study(
        study_name="slurm_example_multionet", storage=storage, load_if_exists=True
    )

    with pytest.raises(ValueError, match="overwrite existing checkpoints"):
        slurm.prepare(config, path, "MultiONet", None)

    study = optuna.load_study(study_name="slurm_example_multionet", storage=storage)
    assert slurm.STUDY_FINGERPRINT not in study.user_attrs
    assert (checkpoints / "multionet_0").read_text() == "existing checkpoint"


def test_requires_postgres_and_array_index(config, setup, monkeypatch):
    path, _ = setup
    slurm.prepare(config, path, "FullyConnected", None)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    with pytest.raises(ValueError, match="SLURM_ARRAY_TASK_ID"):
        slurm.worker(path, "cpu", 1)
    monkeypatch.undo()
    monkeypatch.setenv("CODES_OPTUNA_STORAGE_URL", "sqlite:///test.db")
    with pytest.raises(ValueError, match="PostgreSQL"):
        slurm._storage()


def test_missing_staged_data_does_not_create_trial(config, setup, monkeypatch):
    path, storage = setup
    slurm.prepare(config, path, "FullyConnected", None)
    (path.parent / "datasets" / "dummy" / "data.hdf5").unlink()
    with pytest.raises(FileNotFoundError, match="not staged"):
        slurm.worker(path, "cpu", 1)
    study = optuna.load_study(
        study_name="slurm_example_fullyconnected", storage=storage
    )
    assert study.trials == []


def test_wrong_existing_study_directions(config, setup):
    path, storage = setup
    optuna.create_study(
        study_name="slurm_example_fullyconnected", storage=storage, direction="maximize"
    )
    with pytest.raises(ValueError, match="objective directions"):
        slurm.prepare(config, path, "FullyConnected", None)


@pytest.mark.parametrize("tuning_id", [".", "..", "", "../bad"])
def test_reject_unsafe_tuning_id(config, tuning_id):
    config["tuning_id"] = tuning_id
    with pytest.raises(ValueError, match="directory name"):
        slurm._study_config(config, "FullyConnected")


def test_model_configuration_change_rejected_before_trial(config, setup):
    path, storage = setup
    config["dataset"]["use_optimal_params"] = True
    model_config = path.parent / "datasets" / "dummy" / "surrogates_config.py"
    model_config.write_text("# version 1\n")
    slurm.prepare(config, path, "FullyConnected", None)
    model_config.write_text("# version 2\n")
    with pytest.raises(ValueError, match="configuration changed"):
        slurm.worker(path, "cpu", 1)
    study = optuna.load_study(
        study_name="slurm_example_fullyconnected", storage=storage
    )
    assert study.trials == []


def test_missing_array_index_does_not_start_training(config, setup, monkeypatch):
    path, storage = setup
    slurm.prepare(config, path, "FullyConnected", None)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "-1")
    with pytest.raises(ValueError, match="SLURM_ARRAY_TASK_ID"):
        slurm.worker(path, "cpu", 1)
    study = optuna.load_study(
        study_name="slurm_example_fullyconnected", storage=storage
    )
    assert study.trials == []


def test_prepares_shared_workspace_for_all_models_without_credentials(config, setup):
    path, storage = setup
    first = path.parent / "first.json"
    second = path.parent / "second.json"
    slurm.prepare(config, first, "FullyConnected", None)
    saved = path.parent / "tuned" / "slurm_example" / "optuna_config.yaml"
    contents = saved.read_text()
    assert "top-secret-token-409" not in contents
    assert "postgres_config" not in contents
    slurm.prepare(config, second, "MultiONet", None)
    assert saved.read_text() == contents
    assert optuna.load_study(study_name="slurm_example_multionet", storage=storage)
    config["seed"] = 99
    with pytest.raises(ValueError, match="different configuration"):
        slurm.prepare(config, path.parent / "third.json", "MultiONet", None)
    assert saved.read_text() == contents


def test_rejects_unrelated_workspace_before_initializing_study(config, setup):
    path, storage = setup
    folder = path.parent / "tuned" / "slurm_example"
    folder.mkdir(parents=True)
    (folder / "other").write_text("retain")
    with pytest.raises(ValueError, match="has no configuration"):
        slurm.prepare(config, path, "FullyConnected", None)
    assert optuna.get_all_study_summaries(storage=storage) == []
    assert (folder / "other").read_text() == "retain"


def test_evaluate_tuning_uses_cluster_database_without_admin(
    config, setup, monkeypatch, capsys
):
    from codes.tune import evaluate_tuning

    path, storage = setup
    slurm.prepare(config, path, "FullyConnected", None)
    monkeypatch.setenv(
        "CODES_OPTUNA_STORAGE_URL", "postgresql://user:secret@example/db"
    )
    monkeypatch.setattr(
        evaluate_tuning.psycopg2,
        "connect",
        lambda **kwargs: pytest.fail("database admin attempted"),
    )
    called = []
    monkeypatch.setattr(
        optuna.study,
        "get_all_study_summaries",
        lambda storage: called.append(storage) or [],
    )
    evaluate_tuning.evaluate_tuning("slurm_example")
    assert called == ["postgresql://user:secret@example/db"]
    assert "secret" not in capsys.readouterr().out


def test_legacy_study_config_reader_resolves_slurm_study(config, setup):
    from codes.tune.evaluate_study import load_study_config

    path, _ = setup
    slurm.prepare(config, path, "FullyConnected", None)
    loaded = load_study_config("slurm_example_fullyconnected")
    assert loaded["surrogate"]["name"] == "FullyConnected"
    assert loaded["dataset"] == config["dataset"]


def test_new_dataset_model_config_rejected_before_trial(config, setup):
    path, storage = setup
    config["dataset"]["use_optimal_params"] = True
    model_config = path.parent / "datasets" / "dummy" / "surrogates_config.py"
    slurm.prepare(config, path, "FullyConnected", None)
    model_config.write_text("# newly added\n")
    with pytest.raises(ValueError, match="configuration changed"):
        slurm.worker(path, "cpu", 1)
    study = optuna.load_study(
        study_name="slurm_example_fullyconnected", storage=storage
    )
    assert study.trials == []


def test_model_configuration_rechecked_before_each_objective(
    config, setup, monkeypatch
):
    path, storage = setup
    config["dataset"]["use_optimal_params"] = True
    model_config = path.parent / "datasets" / "dummy" / "surrogates_config.py"
    model_config.write_text("# version 1\n")
    slurm.prepare(config, path, "FullyConnected", None)
    observed = []

    def make_objective(config, study, devices):
        def objective(trial):
            observed.append(trial.number)
            model_config.write_text("# version 2\n")
            return 0.5

        return objective

    monkeypatch.setattr(slurm, "create_objective", make_objective)
    with pytest.raises(ValueError, match="configuration changed"):
        slurm.worker(path, "cpu", 1)
    assert observed == [0]
    study = optuna.load_study(
        study_name="slurm_example_fullyconnected", storage=storage
    )
    assert len(study.trials) == 2
    assert study.trials[0].state == TrialState.COMPLETE
    assert study.trials[1].state == TrialState.FAIL
