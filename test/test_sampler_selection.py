import optuna
import pytest

from codes.tune.tune_utils import create_sampler, sampler_name


def test_sampler_defaults_follow_objective_type():
    assert sampler_name({"multi_objective": False}) == "tpe"
    assert isinstance(
        create_sampler({"multi_objective": False}, 7), optuna.samplers.TPESampler
    )
    assert sampler_name({"multi_objective": True}) == "nsgaii"
    assert isinstance(
        create_sampler({"multi_objective": True, "population_size": 12}, 7),
        optuna.samplers.NSGAIISampler,
    )


def test_tpe_can_be_selected_for_multiobjective():
    config = {"multi_objective": True, "sampler": "tpe"}
    assert sampler_name(config) == "tpe"
    assert isinstance(create_sampler(config, 7), optuna.samplers.TPESampler)


def test_nsgaii_requires_multiobjective():
    with pytest.raises(ValueError, match="requires multi_objective"):
        sampler_name({"multi_objective": False, "sampler": "nsgaii"})


def test_local_multiobjective_tpe_runs_and_refuses_sampler_change(monkeypatch):
    import run_tuning

    storage = optuna.storages.InMemoryStorage()
    config = {
        "multi_objective": True,
        "sampler": "tpe",
        "seed": 5,
        "n_trials": 2,
        "dataset": {"name": "dummy"},
        "time_pruning": False,
    }

    def objective(trial):
        assert isinstance(trial.study.sampler, optuna.samplers.TPESampler)
        trial.set_user_attr(
            "inference_timing", trial.study.user_attrs["inference_timing"]
        )
        return trial.suggest_float("x", 0, 1), 0.01

    monkeypatch.setattr(run_tuning, "download_data", lambda _: None)
    monkeypatch.setattr(run_tuning, "create_objective", lambda *a: objective)
    run_tuning.run_single_study(config, "test", storage, False)
    run_tuning.run_single_study(config, "test", storage, False)
    study = optuna.load_study(study_name="test", storage=storage)
    assert len(study.trials) == 2
    assert study.user_attrs["codes_sampler"] == "tpe"
    with pytest.raises(ValueError, match="created with sampler 'tpe'"):
        run_tuning.run_single_study(
            config | {"sampler": "nsgaii"}, "test", storage, False
        )


def test_legacy_multiobjective_study_rejects_tpe_without_mutating_attrs(tmp_path):
    import run_tuning

    storage = f"sqlite:///{tmp_path / 'legacy.db'}"
    study = optuna.create_study(
        study_name="legacy",
        storage=storage,
        directions=["minimize", "minimize"],
    )
    study.tell(study.ask(), (0.1, 0.2))
    config = {
        "multi_objective": True,
        "sampler": "tpe",
        "seed": 5,
        "population_size": 8,
        "n_trials": 2,
    }
    with pytest.raises(ValueError, match="historical 'nsgaii'"):
        run_tuning.run_single_study(config, "legacy", storage, True)
    assert optuna.load_study(study_name="legacy", storage=storage).user_attrs == {}
