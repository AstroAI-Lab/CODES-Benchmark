import math
import queue
from datetime import datetime, timedelta
from types import SimpleNamespace

import optuna
import pytest
from optuna.trial import TrialState

import codes.surrogates.AbstractSurrogate.abstract_surrogate as abstract_surrogate
from codes.surrogates import FullyConnected
from codes.tune.optuna_fcts import (
    MODULE_REGISTRY,
    create_objective,
    make_optuna_params,
    maybe_set_runtime_threshold,
)


class DummyTrial:
    def __init__(self):
        self.suggested = {}

    def suggest_int(self, name, low, high, step=1):
        # always return low
        self.suggested[name] = low
        return low

    def suggest_float(self, name, low, high, log=False):
        # always return high
        self.suggested[name] = high
        return high

    def suggest_categorical(self, name, choices):
        # return first choice
        val = choices[0]
        self.suggested[name] = val
        return val


class DummyModel:
    def __init__(
        self, device, n_quantities, n_timesteps, n_parameters, config=None, **kwargs
    ):
        pass

    def to(self, device):
        pass

    def prepare_data(self, **kwargs):
        return "train_loader", "test_loader", None

    def fit(self, **kwargs):
        pass

    def predict(self, loader, leave_log=False):
        import torch

        t = torch.zeros((2, 4, 1))
        return t, t

    def save(self, **kwargs):
        pass


@pytest.fixture
def basic_params():
    return {
        "batch_size": {"type": "int", "low": 10, "high": 20, "step": 5},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.01, "log": True},
        "activation": {"choices": ["relu", "tanh", "identity"]},
        "loss_function": {"choices": ["mse", "smoothl1"]},
    }


@pytest.fixture
def conditional_params(basic_params):
    p = basic_params.copy()
    p["scheduler"] = {"choices": ["poly", "cosine"]}
    p["poly_power"] = {"type": "int", "low": 1, "high": 3}
    p["eta_min"] = {"type": "float", "low": 0.0, "high": 0.1}
    return p


def test_make_optuna_params_basic(basic_params):
    trial = DummyTrial()
    # only basic choices
    out = make_optuna_params(
        trial,
        {
            "batch_size": basic_params["batch_size"],
            "learning_rate": basic_params["learning_rate"],
            "activation": basic_params["activation"],
            "loss_function": basic_params["loss_function"],
        },
    )
    # int param should equal low
    assert out["batch_size"] == basic_params["batch_size"]["low"]
    # float param should equal high
    assert math.isclose(out["learning_rate"], basic_params["learning_rate"]["high"])
    # activation returns first choice
    expected_cls = MODULE_REGISTRY[basic_params["activation"]["choices"][0]]
    assert isinstance(out["activation"], expected_cls)
    # loss_function returns first choice
    expected_cls = MODULE_REGISTRY[basic_params["loss_function"]["choices"][0]]
    assert isinstance(out["loss_function"], expected_cls)


def test_make_optuna_params_conditional(conditional_params):
    trial = DummyTrial()
    # include scheduler to trigger poly branch
    params = conditional_params.copy()
    params["scheduler"]["choices"] = ["poly"]
    out = make_optuna_params(trial, params)
    # since scheduler chosen 'poly', poly_power must be suggested
    assert "poly_power" in out
    # cos branch
    trial2 = DummyTrial()
    p2 = conditional_params.copy()
    p2["scheduler"]["choices"] = ["cosine"]
    out2 = make_optuna_params(trial2, p2)
    assert "eta_min" in out2


@pytest.mark.parametrize("loss_name", ["mse", "smoothl1"])
def test_loss_beta_is_only_tuned_for_smooth_l1(loss_name):
    trial = DummyTrial()
    params = make_optuna_params(
        trial,
        {
            "loss_function": {"choices": [loss_name]},
            "beta": {"type": "float", "low": 0.1, "high": 2.5},
        },
    )
    assert isinstance(params["loss_function"], MODULE_REGISTRY[loss_name])
    if loss_name == "smoothl1":
        assert params["beta"] == 2.5
        assert trial.suggested["beta"] == 2.5
    else:
        assert "beta" not in params
        assert "beta" not in trial.suggested


@pytest.fixture
def timed_trial(monkeypatch):
    study = optuna.create_study()
    trial = study.ask()
    model = FullyConnected(
        device="cpu",
        n_quantities=1,
        n_timesteps=2,
        config={"hidden_size": 4, "num_hidden_layers": 1},
    )
    model.optuna_trial = trial
    model._trial_start_time = 90.0
    monkeypatch.setattr(abstract_surrogate, "time", SimpleNamespace(time=lambda: 100.0))
    return model, trial


@pytest.mark.parametrize("threshold", [0.0, 50.0])
def test_time_pruning_stops_slow_trials_and_records_reason(timed_trial, threshold):
    model, trial = timed_trial
    trial.study.set_user_attr("runtime_threshold", threshold)
    # Ten elapsed seconds after ten of 100 epochs projects a 100-second run.
    with pytest.raises(optuna.TrialPruned, match="Projected total time 100.0s"):
        model.time_pruning(current_epoch=10, total_epochs=100)
    assert "100.0s" in trial.user_attrs["prune_reason"]
    assert f"threshold {threshold:.1f}s" in trial.user_attrs["prune_reason"]


@pytest.mark.parametrize("threshold", [None, 100.0, 150.0])
def test_time_pruning_keeps_trials_within_budget(timed_trial, threshold):
    model, trial = timed_trial
    if threshold is not None:
        trial.study.set_user_attr("runtime_threshold", threshold)
    model.time_pruning(current_epoch=10, total_epochs=100)
    assert "prune_reason" not in trial.user_attrs


def test_time_pruning_respects_warmup(timed_trial):
    model, trial = timed_trial
    trial.study.set_user_attr("runtime_threshold", 0.0)
    model.time_pruning(current_epoch=9, total_epochs=100)
    assert "prune_reason" not in trial.user_attrs


def test_time_pruning_allows_training_without_an_optuna_trial(timed_trial):
    model, _ = timed_trial
    model.optuna_trial = None
    model.time_pruning(current_epoch=10, total_epochs=100)


# --------- Tests for maybe_set_runtime_threshold ----------
class FakeTrial:
    def __init__(self, num, state, start, complete=None):
        self.number = num
        self.state = state
        self.datetime_start = start
        self.datetime_complete = complete
        self.user_attrs = {}


class FakeStudy:
    def __init__(self, trials):
        self._trials = trials
        self.user_attrs = {}

    def get_trials(self, deepcopy=False):
        return self._trials

    def set_user_attr(self, key, val):
        self.user_attrs[key] = val


def test_maybe_set_runtime_threshold_not_enough():
    # only 1 complete trial, warmup_target=2
    t1 = FakeTrial(
        0,
        TrialState.COMPLETE,
        datetime.now() - timedelta(seconds=5),
        datetime.now(),
    )
    study = FakeStudy([t1])
    maybe_set_runtime_threshold(study, warmup_target=2)
    assert "runtime_threshold" not in study.user_attrs


def test_maybe_set_runtime_threshold_enough():
    now = datetime.now()
    trials = []
    for i in range(3):
        trials.append(
            FakeTrial(
                i,
                TrialState.COMPLETE,
                now - timedelta(seconds=10 + i),
                now - timedelta(seconds=i),
            )
        )
    study = FakeStudy(trials)
    maybe_set_runtime_threshold(study, warmup_target=3)
    # after enough trials, attrs should be set
    assert "runtime_threshold" in study.user_attrs
    assert "warmup_mean" in study.user_attrs
    assert study.user_attrs["warmup_target"] == 3


# --------- Tests for create_objective ----------


def test_training_run_single_objective(monkeypatch, tmp_path):
    # Prepare dummy config
    config = {
        "dataset": {
            "name": "ds",
            "log10_transform": False,
            "normalise": "none",
            "tolerance": 1e-3,
        },
        "seed": 123,
        "surrogate": {"name": "Surr"},
        "optuna_params": {},
        "batch_size": 16,
        "epochs": 5,
        "target_percentile": 0.5,
        "multi_objective": False,
    }
    # Fake download
    monkeypatch.setattr("codes.tune.optuna_fcts.download_data", lambda *args, **k: None)
    # Fake data loaders
    import numpy as np

    dummy_data = np.zeros((2, 4, 1))
    dummy_params = np.zeros((2, 3))
    dummy_timesteps = np.arange(4)
    dummy_info = {}
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.check_and_load_data",
        lambda *args, **kw: (
            (dummy_data, dummy_data, None),
            (dummy_params, dummy_params, None),
            dummy_timesteps,
            None,
            dummy_info,
            None,
        ),
    )
    monkeypatch.setattr("codes.tune.optuna_fcts.set_random_seeds", lambda *a, **k: None)

    monkeypatch.setattr("codes.tune.optuna_fcts.get_surrogate", lambda name: DummyModel)
    monkeypatch.setattr("codes.tune.optuna_fcts.get_model_config", lambda name, cfg: {})
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.make_optuna_params", lambda trial, params: {}
    )
    # patch the percentile training_run reports
    monkeypatch.setattr("codes.tune.optuna_fcts.np.percentile", lambda x, q: 7.0)
    # Run
    trial = type("T", (object,), {"number": 0})()
    val = __import__("codes.tune.optuna_fcts", fromlist=["training_run"]).training_run(
        trial, "cpu", 0, config, "study1"
    )
    assert isinstance(val, float) and val == 7.0


def test_training_run_multi_objective(monkeypatch, tmp_path):
    config = {
        "dataset": {
            "name": "ds",
            "log10_transform": False,
            "normalise": "none",
            "tolerance": 1e-3,
        },
        "seed": 456,
        "surrogate": {"name": "Surr"},
        "optuna_params": {},
        "batch_size": 8,
        "epochs": 5,
        "target_percentile": 0.5,
        "multi_objective": True,
    }
    # stub all as above
    monkeypatch.setattr("codes.tune.optuna_fcts.download_data", lambda *args, **k: None)
    import numpy as np

    dummy_data = np.ones((3, 5, 1))
    dummy_params = np.ones((3, 2))
    dummy_timesteps = np.arange(5)
    dummy_info = {}
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.check_and_load_data",
        lambda *args, **kw: (
            (dummy_data, dummy_data, None),
            (dummy_params, dummy_params, None),
            dummy_timesteps,
            None,
            dummy_info,
            None,
        ),
    )
    monkeypatch.setattr("codes.tune.optuna_fcts.set_random_seeds", lambda *a, **k: None)

    class DummyModel2(DummyModel):
        def predict(self, loader, leave_log=False):
            import torch

            return torch.zeros((3, 5, 1)), torch.ones((3, 5, 1))

    monkeypatch.setattr(
        "codes.tune.optuna_fcts.get_surrogate", lambda name: DummyModel2
    )
    monkeypatch.setattr("codes.tune.optuna_fcts.get_model_config", lambda name, cfg: {})
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.make_optuna_params", lambda trial, params: {}
    )
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.measure_inference_time", lambda m, dur: [1.0, 2.0, 3.0]
    )
    monkeypatch.setattr("codes.tune.optuna_fcts.np.percentile", lambda x, q: 5.0)

    trial = type("T", (object,), {"number": 2})()
    val = __import__("codes.tune.optuna_fcts", fromlist=["training_run"]).training_run(
        trial, "cpu", 1, config, "study_2"
    )
    # expect (loss, mean_inference)
    assert isinstance(val, tuple) and val[0] == 5.0 and val[1] == pytest.approx(2.0)


def test_create_objective_simple(monkeypatch):
    # stub training_run
    called = {}

    def fake_run(trial, device, slot, config, name):
        called["args"] = (device, slot, config, name)
        return 42.0

    monkeypatch.setattr("codes.tune.optuna_fcts.training_run", fake_run)

    device_queue = queue.Queue()
    device_queue.put(("cpu", 0))
    config = {"dataset": {"name": "ds"}}
    obj = create_objective(config, "study1", device_queue)

    class DummyT:
        number = 5

    trial = DummyT()
    result = obj(trial)
    # training_run returned 42.0
    assert result == 42.0
    # device put back
    assert not device_queue.empty()
    dev, slot = device_queue.get()
    assert dev == "cpu" and slot == 0


def test_prepare_workspace_reuses_identical_config(tmp_path, monkeypatch):
    """Re-running with an unchanged config must not ask whether to overwrite."""
    from codes.tune.optuna_fcts import load_yaml_config
    from codes.tune.tune_utils import prepare_workspace

    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "optuna_config.yaml"
    cfg_path.write_text("tuning_id: run\nseed: 42\nstorage: None\n")

    def no_prompt(_):
        raise AssertionError("prepare_workspace asked about an unchanged config")

    monkeypatch.setattr("builtins.input", no_prompt)
    first = prepare_workspace(cfg_path, load_yaml_config(str(cfg_path)))
    second = prepare_workspace(cfg_path, load_yaml_config(str(cfg_path)))

    assert first["_overwrite_run"] is True
    assert not second.get("_overwrite_run", False)
