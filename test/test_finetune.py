import os

import numpy as np
import pytest
import torch
import yaml

import codes.finetune.finetune_fcts as ff
from codes.benchmark.bench_fcts import fixed_step_rollout
from codes.benchmark.bench_utils import get_model_config
from codes.finetune import (
    eval_step_sizes,
    finetune_config,
    finetune_model,
    rollout_errors,
    rollout_score,
    run_finetuning,
)
from codes.surrogates import (
    AbstractSurrogateModel,
    FullyConnected,
    FullyConnectedResidual,
    MultiONet,
    MultiONetResidual,
)
from codes.surrogates.AbstractSurrogate.abstract_config import (
    AbstractSurrogateBaseConfig,
)
from codes.utils import elapsed_to_dummy, interp_states

S, M, Q, P = 6, 9, 3, 2
# Non uniform with a non zero start, the case where elapsed and absolute time differ.
TIMES = np.array([2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 16.0])
SUPPORTED = [FullyConnected, FullyConnectedResidual, MultiONet, MultiONetResidual]


def make_model(cls, n_params=P, params_branch=True):
    torch.manual_seed(0)
    config = {"params_branch": params_branch} if issubclass(cls, MultiONet) else {}
    model = cls(
        device="cpu",
        n_quantities=Q,
        n_timesteps=M,
        n_parameters=n_params,
        config=config,
    )
    model.normalisation = {"mode": "disabled", "log10_transform": True}
    return model.eval()


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    trajectories = rng.random((S, M, Q)).astype(np.float32)
    params = rng.random((S, P)).astype(np.float32)
    return trajectories, params


class TestTimeHelpers:
    def test_elapsed_to_dummy_uses_elapsed_time(self):
        # 3.0 elapsed from timesteps[0] = 2.0 lands on timesteps[3] = 5.0, index 4.
        assert elapsed_to_dummy(0.0, TIMES) == 0.0
        assert np.isclose(elapsed_to_dummy(3.0, TIMES), 4 / (M - 1))
        assert np.isclose(elapsed_to_dummy(TIMES[-1] - TIMES[0], TIMES), 1.0)

    def test_interp_states_is_linear_in_the_index(self):
        arr = np.arange(M, dtype=float)[None, :, None]
        out = interp_states(arr, TIMES, np.array([2.5, 3.5, 16.0]))
        assert np.allclose(out[0, :, 0], [1.0, 2.5, 8.0])


class TestForwardFromState:
    @pytest.mark.parametrize("cls", SUPPORTED)
    @pytest.mark.parametrize("params_branch", [True, False])
    def test_matches_dataloader_path(self, cls, params_branch, data):
        if params_branch is False and not issubclass(cls, MultiONet):
            pytest.skip("params_branch only applies to MultiONet")
        trajectories, params = data
        model = make_model(cls, params_branch=params_branch)
        times = np.linspace(0.0, 1.0, M)
        loader, _, _ = model.prepare_data(
            dataset_train=trajectories,
            dataset_test=None,
            dataset_val=None,
            timesteps=times,
            batch_size=16,
            shuffle=False,
            dataset_train_params=params,
            dummy_timesteps=True,
        )
        expected, _ = model.predict(data_loader=loader, leave_log=True, leave_norm=True)
        with torch.no_grad():
            out = model.forward_from_state(
                torch.tensor(trajectories[:, 0]),
                torch.tensor(times, dtype=torch.float32),
                torch.tensor(params),
            )
        assert out.shape == (S, M, Q)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_unsupported_surrogate_raises(self):
        class NoRollout(AbstractSurrogateModel):
            def forward(self, inputs):
                return inputs

            def prepare_data(self, *args, **kwargs):
                return None

            def fit(self, *args, **kwargs):
                return None

        model = NoRollout.__new__(NoRollout)
        with pytest.raises(NotImplementedError, match="cannot be used for rollout"):
            model.forward_from_state(torch.zeros(1, Q), torch.zeros(1))


def test_model_config_falls_back_to_parent(tmp_path, monkeypatch):
    folder = tmp_path / "datasets" / "toy"
    folder.mkdir(parents=True)
    (folder / "surrogates_config.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\nclass MultiONetConfig:\n    hidden_size: int = 7\n\n"
        "@dataclass\nclass FullyConnectedResidualConfig:\n    hidden_size: int = 9\n"
    )
    monkeypatch.chdir(tmp_path)
    conf = {"dataset": {"name": "toy"}}
    assert get_model_config("MultiONetResidual", conf) == {"hidden_size": 7}
    assert get_model_config("FullyConnectedResidual", conf) == {"hidden_size": 9}
    assert get_model_config("FullyConnected", conf) == {}


class TestFixedStepRollout:
    def test_steps_to_the_horizon_and_matches_rollout_errors(self, data):
        trajectories, params = data
        model = make_model(MultiONetResidual)
        preds, ref, one_shot, times = fixed_step_rollout(
            model, trajectories, trajectories, TIMES, params, 16, {"dt": 4.0}
        )
        assert preds.shape == ref.shape == one_shot.shape == (S, 4, Q)
        assert np.allclose(times, [2.0, 6.0, 10.0, 14.0])
        errors = rollout_errors(model, trajectories, params, TIMES, 4.0)
        assert np.allclose(errors, np.abs(preds - ref)[:, 1:], atol=1e-5)

    def test_max_steps_caps_the_rollout(self, data):
        trajectories, params = data
        model = make_model(FullyConnected)
        preds, _, _, _ = fixed_step_rollout(
            model,
            trajectories,
            trajectories,
            TIMES,
            params,
            16,
            {"dt": 1.0, "max_steps": 3},
        )
        assert preds.shape[1] == 4

    @pytest.mark.parametrize("it_conf", [{}, {"dt": 100.0}])
    def test_invalid_step_size_raises(self, data, it_conf):
        trajectories, params = data
        with pytest.raises(ValueError):
            fixed_step_rollout(
                make_model(FullyConnected),
                trajectories,
                trajectories,
                TIMES,
                params,
                16,
                it_conf,
            )


class TestFinetuneConfig:
    base = {"training_id": "run", "surrogates": ["MultiONet"], "seed": 3}

    def test_defaults_are_filled_in(self):
        ft = finetune_config({**self.base, "finetune": {"dt_range": [1, 4]}})
        assert ft["training_id"] == "run_finetune"
        assert ft["surrogates"] == ["MultiONet"]
        assert ft["seed"] == 3
        assert ft["eval_dt_range"] == [1.0, 4.0]
        assert ft["rollout_steps"] == ff.FINETUNE_DEFAULTS["rollout_steps"]

    @pytest.mark.parametrize(
        "block",
        [
            None,
            {},
            {"dt_range": [1, 4], "learnig_rate": 1e-3},
            {"dt_range": [1, 4], "rollout_aggregation": "max"},
        ],
    )
    def test_invalid_blocks_raise(self, block):
        conf = dict(self.base)
        if block is not None:
            conf["finetune"] = block
        with pytest.raises(ValueError):
            finetune_config(conf)


def test_eval_step_sizes_cover_the_range_evenly():
    centres = eval_step_sizes([1.0, 16.0], 4)
    assert np.allclose(centres, np.sqrt([2.0, 8.0, 32.0, 128.0]))
    edges = np.exp(np.linspace(0.0, np.log(16.0), 5))
    for seed in range(20):
        drawn = eval_step_sizes([1.0, 16.0], 4, np.random.default_rng(seed))
        assert np.all((drawn >= edges[:-1]) & (drawn <= edges[1:]))


def finetune_settings(**overrides):
    ft = {**ff.FINETUNE_DEFAULTS, "dt_range": [0.5, 3.0], "epochs": 2, "batch_size": 4}
    return {**ft, **overrides}


class ConstantSurrogate(torch.nn.Module):
    """A constant prediction gives known losses even when a rollout stops early."""

    def __init__(self, **config):
        super().__init__()
        self.device = "cpu"
        self.config = AbstractSurrogateBaseConfig(**config)
        self.value = torch.nn.Parameter(torch.tensor(1.0))

    def forward_from_state(self, state, times, params=None):
        return self.value.expand(state.shape[0], len(times), state.shape[1])


class TestFinetuneModel:
    def test_smooth_l1_keeps_the_training_default(self):
        config = AbstractSurrogateBaseConfig(loss_function=torch.nn.SmoothL1Loss())
        assert config.loss.beta == 1.0

    def test_uses_the_configured_loss(self):
        model = ConstantSurrogate(loss_function=torch.nn.SmoothL1Loss(), beta=2.5)
        history = finetune_model(
            model,
            np.zeros((2, 3, 1), dtype=np.float32),
            None,
            np.linspace(0.0, 1.0, 3),
            finetune_settings(epochs=1, learning_rate=0.0, rollout_weight=0.0),
            np.random.default_rng(0),
        )
        # SmoothL1Loss with error 1 and beta 2.5 is 1 / (2 * 2.5).
        assert history[0]["data"] == pytest.approx(0.2)

    @pytest.mark.parametrize(
        "steps,dt,aggregation,expected",
        [
            (1, 0.6, "mean", 1.0),
            (4, 0.6, "mean", 1.0),
            (4, 0.4, "mean", 1.0),
            (4, 0.4, "sum_decay", 1.5),
        ],
    )
    def test_truncated_rollout_loss(self, steps, dt, aggregation, expected):
        history = finetune_model(
            ConstantSurrogate(),
            np.zeros((2, 3, 1), dtype=np.float32),
            None,
            np.linspace(0.0, 1.0, 3),
            finetune_settings(
                epochs=1,
                learning_rate=0.0,
                max_warmup_steps=0,
                dt_range=[dt, dt],
                rollout_steps=steps,
                rollout_aggregation=aggregation,
                rollout_decay=0.5,
                rollout_weight=0.5,
            ),
            np.random.default_rng(0),
        )
        assert history[0]["rollout"] == pytest.approx(expected)
        assert history[0]["total"] == pytest.approx(0.5 + 0.5 * expected)

    @pytest.mark.parametrize("aggregation", ["mean", "sum_decay"])
    def test_updates_the_weights(self, data, aggregation):
        trajectories, params = data
        model = make_model(MultiONetResidual)
        before = [p.detach().clone() for p in model.parameters()]
        history = finetune_model(
            model,
            trajectories,
            params,
            TIMES,
            finetune_settings(rollout_aggregation=aggregation, noise=0.1),
            np.random.default_rng(0),
        )
        assert len(history) == 2
        assert history[-1]["rollout"] > 0 and history[-1]["skipped"] == 0
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    def test_skips_batches_with_non_finite_loss(self, data, monkeypatch):
        trajectories, params = data
        model = make_model(FullyConnected)
        original = model.forward_from_state
        monkeypatch.setattr(
            model, "forward_from_state", lambda *a: original(*a) * float("nan")
        )
        before = [p.detach().clone() for p in model.parameters()]
        history = finetune_model(
            model,
            trajectories,
            params,
            TIMES,
            finetune_settings(),
            np.random.default_rng(0),
        )
        assert history[-1]["skipped"] == 2
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_rollout_score_is_the_geometric_mean_and_filters_dynamic(data):
    trajectories, params = data
    model = make_model(MultiONet)
    score, medians = rollout_score(model, trajectories, params, TIMES, [1.0, 4.0])
    assert np.isclose(score, np.sqrt(medians[0] * medians[1]))
    with pytest.raises(ValueError, match="dynamic_span"):
        rollout_score(model, trajectories, params, TIMES, [1.0], dynamic_span=10.0)


def test_search_resumes_trials_without_timing_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ft = finetune_config(
        {
            "training_id": "base",
            "surrogates": ["MultiONet"],
            "finetune": {"dt_range": [0.5, 3.0], "optuna": {"n_trials": 1}},
        }
    )
    run_dir = tmp_path / "trained" / ft["training_id"]
    trial_dir = run_dir / "multionet_trials"
    trial_dir.mkdir(parents=True)
    study = ff.optuna.create_study(
        study_name=f"{ft['training_id']}_multionet",
        storage=f"sqlite:///{run_dir / 'multionet_study.db'}",
    )
    study.add_trial(ff.optuna.trial.create_trial(value=0.1))
    weights = make_model(MultiONet).state_dict()
    torch.save(weights, trial_dir / "trial0.pth")

    def unexpected_call(*args):
        pytest.fail("A completed search must load the saved trial without retraining.")

    _, trial, restored, duration = ff._search(
        ft, "MultiONet", unexpected_call, unexpected_call
    )
    assert trial == 0
    assert all(torch.equal(restored[k], v) for k, v in weights.items())
    assert duration is None


@pytest.mark.parametrize("search", [False, True])
def test_run_finetuning_writes_a_run_that_run_eval_can_load(
    data, tmp_path, monkeypatch, search
):
    trajectories, params = data
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ff,
        "load_finetune_data",
        lambda conf: {
            "train": trajectories,
            "test": trajectories,
            "train_params": params,
            "test_params": params,
            "timesteps": TIMES,
        },
    )

    def load_model(name, conf, d, device):
        model = make_model(MultiONet)
        model.fit_duration = 1234.0  # The base training duration must be replaced.
        return model

    monkeypatch.setattr(ff, "load_trained_model", load_model)
    clock = iter([10.0, 12.0, 20.0, 25.0])
    monkeypatch.setattr(ff, "perf_counter", lambda: next(clock), raising=False)
    conf = {
        "training_id": "base",
        "surrogates": ["MultiONet"],
        "batch_size": [4],
        "epochs": [1],
        "dataset": {"name": "toy"},
        "interpolation": {"enabled": True, "intervals": [2]},
        "finetune": {
            "dt_range": [0.5, 3.0],
            "epochs": 1,
            "batch_size": 4,
            "eval_n_dt": 2,
        },
    }
    if search:
        conf["finetune"]["optuna"] = {
            "n_trials": 2,
            "params": {"learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4}},
        }
    summary = run_finetuning(conf, "cpu")["MultiONet"]

    assert len(summary["step_sizes"]) == 2
    assert (summary["best_trial"] is not None) == search
    run = os.path.join("trained", "base_finetune")
    assert os.path.isfile(os.path.join(run, "MultiONet", "multionet_main.pth"))
    assert os.path.isfile(
        os.path.join("results", "base_finetune", "multionet_finetune.yaml")
    )
    with open(os.path.join(run, "config.yaml"), encoding="utf-8") as f:
        exported = yaml.safe_load(f)
    assert exported["training_id"] == "base_finetune"
    assert exported["interpolation"]["enabled"] is False
    assert "finetune" not in exported
    duration = [2.0, 5.0][summary["best_trial"]] if search else 2.0
    restored = make_model(MultiONet)
    restored.load("base_finetune", "MultiONet", "multionet_main")
    assert restored.train_duration == duration
    if search:
        # No new trials: restore the selected trial's time along with its weights.
        resumed = run_finetuning(conf, "cpu")["MultiONet"]
        assert resumed["best_trial"] == summary["best_trial"]
        restored.load("base_finetune", "MultiONet", "multionet_main")
        assert restored.train_duration == duration
