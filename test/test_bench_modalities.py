from unittest.mock import patch

import numpy as np
import pytest
import torch

from codes.benchmark.bench_fcts import (
    _mode_dummy_timesteps,
    evaluate_batchsize,
    evaluate_extrapolation,
    evaluate_interpolation,
    evaluate_sparse,
    evaluate_UQ,
)


# Dummy model to record load calls
class DummyModel:
    def __init__(self, device, n_quantities, n_timesteps, n_parameters, config):
        self._loads = []
        self._loaders = []
        self._times = []
        self.n_timesteps = n_timesteps

    def load(self, training_id, surr_name, model_identifier):
        self._loads.append(model_identifier)

    def prepare_data(self, timesteps, dummy_timesteps, **kwargs):
        assert not dummy_timesteps
        self._times.append(timesteps)
        return f"mode loader {len(self._times)}", None, None

    def predict(self, data_loader, leave_log=None, leave_norm=None):
        self._loaders.append(data_loader)
        # targets always zero.  Shape (batch=2, timesteps=4, quantities=1).
        preds = torch.rand(2, 4, 1)
        targets = torch.rand(2, 4, 1)
        return preds, targets

    def __call__(self, inputs):
        # Return a dummy output tensor with 3 dims (B, T, Q)
        return torch.zeros(1, self.n_timesteps, 1), None


# Two standalone fakes: one for heatmap (returns tuple), one for all others (returns None)
def _fake_heatmap(*args, **kwargs):
    return ([], [])


def _fake_noop(*args, **kwargs):
    return None


@pytest.fixture(autouse=True)
def patch_plots():
    import codes.benchmark.bench_fcts as bf

    fake_impl = {}
    for name in dir(bf):
        if not name.startswith("plot_"):
            continue
        if name == "plot_uncertainty_heatmap":
            fake_impl[name] = _fake_heatmap
        else:
            fake_impl[name] = _fake_noop

    with patch.multiple("codes.benchmark.bench_fcts", **fake_impl):
        yield


@pytest.mark.parametrize(
    "raw_vals, cfg_key, func, main_bs, expected_nums",
    [
        ([2, 3], "interpolation", evaluate_interpolation, None, [1, 2, 3]),
        ([2, 3, 4], "extrapolation", evaluate_extrapolation, None, [2, 3, 4]),
        ([2, 4, 8], "sparse", evaluate_sparse, None, [1, 2, 4, 8]),
        ([0.5, 2], "batch_scaling", evaluate_batchsize, 8, [4, 8, 16]),
        (3, "uncertainty", evaluate_UQ, None, [0, 1, 2]),
    ],
)
def test_modality_variations(raw_vals, cfg_key, func, main_bs, expected_nums):
    surr = "TestSurr"
    cfg = {"training_id": "TID", "surrogates": [surr]}
    if cfg_key == "uncertainty":
        cfg["uncertainty"] = {"enabled": True, "ensemble_size": raw_vals}
    else:
        cfg[cfg_key] = {"enabled": True}
        subkey = {
            "interpolation": "intervals",
            "extrapolation": "cutoffs",
            "sparse": "factors",
            "batch_scaling": "sizes",
        }[cfg_key]
        cfg[cfg_key][subkey] = raw_vals
    cfg["batch_size"] = [main_bs or 8]

    timesteps = np.arange(4)
    # minimal iterable loader; evaluate_batchsize will call next(iter(loader))
    loader = [np.zeros((1,))]
    labels = ["q"] if func is evaluate_interpolation else None

    model = DummyModel(None, 1, len(timesteps), 0, {})

    # invoke
    if func in (evaluate_interpolation, evaluate_extrapolation):
        val_data = np.zeros((2, len(timesteps), 1))
        metrics = func(model, surr, loader, timesteps, cfg, labels, val_data=val_data)
    elif func is evaluate_sparse:
        metrics = func(model, surr, loader, timesteps, n_train_samples=10, conf=cfg)
    elif func is evaluate_batchsize:
        metrics = func(model, surr, loader, timesteps, cfg)
    else:
        metrics = func(model, surr, loader, timesteps, cfg, labels=None)

    lower = surr.lower()
    # build expected identifiers
    ids = []
    if cfg_key == "interpolation":
        for i in expected_nums:
            ids.append(f"{lower}_main" if i == 1 else f"{lower}_interpolation_{i}")
    elif cfg_key == "extrapolation":
        max_c = len(timesteps)
        for c in expected_nums:
            ids.append(f"{lower}_main" if c == max_c else f"{lower}_extrapolation_{c}")
    elif cfg_key == "sparse":
        for f in expected_nums:
            ids.append(f"{lower}_main" if f == 1 else f"{lower}_sparse_{f}")
    elif cfg_key == "batch_scaling":
        for bs in expected_nums:
            ids.append(f"{lower}_main" if bs == main_bs else f"{lower}_batchsize_{bs}")
    else:  # uncertainty
        for idx in expected_nums:
            ids.append(f"{lower}_main" if idx == 0 else f"{lower}_UQ_{idx}")

    assert model._loads == ids

    prefix = {
        "interpolation": "interval",
        "extrapolation": "cutoff",
        "sparse": "factor",
        "batch_scaling": "batch_size",
        "uncertainty": None,
    }[cfg_key]

    if cfg_key != "uncertainty":
        for num in expected_nums:
            assert f"{prefix} {num}" in metrics
    else:
        assert "average_uncertainty_log" in metrics


@pytest.mark.parametrize(
    "mode, metric, expected",
    [
        # Trained on steps 0, 2, 4, 6 of 8: step 6 is the last one, at time 1.
        ("interpolation", 2, np.arange(8) / 6),
        # 7 is divisible by 7, so the grid matches np.linspace(0, 1, 8).
        ("interpolation", 7, np.linspace(0, 1, 8)),
        # Trained on steps 0..3: later steps map past 1.
        ("extrapolation", 4, np.arange(8) / 3),
    ],
)
def test_mode_dummy_timesteps_match_training(mode, metric, expected):
    np.testing.assert_allclose(_mode_dummy_timesteps(8, mode, metric), expected)


def test_mode_dummy_timesteps_reject_single_timestep():
    with pytest.raises(ValueError, match="single timestep"):
        _mode_dummy_timesteps(8, "extrapolation", 1)


@pytest.mark.parametrize(
    "func, cfg_key, subkey, values",
    [
        (evaluate_interpolation, "interpolation", "intervals", [2]),
        (evaluate_extrapolation, "extrapolation", "cutoffs", [3]),
    ],
)
def test_mode_models_get_training_consistent_loaders(func, cfg_key, subkey, values):
    """The main model uses the given loader; mode models get their own time input."""
    cfg = {
        "training_id": "TID",
        "surrogates": ["S"],
        "batch_size": 8,
        cfg_key: {"enabled": True, subkey: values},
    }
    timesteps = np.arange(4)
    model = DummyModel(None, 1, len(timesteps), 0, {})

    func(model, "S", "main loader", timesteps, cfg, val_data=np.zeros((2, 4, 1)))

    assert sorted(model._loaders) == ["main loader", "mode loader 1"]
    np.testing.assert_allclose(
        model._times[0], _mode_dummy_timesteps(4, cfg_key, values[0])
    )
