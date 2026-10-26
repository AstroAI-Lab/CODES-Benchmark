"""
End-to-end tests: train every surrogate on a tiny synthetic dataset with
run_training.py, then evaluate the models with run_eval.py.

The golden values in test/data/golden_metrics.yaml pin the evaluation results of this
toy run, so a change that alters benchmark results makes them fail. If the change is
intended, regenerate the values and record the change in CHANGELOG.md:

    CODES_UPDATE_GOLDEN=1 pytest test/test_end_to_end.py
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import run_eval
import run_training
from codes.utils.data_utils import create_dataset

GOLDEN_FILE = Path(__file__).parent / "data" / "golden_metrics.yaml"
UPDATE_GOLDEN = os.environ.get("CODES_UPDATE_GOLDEN") == "1"
# Loose enough for numerical noise across platforms and library versions, tight
# enough to catch changes to training or evaluation. LatentNeuralODE trains with an
# adaptive ODE solver, which turns rounding differences between CPUs into differences
# of up to about 10% after a few epochs, so it gets a looser tolerance.
GOLDEN_RTOL = 1e-2
GOLDEN_RTOL_PER_SURROGATE = {"LatentNeuralODE": 0.25}

SURROGATES = ["MultiONet", "FullyConnected", "LatentNeuralODE", "LatentPoly"]


# Small networks with high learning rates, so a few epochs learn the time dependence.
TOY_SURROGATES_CONFIG = """
from dataclasses import dataclass


@dataclass
class MultiONetConfig:
    hidden_size: int = 32
    branch_hidden_layers: int = 2
    trunk_hidden_layers: int = 2
    output_factor: int = 4
    learning_rate: float = 3e-3


@dataclass
class FullyConnectedConfig:
    hidden_size: int = 32
    num_hidden_layers: int = 2
    learning_rate: float = 3e-3


@dataclass
class LatentNeuralODEConfig:
    latent_features: int = 3
    coder_layers: int = 2
    coder_width: int = 16
    ode_layers: int = 2
    ode_width: int = 16
    learning_rate: float = 1e-2


@dataclass
class LatentPolyConfig:
    latent_features: int = 3
    degree: int = 3
    coder_layers: int = 2
    coder_width: int = 16
    learning_rate: float = 1e-2
"""


def make_toy_dataset() -> None:
    """Create a small parametric dataset of positive, smooth trajectories."""
    rng = np.random.default_rng(0)
    n_samples, n_timesteps = 30, 16
    t = np.linspace(0.0, 1.0, n_timesteps)
    params = rng.uniform(1.0, 10.0, size=(n_samples, 2)).astype(np.float32)
    rates = np.stack(
        [-np.log10(params[:, 0]), 0.5 * np.log10(params[:, 1]), -0.2 * np.ones(30)],
        axis=1,
    )
    start = rng.uniform(0.5, 2.0, size=(n_samples, 3))
    data = start[:, None, :] * 10.0 ** (rates[:, None, :] * t[None, :, None])
    data = data.astype(np.float32)
    # Pass the split explicitly; create_dataset shuffles single arrays unseeded.
    splits = (slice(0, 18), slice(18, 24), slice(24, 30))
    create_dataset(
        "toy",
        tuple(data[s] for s in splits),
        params=tuple(params[s] for s in splits),
        timesteps=t,
        labels=["A", "B", "C"],
    )
    Path("datasets/toy/surrogates_config.py").write_text(TOY_SURROGATES_CONFIG)


def make_config(training_id: str, devices: list[str]) -> dict:
    return {
        "training_id": training_id,
        "surrogates": SURROGATES,
        "devices": devices,
        "seed": 42,
        "verbose": False,
        # Scalars exercise the "same value for every surrogate" form.
        "batch_size": 32,
        "epochs": 30,
        "checkpoint": True,
        "dataset": {
            "name": "toy",
            "log10_transform": True,
            "log10_transform_params": True,
            "normalise": "minmax",
            "normalise_per_species": False,
            "tolerance": 1e-20,
            "log_timesteps": False,
            "use_optimal_params": True,
            "subset_factor": 1,
        },
        "interpolation": {"enabled": True, "intervals": [2]},
        "extrapolation": {"enabled": True, "cutoffs": [10]},
        "sparse": {"enabled": False, "factors": [2]},
        "batch_scaling": {"enabled": False, "sizes": ["1/2"]},
        "uncertainty": {"enabled": True, "ensemble_size": 2},
        "losses": False,
        "iterative": False,
        "gradients": False,
        "timing": False,
        "compute": False,
        "compare": True,
        "relative_error_threshold": 1e-10,
        "error_percentile": 99,
    }


def train(config: dict) -> None:
    path = Path(f"{config['training_id']}.yaml")
    path.write_text(yaml.safe_dump(config))
    run_training.main(SimpleNamespace(config=str(path)))


def saved_weights(training_id: str) -> dict[str, dict[str, torch.Tensor]]:
    """The state dicts of all models of a training run, by relative path."""
    directory = Path("trained") / training_id
    return {
        str(path.relative_to(directory)): torch.load(
            path, map_location="cpu", weights_only=False
        )["state_dict"]
        for path in sorted(directory.glob("*/*.pth"))
    }


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Train the toy models sequentially and in parallel, then evaluate them."""
    root = tmp_path_factory.mktemp("e2e")
    cwd = os.getcwd()
    os.chdir(root)
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        make_toy_dataset()
        sequential = make_config("sequential", ["cpu"])
        train(sequential)
        train(make_config("parallel", ["cpu", "cpu"]))

        eval_config = Path("eval.yaml")
        eval_config.write_text(yaml.safe_dump(sequential))
        run_eval.main(SimpleNamespace(config=str(eval_config), device="None"))
        yield root
    finally:
        torch.use_deterministic_algorithms(deterministic)
        os.chdir(cwd)


def test_all_models_trained(workspace):
    names = set(saved_weights("sequential"))
    for surrogate in SURROGATES:
        stem = f"{surrogate}/{surrogate.lower()}"
        assert {
            f"{stem}_main.pth",
            f"{stem}_interpolation_2.pth",
            f"{stem}_extrapolation_10.pth",
            f"{stem}_UQ_1.pth",
        } <= names
    assert (workspace / "trained" / "sequential" / "completed.txt").exists()
    assert not list((workspace / "trained").rglob("best_checkpoint.pth"))


def test_parallel_training_matches_sequential(workspace):
    """The seed alone must determine a model, however tasks are spread over devices."""
    sequential = saved_weights("sequential")
    parallel = saved_weights("parallel")
    assert sequential.keys() == parallel.keys()
    for name, state in sequential.items():
        for key, value in state.items():
            assert torch.equal(value, parallel[name][key]), f"{name}: {key} differs"


def test_ensemble_members_differ(workspace):
    weights = saved_weights("sequential")
    for surrogate in SURROGATES:
        stem = f"{surrogate}/{surrogate.lower()}"
        main, member = weights[f"{stem}_main.pth"], weights[f"{stem}_UQ_1.pth"]
        assert any(not torch.equal(main[k], member[k]) for k in main)


def load_metrics(surrogate: str) -> dict:
    path = Path("results") / "sequential" / f"{surrogate.lower()}_metrics.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def golden_candidates() -> dict[str, dict[str, float]]:
    """The metrics pinned by the golden-value test, per surrogate."""
    values = {}
    for surrogate in SURROGATES:
        metrics = load_metrics(surrogate)
        accuracy = metrics["accuracy"]
        values[surrogate] = {
            "accuracy.mean_absolute_error_log": accuracy["mean_absolute_error_log"],
            "accuracy.percentile_absolute_error_log": accuracy[
                "percentile_absolute_error_log"
            ],
            "accuracy.mean_relative_error": accuracy["mean_relative_error"],
            "interpolation.interval_2": metrics["interpolation"]["interval 2"][
                "MAE_log"
            ],
            "extrapolation.cutoff_10": metrics["extrapolation"]["cutoff 10"]["MAE_log"],
            "UQ.average_uncertainty_log": metrics["UQ"]["average_uncertainty_log"],
        }
    return {s: {k: float(v) for k, v in m.items()} for s, m in values.items()}


def test_metrics_are_finite(workspace):
    for surrogate, values in golden_candidates().items():
        for name, value in values.items():
            assert np.isfinite(value), f"{surrogate} {name} = {value}"


def test_golden_values(workspace):
    actual = golden_candidates()
    if UPDATE_GOLDEN:
        GOLDEN_FILE.parent.mkdir(exist_ok=True)
        with open(GOLDEN_FILE, "w", encoding="utf-8") as f:
            f.write(f"# Generated by {Path(__file__).name}; see its docstring.\n")
            yaml.safe_dump(actual, f)
        pytest.skip("Golden values regenerated.")

    with open(GOLDEN_FILE, encoding="utf-8") as f:
        expected = yaml.safe_load(f)
    assert expected.keys() == actual.keys()
    mismatches = [
        f"{surrogate} {name}: {actual[surrogate][name]:.6g} instead of {value:.6g}"
        for surrogate, values in expected.items()
        for name, value in values.items()
        if actual[surrogate][name]
        != pytest.approx(
            value, rel=GOLDEN_RTOL_PER_SURROGATE.get(surrogate, GOLDEN_RTOL)
        )
    ]
    assert not mismatches, "\n".join(mismatches)
