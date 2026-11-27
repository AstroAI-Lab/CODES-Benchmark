import math

import optuna
import pytest

from codes.tune.pareto_pruning import (
    PARETO_PRUNED_ATTR,
    assess_checkpoint,
    checkpoint_epochs,
    parse_pareto_pruning,
)


def _settings(mode="prune", **overrides):
    pruning = {"mode": mode, "peer_warmup": 2, "latency_margin": 0, **overrides}
    return parse_pareto_pruning(
        {"multi_objective": True, "multi_objective_pruning": pruning}
    )


def _study_with_peers(gates, error=0.1, latency=1.0, count=2):
    study = optuna.create_study(directions=["minimize", "minimize"])
    for _ in range(count):

        def objective(trial):
            trial.set_user_attr(
                "pareto_history",
                [
                    {"epoch": gate, "error": error, "latency_ms": latency}
                    for gate in gates
                ],
            )
            return error, latency

        study.optimize(objective, n_trials=1)
    return study


def test_defaults_and_checkpoint_rounding():
    assert parse_pareto_pruning({}) is None
    settings = parse_pareto_pruning(
        {"multi_objective": True, "multi_objective_pruning": {"mode": "shadow"}}
    )
    assert settings["peer_warmup"] == 8
    assert settings["latency_margin"] == 0.10
    assert settings["checkpoints"] == [0.2, 0.4, 0.7]
    assert checkpoint_epochs(10, settings) == (2, 4, 7)
    assert checkpoint_epochs(2, settings) == ()


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "active"},
        {"mode": "prune", "unknown": 1},
        {"mode": "prune", "checkpoints": [0.5]},
        {"mode": "prune", "checkpoints": [0.5, 1.0]},
        {"mode": "prune", "checkpoints": [0.5, float("nan")]},
        {"mode": "prune", "peer_warmup": True},
        {"mode": "prune", "error_gap": float("nan")},
    ],
)
def test_rejects_invalid_settings(bad):
    with pytest.raises(ValueError):
        parse_pareto_pruning({"multi_objective": True, "multi_objective_pruning": bad})


def test_shadow_writes_diagnostic_only_at_full_decision():
    settings = _settings("shadow", checkpoints=[0.2, 0.4, 0.7], error_gap=0.01)
    study = _study_with_peers([2, 4, 7])
    trial = study.ask()
    assert not assess_checkpoint(trial, 2, 10, 0.2, 1.0, settings)
    assert "pareto_would_prune" not in trial.user_attrs
    assert not assess_checkpoint(trial, 4, 10, 0.2, 1.0, settings)
    assert trial.user_attrs["pareto_would_prune"]["epoch"] == 4
    assert PARETO_PRUNED_ATTR not in trial.user_attrs


def test_prunes_on_second_qualifying_consecutive_checkpoint_only():
    settings = _settings(checkpoints=[0.2, 0.4, 0.7], error_gap=0.01)
    study = _study_with_peers([2, 4, 7])
    trial = study.ask()
    assert not assess_checkpoint(trial, 2, 10, 0.2, 1.0, settings)
    assert assess_checkpoint(trial, 4, 10, 0.2, 1.0, settings)
    assert trial.user_attrs[PARETO_PRUNED_ATTR] is True


def test_missing_previous_gate_breaks_sequence_and_requires_trend():
    settings = _settings(checkpoints=[0.2, 0.4, 0.7], error_gap=0.01)
    study = _study_with_peers([2, 4, 7])
    trial = study.ask()
    assert not assess_checkpoint(trial, 2, 10, 0.2, 1.0, settings)
    assert not assess_checkpoint(trial, 7, 10, 0.2, 1.0, settings)
    assert "pareto_would_prune" not in trial.user_attrs


def test_small_measured_latency_gap_cannot_prune_fast_candidate():
    settings = _settings(latency_margin=0.10)
    study = _study_with_peers([2, 4, 7], latency=0.96)
    trial = study.ask()
    for epoch in (2, 4, 7):
        assert not assess_checkpoint(trial, epoch, 10, 0.5, 1.0, settings)
    assert "pareto_would_prune" not in trial.user_attrs


def test_improving_error_and_material_latency_tradeoff_are_protected():
    settings = _settings(checkpoints=[0.2, 0.4, 0.7], error_gap=0.01)
    study = _study_with_peers([2, 4, 7])
    trial = study.ask()
    assert not assess_checkpoint(trial, 2, 10, 0.2, 0.7, settings)
    assert not assess_checkpoint(trial, 4, 10, 0.18, 0.7, settings)
    assert not assess_checkpoint(trial, 7, 10, 0.17, 0.7, settings)
    assert "pareto_would_prune" not in trial.user_attrs


@pytest.mark.parametrize("errors", [(0.30, 0.25, 0.25), (0.30, 0.25, 0.20)])
def test_big_recent_learning_leg_protects_slow_learner_through_final_rung(errors):
    settings = _settings(checkpoints=[0.2, 0.4, 0.7], error_gap=0.01)
    study = _study_with_peers([2, 4, 7])
    trial = study.ask()
    assert not assess_checkpoint(trial, 2, 10, errors[0], 1.0, settings)
    assert not assess_checkpoint(trial, 4, 10, errors[1], 1.0, settings)
    assert not assess_checkpoint(trial, 7, 10, errors[2], 1.0, settings)
    assert "pareto_would_prune" not in trial.user_attrs


def test_shadow_keeps_first_decision_and_later_metrics_out_of_peer_pool():
    settings = _settings("shadow", checkpoints=[0.2, 0.4, 0.7], error_gap=0.01)
    study = _study_with_peers([2, 4, 7])
    shadow = study.ask()
    assert not assess_checkpoint(shadow, 2, 10, 0.2, 1.0, settings)
    assert not assess_checkpoint(shadow, 4, 10, 0.2, 1.0, settings)
    first_diagnostic = shadow.user_attrs["pareto_would_prune"]
    assert not assess_checkpoint(shadow, 7, 10, 0.2, 1.0, settings)
    assert shadow.user_attrs["pareto_would_prune"] == first_diagnostic

    other = study.ask()
    assert not assess_checkpoint(other, 7, 10, 0.2, 1.0, settings)
    # The two completed reference trials qualify; the shadow trial's post-flag
    # observation at epoch 7 is not considered as a reference.
    assert other.user_attrs["pareto_history"][-1]["witness_count"] == 2


def test_bad_metric_is_not_published():
    settings = _settings(checkpoints=[0.2, 0.4])
    trial = optuna.create_study(directions=["minimize", "minimize"]).ask()
    assert not assess_checkpoint(trial, 2, 10, math.nan, 1.0, settings)
    assert "pareto_history" not in trial.user_attrs


def test_negative_error_is_not_published():
    settings = _settings()
    trial = optuna.create_study(directions=["minimize", "minimize"]).ask()
    assert not assess_checkpoint(trial, 2, 10, -0.1, 1.0, settings)
    assert "pareto_history" not in trial.user_attrs
