"""Checkpoint-level shadow or active pruning for multi-objective Optuna studies."""

from __future__ import annotations

import math
from numbers import Real

from optuna.trial import FrozenTrial, Trial, TrialState

PARETO_PRUNED_ATTR = "pareto_pruned"
_HISTORY_ATTR = "pareto_history"
_DIAGNOSTIC_ATTR = "pareto_would_prune"

DEFAULTS = {
    "mode": "off",  # off | shadow | prune
    "checkpoints": [0.20, 0.40, 0.70],
    "peer_warmup": 8,
    "error_gap": 0.02,
    "latency_margin": 0.10,
    "max_recent_error_improvement": 0.05,
}
_ALLOWED_KEYS = set(DEFAULTS)


def parse_pareto_pruning(config: dict) -> dict | None:
    """Normalize ``multi_objective_pruning``; shadow/prune require MO studies."""
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    raw = config.get("multi_objective_pruning")
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError("multi_objective_pruning must be a mapping")
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown multi_objective_pruning keys: {sorted(unknown)}")
    settings = dict(DEFAULTS)
    settings.update(raw)
    mode = settings["mode"]
    if not isinstance(mode, str):
        raise ValueError("multi_objective_pruning.mode must be a string")
    if mode not in {"off", "shadow", "prune"}:
        raise ValueError("multi_objective_pruning.mode must be off, shadow, or prune")
    if mode == "off":
        return None
    if config.get("multi_objective") is not True:
        raise ValueError("multi_objective must be true when Pareto pruning is enabled")

    fractions = settings["checkpoints"]
    if not isinstance(fractions, (list, tuple)) or len(fractions) < 2:
        raise ValueError("checkpoints must contain at least two fractions")
    if any(
        isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
        for x in fractions
    ):
        raise ValueError("checkpoint fractions must be finite numbers")
    fractions = [float(x) for x in fractions]
    if any(x <= 0 or x >= 1 for x in fractions) or fractions != sorted(set(fractions)):
        raise ValueError("checkpoints must be unique increasing fractions in (0, 1)")
    settings["checkpoints"] = fractions  # JSON-stable for manifests and reloads.

    warmup = settings["peer_warmup"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
        raise ValueError("peer_warmup must be a positive integer")
    for key, lower, upper in (
        ("error_gap", 0, None),
        ("latency_margin", 0, 1),
        ("max_recent_error_improvement", 0, 1),
    ):
        value = settings[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")
        if value < lower or (upper is not None and value > upper):
            raise ValueError(f"{key} is out of range")
        settings[key] = float(value)
    return settings


def checkpoint_epochs(total_epochs: int, settings: dict) -> tuple[int, ...]:
    """Return distinct one-based rung epochs, excluding the final epoch."""
    if (
        isinstance(total_epochs, bool)
        or not isinstance(total_epochs, int)
        or total_epochs < 1
    ):
        return ()
    rounded = {
        max(1, math.ceil(total_epochs * fraction))
        for fraction in settings["checkpoints"]
    }
    gates = tuple(epoch for epoch in sorted(rounded) if epoch < total_epochs)
    return gates if len(gates) >= 2 else ()


def _record(history: list[dict], epoch: int | None) -> dict | None:
    if epoch is None:
        return None
    return next((item for item in history if item["epoch"] == epoch), None)


def _peer_witness(
    trial: Trial,
    peers: list[FrozenTrial],
    epoch: int,
    error: float,
    latency: float,
    settings: dict,
) -> tuple[dict | None, int]:
    best = None
    peer_count = 0
    for peer in peers:
        if peer.number == trial.number:
            continue
        if peer.state not in {
            TrialState.COMPLETE,
            TrialState.RUNNING,
            TrialState.PRUNED,
        }:
            continue
        policy_pruned = peer.user_attrs.get(PARETO_PRUNED_ATTR) is True
        if not policy_pruned and (
            peer.state == TrialState.PRUNED or "exception" in peer.user_attrs
        ):
            continue
        diagnostic = peer.user_attrs.get(_DIAGNOSTIC_ATTR)
        if diagnostic and epoch > diagnostic["epoch"]:
            continue
        record = _record(peer.user_attrs.get(_HISTORY_ATTR, []), epoch)
        if record is None:
            continue
        peer_count += 1
        peer_error, peer_latency = record["error"], record["latency_ms"]
        if (
            peer_latency <= latency * (1 - settings["latency_margin"])
            and peer_error < error
            and peer_error <= error - settings["error_gap"]
        ):
            candidate = (peer_error, peer.number, peer_latency)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None, peer_count
    best_error, number, peer_latency = best
    return {
        "trial_number": number,
        "error": best_error,
        "latency_ms": peer_latency,
    }, peer_count


def assess_checkpoint(
    trial: Trial,
    epoch: int,
    total_epochs: int,
    error: float,
    latency: float,
    settings: dict | None,
) -> bool:
    """Publish checkpoint metrics and return whether an active prune is due."""
    if settings is None or settings["mode"] == "off":
        return False
    gates = checkpoint_epochs(total_epochs, settings)
    if epoch not in gates:
        return False
    if (
        isinstance(error, bool)
        or isinstance(latency, bool)
        or not isinstance(error, Real)
        or not isinstance(latency, Real)
        or not math.isfinite(error)
        or not math.isfinite(latency)
        or error < 0
        or latency <= 0
    ):
        return False
    history = list(trial.user_attrs.get(_HISTORY_ATTR, []))
    if _record(history, epoch) is not None:
        return False
    gate_index = gates.index(epoch)
    prior_gate = gates[gate_index - 1] if gate_index else None
    previous = _record(history, prior_gate)
    peers = trial.study.get_trials(deepcopy=False)
    witness, distinct_peers = _peer_witness(
        trial, peers, epoch, error, latency, settings
    )
    enough_peers = distinct_peers >= settings["peer_warmup"]
    dominance = witness is not None and enough_peers
    # Missing/skipped prior rung breaks the streak and cannot satisfy the trend brake.
    improvement = None
    previous_improvement = 0.0
    if previous is not None:
        old_error = previous["error"]
        improvement = max(old_error - error, 0.0) / old_error if old_error else 0.0
        previous_improvement = previous["recent_error_improvement"] or 0.0
    streak = int(dominance)
    if dominance and previous is not None:
        streak += previous["streak"]
    decision = (
        streak >= 2
        and improvement is not None
        and max(improvement, previous_improvement)
        <= settings["max_recent_error_improvement"]
    )
    history.append(
        {
            "epoch": epoch,
            "error": float(error),
            "latency_ms": float(latency),
            "witness": witness,
            "witness_count": distinct_peers,
            "dominance": dominance,
            "streak": streak,
            "recent_error_improvement": improvement,
        }
    )
    trial.set_user_attr(_HISTORY_ATTR, history)
    if decision:
        diagnostic = {
            "epoch": epoch,
            "witness": witness,
            "streak": streak,
            "recent_error_improvement": improvement,
        }
        if _DIAGNOSTIC_ATTR not in trial.user_attrs:
            trial.set_user_attr(_DIAGNOSTIC_ATTR, diagnostic)
        if settings["mode"] == "prune":
            trial.set_user_attr(PARETO_PRUNED_ATTR, True)
            return True
    return False
