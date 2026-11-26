"""Device-resident timing for endpoint and trajectory surrogate requests."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from codes.utils import elapsed_to_dummy

DEFAULT_BATCH_SIZES = (1, 8, 32, 128, 512, 2048, 8192)
TRAJECTORY_QUERIES = 16
TIMING_REPETITIONS = 5


def select_horizons(timesteps: np.ndarray) -> np.ndarray:
    """Choose the first positive elapsed time, geometric midpoint, and full span."""
    grid = np.asarray(timesteps, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("timesteps must be a finite 1D array with at least two values")
    positive = grid[1:] - grid[0]
    if not np.all(np.isfinite(positive)) or np.any(np.diff(grid) <= 0):
        raise ValueError("timesteps must be strictly increasing with a finite span")
    log_times = np.log(positive)
    midpoint = (log_times[0] + log_times[-1]) / 2
    middle = positive[np.argmin(np.abs(log_times - midpoint))]
    return np.unique([positive[0], middle, positive[-1]])


def _device(model: Any) -> torch.device:
    value = getattr(model, "device", None)
    return torch.device(value) if value is not None else next(model.parameters()).device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _batch_size(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Inference batch sizes must be positive integers")
    return int(size)


def _check_rollout_step(step: float | None) -> None:
    if step is not None and (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(step)
        or step <= 0
    ):
        raise ValueError("rollout_step must be a finite positive duration")


def _inputs(
    states: np.ndarray, params: np.ndarray | None, size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, np.ndarray]:
    states = np.asarray(states)
    if states.ndim != 2 or not len(states):
        raise ValueError("states must have shape (N, C) with N > 0")
    indices = np.arange(size) % len(states)
    state = torch.as_tensor(states[indices], dtype=torch.float32, device=device)
    if params is None:
        return state, None, indices
    params = np.asarray(params)
    if params.ndim != 2 or len(params) != len(states):
        raise ValueError("params must have shape (N, P) matching states")
    parameters = torch.as_tensor(params[indices], dtype=torch.float32, device=device)
    return state, parameters, indices


def _make_request(
    model: Any,
    state: torch.Tensor,
    params: torch.Tensor | None,
    queries: np.ndarray,
    timesteps: np.ndarray,
    rollout_step: float | None,
) -> Callable[[], torch.Tensor]:
    """Preload query times so device transfers and time mapping are outside timing."""
    if rollout_step is None:
        times = torch.as_tensor(
            elapsed_to_dummy(queries, timesteps), dtype=state.dtype, device=state.device
        )
        return lambda: model.forward_from_state(state, times, params)

    full_step = torch.as_tensor(
        elapsed_to_dummy([rollout_step], timesteps),
        dtype=state.dtype,
        device=state.device,
    )
    groups = []
    for interval in np.diff(queries, prepend=0):
        ratio = float(interval) / rollout_step
        if not math.isfinite(ratio):
            raise ValueError("rollout_step is too small for the requested horizon")
        nearest = round(ratio)
        count = max(
            1,
            nearest
            if math.isclose(ratio, nearest, rel_tol=1e-12)
            else math.ceil(ratio),
        )
        full_steps = count - 1
        remainder = float(interval) - full_steps * rollout_step
        last_step = torch.as_tensor(
            elapsed_to_dummy([remainder], timesteps),
            dtype=state.dtype,
            device=state.device,
        )
        groups.append((full_steps, last_step))

    def run() -> torch.Tensor:
        current = state
        saved = []
        for full_steps, last_step in groups:
            for _ in range(full_steps):
                current = model.forward_from_state(current, full_step, params)[:, 0, :]
            current = model.forward_from_state(current, last_step, params)[:, 0, :]
            saved.append(current)
        return torch.stack(saved, dim=1)

    return run


def _measure(
    request: Callable[[], torch.Tensor], device: torch.device, size: int
) -> dict[str, Any]:
    with torch.inference_mode():
        request()
        _sync(device)
        if device.type == "cuda":
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(TIMING_REPETITIONS):
            _sync(device)
            start = time.perf_counter()
            request()
            _sync(device)
            samples.append((time.perf_counter() - start) * 1000)
        peak = (
            torch.cuda.max_memory_allocated(device) - before
            if device.type == "cuda"
            else None
        )
    median = float(np.median(samples))
    return {
        "median_ms": median,
        "p95_ms": float(np.percentile(samples, 95)),
        "cells_per_second": size * 1000 / median,
        "peak_memory_bytes": peak,
    }


def _endpoint_references(
    references: np.ndarray | None, states: np.ndarray, timesteps: np.ndarray
) -> np.ndarray | None:
    if references is None:
        return None
    references = np.asarray(references)
    if references.shape != (len(states), len(timesteps), np.shape(states)[1]):
        raise ValueError(
            "reference_trajectories must have shape (N, T, C) matching states and timesteps"
        )
    return references


def _reference_stats(
    request: Callable[[], torch.Tensor], reference: np.ndarray
) -> dict[str, float | None]:
    with torch.inference_mode():
        pred = request()[:, -1, :].cpu().numpy()
    finite = np.isfinite(pred)
    valid = finite & np.isfinite(reference)
    return {
        "normalized_mae": float(np.abs(pred[valid] - reference[valid]).mean())
        if valid.any()
        else None,
        "nonfinite_fraction": float((~finite).mean()),
    }


def _metadata(rollout_step: float | None) -> dict[str, Any]:
    return {
        "warmup": 1,
        "repetitions": TIMING_REPETITIONS,
        "input_residency": "device",
        "resampling": "deterministic cyclic",
        "rollout_step": rollout_step,
    }


def time_endpoint_workload(
    model: Any,
    states: np.ndarray,
    params: np.ndarray | None,
    timesteps: np.ndarray,
    *,
    batch_size: int = 512,
    rollout_step: float | None = None,
    reference_trajectories: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure one endpoint request per horizon at the fixed HPO batch size."""
    size = _batch_size(batch_size)
    _check_rollout_step(rollout_step)
    device = _device(model)
    grid = np.asarray(timesteps, dtype=float)
    horizons = select_horizons(grid)
    state, parameters, indices = _inputs(states, params, size, device)
    refs = _endpoint_references(reference_trajectories, states, grid)
    results = {}
    for horizon in horizons:
        request = _make_request(
            model, state, parameters, np.array([horizon]), grid, rollout_step
        )
        stats = _measure(request, device, size)
        if refs is not None:
            index = np.searchsorted(grid - grid[0], horizon)
            stats.update(_reference_stats(request, refs[indices, index, :]))
        results[str(float(horizon))] = stats
    latencies = [row["median_ms"] for row in results.values()]
    return {
        "geomean_objective_ms": float(np.exp(np.log(latencies).mean())),
        "batch_size": size,
        "horizons": results,
        "device": str(device),
        "metadata": _metadata(rollout_step),
    }


def time_inference_workloads(
    model: Any,
    states: np.ndarray,
    params: np.ndarray | None,
    timesteps: np.ndarray,
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    rollout_step: float | None = None,
    reference_trajectories: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return per-batch endpoint and trajectory curves for the main benchmark."""
    sizes = [_batch_size(size) for size in batch_sizes]
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError(
            "inference_timing.batch_sizes must contain distinct batch sizes"
        )
    _check_rollout_step(rollout_step)
    device = _device(model)
    grid = np.asarray(timesteps, dtype=float)
    horizons = select_horizons(grid)
    refs = _endpoint_references(reference_trajectories, states, grid)
    curve: dict[str, dict[str, dict]] = {"endpoint": {}, "trajectory": {}}
    for size in sizes:
        rows = {kind: curve[kind].setdefault(str(size), {}) for kind in curve}
        try:
            state, parameters, indices = _inputs(states, params, size, device)
        except torch.OutOfMemoryError:
            for row in rows.values():
                row.update({str(float(h)): {"status": "oom"} for h in horizons})
            continue
        for horizon in horizons:
            queries = {
                "endpoint": np.array([horizon]),
                "trajectory": np.linspace(
                    horizon / TRAJECTORY_QUERIES, horizon, TRAJECTORY_QUERIES
                ),
            }
            for kind, times in queries.items():
                try:
                    request = _make_request(
                        model, state, parameters, times, grid, rollout_step
                    )
                    stats = _measure(request, device, size)
                    stats["query_count"] = len(times)
                    if refs is not None and kind == "endpoint":
                        index = np.searchsorted(grid - grid[0], horizon)
                        stats.update(_reference_stats(request, refs[indices, index, :]))
                except torch.OutOfMemoryError:
                    stats = {"status": "oom", "query_count": len(times)}
                rows[kind][str(float(horizon))] = stats
    return {
        **curve,
        "device": str(device),
        "metadata": {
            **_metadata(rollout_step),
            "trajectory_query_count": TRAJECTORY_QUERIES,
        },
    }
