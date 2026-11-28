import numpy as np
import pytest
import torch

from codes.benchmark.workload_timing import (
    select_horizons,
    time_endpoint_workload,
    time_host_to_host_workloads,
    time_inference_workloads,
)


def test_select_horizons_uses_elapsed_geometric_midpoint_and_deduplicates():
    got = select_horizons(np.array([2.0, 3.0, 6.0, 12.0]))
    np.testing.assert_allclose(got, [1.0, 4.0, 10.0])
    np.testing.assert_array_equal(select_horizons(np.array([0.0, 1.0])), [1.0])


class _Echo:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def forward_from_state(self, state, times, params=None):
        self.calls.append(times.detach().cpu().numpy().copy())
        return state[:, None, :].expand(-1, len(times), -1)


def test_endpoint_timing_resamples_and_reports_reference_metrics():
    result = time_endpoint_workload(
        _Echo(),
        np.ones((2, 3), dtype=np.float32),
        None,
        np.array([0.0, 1.0, 2.0]),
        batch_size=5,
        reference_trajectories=np.ones((2, 3, 3), dtype=np.float32),
    )
    assert result["batch_size"] == 5
    assert result["metadata"]["repetitions"] == 5
    assert result["geomean_objective_ms"] > 0
    for metrics in result["horizons"].values():
        assert metrics["normalized_mae"] == 0
        assert metrics["nonfinite_fraction"] == 0


def test_trajectory_direct_mode_uses_one_vector_query_and_reports_cell_rate():
    model = _Echo()
    result = time_inference_workloads(
        model,
        np.ones((2, 1), dtype=np.float32),
        None,
        np.array([0.0, 1.0]),
        batch_sizes=(3,),
    )
    measured_calls = [times for times in model.calls if len(times) == 16]
    assert measured_calls
    stats = result["trajectory"]["3"]["1.0"]
    assert stats["query_count"] == 16
    assert stats["cells_per_second"] > 0


def test_host_to_host_timing_packs_and_returns_physical_states(monkeypatch):
    from codes.benchmark import workload_timing

    returned = []
    measure = workload_timing._measure

    def capture_output(request, *args, **kwargs):
        returned.append(request().copy())
        return measure(request, *args, **kwargs)

    monkeypatch.setattr(workload_timing, "_measure", capture_output)

    class Capture(_Echo):
        def forward_from_state(self, state, times, params=None):
            self.states = state.detach().cpu().numpy().copy()
            return super().forward_from_state(state, times, params)

    model = Capture()
    result = time_host_to_host_workloads(
        model,
        np.array([[0.0], [10.0]], dtype=np.float32),
        None,
        np.array([0.0, 1.0]),
        batch_sizes=(3,),
        normalisation={"mode": "minmax", "min": [0.0], "max": [10.0]},
    )
    np.testing.assert_allclose(model.states[:, 0], [-1.0, 1.0, -1.0])
    for states in returned:
        np.testing.assert_allclose(
            states[:, :, 0], np.broadcast_to([[0.0], [10.0], [0.0]], states.shape[:2])
        )
    assert result["metadata"]["input_residency"] == "host"
    assert result["metadata"]["output_residency"] == "host"
    assert result["endpoint"]["3"]["1.0"]["query_count"] == 1
    assert result["trajectory"]["3"]["1.0"]["query_count"] == 16


def test_endpoint_rollout_policy_advances_in_fixed_steps_and_remainder():
    model = _Echo()
    time_endpoint_workload(
        model,
        np.ones((1, 1), dtype=np.float32),
        None,
        np.array([0.0, 1.0]),
        batch_size=1,
        rollout_step=0.4,
    )
    # The full-span endpoint is reached by .4 + .4 + .2, with each call receiving
    # an input device tensor prepared before timing.
    np.testing.assert_allclose(np.concatenate(model.calls[:3]), [0.4, 0.4, 0.2])
    assert len(model.calls) == 18


def test_rollout_substeps_restart_dummy_time_on_nonuniform_grid():
    model = _Echo()
    time_endpoint_workload(
        model,
        np.ones((1, 1), dtype=np.float32),
        None,
        np.array([0.0, 1.0, 3.0]),
        batch_size=1,
        rollout_step=1.0,
    )
    # Each one-unit advance uses the time from a fresh initial state.
    np.testing.assert_allclose(np.concatenate(model.calls[-3:]), [0.5, 0.5, 0.5])


def test_trajectory_oom_does_not_remove_endpoint_results():
    class Limited(_Echo):
        def forward_from_state(self, state, times, params=None):
            if len(times) > 1 and len(state) > 2:
                raise torch.OutOfMemoryError("test allocation failed")
            return super().forward_from_state(state, times, params)

    result = time_inference_workloads(
        Limited(),
        np.ones((2, 1), dtype=np.float32),
        None,
        np.array([0.0, 1.0]),
        batch_sizes=(2, 4),
    )
    assert result["trajectory"]["4"]["1.0"]["status"] == "oom"
    assert result["endpoint"]["4"]["1.0"]["median_ms"] > 0


def test_fixed_hpo_request_propagates_oom():
    def oom(*args):
        raise torch.OutOfMemoryError("test allocation failed")

    model = _Echo()
    model.forward_from_state = oom
    with pytest.raises(torch.OutOfMemoryError):
        time_endpoint_workload(
            model, np.ones((1, 1), dtype=np.float32), None, np.array([0.0, 1.0])
        )


def test_registered_surrogate_serves_both_timed_workloads(small_surrogate_factory):
    model = small_surrogate_factory(n_parameters=1).eval()
    grid = np.linspace(0.0, 1.0, 8)
    states = np.full((2, 2), 0.5, dtype=np.float32)
    result = time_inference_workloads(
        model, states, np.ones((2, 1)), grid, batch_sizes=(2,)
    )
    for kind, queries in (("endpoint", 1), ("trajectory", 16)):
        row = result[kind]["2"]["1.0"]
        assert row["query_count"] == queries
        assert np.isfinite(row["median_ms"]) and row["median_ms"] > 0
