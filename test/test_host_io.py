import h5py
import numpy as np
import pytest

from codes.benchmark.host_io import (
    load_physical_validation_inputs,
    pack_model_inputs,
    unpack_model_outputs,
)
from codes.utils import check_and_load_data


@pytest.mark.parametrize("mode", ["minmax", "standardise", "disable"])
def test_physical_io_matches_the_validation_loader(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "datasets" / "test"
    folder.mkdir(parents=True)
    trajectories = np.array(
        [[[0.0, 20.0], [2.0, 120.0]], [[1.0, 10.0], [3.0, 80.0]]], dtype=np.float32
    )
    params = np.array([[0.01, 10.0], [0.02, 20.0]], dtype=np.float32)
    with h5py.File(folder / "data.hdf5", "w") as data:
        for name in ("train", "test", "val"):
            data[name] = trajectories
            data[f"{name}_params"] = params

    (_, _, expected), (_, _, expected_params), _, _, stats, _ = check_and_load_data(
        "test", verbose=False, normalisation_mode=mode, per_species=True, tolerance=0.1
    )
    raw, raw_params = load_physical_validation_inputs("test")
    np.testing.assert_equal(raw, trajectories[:, 0, :])
    np.testing.assert_equal(raw_params, params)

    packed = np.empty_like(raw)
    packed_params = np.empty_like(params)
    pack_model_inputs(raw, packed, stats, log=True, tolerance=0.1)
    pack_model_inputs(raw_params, packed_params, stats, parameters=True, log=True)
    np.testing.assert_allclose(packed, expected[:, 0, :], atol=1e-6)
    np.testing.assert_allclose(packed_params, expected_params, atol=1e-6)

    output = np.empty_like(expected)
    unpack_model_outputs(stats, expected, output, log=True)
    np.testing.assert_allclose(output, np.maximum(trajectories, 0.1), rtol=1e-6)
