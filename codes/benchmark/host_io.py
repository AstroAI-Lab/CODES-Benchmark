"""Host-side input and output transforms for inference timing.

These helpers keep physical-unit conversion explicit so benchmark callers can
stage their buffers before starting a timed host-to-host measurement.
"""

from __future__ import annotations

import os

import h5py
import numpy as np


def load_physical_validation_inputs(
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load only the physical initial states and optional validation parameters.

    Parameter availability follows ``check_and_load_data``: all three split
    parameter datasets must exist before validation parameters are returned.
    """
    dataset_path = os.path.join("datasets", dataset_name.lower(), "data.hdf5")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"data.hdf5 file not found for dataset {dataset_name!r}: {dataset_path}"
        )
    with h5py.File(dataset_path, "r") as handle:
        if "val" not in handle:
            raise KeyError(f"'val' data not found in {dataset_path}")
        # Read the hyperslab directly; do not load complete trajectories.
        states = np.asarray(handle["val"][:, 0, :], dtype=np.float32)
        has_params = all(
            key in handle for key in ("train_params", "test_params", "val_params")
        )
        params = (
            np.asarray(handle["val_params"], dtype=np.float32) if has_params else None
        )
    return states, params


def _stats(normalisation: dict, parameters: bool) -> tuple[str, object, object]:
    mode = normalisation.get("mode", "disable")
    if parameters:
        if mode == "minmax":
            return mode, normalisation["min_params"], normalisation["max_params"]
        if mode in ("standardise", "standardize"):
            return mode, normalisation["mean_params"], normalisation["std_params"]
        return mode, None, None
    if mode == "minmax":
        return mode, normalisation["min"], normalisation["max"]
    if mode in ("standardise", "standardize"):
        return mode, normalisation["mean"], normalisation["std"]
    return mode, None, None


def pack_model_inputs(
    source: np.ndarray,
    destination: np.ndarray,
    normalisation: dict,
    *,
    parameters: bool = False,
    log: bool = False,
    tolerance: float | None = None,
) -> None:
    """Convert physical float inputs into CODES model units in-place.

    ``destination`` must be a writable float32 buffer with the same shape as
    ``source``. The function intentionally returns nothing to avoid replacing
    caller-owned (for example pinned) storage.
    """
    if destination.dtype != np.float32:
        raise TypeError("destination must have dtype float32")
    if destination.shape != source.shape:
        raise ValueError("source and destination must have the same shape")
    np.copyto(destination, source, casting="unsafe")
    if tolerance is not None:
        np.maximum(destination, np.float32(tolerance), out=destination)
    if log:
        if np.any(destination <= 0):
            raise ValueError(
                "Data contains non-positive values and cannot be log-transformed"
            )
        np.log10(destination, out=destination)

    mode, first, second = _stats(normalisation, parameters)
    if mode == "minmax":
        # Keep the same float32 operation order as normalize_data.
        np.subtract(destination, first, out=destination)
        np.multiply(destination, np.float32(2), out=destination)
        np.divide(destination, np.subtract(second, first), out=destination)
        np.subtract(destination, np.float32(1), out=destination)
    elif mode in ("standardise", "standardize"):
        np.subtract(destination, first, out=destination)
        np.divide(destination, second, out=destination)
    elif mode not in ("disable", "disabled"):
        raise ValueError(f"Unsupported normalisation mode: {mode!r}")


def unpack_model_outputs(
    normalisation: dict,
    normalized_numpy: np.ndarray,
    output_numpy: np.ndarray,
    *,
    log: bool = False,
) -> None:
    """Convert normalized model outputs to physical units in a preallocated buffer."""
    if output_numpy.dtype != np.float32:
        raise TypeError("output_numpy must have dtype float32")
    if output_numpy.shape != normalized_numpy.shape:
        raise ValueError("normalized_numpy and output_numpy must have the same shape")
    np.copyto(output_numpy, normalized_numpy, casting="unsafe")
    mode, first, second = _stats(normalisation, parameters=False)
    if mode == "minmax":
        # AbstractSurrogate.denormalize: (x + 1) * (max - min) / 2 + min.
        np.add(output_numpy, np.float32(1), out=output_numpy)
        np.multiply(output_numpy, np.subtract(second, first), out=output_numpy)
        np.divide(output_numpy, np.float32(2), out=output_numpy)
        np.add(output_numpy, first, out=output_numpy)
    elif mode in ("standardise", "standardize"):
        np.multiply(output_numpy, second, out=output_numpy)
        np.add(output_numpy, first, out=output_numpy)
    elif mode not in ("disable", "disabled"):
        raise ValueError(f"Unsupported normalisation mode: {mode!r}")
    if log:
        np.power(np.float32(10), output_numpy, out=output_numpy)
