"""
Pytest configuration and shared fixtures for the CODES benchmark test suite.
"""

import os
import tempfile

import numpy as np
import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "download: marks tests that require downloading data"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


@pytest.fixture(scope="session")
def device():
    """Fixture providing the device to use for testing."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def test_constants():
    """Fixture providing test constants used across multiple test files."""
    return {
        "n_chemicals": 10,
        "n_timesteps": 50,
        "n_parameters": 5,
        "batch_size": 4,
        "n_samples": 8,
        "n_train_samples": 6,
        "n_test_samples": 1,
        "n_val_samples": 1,
    }


@pytest.fixture
def random_seed():
    """Fixture providing a consistent random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory that's cleaned up after tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_3d_data(test_constants, random_seed):
    """Fixture providing sample 3D data arrays for testing."""
    constants = test_constants
    return {
        "train": np.random.rand(
            constants["n_train_samples"],
            constants["n_timesteps"],
            constants["n_chemicals"],
        ).astype(np.float32),
        "test": np.random.rand(
            constants["n_test_samples"],
            constants["n_timesteps"],
            constants["n_chemicals"],
        ).astype(np.float32),
        "val": np.random.rand(
            constants["n_val_samples"],
            constants["n_timesteps"],
            constants["n_chemicals"],
        ).astype(np.float32),
        "timesteps": np.linspace(0, 1, constants["n_timesteps"]).astype(np.float32),
    }


@pytest.fixture
def sample_parameters(test_constants, random_seed):
    """Fixture providing sample parameter arrays for testing."""
    constants = test_constants
    return {
        "train_params": np.random.rand(
            constants["n_train_samples"], constants["n_parameters"]
        ).astype(np.float32),
        "test_params": np.random.rand(
            constants["n_test_samples"], constants["n_parameters"]
        ).astype(np.float32),
        "val_params": np.random.rand(
            constants["n_val_samples"], constants["n_parameters"]
        ).astype(np.float32),
    }


@pytest.fixture
def mock_normalisation():
    """Fixture providing mock normalization parameters for testing."""
    return {
        "mode": "standardise",
        "mean": np.array([0.5, 0.3, 0.7], dtype=np.float32),
        "std": np.array([0.2, 0.1, 0.3], dtype=np.float32),
        "log10_transform": False,
    }


@pytest.fixture
def skip_if_no_gpu():
    """Fixture to skip tests if no GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def original_cwd():
    """Fixture to preserve and restore the original working directory."""
    original = os.getcwd()
    yield original
    os.chdir(original)


# Utility functions for tests
def assert_tensor_properties(
    tensor, expected_shape=None, expected_device=None, expected_dtype=None
):
    """Helper function to assert tensor properties."""
    assert isinstance(tensor, torch.Tensor), (
        f"Expected torch.Tensor, got {type(tensor)}"
    )

    if expected_shape is not None:
        assert tensor.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {tensor.shape}"
        )

    if expected_device is not None:
        expected_device_type = expected_device.split(":")[0]
        assert tensor.device.type == expected_device_type, (
            f"Expected device {expected_device_type}, got {tensor.device.type}"
        )

    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, (
            f"Expected dtype {expected_dtype}, got {tensor.dtype}"
        )


def assert_model_state_consistency(
    model1, model2, check_parameters=True, check_attributes=True
):
    """Helper function to assert that two models have consistent state."""
    assert isinstance(model1, type(model2)), "Models should be of the same type"

    if check_parameters:
        # Check that parameter shapes match
        params1 = list(model1.parameters())
        params2 = list(model2.parameters())
        assert len(params1) == len(params2), (
            "Models should have same number of parameters"
        )

        for p1, p2 in zip(params1, params2):
            assert p1.shape == p2.shape, (
                f"Parameter shapes don't match: {p1.shape} vs {p2.shape}"
            )

    if check_attributes:
        # Check basic attributes
        basic_attrs = ["n_quantities", "n_timesteps", "n_parameters"]
        for attr in basic_attrs:
            if hasattr(model1, attr) and hasattr(model2, attr):
                assert getattr(model1, attr) == getattr(model2, attr), (
                    f"Attribute {attr} doesn't match"
                )


# Export utility functions for use in test modules
__all__ = [
    "assert_tensor_properties",
    "assert_model_state_consistency",
]
