import os
import tempfile

import h5py
import numpy as np
import pytest
import yaml

from codes.utils.data_utils import (
    DatasetError,
    check_and_load_data,
    create_dataset,
    download_data,
    normalize_data,
)


# Load available datasets from data_sources.yaml
def get_available_datasets():
    """Get list of available datasets from data_sources.yaml."""
    data_sources_path = "datasets/data_sources.yaml"
    if not os.path.exists(data_sources_path):
        return []

    with open(data_sources_path, "r", encoding="utf-8") as f:
        data_sources = yaml.safe_load(f)

    return list(data_sources.keys())


def get_local_datasets():
    """Get list of locally available datasets."""
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        return []

    local_datasets = []
    for item in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, item)
        if os.path.isdir(dataset_path):
            data_file = os.path.join(dataset_path, "data.hdf5")
            if os.path.exists(data_file):
                local_datasets.append(item)

    return local_datasets


AVAILABLE_DATASETS = get_available_datasets()
LOCAL_DATASETS = get_local_datasets()


@pytest.fixture
def sample_dataset_data(test_constants, random_seed):
    """Create sample dataset data for testing."""
    constants = test_constants
    n_samples = 20  # Use a larger sample count for dataset tests

    train_data = np.random.rand(
        n_samples, constants["n_timesteps"], constants["n_chemicals"]
    ).astype(np.float32)
    test_data = np.random.rand(
        n_samples // 4, constants["n_timesteps"], constants["n_chemicals"]
    ).astype(np.float32)
    val_data = np.random.rand(
        n_samples // 4, constants["n_timesteps"], constants["n_chemicals"]
    ).astype(np.float32)
    timesteps = np.linspace(0, 1, constants["n_timesteps"]).astype(np.float32)

    return train_data, test_data, val_data, timesteps


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory for dataset testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestDataSourcesYaml:
    """Test the data_sources.yaml configuration file."""

    def test_data_sources_yaml_exists(self):
        """Test that data_sources.yaml exists and is readable."""
        assert os.path.exists(
            "datasets/data_sources.yaml"
        ), "data_sources.yaml file not found"

    def test_data_sources_yaml_format(self):
        """Test that data_sources.yaml has correct format."""
        with open("datasets/data_sources.yaml", "r", encoding="utf-8") as f:
            data_sources = yaml.safe_load(f)

        assert isinstance(
            data_sources, dict
        ), "data_sources.yaml should contain a dictionary"
        assert len(data_sources) > 0, "data_sources.yaml should not be empty"

    def test_data_sources_urls(self):
        """Test that all datasets have valid URL format."""
        with open("datasets/data_sources.yaml", "r", encoding="utf-8") as f:
            data_sources = yaml.safe_load(f)

        for dataset_name, url in data_sources.items():
            assert isinstance(
                dataset_name, str
            ), f"Dataset name {dataset_name} should be string"
            assert isinstance(url, str), f"URL for {dataset_name} should be string"
            assert url.startswith(
                ("http://", "https://")
            ), f"URL for {dataset_name} should start with http:// or https://"

    def test_expected_datasets_present(self):
        """Test that expected datasets are present in data_sources.yaml."""
        expected_datasets = [
            "osu2008",
            "simple_reaction",
            "simple_ode",
            "lotka_volterra",
            "branca24",
            "simple_primordial",
        ]

        for dataset in expected_datasets:
            assert (
                dataset in AVAILABLE_DATASETS
            ), f"Expected dataset {dataset} not found in data_sources.yaml"


class TestDownloadData:
    """Test dataset downloading functionality."""

    def test_download_data_function_exists(self):
        """Test that download_data function exists and is callable."""
        assert callable(download_data)

    def test_download_data_invalid_dataset(self):
        """Test download_data raises error for invalid dataset."""
        with pytest.raises(ValueError, match="not found in data_sources.yaml"):
            download_data("nonexistent_dataset")

    @pytest.mark.parametrize(
        "dataset_name", AVAILABLE_DATASETS[:2]
    )  # Test only first 2 to avoid long downloads
    def test_download_data_valid_dataset(self, dataset_name, temp_dataset_dir):
        """Test downloading valid datasets to temporary location."""
        download_path = os.path.join(temp_dataset_dir, "data.hdf5")

        try:
            download_data(dataset_name, path=download_path, verbose=False)
            assert os.path.exists(
                download_path
            ), f"Dataset {dataset_name} was not downloaded"
            assert (
                os.path.getsize(download_path) > 0
            ), f"Downloaded file for {dataset_name} is empty"
        except Exception as e:
            pytest.skip(f"Download failed for {dataset_name}: {e}")

    def test_download_data_already_exists(self, temp_dataset_dir):
        """Test that download_data handles existing files correctly."""
        # Create a dummy file
        dataset_name = list(AVAILABLE_DATASETS)[0] if AVAILABLE_DATASETS else "test"
        download_path = os.path.join(temp_dataset_dir, "data.hdf5")

        with open(download_path, "w") as f:
            f.write("dummy content")

        # Should not overwrite existing file
        if AVAILABLE_DATASETS:
            download_data(dataset_name, path=download_path, verbose=False)

            with open(download_path, "r") as f:
                content = f.read()
            assert content == "dummy content", "Existing file should not be overwritten"


class TestLocalDatasets:
    """Test loading of locally available datasets."""

    @pytest.mark.parametrize("dataset_name", LOCAL_DATASETS)
    def test_local_dataset_structure(self, dataset_name):
        """Test that local datasets have correct HDF5 structure."""
        dataset_path = os.path.join("datasets", dataset_name, "data.hdf5")

        with h5py.File(dataset_path, "r") as f:
            # Check required datasets exist
            required_datasets = ["train", "test", "val"]
            for req_dataset in required_datasets:
                assert (
                    req_dataset in f
                ), f"Dataset {dataset_name} missing required '{req_dataset}' data"

            # Check data shapes
            train_data = f["train"]
            test_data = f["test"]
            val_data = f["val"]

            assert (
                len(train_data.shape) == 3
            ), f"Train data in {dataset_name} should be 3D"
            assert (
                len(test_data.shape) == 3
            ), f"Test data in {dataset_name} should be 3D"
            assert len(val_data.shape) == 3, f"Val data in {dataset_name} should be 3D"

            # Check consistent dimensions
            assert (
                train_data.shape[1] == test_data.shape[1] == val_data.shape[1]
            ), f"Inconsistent timestep dimensions in {dataset_name}"
            assert (
                train_data.shape[2] == test_data.shape[2] == val_data.shape[2]
            ), f"Inconsistent quantity dimensions in {dataset_name}"

    @pytest.mark.parametrize("dataset_name", LOCAL_DATASETS)
    def test_local_dataset_timesteps(self, dataset_name):
        """Test that local datasets have timesteps if available."""
        dataset_path = os.path.join("datasets", dataset_name, "data.hdf5")

        with h5py.File(dataset_path, "r") as f:
            if "timesteps" in f:
                timesteps = f["timesteps"]
                train_data = f["train"]

                assert (
                    len(timesteps.shape) == 1
                ), f"Timesteps in {dataset_name} should be 1D"
                assert (
                    timesteps.shape[0] == train_data.shape[1]
                ), f"Timesteps length doesn't match data timesteps in {dataset_name}"

    @pytest.mark.parametrize("dataset_name", LOCAL_DATASETS)
    def test_local_dataset_parameters(self, dataset_name):
        """Test parameter data if available in local datasets."""
        dataset_path = os.path.join("datasets", dataset_name, "data.hdf5")

        with h5py.File(dataset_path, "r") as f:
            param_datasets = ["train_params", "test_params", "val_params"]
            has_params = all(param in f for param in param_datasets)

            if has_params:
                train_params = f["train_params"]
                test_params = f["test_params"]
                val_params = f["val_params"]

                train_data = f["train"]
                test_data = f["test"]
                val_data = f["val"]

                # Check parameter shapes
                assert (
                    len(train_params.shape) == 2
                ), f"Train params in {dataset_name} should be 2D"
                assert (
                    train_params.shape[0] == train_data.shape[0]
                ), f"Train params count doesn't match train data samples in {dataset_name}"

                assert (
                    test_params.shape[0] == test_data.shape[0]
                ), f"Test params count doesn't match test data samples in {dataset_name}"
                assert (
                    val_params.shape[0] == val_data.shape[0]
                ), f"Val params count doesn't match val data samples in {dataset_name}"


class TestCheckAndLoadData:
    """Test the check_and_load_data function."""

    def test_check_and_load_data_function_exists(self):
        """Test that check_and_load_data function exists."""
        assert callable(check_and_load_data)

    def test_check_and_load_data_invalid_dataset(self):
        """Test check_and_load_data raises error for invalid dataset."""
        with pytest.raises(DatasetError, match="not found"):
            check_and_load_data("nonexistent_dataset")

    @pytest.mark.parametrize(
        "dataset_name", LOCAL_DATASETS[:3]
    )  # Test first 3 local datasets
    def test_check_and_load_data_valid_dataset(self, dataset_name):
        """Test loading valid local datasets."""
        try:
            result = check_and_load_data(dataset_name, verbose=False, tolerance=1e-30)

            # Check return tuple structure
            assert len(result) == 6, "check_and_load_data should return 6 elements"

            (
                (train_data, test_data, val_data),
                params,
                timesteps,
                n_train_samples,
                data_info,
                labels,
            ) = result

            # Check data types and shapes
            assert isinstance(train_data, np.ndarray)
            assert isinstance(test_data, np.ndarray)
            assert isinstance(val_data, np.ndarray)
            assert isinstance(timesteps, np.ndarray)
            assert isinstance(
                n_train_samples, (int, np.integer)
            )  # Accept both int and numpy integer types
            assert isinstance(data_info, dict)

            # Check data shapes
            assert train_data.ndim == 3
            assert test_data.ndim == 3
            assert val_data.ndim == 3
            assert timesteps.ndim == 1

            # Check consistent dimensions
            assert (
                train_data.shape[1]
                == test_data.shape[1]
                == val_data.shape[1]
                == timesteps.shape[0]
            )
            assert train_data.shape[2] == test_data.shape[2] == val_data.shape[2]

            # Check sample count
            assert n_train_samples == train_data.shape[0]

        except Exception as e:
            pytest.fail(f"Failed to load dataset {dataset_name}: {e}")

    def test_check_and_load_data_normalization_modes(self):
        """Test different normalization modes."""
        if not LOCAL_DATASETS:
            pytest.skip("No local datasets available")

        dataset_name = LOCAL_DATASETS[0]

        # Test different normalization modes
        modes = ["disable", "minmax", "standardise"]

        for mode in modes:
            try:
                result = check_and_load_data(
                    dataset_name, normalisation_mode=mode, verbose=False
                )

                data_info = result[4]
                assert (
                    "mode" in data_info
                ), f"Data info should contain normalization mode for {mode}"

            except Exception as e:
                pytest.fail(f"Failed to load dataset with {mode} normalization: {e}")

    def test_check_and_load_data_log_transform(self):
        """Test log transformation option."""
        if not LOCAL_DATASETS:
            pytest.skip("No local datasets available")

        dataset_name = LOCAL_DATASETS[0]

        # Test with and without log transform
        for log_transform in [True, False]:
            try:
                result = check_and_load_data(
                    dataset_name, log=log_transform, verbose=False
                )

                data_info = result[4]
                assert "log10_transform" in data_info
                assert data_info["log10_transform"] == log_transform

            except Exception as e:
                pytest.fail(f"Failed to load dataset with log={log_transform}: {e}")

    def test_check_and_load_data_tolerance(self):
        """Test tolerance parameter for log transformation."""
        if not LOCAL_DATASETS:
            pytest.skip("No local datasets available")

        dataset_name = LOCAL_DATASETS[0]

        try:
            result = check_and_load_data(
                dataset_name, log=True, tolerance=1e-10, verbose=False
            )

            # Should complete without error
            assert result is not None

        except Exception as e:
            pytest.fail(f"Failed to load dataset with tolerance: {e}")


class TestCreateDataset:
    """Test dataset creation functionality."""

    def test_create_dataset_function_exists(self):
        """Test that create_dataset function exists."""
        assert callable(create_dataset)

    def test_create_dataset_with_single_array(
        self, sample_dataset_data, temp_dataset_dir
    ):
        """Test creating dataset from single data array."""
        train_data, test_data, val_data, timesteps = sample_dataset_data

        # Combine all data
        all_data = np.concatenate([train_data, test_data, val_data], axis=0)

        # Change to temp directory temporarily
        original_cwd = os.getcwd()
        temp_datasets_dir = os.path.join(temp_dataset_dir, "datasets")
        os.makedirs(temp_datasets_dir)

        try:
            os.chdir(temp_dataset_dir)

            create_dataset(
                name="test_dataset",
                data=all_data,
                timesteps=timesteps,
                split=(0.7, 0.15, 0.15),
            )

            # Check that dataset was created
            dataset_path = os.path.join(temp_datasets_dir, "test_dataset", "data.hdf5")
            assert os.path.exists(dataset_path), "Dataset file was not created"

            # Check dataset structure
            with h5py.File(dataset_path, "r") as f:
                assert "train" in f
                assert "test" in f
                assert "val" in f
                assert "timesteps" in f

        finally:
            os.chdir(original_cwd)

    def test_create_dataset_with_tuple_data(
        self, sample_dataset_data, temp_dataset_dir
    ):
        """Test creating dataset from tuple of data arrays."""
        train_data, test_data, val_data, timesteps = sample_dataset_data

        original_cwd = os.getcwd()
        temp_datasets_dir = os.path.join(temp_dataset_dir, "datasets")
        os.makedirs(temp_datasets_dir)

        try:
            os.chdir(temp_dataset_dir)

            create_dataset(
                name="test_dataset_tuple",
                data=(train_data, test_data, val_data),
                timesteps=timesteps,
            )

            # Check that dataset was created
            dataset_path = os.path.join(
                temp_datasets_dir, "test_dataset_tuple", "data.hdf5"
            )
            assert os.path.exists(dataset_path), "Dataset file was not created"

            # Check data shapes match
            with h5py.File(dataset_path, "r") as f:
                assert f["train"].shape == train_data.shape
                assert f["test"].shape == test_data.shape
                assert f["val"].shape == val_data.shape

        finally:
            os.chdir(original_cwd)

    def test_create_dataset_with_parameters(
        self, sample_dataset_data, temp_dataset_dir
    ):
        """Test creating dataset with parameter data."""
        train_data, test_data, val_data, timesteps = sample_dataset_data

        # Create sample parameters
        n_params = 5
        train_params = np.random.rand(train_data.shape[0], n_params).astype(np.float32)
        test_params = np.random.rand(test_data.shape[0], n_params).astype(np.float32)
        val_params = np.random.rand(val_data.shape[0], n_params).astype(np.float32)

        original_cwd = os.getcwd()
        temp_datasets_dir = os.path.join(temp_dataset_dir, "datasets")
        os.makedirs(temp_datasets_dir)

        try:
            os.chdir(temp_dataset_dir)

            create_dataset(
                name="test_dataset_params",
                data=(train_data, test_data, val_data),
                params=(train_params, test_params, val_params),
                timesteps=timesteps,
            )

            # Check that dataset was created with parameters
            dataset_path = os.path.join(
                temp_datasets_dir, "test_dataset_params", "data.hdf5"
            )
            assert os.path.exists(dataset_path), "Dataset file was not created"

            with h5py.File(dataset_path, "r") as f:
                assert "train_params" in f
                assert "test_params" in f
                assert "val_params" in f
                assert f["train_params"].shape == train_params.shape

        finally:
            os.chdir(original_cwd)

    def test_create_dataset_with_labels(self, sample_dataset_data, temp_dataset_dir):
        """Test creating dataset with quantity labels."""
        train_data, test_data, val_data, timesteps = sample_dataset_data

        labels = [f"species_{i}" for i in range(train_data.shape[2])]

        original_cwd = os.getcwd()
        temp_datasets_dir = os.path.join(temp_dataset_dir, "datasets")
        os.makedirs(temp_datasets_dir)

        try:
            os.chdir(temp_dataset_dir)

            create_dataset(
                name="test_dataset_labels",
                data=(train_data, test_data, val_data),
                timesteps=timesteps,
                labels=labels,
            )

            # Check that dataset was created with labels
            dataset_path = os.path.join(
                temp_datasets_dir, "test_dataset_labels", "data.hdf5"
            )
            assert os.path.exists(dataset_path), "Dataset file was not created"

            with h5py.File(dataset_path, "r") as f:
                assert "labels" in f.attrs
                stored_labels = [
                    label.decode("utf-8") if isinstance(label, bytes) else label
                    for label in f.attrs["labels"]
                ]
                assert stored_labels == labels

        finally:
            os.chdir(original_cwd)


class TestNormalizeData:
    """Test data normalization functionality."""

    def test_normalize_data_function_exists(self):
        """Test that normalize_data function exists."""
        assert callable(normalize_data)

    def test_normalize_data_minmax(self, sample_dataset_data):
        """Test minmax normalization."""
        train_data, test_data, val_data, _ = sample_dataset_data

        data_info, train_norm, test_norm, val_norm = normalize_data(
            train_data, test_data, val_data, mode="minmax"
        )

        assert data_info["mode"] == "minmax"
        assert "min" in data_info
        assert "max" in data_info

        # Check normalization range [-1, 1]
        assert train_norm.min() >= -1.0
        assert train_norm.max() <= 1.0

    def test_normalize_data_standardize(self, sample_dataset_data):
        """Test standardization normalization."""
        train_data, test_data, val_data, _ = sample_dataset_data

        data_info, train_norm, test_norm, val_norm = normalize_data(
            train_data, test_data, val_data, mode="standardise"
        )

        assert data_info["mode"] == "standardise"
        assert "mean" in data_info
        assert "std" in data_info

        # Check standardization (mean ≈ 0, std ≈ 1)
        assert abs(train_norm.mean()) < 0.1
        assert abs(train_norm.std() - 1.0) < 0.1

    def test_normalize_data_per_species(self, sample_dataset_data):
        """Test per-species normalization."""
        train_data, test_data, val_data, _ = sample_dataset_data

        data_info, train_norm, test_norm, val_norm = normalize_data(
            train_data, test_data, val_data, mode="minmax", per_species=True
        )

        # Check that normalization parameters have correct shape
        assert hasattr(data_info["min"], "shape")
        assert hasattr(data_info["max"], "shape")
        assert data_info["min"].shape == (train_data.shape[2],)
        assert data_info["max"].shape == (train_data.shape[2],)

    def test_normalize_data_invalid_mode(self, sample_dataset_data):
        """Test that invalid normalization mode raises error."""
        train_data, test_data, val_data, _ = sample_dataset_data

        with pytest.raises(ValueError, match="Mode must be either"):
            normalize_data(train_data, test_data, val_data, mode="invalid")

    def test_normalize_data_none_inputs(self, sample_dataset_data):
        """Test normalization with None test/val data."""
        train_data, _, _, _ = sample_dataset_data

        data_info, train_norm, test_norm, val_norm = normalize_data(
            train_data, None, None, mode="minmax"
        )

        assert train_norm is not None
        assert test_norm is None
        assert val_norm is None


class TestDatasetError:
    """Test DatasetError exception."""

    def test_dataset_error_is_exception(self):
        """Test that DatasetError is a proper exception."""
        assert issubclass(DatasetError, Exception)

    def test_dataset_error_can_be_raised(self):
        """Test that DatasetError can be raised with message."""
        with pytest.raises(DatasetError, match="test error"):
            raise DatasetError("test error")


class TestIntegration:
    """Integration tests combining multiple dataset operations."""

    def test_full_dataset_workflow(self, sample_dataset_data, temp_dataset_dir):
        """Test complete workflow: create, save, and load dataset."""
        train_data, test_data, val_data, timesteps = sample_dataset_data

        original_cwd = os.getcwd()
        temp_datasets_dir = os.path.join(temp_dataset_dir, "datasets")
        os.makedirs(temp_datasets_dir)

        try:
            os.chdir(temp_dataset_dir)

            # Create dataset
            dataset_name = "integration_test"
            create_dataset(
                name=dataset_name,
                data=(train_data, test_data, val_data),
                timesteps=timesteps,
            )

            # Load dataset
            result = check_and_load_data(dataset_name, verbose=False)
            loaded_data, params, loaded_timesteps, n_samples, data_info, labels = result

            # Verify loaded data matches original
            loaded_train, loaded_test, loaded_val = loaded_data

            # Shapes should match (allowing for normalization effects)
            assert loaded_train.shape[1:] == train_data.shape[1:]
            assert loaded_test.shape[1:] == test_data.shape[1:]
            assert loaded_val.shape[1:] == val_data.shape[1:]
            assert loaded_timesteps.shape == timesteps.shape

        finally:
            os.chdir(original_cwd)


class TestCheckAndLoadDataEdgeCases:
    """Loading a small synthetic dataset with transformations that used to fail."""

    @pytest.fixture
    def toy_dataset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rng = np.random.default_rng(0)
        data = rng.uniform(0.1, 1.0, size=(20, 6, 3)).astype(np.float32)
        data[0, 0, 0] = 0.0
        params = rng.uniform(1.0, 2.0, size=(20, 2)).astype(np.float32)
        create_dataset("toy", data, params=params)
        return "toy"

    def test_log_transform_rejects_non_positive_data(self, toy_dataset):
        """Non-positive values must not silently turn into -inf or NaN."""
        with pytest.raises(DatasetError, match="non-positive"):
            check_and_load_data(toy_dataset, verbose=False, log=True)

    def test_tolerance_makes_data_loggable(self, toy_dataset):
        (train, test, val), *_ = check_and_load_data(
            toy_dataset, verbose=False, log=True, tolerance=1e-10
        )
        assert all(np.all(np.isfinite(d)) for d in (train, test, val))

    @pytest.mark.parametrize("verbose", [False, True])
    def test_parameters_without_normalisation(self, toy_dataset, verbose):
        """Parametric data must load with normalisation disabled."""
        _, (train_params, _, _), _, _, data_info, _ = check_and_load_data(
            toy_dataset, verbose=verbose, log=False, normalisation_mode="disable"
        )
        assert train_params is not None
        assert data_info["mode"] == "disable"
