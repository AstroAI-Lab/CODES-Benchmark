import threading
from unittest.mock import Mock, patch

import pytest

from codes.train.tasks import train_and_save_model

# import *from* the module that actually defines them
from codes.train.train_fcts import (
    create_task_list_for_surrogate,
)
from codes.utils import load_task_list, save_task_list

# — fixtures —


@pytest.fixture
def sample_config():
    return {
        "training_id": "t123",
        "seed": 10,
        "surrogates": ["A", "B"],
        "epochs": [5, 10],
        "dataset": {
            "name": "ds",
            "log10_transform": False,
            "normalise": "minmax",
            "tolerance": 1e-3,
        },
        "interpolation": {"enabled": True, "intervals": [1, 2]},
        "extrapolation": {"enabled": False},
        "sparse": {"enabled": False},
        "uncertainty": {"enabled": False},
        "batch_scaling": {"enabled": False},
    }


@pytest.fixture
def minimal_config():
    return {
        "training_id": "mini",
        "seed": 0,
        "surrogates": ["A"],
        "epochs": 3,
        "dataset": {
            "name": "ds",
            "log10_transform": False,
            "normalise": "none",
            "tolerance": 1e-3,
        },
        "interpolation": {"enabled": False},
        "extrapolation": {"enabled": False},
        "sparse": {"enabled": False},
        "uncertainty": {"enabled": False},
        "batch_scaling": {"enabled": False},
    }


@pytest.fixture
def tmp_train_dir(tmp_path):
    # create a "trained/<id>" tree for tests that write files
    d = tmp_path / "trained" / "some_id"
    d.mkdir(parents=True)
    return tmp_path


# — create_task_list_for_surrogate —


def test_create_task_list_minimal(minimal_config):
    tasks = create_task_list_for_surrogate(minimal_config, "A")
    assert tasks == [("A", "main", "", "mini", 0, 3)]


def test_create_task_list_with_interpolation(sample_config):
    tasks = create_task_list_for_surrogate(sample_config, "A")
    # main + 2 interpolation entries
    assert len(tasks) == 1 + 2
    interpol = [t for t in tasks if t[1] == "interpolation"]
    assert [t[2] for t in interpol] == [1, 2]


def test_uq_members_get_distinct_seeds(minimal_config):
    minimal_config["uncertainty"] = {"enabled": True, "ensemble_size": 5}
    tasks = create_task_list_for_surrogate(minimal_config, "A")
    seeds = {t[2] or "main": t[4] for t in tasks}
    # Member k uses seed + k, so no member repeats the main model's seed.
    assert seeds == {"main": 0, 1: 1, 2: 2, 3: 3, 4: 4}


# — train_and_save_model —


class TestTrainAndSaveModel:
    @patch("codes.train.tasks.load_and_save_config")
    @patch("codes.train.tasks.make_description")
    @patch("codes.train.tasks.set_random_seeds")
    @patch("codes.train.tasks.determine_batch_size")
    @patch("codes.train.tasks.get_model_config")
    @patch("codes.train.tasks.get_surrogate")
    @patch("codes.train.tasks.get_data_subset")
    @patch("codes.train.tasks.check_and_load_data")
    def test_train_and_save_model_success(
        self,
        mock_check_and_load_data,
        mock_get_data_subset,
        mock_get_surrogate,
        mock_get_model_config,
        mock_determine_batch_size,
        mock_set_random_seeds,
        mock_make_description,
        mock_load_and_save_config,
        sample_config,
    ):
        """Test successful model training and saving."""

        # load_and_save_config returns our fixture (so no real file I/O)
        mock_load_and_save_config.return_value = sample_config

        # stub the helpers
        mock_make_description.return_value = "Test Description"
        mock_determine_batch_size.return_value = 32
        mock_get_model_config.return_value = {}

        # check_and_load_data => train_data, params, etc.
        train_data = Mock()
        train_data.shape = (100, 50, 10)
        params = Mock()
        params.shape = (100, 5)
        timesteps = Mock()
        info = Mock()
        mock_check_and_load_data.return_value = (
            (train_data, Mock(), None),
            (params, Mock(), None),
            timesteps,
            None,
            info,
            None,
        )

        # get_data_subset => subsets
        mock_get_data_subset.return_value = (
            (train_data, Mock()),
            (params, Mock()),
            timesteps,
        )

        # surrogate factory => a dummy model
        model = Mock()
        model.prepare_data.return_value = (Mock(), Mock(), None)
        mock_get_surrogate.return_value = lambda *args, **kw: model

        # run it
        train_and_save_model(
            surr_name="A",
            mode="main",
            metric="",
            training_id="test_123",
            seed=42,
            epochs=100,
            device="cpu",
        )

        # assertions: ensure we at least fit & save
        mock_load_and_save_config.assert_called_once()  # was invoked
        model.fit.assert_called_once()  # training happened
        model.save.assert_called_once()  # model was saved

    @patch("codes.train.tasks.load_and_save_config")
    @patch("codes.train.tasks.check_and_load_data")
    @patch("codes.train.tasks.get_data_subset")
    @patch("codes.train.tasks.get_surrogate")
    @patch("codes.train.tasks.set_random_seeds")
    @patch("codes.train.tasks.determine_batch_size")
    @patch("codes.train.tasks.get_model_config")
    @patch("codes.train.tasks.make_description")
    def test_train_and_save_model_with_thread_lock(
        self,
        mock_make_description,
        mock_get_model_config,
        mock_determine_batch_size,
        mock_set_random_seeds,
        mock_get_surrogate,
        mock_get_data_subset,
        mock_check_and_load_data,
        mock_load_and_save_config,
        minimal_config,
    ):
        """Test that thread lock is used properly."""

        # stub config load and helpers
        mock_load_and_save_config.return_value = minimal_config
        mock_determine_batch_size.return_value = 1
        mock_get_model_config.return_value = {}
        mock_make_description.return_value = "desc"

        # minimal data mocks
        train = Mock()
        train.shape = (10, 2, 1)
        mock_check_and_load_data.return_value = (
            (train, Mock(), None),
            (None, None, None),
            Mock(),
            None,
            Mock(),
            None,
        )
        mock_get_data_subset.return_value = ((train, Mock()), (None, None), Mock())

        # surrogate => model stub
        model = Mock()
        model.prepare_data.return_value = (Mock(), Mock(), None)
        mock_get_surrogate.return_value = lambda *args, **kw: model

        # real lock
        real_lock = threading.Lock()

        # call under test (we omit epochs here, it falls back)
        train_and_save_model(
            surr_name="A",
            mode="main",
            metric="",
            training_id="minimal_test",
            threadlock=real_lock,
        )

        # should seed twice (once per locked region)
        assert mock_set_random_seeds.call_count == 2


# — task-list utils sanity —


def test_save_and_load_task_list(tmp_train_dir):
    tasks = [["x"], ["y"]]
    f = tmp_train_dir / "my.json"
    save_task_list(tasks, str(f))
    assert load_task_list(str(f)) == tasks
    assert load_task_list(str(tmp_train_dir / "nope.json")) == []
