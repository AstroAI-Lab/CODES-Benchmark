import threading
from queue import Queue
from unittest.mock import Mock, patch

import pytest

# import *from* the module that actually defines them
from codes.train.train_fcts import (
    DummyLock,
    create_task_list_for_surrogate,
    parallel_training,
    sequential_training,
    train_and_save_model,
    worker,
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


# — DummyLock —


def test_dummy_lock_noops():
    lock = DummyLock()
    lock.acquire()
    lock.release()
    with lock:
        pass
    # __exit__ is no-op even with args
    lock.__exit__(ValueError, ValueError("e"), None)


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

    # — train_and_save_model —

    @patch("codes.train.train_fcts.load_and_save_config")
    @patch("codes.train.train_fcts.make_description")
    @patch("codes.train.train_fcts.set_random_seeds")
    @patch("codes.train.train_fcts.determine_batch_size")
    @patch("codes.train.train_fcts.get_model_config")
    @patch("codes.train.train_fcts.get_surrogate")
    @patch("codes.train.train_fcts.get_data_subset")
    @patch("codes.train.train_fcts.check_and_load_data")
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
            surr_name="test_surrogate",
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

    @patch("codes.train.train_fcts.load_and_save_config")
    @patch("codes.train.train_fcts.check_and_load_data")
    @patch("codes.train.train_fcts.get_data_subset")
    @patch("codes.train.train_fcts.get_surrogate")
    @patch("codes.train.train_fcts.set_random_seeds")
    @patch("codes.train.train_fcts.determine_batch_size")
    @patch("codes.train.train_fcts.get_model_config")
    @patch("codes.train.train_fcts.make_description")
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
            surr_name="test_surrogate",
            mode="main",
            metric="",
            training_id="minimal_test",
            threadlock=real_lock,
        )

        # should seed twice (once per locked region)
        assert mock_set_random_seeds.call_count == 2


# — worker —


def test_worker_success(tmp_train_dir):
    queue = Queue()
    task = ("A", "main", "", "some_id", 0, 1)
    queue.put(task)

    # write a dummy JSON task-list
    tfile = tmp_train_dir / "tasks.json"
    save_task_list([list(task)], str(tfile))

    # patch the _module_ reference that worker actually uses:
    with patch("codes.train.train_fcts.train_and_save_model") as mock_train:
        progress = Mock()
        errors = [False]
        worker(
            task_queue=queue,
            device="cpu",
            device_idx=0,
            overall_progress_bar=progress,
            task_list_filepath=str(tfile),
            errors_encountered=errors,
            threadlock=threading.Lock(),
        )
    mock_train.assert_called_once()
    progress.update.assert_called_once_with(1)
    assert queue.empty()
    assert not errors[0]


def test_worker_exception(tmp_train_dir):
    queue = Queue()
    task = ("A", "main", "", "some_id", 0, 1)
    queue.put(task)
    tfile = tmp_train_dir / "tasks.json"
    save_task_list([list(task)], str(tfile))

    with patch("codes.train.train_fcts.train_and_save_model", side_effect=RuntimeError):
        progress = Mock()
        errors = [False]
        worker(
            task_queue=queue,
            device="cpu",
            device_idx=0,
            overall_progress_bar=progress,
            task_list_filepath=str(tfile),
            errors_encountered=errors,
            threadlock=threading.Lock(),
        )
    progress.update.assert_called_once_with(1)
    assert queue.empty()
    assert errors[0]


# — parallel & sequential training —


@patch("codes.train.train_fcts.get_progress_bar")
@patch("codes.train.train_fcts.train_and_save_model")
def test_sequential_training(mock_train, mock_pbar, tmp_train_dir):
    tasks = [("A", "main", "", "some_id", 0, 1)] * 3
    # create a real JSON file so load/save work
    tfile = tmp_train_dir / "tasks.json"
    save_task_list([list(t) for t in tasks], str(tfile))

    bar = Mock(format_dict={"elapsed": 2}, update=Mock(), close=Mock())
    mock_pbar.return_value = bar

    elapsed = sequential_training(tasks, ["cpu"], str(tfile))
    assert mock_train.call_count == 3
    assert elapsed == 2
    # file should be removed on success
    assert not tfile.exists()


@patch("codes.train.train_fcts.get_progress_bar")
@patch("codes.train.train_fcts.train_and_save_model")
def test_parallel_training(mock_train, mock_pbar, tmp_train_dir):
    tasks = [("A", "main", "", "some_id", 0, 1)]
    bar = Mock(format_dict={"elapsed": 5}, update=Mock(), close=Mock())
    mock_pbar.return_value = bar

    # no tasks.json needed here; worker will create+remove it itself
    tfile = tmp_train_dir / "tasks.json"
    elapsed = parallel_training(tasks, ["cpu", "cpu"], str(tfile))

    # train called exactly once
    assert mock_train.call_count == 1
    assert elapsed == 5
    # file cleaned up on success
    assert not tfile.exists()


# — task-list utils sanity —


def test_save_and_load_task_list(tmp_train_dir):
    tasks = [["x"], ["y"]]
    f = tmp_train_dir / "my.json"
    save_task_list(tasks, str(f))
    assert load_task_list(str(f)) == tasks
    assert load_task_list(str(tmp_train_dir / "nope.json")) == []
