import numpy as np
import pytest
import torch

import codes.benchmark.bench_fcts as bf


class FakeModel:
    def __init__(self, *, train_duration=2.5, n_quantities=2):
        self.train_duration = train_duration
        self.n_quantities = n_quantities
        self.load_calls = []

    def load(self, training_id, surr_name, model_identifier):
        self.load_calls.append((training_id, surr_name, model_identifier))

    def predict(self, data_loader, leave_log=None, leave_norm=None):
        """
        Return preds and targets of shape [1, 3, n_quantities] where
        preds are always 2x targets.
        """
        batch, T, Q = 1, 3, self.n_quantities
        # build a time axis [1,2,3] and expand to [1,3,1]
        t = torch.arange(1, T + 1, dtype=torch.float32).reshape(1, T, 1)
        targets = (t**2).expand(batch, T, Q)  # [[0,1,4], …]
        preds = (targets * 2).expand(batch, T, Q)  # [[0,2,8], …]
        return preds, targets

    def denormalize(self, arr, leave_log=None, leave_norm=None):
        # return the array as is, no normalization
        return arr


@pytest.fixture(autouse=True)
def no_plots(monkeypatch):
    # patch out all the plotting functions so they don't error or try to open displays
    for fn in [
        "plot_error_percentiles_over_time",
        "plot_error_distribution_per_quantity",
        "plot_gradients_heatmap",
    ]:
        monkeypatch.setattr(
            bf,
            fn,
            lambda *args, **kwargs: (
                (  # for dynamic corr return dummy tuple
                    5,
                    0.1,
                    0.2,
                )
                if fn == "plot_gradients_heatmap"
                else None
            ),
        )


@pytest.fixture
def simple_conf():
    return {
        "training_id": "TID",
        "surrogates": ["SurrA", "SurrB"],
        "epochs": [10, 20],
        "relative_error_threshold": 0.0,
        "dataset": {"log_timesteps": False},
    }


@pytest.fixture
def simple_loader():
    # loader doesn't matter, it's not actually iterated for accuracy functions
    return "dummy_loader"


def test_evaluate_accuracy(simple_conf, simple_loader):
    timesteps = np.array([1.0, 2.0, 3.0])
    model = FakeModel(train_duration=3.14, n_quantities=2)
    # insert the right surrogate name into conf
    surr = "SurrB"
    simple_conf["surrogates"] = ["SurrA", surr]
    # call
    metrics = bf.evaluate_accuracy(
        model=model,
        surr_name=surr,
        timesteps=timesteps,
        test_loader=simple_loader,
        conf=simple_conf,
        labels=["q1", "q2"],
    )
    # check load called with main identifier
    assert model.load_calls == [("TID", surr, f"{surr.lower()}_main")]
    # mean squared error ≈ 98/3
    assert metrics["root_mean_squared_error_real"] == pytest.approx(np.sqrt(98 / 3))

    # mean absolute error ≈ 14/3
    assert metrics["mean_absolute_error_real"] == pytest.approx(14 / 3)
    # relative errors: abs(1)/max(abs(0),0.0) -> 1/0 -> inf; but threshold=0 so yields 1.0
    assert metrics["mean_relative_error"] == pytest.approx(1.0)
    assert metrics["main_model_training_time"] == 3.14
    # epoch picked from conf
    assert metrics["main_model_epochs"] == simple_conf["epochs"][1]
    # absolute_errors array shape should match
    assert metrics["absolute_errors"].shape == (1, 3, 2)
    assert metrics["relative_errors"].shape == (1, 3, 2)


def test_evaluate_gradients(simple_conf, simple_loader):
    model = FakeModel(n_quantities=1)
    surr = "SurrA"
    simple_conf["surrogates"] = [surr]
    # call
    out = bf.evaluate_gradients(
        model=model,
        surr_name=surr,
        test_loader=simple_loader,
        conf=simple_conf,
        species_names=["X"],
    )
    # load with main
    assert model.load_calls == [("TID", surr, f"{surr.lower()}_main")]
    # gradients: since targets zeros -> gradient zeros -> normalized stays nan or zero,
    # but our fake plot returns max_count=5, max_grad=0.1, max_err=0.2
    assert out["max_counts"] == 5
    assert out["max_gradient"] == pytest.approx(0.1)
    assert out["max_error"] == pytest.approx(0.2)
    # species_correlations should have one entry for "X"
    assert "X" in out["species_correlations"]
    assert isinstance(out["avg_correlation"], np.float32)


def test_time_inference(monkeypatch):
    # patch measure_inference_time to return fixed list
    monkeypatch.setattr(
        bf, "measure_inference_time", lambda model, loader, n_runs=None: [1.0, 3.0, 5.0]
    )
    model = FakeModel()
    surr = "SurrA"
    conf = {"training_id": "TID"}
    # call
    result = bf.time_inference(
        model, surr, test_loader="L", conf=conf, n_test_samples=2, n_runs=3
    )
    assert result["mean_inference_time_per_run"] == pytest.approx(3.0)
    assert result["std_inference_time_per_run"] == pytest.approx(
        np.std([1.0, 3.0, 5.0])
    )
    assert result["num_predictions"] == 2
    assert result["mean_inference_time_per_prediction"] == pytest.approx(3.0 / 2)


def test_evaluate_compute_cpu_path(monkeypatch):
    # Force non-CUDA execution
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # measure_memory_footprint should not be called
    called = {"mem": False}

    def fake_measure(model, inputs, device):
        called["mem"] = True
        return {}, model

    monkeypatch.setattr(bf, "measure_memory_footprint", fake_measure)
    monkeypatch.setattr(bf, "count_trainable_parameters", lambda m: 12345)

    class DummyLoader:
        def __iter__(self):
            yield ("inp",)

    loader = DummyLoader()
    model = FakeModel()
    surr = "SurrB"
    conf = {"training_id": "TID"}
    out = bf.evaluate_compute(model, surr, test_loader=loader, conf=conf)
    assert model.load_calls == [("TID", surr, f"{surr.lower()}_main")]
    assert out["num_trainable_parameters"] == 12345
    assert out["memory_footprint"] == {}
    assert not called["mem"]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable for memory profiling test."
)
def test_evaluate_compute_cuda_path(monkeypatch):
    fake_mem = {"model_memory": 100, "forward_memory_nograd": 50}

    def fake_measure(model, inputs, device):
        return fake_mem, model

    monkeypatch.setattr(bf, "measure_memory_footprint", fake_measure)
    monkeypatch.setattr(bf, "count_trainable_parameters", lambda m: 12345)

    class DummyLoader:
        def __iter__(self):
            yield ("inp",)

    loader = DummyLoader()
    model = FakeModel()
    surr = "SurrB"
    conf = {"training_id": "TID"}
    out = bf.evaluate_compute(model, surr, test_loader=loader, conf=conf)
    assert model.load_calls == [("TID", surr, f"{surr.lower()}_main")]
    assert out["num_trainable_parameters"] == 12345
    assert out["memory_footprint"] is fake_mem


def test_evaluate_iterative_predictions(simple_conf, simple_loader, monkeypatch):
    import numpy as np
    import torch

    import codes.benchmark.bench_fcts as bf

    # stub out the final plot_example_iterative_predictions so it doesn't call save_plot
    monkeypatch.setattr(
        bf, "plot_example_iterative_predictions", lambda *args, **kwargs: None
    )

    # model with T=12 timesteps and Q=2 quantities
    T, Q = 12, 2
    timesteps = np.arange(1.0, T + 1.0)

    class FakeIterModel:
        def __init__(self, n_timesteps, n_quantities):
            self.n_timesteps = n_timesteps
            self.n_quantities = n_quantities
            self.load_calls = []

        def load(self, training_id, surr_name, model_identifier):
            self.load_calls.append((training_id, surr_name, model_identifier))

        def predict(self, data_loader, leave_log: bool = False, leave_norm=None):
            # preds == targets == ones
            shape = (1, self.n_timesteps, self.n_quantities)
            ones = torch.ones(shape, dtype=torch.float32)
            return ones, ones

        def prepare_data(self, **kwargs):
            # only the returned loader is used by predict
            return "train_loader", None, None

        def denormalize(self, arr, leave_log=None, leave_norm=None):
            return arr

    model = FakeIterModel(n_timesteps=T, n_quantities=Q)

    surr = "SurrA"
    simple_conf["surrogates"] = [surr]
    simple_conf["batch_size"] = 4
    simple_conf["relative_error_threshold"] = 0.0
    simple_conf["training_id"] = "TID"  # ensure load uses this

    metrics = bf.evaluate_iterative_predictions(
        model=model,
        surr_name=surr,
        timesteps=timesteps,
        val_loader="dummy_val_loader",
        val_params=None,
        conf=simple_conf,
        labels=["q1", "q2"],
    )

    # ensure we loaded the main model
    assert model.load_calls == [("TID", surr, f"{surr.lower()}_main")]

    # ensure that each m
    for key in [
        "root_mean_squared_error_log",
        "mean_absolute_error_log",
        "percentile_absolute_error_log",
        "mean_squared_error",
        "mean_absolute_error",
        "absolute_errors",
    ]:
        print(metrics[key])
        assert metrics[key] == pytest.approx(0.0)

    # array shapes should be (1, T, Q)
    assert metrics["absolute_errors"].shape == (1, T, Q)
