import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from codes.benchmark import bench_utils as bu


@pytest.fixture(autouse=True)
def chdir_tmp_path(tmp_path, monkeypatch):
    """Run each test in a temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_get_required_models_list():
    s = "TestSurr"
    conf = {
        "gradients": False,
        "interpolation": {"enabled": True, "intervals": [2, 4]},
        "extrapolation": {"enabled": True, "cutoffs": [3]},
        "sparse": {"enabled": True, "factors": [2]},
        "uncertainty": {"enabled": True, "ensemble_size": 3},
    }
    out = bu.get_required_models_list(s, conf)
    # main
    assert "testsurr_main.pth" in out
    # interpolation
    assert "testsurr_interpolation_2.pth" in out
    assert "testsurr_interpolation_4.pth" in out
    # extrapolation
    assert "testsurr_extrapolation_3.pth" in out
    # sparse
    assert "testsurr_sparse_2.pth" in out
    # UQ: ensemble_size=3 → two extra
    assert "testsurr_UQ_1.pth" in out and "testsurr_UQ_2.pth" in out


def test_convert_and_discard_numpy_types():
    arr = np.array([1, 2, 3])
    scalar = np.float32(2.5)
    nested = {"a": arr, "b": {"c": scalar, "d": 5}, "e": [arr, 7]}
    converted = bu.convert_to_standard_types(nested)
    # arrays → lists
    assert isinstance(converted["b"]["c"], float)
    assert converted["b"]["d"] == 5
    assert isinstance(converted["e"][0], list)
    # discard_numpy_entries removes dict entries whose *value* is a numpy array,
    # but it does not traverse into lists, so 'e' remains as a list.
    cleaned = bu.discard_numpy_entries(nested)
    assert "a" not in cleaned
    assert "d" in cleaned["b"]
    assert "e" in cleaned
    assert cleaned["e"] == nested["e"]


@pytest.mark.parametrize(
    "mean,std,expected",
    [
        (1e-7, 2e-7, "100.00 ns ± 200.00 ns"),
        (1e-5, 2e-5, "10.00 µs ± 20.00 µs"),
        (1e-2, 2e-2, "10.00 ms ± 20.00 ms"),
        (2.5, 3.5, "2.50 s ± 3.50 s"),
    ],
)
def test_format_time(mean, std, expected):
    out = bu.format_time(mean, std)
    assert out == expected


def test_format_seconds():
    assert bu.format_seconds(3661) == "01:01:01"
    assert bu.format_seconds(59) == "00:00:59"
    assert bu.format_seconds(3600) == "01:00:00"


def test_flatten_and_scientific_notation():
    nested = {"x": 1, "y": {"a": 0.001, "b": 2}}
    flat = bu.flatten_dict(nested)
    assert flat == {"x": 1, "y - a": 0.001, "y - b": 2}
    sci = bu.convert_dict_to_scientific_notation(flat, precision=3)
    # 0.001 → 1.000e-03
    assert sci["y - a"].endswith("e-03") and sci["x"] == "1.000e+00"


def test_make_and_save_csv(tmp_path):
    # prepare minimal metrics and config
    metrics = {
        "A": {
            "accuracy": {
                "mean_squared_error": 0.1,
                "mean_absolute_error": 0.2,
                "mean_relative_error": 0.3,
                "median_relative_error": 0.4,
                "max_relative_error": 0.5,
                "min_relative_error": 0.6,
                "main_model_epochs": 5,
                "main_model_training_time": 2.0,
            }
        },
        "B": {
            "accuracy": {
                "mean_squared_error": 0.01,
                "mean_absolute_error": 0.02,
                "mean_relative_error": 0.03,
                "median_relative_error": 0.04,
                "max_relative_error": 0.05,
                "min_relative_error": 0.06,
                "main_model_epochs": 7,
                "main_model_training_time": 1.0,
            }
        },
    }
    config = {
        "training_id": "tid",
        "gradients": False,
        "interpolation": {"enabled": False},
        "extrapolation": {"enabled": False},
        "sparse": {"enabled": False},
        "uncertainty": {"enabled": False},
        "verbose": False,
        "iterative": False,
    }
    # run
    bu.make_comparison_csv(metrics, config)
    csv_path = Path("results/tid/all_metrics.csv")
    assert csv_path.exists()
    # header
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["Category", "A", "B"]
        # ensure one of our keys appears
        rows = list(reader)
        cats = [row[0] for row in rows]
        assert "accuracy - mean_squared_error" in cats

    # now test save_table_csv
    headers = ["H1", "H2"]
    rows = [["* v1 *", "  x "], ["y", "*z*"]]
    bu.save_table_csv(headers, rows, config)
    tbl_path = Path("results/tid/metrics_table.csv")
    assert tbl_path.exists()
    with open(tbl_path) as f:
        lines = f.read().splitlines()
    # asterisks removed, whitespace stripped
    assert lines[0] == "H1,H2"
    assert "v1,x" in lines[1]
    assert "y,z" in lines[2]


def test_get_model_config(tmp_path, monkeypatch):
    # no file → {}
    config = {"dataset": {"name": "nosuch", "use_optimal_params": True}}
    assert bu.get_model_config("M", config) == {}

    # create a fake surrogate config module
    ds = tmp_path / "datasets" / "myds"
    ds.mkdir(parents=True)
    py = ds / "surrogates_config.py"
    py.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class MyModelConfig:\n"
        "    alpha: float = 1.23\n"
    )
    monkeypatch.chdir(tmp_path)
    cfg = {"dataset": {"name": "myds", "use_optimal_params": True}}
    out = bu.get_model_config("MyModel", cfg)
    assert out == {"alpha": 1.23}


def test_measure_inference_time():
    class Dummy(torch.nn.Module):
        def forward(self, x):
            return x, x

    model = Dummy()
    t0 = torch.ones((2,))  # each "batch"
    # our loader yields tensors
    loader = [t0, t0, t0]
    times = bu.measure_inference_time(model, loader, n_runs=4)
    assert isinstance(times, list) and len(times) == 4
    assert all(isinstance(t, float) for t in times)
    assert all(t >= 0.0 for t in times)


@pytest.mark.parametrize(
    "train_bs, eval_bs", [(8, 8), ([8, 16], [8, 16]), (8, [8, 8]), ([8, 8], 8)]
)
def test_check_benchmark_accepts_scalar_and_list_batch_sizes(train_bs, eval_bs):
    """A batch size may be one value for all surrogates or one per surrogate."""
    trained = Path("trained") / "tid"
    trained.mkdir(parents=True)
    surrogates = ["A", "B"]
    with open(trained / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"surrogates": surrogates, "batch_size": train_bs}, f)
    conf = {
        "training_id": "tid",
        "surrogates": surrogates,
        "batch_size": eval_bs,
        "devices": ["cpu"],
    }

    bu.check_benchmark(conf)
