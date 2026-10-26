from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import run_eval
from codes.benchmark.bench_fcts import run_benchmark


@pytest.fixture
def minimal_bench_config():
    return {
        "devices": ["cpu"],
        "batch_size": 8,
        "surrogates": ["M1"],
        "epochs": [1],
        "dataset": {
            "name": "ds",
            "log10_transform": False,
            "normalise": "none",
            "tolerance": 1e-3,
        },
        "losses": False,
        "gradients": False,
        "timing": False,
        "compute": False,
        "iterative": False,
        "interpolation": {"enabled": False},
        "extrapolation": {"enabled": False},
        "sparse": {"enabled": False},
        "batch_scaling": {"enabled": False},
        "uncertainty": {"enabled": False},
    }


@pytest.fixture
def interp_bench_config(minimal_bench_config):
    cfg = minimal_bench_config.copy()
    cfg["batch_size"] = [4]
    cfg["interpolation"] = {"enabled": True, "intervals": [1, 2, 3]}
    return cfg


class TestRunEvalMain:
    @pytest.fixture(autouse=True)
    def patches(self):
        # Patch the names imported into run_eval
        with (
            patch.object(run_eval, "read_yaml_config") as m_read,
            patch.object(run_eval, "check_benchmark") as m_chk_bench,
            patch.object(run_eval, "download_data") as m_download,
            patch.object(run_eval, "get_surrogate") as m_get_surr,
            patch.object(run_eval, "check_surrogate") as m_chk_surr,
            patch.object(run_eval, "run_benchmark") as m_run_bench,
            patch.object(run_eval, "compare_models") as m_cmp_models,
            patch.object(run_eval, "nice_print") as m_nice,
        ):
            yield {
                "read": m_read,
                "chk_bench": m_chk_bench,
                "download": m_download,
                "get_surr": m_get_surr,
                "chk_surr": m_chk_surr,
                "run_bench": m_run_bench,
                "cmp_models": m_cmp_models,
                "nice": m_nice,
            }

    def test_all_surrogates_and_compare(self, patches):
        cfg = {
            "dataset": {"name": "ds"},
            "surrogates": ["S1", "S2"],
            "compare": True,
            "verbose": True,
        }
        patches["read"].return_value = cfg

        # have distinct dummy metric dicts per surrogate
        dummy1 = {"m": 1}
        dummy2 = {"m": 2}
        patches["run_bench"].side_effect = [dummy1, dummy2]
        patches["get_surr"].return_value = object()

        args = SimpleNamespace(config="cfg.yaml", device="cpu")
        run_eval.main(args)

        # check basic calls
        patches["read"].assert_called_once_with("cfg.yaml")
        patches["chk_bench"].assert_called_once_with(cfg)
        patches["download"].assert_called_once_with("ds", verbose=True)

        # run_benchmark once per surrogate
        assert patches["run_bench"].call_count == 2

        # compare_models should be called once
        assert patches["cmp_models"].call_count == 1

        # first argument to compare_models is the dict of metrics,
        # check it has the right keys and values
        metrics_dict, passed_cfg = patches["cmp_models"].call_args[0]
        assert set(metrics_dict.keys()) == {"S1", "S2"}
        assert metrics_dict["S1"] is dummy1 and metrics_dict["S2"] is dummy2
        assert passed_cfg is cfg

    @pytest.mark.parametrize(
        "device, expected", [("cuda:1", ["cuda:1"]), ("None", ["cpu"])]
    )
    def test_device_argument_overrides_config(self, patches, device, expected):
        cfg = {"dataset": {"name": "ds"}, "surrogates": [], "devices": ["cpu"]}
        patches["read"].return_value = cfg

        run_eval.main(SimpleNamespace(config="cfg.yaml", device=device))

        checked_cfg = patches["chk_bench"].call_args[0][0]
        assert checked_cfg["devices"] == expected

    def test_skip_unknown_surrogate(self, patches, capsys):
        cfg = {
            "dataset": {"name": "ds"},
            "surrogates": ["GOOD", "BAD"],
            "compare": False,
            "verbose": False,
        }
        patches["read"].return_value = cfg

        good_cls = object()
        patches["get_surr"].side_effect = lambda name: (
            good_cls if name == "GOOD" else None
        )

        args = SimpleNamespace(config="cfg.yaml", device="cpu")
        run_eval.main(args)

        # run_benchmark only for GOOD
        patches["run_bench"].assert_called_once_with("GOOD", good_cls, cfg)

        out = capsys.readouterr().out
        assert "Surrogate BAD not recognized. Skipping." in out

        # compare_models not invoked
        patches["cmp_models"].assert_not_called()

    def test_no_compare_if_single(self, patches):
        cfg = {
            "dataset": {"name": "ds"},
            "surrogates": ["ONLYONE"],
            "compare": True,
            "verbose": False,
        }
        patches["read"].return_value = cfg
        patches["get_surr"].return_value = object()

        args = SimpleNamespace(config="cfg.yaml", device="cpu")
        run_eval.main(args)

        # nice_print should be called to warn about needing 2 models
        patches["nice"].assert_any_call(
            "At least two surrogate models are required to compare."
        )


class DummyModel:
    def __init__(self, device, n_quantities, n_timesteps, n_parameters, config):
        self.train_duration = 0.123
        self.n_quantities = n_quantities

    def prepare_data(self, **kw):
        return None, None, "VAL_LOADER"


@patch("codes.benchmark.bench_fcts.write_metrics_to_yaml")
@patch("codes.benchmark.bench_fcts.evaluate_accuracy")
@patch("codes.benchmark.bench_fcts.get_model_config")
@patch("codes.benchmark.bench_fcts.check_and_load_data")
def test_run_benchmark_minimal(
    mock_load_data,
    mock_get_model_cfg,
    mock_eval_acc,
    mock_write_yaml,
    minimal_bench_config,
):
    cfg = minimal_bench_config
    td = np.zeros((2, 5, 6))
    tp = np.zeros((2, 3))
    mock_load_data.return_value = (
        (td, td, td),
        (tp, tp, tp),
        np.arange(5),
        2,
        None,
        ["q1", "q2"],
    )
    mock_get_model_cfg.return_value = {}
    mock_eval_acc.return_value = {"mse": 0.0}

    metrics = run_benchmark("M1", DummyModel, cfg)

    assert metrics["accuracy"] == {"mse": 0.0}
    mock_write_yaml.assert_called_once_with("M1", cfg, metrics)
