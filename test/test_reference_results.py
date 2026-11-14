import hashlib
import json

import pytest
import yaml

from benchmarks.results import MODELS, export, render


@pytest.fixture(autouse=True)
def skip_external_identity_check(monkeypatch):
    monkeypatch.setattr("benchmarks.results.verify", lambda work, manifest: None)


def make_run(tmp_path, dataset="primordial"):
    names = ("primordial", "primordial_parametric", "cloud", "cloud_parametric")
    protocol = {
        "models": list(MODELS),
        "seeds": [42, 142, 242],
        "pilot_epochs": [2, 4],
        "devices": ["cuda:0"],
        "datasets": {},
    }
    for name in names:
        ds = {
            "name": name,
            "log10_transform": True,
            "log10_transform_params": False,
            "normalise": "standardise",
            "normalise_per_species": False,
            "tolerance": 1e-8,
            "log_timesteps": False,
            "use_optimal_params": True,
            "subset_factor": 1,
        }
        protocol["datasets"][name] = {
            "dataset": ds,
            "epochs": dict.fromkeys(MODELS, 100),
            "batch_size": dict.fromkeys(MODELS, 16),
            "ensemble_size": 5,
            "checkpoint": False,
        }
    (tmp_path / "protocol.yaml").write_text(yaml.safe_dump(protocol))
    manifest = dict(
        dataset=dataset,
        phase="reference",
        ensemble_size=5,
        data_sources_sha256="sources",
        code_revision="revision",
        code_sha256="code",
        dataset_sha256="data-" + dataset,
        protocol_sha256="protocol",
        hyperparameters_sha256="params-" + dataset,
        software={"gpu": "GPU, driver"},
        units=[],
    )
    settings = protocol["datasets"][dataset]
    for model in MODELS:
        for seed in protocol["seeds"]:
            training_id = f"{model}-{seed}"
            config = {
                "training_id": training_id,
                "surrogates": [model],
                "seed": seed,
                "epochs": [100],
                "batch_size": [16],
                "dataset": settings["dataset"],
                "devices": protocol["devices"],
                "checkpoint": False,
                "uncertainty": {"enabled": True, "ensemble_size": 5},
                "iterative": {"mode": "grid", "interval": 10},
                "synchronize_timing": True,
            }
            config_path = tmp_path / f"{training_id}.yaml"
            config_path.write_text(yaml.safe_dump(config))
            manifest["units"].append(
                dict(
                    model=model,
                    seed=seed,
                    epochs=100,
                    training_id=training_id,
                    config=config_path.name,
                    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                )
            )
            path = tmp_path / "results" / training_id
            path.mkdir(parents=True)
            metrics = {
                "accuracy": {
                    "error_percentile": 99,
                    "mean_absolute_error_log": protocol["seeds"].index(seed) + 1,
                    "percentile_absolute_error_log": 2,
                    "mean_relative_error": 0.2,
                },
                "timing": {
                    "mean_inference_time_per_prediction": 0.000002,
                    "mean_inference_time_per_run": 0.002,
                    "std_inference_time_per_run": 0.0001,
                    "num_predictions": 1000,
                    "timing_method": "synchronized_forward_wall_time",
                },
                "compute": {"num_trainable_parameters": 2000},
                "iterative": {"mean_absolute_error_log": 3},
                "UQ": {
                    "average_uncertainty_log": 4,
                    "MAE_log": 5,
                    "correlation_metrics_log": 0.5,
                },
            }
            (path / f"{model.lower()}_metrics.yaml").write_text(yaml.safe_dump(metrics))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "_software": {"gpu": "GPU, driver"},
                **{
                    unit["training_id"]: {
                        "train": {"success": True},
                        "eval": {"success": True},
                    }
                    for unit in manifest["units"]
                },
            }
        )
    )
    return manifest


def test_export_units_aggregation_and_render(tmp_path):
    make_run(tmp_path)
    bundle = export([tmp_path])
    model = bundle["datasets"]["primordial"][MODELS[0]]
    assert model["aggregate"]["mlae_dex"] == {"mean": 2, "sample_std": 1}
    metrics = model["replicas"][0]["metrics"]
    assert metrics["mre_percent"] == 20
    assert metrics["inference_us_per_prediction"] == 2
    assert metrics["parameters_k"] == 2
    assert render(bundle) == render(json.loads(json.dumps(bundle)))


def test_missing_replica_and_nonfinite_metric(tmp_path):
    manifest = make_run(tmp_path)
    manifest["units"].pop()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="three distinct seeds"):
        export([tmp_path])
    make_run_again = manifest["units"][0]
    path = (
        tmp_path
        / "results"
        / make_run_again["training_id"]
        / f"{MODELS[0].lower()}_metrics.yaml"
    )
    metrics = yaml.safe_load(path.read_text())
    metrics["accuracy"]["mean_relative_error"] = float("inf")
    path.write_text(yaml.safe_dump(metrics))
    with pytest.raises(ValueError, match="finite scalar"):
        export([tmp_path])


def test_mixed_provenance_is_rejected(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    make_run(first)
    manifest = make_run(second)
    manifest["code_revision"] = "other"
    (second / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mixed"):
        export([first, second])


def test_missing_metric_pilot_and_unsynchronized_timing(tmp_path):
    manifest = make_run(tmp_path)
    manifest["phase"] = "pilot"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reference-phase"):
        export([tmp_path])
    manifest["phase"] = "reference"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    unit = manifest["units"][0]
    path = (
        tmp_path
        / "results"
        / unit["training_id"]
        / f"{unit['model'].lower()}_metrics.yaml"
    )
    metrics = yaml.safe_load(path.read_text())
    metrics["timing"]["timing_method"] = "legacy_forward_wall_time"
    path.write_text(yaml.safe_dump(metrics))
    with pytest.raises(ValueError, match="synchronized timing"):
        export([tmp_path])
    metrics["timing"]["timing_method"] = "synchronized_forward_wall_time"
    del metrics["UQ"]["MAE_log"]
    path.write_text(yaml.safe_dump(metrics))
    with pytest.raises(ValueError, match="missing required metric"):
        export([tmp_path])


def test_multiple_datasets_preserve_configuration_and_timings(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    make_run(first)
    make_run(second, "cloud")
    bundle = export([first, second])
    assert (
        bundle["dataset_provenance"]["primordial"]["hyperparameters_sha256"]
        != bundle["dataset_provenance"]["cloud"]["hyperparameters_sha256"]
    )
    assert yaml.safe_load(bundle["protocol_yaml"])["seeds"] == [42, 142, 242]
    replica = bundle["datasets"]["cloud"][MODELS[0]]["replicas"][0]
    assert yaml.safe_load(replica["config_yaml"])["seed"] == 42
    assert replica["timing_details"]["mean_inference_time_per_run"] == 0.002
    assert replica["timing_details"]["std_inference_time_per_run"] == 0.0001


def test_reject_mismatched_config(tmp_path):
    manifest = make_run(tmp_path)
    unit = manifest["units"][0]
    path = tmp_path / unit["config"]
    config = yaml.safe_load(path.read_text())
    config["seed"] = 999
    path.write_text(yaml.safe_dump(config))
    unit["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="configuration disagrees"):
        export([tmp_path])


def test_public_export_omits_local_execution_details(tmp_path):
    manifest = make_run(tmp_path)
    private_marker = "private-execution-detail"
    manifest.update(
        repo=f"/home/{private_marker}/checkout",
        workdir=f"/scratch/{private_marker}/run",
        data_path=f"/scratch/{private_marker}/data.hdf5",
        account=private_marker,
        hostname=private_marker,
    )
    protocol_path = tmp_path / "protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    protocol["local_metadata"] = {"account": private_marker, "hostname": private_marker}
    for settings in protocol["datasets"].values():
        settings["dataset"]["local_metadata"] = {"hostname": private_marker}
    protocol_path.write_text(f"# {private_marker}\n" + yaml.safe_dump(protocol))
    for unit in manifest["units"]:
        path = tmp_path / unit["config"]
        config = yaml.safe_load(path.read_text())
        config["local_metadata"] = {"account": private_marker}
        config["dataset"]["local_metadata"] = {"hostname": private_marker}
        path.write_text(f"# {private_marker}\n" + yaml.safe_dump(config))
        unit["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text())
    state["_software"].update(
        hostname=private_marker, executable=f"/home/{private_marker}/python"
    )
    state["_software"]["packages"] = {"torch": "2.0", "local_metadata": private_marker}
    state_path.write_text(json.dumps(state))

    bundle = export([tmp_path])
    assert private_marker not in json.dumps(bundle)
    assert private_marker not in render(bundle)
    assert bundle["provenance"]["software"]["gpu"] == "GPU, driver"
    assert bundle["provenance"]["software"]["packages"] == {"torch": "2.0"}
    assert yaml.safe_load(bundle["protocol_yaml"])["seeds"] == [42, 142, 242]
    replica = bundle["datasets"]["primordial"][MODELS[0]]["replicas"][0]
    assert yaml.safe_load(replica["config_yaml"])["epochs"] == [100]
