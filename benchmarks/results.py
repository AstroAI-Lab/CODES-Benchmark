"""Export completed reference runs to a portable, reviewable results bundle.

Run ``python -m benchmarks.results --workdir RUN [--workdir RUN ...]
--output results.json --markdown report.md`` after evaluation has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import yaml

from benchmarks.reference import validate, verify

DATASETS = ("primordial", "primordial_parametric", "cloud", "cloud_parametric")
MODELS = ("MultiONet", "FullyConnected", "LatentNeuralODE", "LatentPoly")
DATASET_FIELDS = (
    "name",
    "log10_transform",
    "log10_transform_params",
    "normalise",
    "normalise_per_species",
    "tolerance",
    "log_timesteps",
    "use_optimal_params",
    "subset_factor",
)
CONFIG_FIELDS = (
    "training_id",
    "surrogates",
    "seed",
    "epochs",
    "batch_size",
    "devices",
    "checkpoint",
    "synchronize_timing",
    "error_percentile",
    "relative_error_threshold",
    "losses",
    "gradients",
    "timing",
    "compute",
    "compare",
    "cpu_threads",
    "progress",
    "verbose",
)
PACKAGES = (
    "torch",
    "torchode",
    "schedulefree",
    "numpy",
    "scipy",
    "h5py",
    "PyYAML",
    "codes",
)
# Output key: (YAML path, multiplier, display heading).
METRICS = {
    "mlae_dex": (("accuracy", "mean_absolute_error_log"), 1, "mLAE (dex)"),
    "lae99_dex": (("accuracy", "percentile_absolute_error_log"), 1, "LAE99 (dex)"),
    "mre_percent": (("accuracy", "mean_relative_error"), 100, "mRE (%)"),
    "inference_us_per_prediction": (
        ("timing", "mean_inference_time_per_prediction"),
        1e6,
        "Inference (µs/prediction)",
    ),
    "parameters_k": (("compute", "num_trainable_parameters"), 0.001, "Parameters (k)"),
    "iterative_mlae_dex": (
        ("iterative", "mean_absolute_error_log"),
        1,
        "Iterative mLAE (dex)",
    ),
    "ensemble_uncertainty_dex": (
        ("UQ", "average_uncertainty_log"),
        1,
        "Ensemble uncertainty (dex)",
    ),
    "ensemble_mlae_dex": (("UQ", "MAE_log"), 1, "Ensemble mLAE (dex)"),
    "uq_pearson": (("UQ", "correlation_metrics_log"), 1, "UQ Pearson"),
}


def _scalar(value, context):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{context} must be a finite scalar")
    return float(value)


def _read_metrics(path):
    with path.open(encoding="utf-8") as stream:
        source = yaml.safe_load(stream)
    try:
        if source["timing"]["timing_method"] != "synchronized_forward_wall_time":
            raise ValueError(f"{path}: synchronized timing required")
        if source["accuracy"]["error_percentile"] != 99:
            raise ValueError(f"{path}: expected the 99th percentile")
        values = {}
        for key, (parts, multiplier, _) in METRICS.items():
            value = source
            for part in parts:
                value = value[part]
            values[key] = _scalar(value, f"{path}: {'.'.join(parts)}") * multiplier
        timing = {
            key: _scalar(source["timing"][key], f"{path}: timing.{key}")
            for key in (
                "mean_inference_time_per_run",
                "std_inference_time_per_run",
                "num_predictions",
            )
        }
        return values, timing
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing required metric {exc}") from exc


def _required(mapping, name):
    value = mapping.get(name)
    if value is None or value == "" or value == {} or value == []:
        raise ValueError(f"manifest missing {name}")
    return value


def _public_dataset(dataset):
    return {key: dataset[key] for key in DATASET_FIELDS}


def _public_protocol(protocol):
    """Publish scientific settings, excluding local metadata and YAML comments."""
    public = {
        key: protocol[key] for key in ("models", "seeds", "pilot_epochs", "devices")
    }
    public["datasets"] = {}
    for name in DATASETS:
        if name not in protocol["datasets"]:
            continue
        settings = protocol["datasets"][name]
        public["datasets"][name] = {
            "dataset": _public_dataset(settings["dataset"]),
            "epochs": {model: settings["epochs"][model] for model in MODELS},
            "batch_size": {model: settings["batch_size"][model] for model in MODELS},
            "ensemble_size": settings["ensemble_size"],
            "checkpoint": settings["checkpoint"],
        }
    return public


def _public_config(config):
    public = {key: config[key] for key in CONFIG_FIELDS if key in config}
    public["dataset"] = _public_dataset(config["dataset"])
    public["uncertainty"] = {
        key: config["uncertainty"][key] for key in ("enabled", "ensemble_size")
    }
    public["iterative"] = {
        key: config["iterative"][key] for key in ("mode", "interval")
    }
    for modality in ("interpolation", "extrapolation", "sparse", "batch_scaling"):
        if modality in config:
            public[modality] = {"enabled": config[modality]["enabled"]}
    return public


def _public_software(runtime):
    public = {
        key: runtime[key] for key in ("python", "gpu", "cuda_runtime") if key in runtime
    }
    if "packages" in runtime:
        public["packages"] = {
            key: runtime["packages"][key]
            for key in PACKAGES
            if key in runtime["packages"]
        }
    return public


def export(workdirs):
    """Read complete runs, refusing incomplete or incompatible provenance."""
    records = {}
    common = None
    datasets = {}
    protocol_text = None
    protocol_seeds = None
    for workdir in map(Path, workdirs):
        manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("phase") != "reference":
            raise ValueError("only reference-phase runs can be exported")
        verify(workdir.resolve(), manifest)
        dataset = _required(manifest, "dataset")
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset}")
        frozen_protocol = (workdir / "protocol.yaml").read_text(encoding="utf-8")
        protocol = yaml.safe_load(frozen_protocol)
        settings = validate(protocol, dataset, "reference")
        if protocol_text is not None and protocol_text != frozen_protocol:
            raise ValueError("mixed protocol content")
        protocol_text = frozen_protocol
        protocol_seeds = sorted(protocol["seeds"])
        if manifest.get("ensemble_size") != settings["ensemble_size"]:
            raise ValueError("manifest ensemble size differs from protocol")
        provenance = {
            key: _required(manifest, key)
            for key in (
                "code_revision",
                "code_sha256",
                "protocol_sha256",
                "data_sources_sha256",
            )
        }
        state = json.loads((workdir / "state.json").read_text(encoding="utf-8"))
        runtime = _required(state, "_software")
        _required(runtime, "gpu")
        provenance["software"] = _public_software(runtime)
        if common is None:
            common = provenance
        elif common != provenance:
            raise ValueError(
                "mixed code, protocol, hyperparameter or software provenance"
            )
        data_hash = {
            key: _required(manifest, key)
            for key in ("dataset_sha256", "hyperparameters_sha256")
        }
        if dataset in datasets and datasets[dataset] != data_hash:
            raise ValueError(f"mixed dataset provenance for {dataset}")
        datasets[dataset] = data_hash
        for unit in _required(manifest, "units"):
            model, seed = _required(unit, "model"), unit.get("seed")
            if (
                model not in MODELS
                or not isinstance(seed, int)
                or isinstance(seed, bool)
            ):
                raise ValueError(f"invalid model or seed: {model}, {seed}")
            key = (dataset, model, seed)
            if key in records:
                raise ValueError(f"duplicate replica: {key}")
            config = workdir / _required(unit, "config")
            config_hash = _required(unit, "config_sha256")
            if hashlib.sha256(config.read_bytes()).hexdigest() != config_hash:
                raise ValueError(f"config hash mismatch: {config}")
            training_id = _required(unit, "training_id")
            config_text = config.read_text(encoding="utf-8")
            config_data = yaml.safe_load(config_text)
            expected = {
                "training_id": training_id,
                "surrogates": [model],
                "seed": seed,
                "epochs": [settings["epochs"][model]],
                "batch_size": [settings["batch_size"][model]],
                "dataset": settings["dataset"],
                "devices": protocol["devices"],
                "checkpoint": settings["checkpoint"],
                "uncertainty": {
                    "enabled": True,
                    "ensemble_size": settings["ensemble_size"],
                },
                "iterative": {"mode": "grid", "interval": 10},
                "synchronize_timing": True,
            }
            if (
                seed not in protocol_seeds
                or unit.get("epochs") != settings["epochs"][model]
                or any(
                    config_data.get(field) != value for field, value in expected.items()
                )
            ):
                raise ValueError(
                    f"{training_id}: configuration disagrees with manifest or protocol"
                )
            status = state.get(training_id, {})
            if any(
                status.get(phase, {}).get("success") is not True
                for phase in ("train", "eval")
            ):
                raise ValueError(
                    f"{training_id}: train and evaluation must both succeed"
                )
            metric_path = (
                workdir / "results" / training_id / f"{model.lower()}_metrics.yaml"
            )
            values, timing = _read_metrics(metric_path)
            records[key] = {
                "seed": seed,
                "training_id": training_id,
                "config_sha256": config_hash,
                "config_yaml": yaml.safe_dump(
                    _public_config(config_data), sort_keys=True
                ),
                "metrics": values,
                "timing_details": timing,
            }
    if not records:
        raise ValueError("no replicas supplied")
    result = {}
    for dataset in sorted(datasets):
        groups = {
            model: sorted(
                (r for (d, m, _), r in records.items() if d == dataset and m == model),
                key=lambda r: r["seed"],
            )
            for model in MODELS
        }
        seeds = [r["seed"] for r in groups[MODELS[0]]]
        if seeds != protocol_seeds or any(
            [r["seed"] for r in group] != seeds for group in groups.values()
        ):
            raise ValueError(
                f"{dataset}: expected the same three distinct seeds for all four models"
            )
        result[dataset] = {
            model: {
                "replicas": group,
                "aggregate": {
                    metric: {
                        "mean": statistics.mean(values),
                        "sample_std": statistics.stdev(values),
                    }
                    for metric in METRICS
                    for values in [[r["metrics"][metric] for r in group]]
                },
            }
            for model, group in groups.items()
        }
    return {
        "schema_version": 1,
        "provenance": common,
        "dataset_provenance": datasets,
        "protocol_yaml": yaml.safe_dump(_public_protocol(protocol), sort_keys=True),
        "datasets": result,
    }


def render(bundle):
    """Render deterministic Markdown from an exported bundle."""
    lines = [
        "# Reference benchmark results",
        "",
        "Values are mean ± sample standard deviation across three independent replicas.",
        "Inference is synchronized forward-call wall time in microseconds per prediction, normalized by validation trajectories × stored timesteps. Hardware, software, and validation workload affect timing.",
        "Ensemble uncertainty is the mean within-ensemble member spread; it is distinct from the across-replica standard deviation shown after ±.",
        "Iterative scores use the rollout grid specified by the recorded protocol.",
        "",
    ]
    lines += [
        f"Code revision: `{bundle['provenance']['code_revision']}`; code SHA-256: `{bundle['provenance']['code_sha256']}`.",
        f"Runtime GPU, memory and driver: `{bundle['provenance']['software']['gpu']}`.",
        "",
    ]
    for dataset, models in sorted(bundle["datasets"].items()):
        lines += [
            f"## {dataset}",
            "",
            "| Model | " + " | ".join(v[2] for v in METRICS.values()) + " |",
            "| --- | " + " | ".join("---:" for _ in METRICS) + " |",
        ]
        for model in MODELS:
            agg = models[model]["aggregate"]
            lines.append(
                "| "
                + model
                + " | "
                + " | ".join(
                    f"{agg[k]['mean']:.6g} ± {agg[k]['sample_std']:.6g}"
                    for k in METRICS
                )
                + " |"
            )
        lines.append("")
    lines += [
        "The accompanying JSON bundle contains individual replicas and provenance.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        action="append",
        required=True,
        help="Reference run directory; repeat for multiple runs",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON bundle")
    parser.add_argument(
        "--markdown", type=Path, help="Optional generated Markdown report"
    )
    args = parser.parse_args(argv)
    bundle = export(args.workdir)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        args.markdown.write_text(render(bundle), encoding="utf-8")


if __name__ == "__main__":
    main()
