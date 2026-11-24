"""Build versioned benchmark pages and downloads from the canonical bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from statistics import mean, median, stdev

import yaml
from sphinx.errors import ExtensionError

DATASET_NAMES = {
    "primordial": "Primordial",
    "primordial_parametric": "Primordial Parametric",
    "cloud": "Cloud",
    "cloud_parametric": "Cloud Parametric",
}
MODEL_ORDER = ("FullyConnected", "MultiONet", "LatentNeuralODE", "LatentPoly")
DIRECT = (
    ("mLAE ↓ (dex)", "mlae_dex", 3),
    ("LAE₉₉ ↓ (dex)", "lae99_dex", 3),
    ("mRE ↓ (%)", "mre_percent", 2),
    ("Inference ↓ (µs/state)", "inference_us_per_prediction", 3),
    ("Parameters (k)", "parameters_k", 1),
)
ENSEMBLE = (
    ("Ensemble mLAE ↓ (dex)", "ensemble_mlae_dex", 3),
    ("Error–uncertainty Pearson r ↑", "uq_pearson", 3),
)
ROLLOUT = "iterative_mlae_dex"
METRIC_KEYS = {column[1] for column in (*DIRECT, *ENSEMBLE)} | {
    ROLLOUT,
    "ensemble_uncertainty_dex",
}
REPO = "https://github.com/AstroAI-Lab/CODES-Benchmark"
DIAGNOSTIC_FIGURES = (
    (
        "error-distribution.svg",
        "Absolute log-error distribution",
        "Smoothed error density versus absolute log error, with one line per surrogate.",
        "Each line averages the seed-level smoothed histograms for direct predictions, "
        "giving equal weight to each seed. Bins are evenly spaced in log space. "
        "The download records exact zeros separately.",
    ),
    (
        "catastrophic-error-recall.svg",
        "Catastrophic-error recall",
        "Recall of the largest ensemble errors versus the percentage of estimates flagged, "
        "with one curve per surrogate and a diagonal baseline for random flagging.",
        "Flag individual predicted quantities in order of decreasing ensemble spread. "
        "For each surrogate and seed, catastrophic errors are at or above the 99th "
        "percentile of the ensemble-mean absolute log error. Lines show mean recall, "
        "shading shows the minimum and maximum across seeds, and the diagonal shows "
        "what random flagging would catch on average.",
    ),
)
DIAGNOSTIC_FILES = ("plot-data.json", *(figure[0] for figure in DIAGNOSTIC_FIGURES))


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _require_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
        raise ValueError(f"Invalid snapshot identifier: {value!r}")


def _load_snapshot(directory, meta):
    artifact = directory / "results.json"
    raw = artifact.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["results_sha256"]:
        raise ValueError(f"{directory.name}: results.json checksum mismatch")
    bundle = json.loads(raw)
    if bundle["schema_version"] != 1:
        raise ValueError(f"{directory.name}: unsupported results schema")
    revision = bundle["provenance"]["code_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{directory.name}: invalid source revision")
    seeds = sorted(yaml.safe_load(bundle["protocol_yaml"])["seeds"])
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError(f"{directory.name}: expected at least two distinct seeds")
    if not bundle["datasets"]:
        raise ValueError(f"{directory.name}: no datasets")
    for dataset, models in bundle["datasets"].items():
        if dataset not in DATASET_NAMES or not models:
            raise ValueError(f"{directory.name}: invalid dataset {dataset}")
        bundle["dataset_provenance"][dataset]["dataset_sha256"]
        for model, result in models.items():
            label = f"{directory.name}/{dataset}/{model}"
            replicas = sorted(result["replicas"], key=lambda row: row["seed"])
            if [row["seed"] for row in replicas] != seeds:
                raise ValueError(f"{label}: replica seeds differ from the protocol")
            for field in METRIC_KEYS:
                values = [row["metrics"][field] for row in replicas]
                agg = result["aggregate"][field]
                if not all(map(_finite, (*values, agg["mean"], agg["sample_std"]))):
                    raise ValueError(f"{label}: nonfinite {field}")
                if not math.isclose(agg["mean"], mean(values), abs_tol=1e-12) or not (
                    math.isclose(agg["sample_std"], stdev(values), abs_tol=1e-12)
                ):
                    raise ValueError(
                        f"{label}: aggregate disagrees with replicas for {field}"
                    )
            for row in replicas:
                timing = row["timing_details"]
                count = timing["num_predictions"]
                full_time = timing["mean_inference_time_per_run"]
                if (
                    not _finite(count)
                    or not _finite(full_time)
                    or count <= 0
                    or full_time <= 0
                ):
                    raise ValueError(f"{label}: invalid inference workload")
                if not math.isclose(
                    row["metrics"]["inference_us_per_prediction"],
                    1e6 * full_time / count,
                ):
                    raise ValueError(
                        f"{label}: normalized inference time does not match"
                    )
    return bundle


def _table(rows, widths):
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :class: benchmark-results",
        "   :widths: " + " ".join(map(str, widths)),
        "",
    ]
    for first, *rest in rows:
        lines.append(f"   * - {first}")
        lines.extend(f"     - {cell}" for cell in rest)
    return "\n".join(lines) + "\n\n"


def _heading(title, underline="="):
    return title + "\n" + (underline * len(title)) + "\n\n"


def _doc(label, path):
    return f":doc:`{label} </benchmarks/snapshots/{path}>`"


def _bold(value):
    return f"**{value}**"


def _ordered_models(models):
    return [name for name in MODEL_ORDER if name in models] + sorted(
        models.keys() - set(MODEL_ORDER)
    )


def _aggregate_table(models, columns, widths):
    names = _ordered_models(models)
    rows = [["Model", *(heading for heading, _, _ in columns)]]
    leaders = {}
    for heading, field, precision in columns:
        if "↑" in heading or "↓" in heading:
            values = [
                round(models[m]["aggregate"][field]["mean"], precision) for m in names
            ]
            leaders[field] = (max if "↑" in heading else min)(values)
    for model in names:
        cells = [model]
        for _, field, precision in columns:
            stats = models[model]["aggregate"][field]
            value = stats["mean"]
            cell = f"{value:.{precision}f}"
            if field != "parameters_k":
                cell += f" ± {stats['sample_std']:.{precision}f}"
            if field in leaders and round(value, precision) == leaders[field]:
                cell = _bold(cell)
            cells.append(cell)
        rows.append(cells)
    return _table(rows, widths)


def _rollout_table(models):
    names = _ordered_models(models)
    replicas = {
        name: sorted(models[name]["replicas"], key=lambda row: row["seed"])
        for name in names
    }
    seeds = [row["seed"] for row in replicas[names[0]]]
    rows = [["Model", *(f"Seed {seed}" for seed in seeds), "Median ↓"]]
    values = {
        name: [row["metrics"][ROLLOUT] for row in replicas[name]] for name in names
    }
    best = min(round(median(v), 3) for v in values.values())
    for name, errors in values.items():
        med = median(errors)
        formatted = f"{med:.3f}"
        rows.append(
            [
                name,
                *(f"{value:.3f}" for value in errors),
                _bold(formatted) if round(med, 3) == best else formatted,
            ]
        )
    return _table(rows, [30, *([17] * len(seeds)), 19])


def _csv(bundle, destination):
    fields = [
        "dataset",
        "model",
        "seed",
        "config_sha256",
        *sorted(METRIC_KEYS),
        "num_predictions",
        "mean_inference_time_per_run",
        "std_inference_time_per_run",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for dataset, models in bundle["datasets"].items():
            for model, result in models.items():
                for row in sorted(result["replicas"], key=lambda row: row["seed"]):
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": row["seed"],
                            "config_sha256": row["config_sha256"],
                            **{key: row["metrics"][key] for key in METRIC_KEYS},
                            **{key: row["timing_details"][key] for key in fields[-3:]},
                        }
                    )


def _datasets(bundle):
    return [name for name in DATASET_NAMES if name in bundle["datasets"]]


def _downloads(bundle, meta, prefix=""):
    revision = bundle["provenance"]["code_revision"]
    return (
        f":download:`Results bundle (JSON) <{prefix}results.json>`"
        f" · :download:`Per-seed metrics (CSV) <{prefix}replicas.csv>`\n\n"
        f"`Run code at {revision[:7]} <{REPO}/tree/{revision}>`_"
        f" · `Dataset archive <{meta['dataset_record']}>`_"
        f" · `A&A paper <{meta['paper']}>`_\n\n"
    )


def _diagnostics(dataset, meta):
    if dataset not in meta.get("diagnostics", {}):
        return ""
    base = f"diagnostics/{dataset}"
    content = _heading("Diagnostic plots", "-")
    for filename, title, alt, caption in DIAGNOSTIC_FIGURES:
        content += _heading(title, "~")
        content += f".. figure:: {base}/{filename}\n   :alt: {alt}\n\n   {caption}\n\n"
    return (
        content + f":download:`Per-seed plot data (JSON) <{base}/plot-data.json>`\n\n"
    )


def _dataset_page(sid, dataset, meta, bundle, catalog, bundles):
    models = bundle["datasets"][dataset]
    runtime = bundle["provenance"]["software"]
    link = _doc("Snapshot overview", f"{sid}/index")
    history = _doc("All snapshots", "index")
    gpu, _, gpu_details = runtime["gpu"].partition(", ")
    counts = [
        r["timing_details"]["num_predictions"]
        for m in models.values()
        for r in m["replicas"]
    ]
    content = _heading(DATASET_NAMES[dataset])
    content += f"{link} · {history}\n\n"
    content += f"**{meta['title']}**\n\n"
    others = [
        _doc(label["title"], f"{oid}/{dataset}")
        for oid, label in catalog.items()
        if oid != sid and dataset in bundles[oid]["datasets"]
    ]
    if others:
        content += "Other versions: " + " · ".join(others) + "\n\n"
    workload = f"{int(counts[0]):,}"
    if min(counts) != max(counts):
        workload = f"{int(min(counts)):,}–{int(max(counts)):,}"
    content += (
        f"**Evaluation split** {meta['evaluation_split']} · **Timing GPU** {gpu} · "
        f"**Workload** {workload} predicted states per pass\n\n"
    )
    content += (
        "Direct and ensemble metrics show mean ± sample standard deviation across seeds. "
        "Bold marks the best displayed value for each metric, including displayed ties. "
        "Numerical rankings describe this snapshot and protocol.\n\n"
    )
    content += _heading("Direct prediction", "-")
    content += _aggregate_table(models, DIRECT, [23, 15, 15, 13, 19, 15])
    content += (
        "mLAE is mean absolute log error. LAE₉₉ is the 99th percentile of absolute log error. "
        "mRE is calculated as the median relative error within each seed.\n\n"
        + meta["timing"]
        + "\n\n"
    )
    content += _heading("Five-model ensembles", "-")
    content += _aggregate_table(models, ENSEMBLE, [36, 30, 34])
    content += "Pearson r measures how ensemble spread tracks error magnitude.\n\n"
    content += _heading("Rollout", "-")
    content += f"{meta['rollout']}. Entries are rollout mLAE in dex.\n\n"
    content += _rollout_table(models)
    note = meta.get("notes", {}).get(dataset)
    if note:
        content += f".. note:: {note}\n\n"
    content += _diagnostics(dataset, meta)
    content += _heading("Data and provenance", "-")
    content += _downloads(bundle, meta)
    content += (
        f"Timing hardware: {gpu}, {gpu_details}. PyTorch {runtime['packages']['torch']}, "
        f"CUDA {runtime['cuda_runtime']}. The JSON bundle includes the protocol, "
        "configuration hashes, evaluation settings, and full-precision seed results.\n"
    )
    return content


def _write(path, content):
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def _copy_diagnostics(source, target, meta, datasets):
    for dataset, digests in meta.get("diagnostics", {}).items():
        if dataset not in datasets or set(digests) != set(DIAGNOSTIC_FILES):
            raise ValueError(f"{source.name}/{dataset}: invalid diagnostic files")
        destination = target / "diagnostics" / dataset
        destination.mkdir(parents=True, exist_ok=True)
        for filename, digest in digests.items():
            raw = (source / "diagnostics" / dataset / filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(
                    f"{source.name}/{dataset}: {filename} checksum mismatch"
                )
            path = destination / filename
            if not path.exists() or path.read_bytes() != raw:
                path.write_bytes(raw)


def generate(catalog_root: Path, target: Path):
    registry = json.loads((catalog_root / "index.json").read_text(encoding="utf-8"))
    current, catalog = registry["current"], registry["snapshots"]
    bundles = {}
    for sid, meta in catalog.items():
        _require_id(sid)
        bundles[sid] = _load_snapshot(catalog_root / sid, meta)
    if current not in bundles:
        raise ValueError("Current benchmark snapshot does not exist")

    target.mkdir(parents=True, exist_ok=True)
    landing = (
        "**Current snapshot**\n\n"
        + _heading(catalog[current]["title"], "-")
        + catalog[current]["summary"]
        + "\n\n"
    )
    landing += "\n".join(
        f"* {_doc(DATASET_NAMES[dataset], f'{current}/{dataset}')}"
        for dataset in _datasets(bundles[current])
    )
    landing += (
        f"\n\n{_doc('Snapshot overview and downloads', f'{current}/index')} · "
        f"{_doc('Previous snapshots and version history', 'index')}\n"
    )
    _write(target / "current.inc", landing)
    history = _heading("Benchmark snapshot history")
    history += "Each snapshot has a permanent URL and full-precision downloads.\n\n"
    history += "\n".join(
        f"* {_doc(meta['title'], f'{sid}/index')}"
        + (" (current)" if sid == current else "")
        for sid, meta in catalog.items()
    )
    history += "\n\n.. toctree::\n   :hidden:\n\n"
    history += "".join(f"   {sid}/index\n" for sid in catalog)
    _write(target / "index.rst", history)

    for sid, meta in catalog.items():
        bundle = bundles[sid]
        directory = target / sid
        directory.mkdir(exist_ok=True)
        artifact = catalog_root / sid / "results.json"
        if (
            not (directory / "results.json").exists()
            or artifact.read_bytes() != (directory / "results.json").read_bytes()
        ):
            shutil.copyfile(artifact, directory / "results.json")
        _csv(bundle, directory / "replicas.csv")
        _copy_diagnostics(catalog_root / sid, directory, meta, bundle["datasets"])
        overview = (
            _heading(meta["title"])
            + f"Snapshot ID: ``{sid}``\n\n"
            + meta["summary"]
            + "\n\n"
            + _downloads(bundle, meta)
            + ".. toctree::\n   :maxdepth: 1\n\n"
        )
        for dataset in _datasets(bundle):
            overview += f"   {dataset}\n"
            page = _dataset_page(sid, dataset, meta, bundle, catalog, bundles)
            _write(directory / f"{dataset}.rst", page)
        _write(directory / "index.rst", overview)


def _build(app):
    docs = Path(app.srcdir)
    try:
        generate(
            docs.parents[1] / "benchmarks" / "snapshots",
            docs / "benchmarks" / "snapshots",
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ExtensionError(f"Cannot build benchmark pages: {exc}") from exc


def setup(app):
    app.add_css_file("benchmark-tables.css")
    app.connect("builder-inited", _build)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
