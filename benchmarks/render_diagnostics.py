"""Render the two benchmark figures from a dataset's compact plot-data export."""

import argparse
import json
from io import StringIO
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

MODEL_ORDER = ("MultiONet", "FullyConnected", "LatentNeuralODE", "LatentPoly")


def _figure(data, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 3.55), layout="constrained")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DCE2E8", linewidth=0.7)
    ax.set_axisbelow(True)
    ordered = [model for model in MODEL_ORDER if model in data["models"]]
    ordered += sorted(set(data["models"]) - set(MODEL_ORDER))
    # Match the viridis range and reference model order of the benchmark plots.
    colors = matplotlib.colormaps["viridis"](np.linspace(0, 0.95, len(ordered)))
    series = [
        (name, color, data["models"][name])
        for name, color in zip(ordered, colors, strict=True)
    ]
    return fig, ax, series


def _save(fig, destination):
    svg = StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    plt.close(fig)
    clean = "\n".join(line.rstrip() for line in svg.getvalue().splitlines())
    destination.write_text(clean + "\n", encoding="utf-8")


def render(data_path):
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("Unsupported benchmark plot-data version")
    destination = data_path.parent
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "codes-benchmark",
            "font.size": 11,
            "axes.labelsize": 12,
        }
    )

    fig, ax, series = _figure(data, "Absolute log error (dex)", "Density per decade")
    edges = np.array(data["error_distribution"]["log10_bin_edges"])
    centers = 10 ** ((edges[:-1] + edges[1:]) / 2)
    shown = np.zeros(centers.size, dtype=bool)
    for name, color, replicas in series:
        density = np.mean(
            [row["error_distribution"]["density"] for row in replicas], axis=0
        )
        shown |= density > 0
        ax.plot(centers, density, color=color, linestyle="-", label=name, lw=2)
    active_edges = np.flatnonzero(shown)
    ax.set(xscale="log", xlim=10 ** edges[[active_edges[0], active_edges[-1] + 1]])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, destination / "error-distribution.svg")

    fig, ax, series = _figure(
        data, "Predictions flagged (%)", "Catastrophic errors caught (%)"
    )
    budget = 100 * np.array(data["catastrophic_recall"]["flagged_fractions"])
    for name, color, replicas in series:
        recall = 100 * np.array(
            [row["catastrophic_recall"]["recall"] for row in replicas]
        )
        ax.fill_between(budget, recall.min(0), recall.max(0), color=color, alpha=0.13)
        ax.plot(
            budget,
            recall.mean(0),
            color=color,
            linestyle="-",
            lw=2,
            marker="o",
            ms=2.8,
            label=name,
        )
    ax.plot(budget, budget, "k--", lw=1, alpha=0.65, label="Random flagging")
    ax.set(xlim=(0, budget[-1]), ylim=(0, 100))
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, destination / "catastrophic-error-recall.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_data", type=Path, nargs="+")
    for path in parser.parse_args().plot_data:
        render(path)
