import argparse
import math
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import psycopg2
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from optuna.trial import TrialState
from psycopg2 import sql

# add codes directory to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from codes.tune import load_yaml_config
from codes.utils import nice_print

os.environ.setdefault("PGCONNECT_TIMEOUT", "3")

TRIAL_COLOR = "#c7c7c7"
PARETO_COLOR = "#1f77b4"
BEST_F1_COLOR = "#d62728"
CHOSEN_COLOR = "#2ca02c"
EDGE_COLOR = "#000000"
PARETO_X_RANGE = (0, 6)
SURROGATE_ORDER = [
    "multionet",
    "fullyconnected",
    "latentpoly",
    "latentneuralode",
]


def format_surrogate_title(name: str) -> str:
    mapping = {
        "multionet": "MultiONet",
        "fullyconnected": "FullyConnected",
        "latentneuralode": "LatentNeuralODE",
        "latentpoly": "LatentPoly",
    }
    return mapping.get(name.lower(), name)


def pareto_front_mask(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.array([], dtype=bool)
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if not is_efficient[i]:
            continue
        # any other point strictly better in both dims dominates p
        better = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
        dominated = better & (np.arange(points.shape[0]) != i)
        if np.any(dominated):
            is_efficient[i] = False
    return is_efficient


def pareto_front(points: np.ndarray) -> np.ndarray:
    mask = pareto_front_mask(points)
    return points[mask]


def hypervolume_2d(pareto_points: np.ndarray, reference: np.ndarray) -> float:
    # assumes minimize-minimize; reference worse than all pareto_points
    if pareto_points.size == 0:
        return 0.0
    pts = pareto_points[np.argsort(pareto_points[:, 0])]  # sort by first objective
    hv = 0.0
    prev_f2 = reference[1]
    for f1, f2 in pts:
        width = reference[0] - f1
        height = prev_f2 - f2
        if width > 0 and height > 0:
            hv += width * height
        prev_f2 = f2
    return hv


def compute_hypervolume_over_time(
    study: optuna.Study, ref_slack: float = 1.1, ignore_last_n: int = 10
):
    """
    Compute the 2D hypervolume curve over time while excluding the last N trials
    from the Pareto calculation (by completion time). This prevents unreliable
    late-trial improvements (e.g., inference time) from appearing on the Pareto front.

    Returns (hypervolumes, reference_point).
    """
    from optuna.trial import TrialState

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return [], None

    # Order by completion time
    completed.sort(key=lambda t: t.datetime_complete or t.datetime_start)

    # Determine index cutoff for eligible trials (exclude last N)
    cutoff_idx = max(0, len(completed) - max(0, ignore_last_n))
    eligible = completed[:cutoff_idx]
    if not eligible:
        # No eligible trials to compute HV
        return [], None

    # Reference point from eligible trials only
    eligible_vals = np.array([t.values for t in eligible])  # shape (M, 2)
    reference = eligible_vals.max(axis=0) * ref_slack  # slightly worse than worst seen

    hypervolumes = []
    total = len(completed)
    for k in range(1, total + 1):
        # Freeze the Pareto set after cutoff: ignore last N trials entirely
        subset_upto_k = min(k, cutoff_idx)
        if subset_upto_k == 0:
            hypervolumes.append(0.0)
            continue
        subset = completed[:subset_upto_k]
        pts = np.array([t.values for t in subset])
        pareto = pareto_front(pts)
        hv = hypervolume_2d(pareto, reference)
        hypervolumes.append(hv)
    return hypervolumes, reference


def _pareto_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TRIAL_COLOR,
            markeredgecolor=TRIAL_COLOR,
            label="Trial Outcome",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PARETO_COLOR,
            markeredgecolor=EDGE_COLOR,
            label="Pareto front",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="none",
            markerfacecolor=BEST_F1_COLOR,
            markeredgecolor=EDGE_COLOR,
            label="Lowest Δdex99",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=CHOSEN_COLOR,
            markeredgecolor=EDGE_COLOR,
            label="Chosen Trial",
            markersize=6,
        ),
    ]


def compute_pareto_plot_data(
    study: optuna.Study,
    suffix: str,
    ignore_last_n: int = 10,
    chosen_trial_number: int | None = None,
):
    if len(study.directions) != 2:
        return None
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        print(f"Skipping Pareto plot for {suffix}: no completed trials.")
        return None
    completed.sort(key=lambda t: t.datetime_complete or t.datetime_start)
    cutoff = max(0, len(completed) - max(0, ignore_last_n))
    eligible_trials = completed[:cutoff]
    filtered = []
    for t in eligible_trials:
        if t.values is None:
            continue
        vals = np.array(t.values, dtype=float)
        if vals.shape[0] != 2 or np.any(~np.isfinite(vals)):
            continue
        filtered.append((vals, t.number))
    if not filtered:
        print(
            f"Skipping Pareto plot for {suffix}: no eligible trials after ignoring last {ignore_last_n}."
        )
        return None
    points = np.vstack([vals for vals, _ in filtered])
    trial_numbers = np.array([num for _, num in filtered])
    mask = pareto_front_mask(points)
    best_idx = int(np.argmin(points[:, 0]))
    best_point = points[best_idx]
    chosen_point = None
    if chosen_trial_number is not None:
        matches = np.where(trial_numbers == chosen_trial_number)[0]
        if matches.size:
            chosen_point = points[matches[0]]
        else:
            print(
                f"Chosen trial {chosen_trial_number} not in eligible set for {suffix}; "
                "skipping highlight."
            )

    y_min = 0
    y_max = best_point[1] * 1.3

    return {
        "suffix": suffix,
        "points": points,
        "mask": mask,
        "best_point": best_point,
        "chosen_point": chosen_point,
        "xlim": PARETO_X_RANGE,
        "ylim": (y_min, y_max),
        "ignore_last_n": ignore_last_n,
    }


def summarize_pareto_tradeoff(data: dict):
    suffix = data["suffix"]
    best_point = data["best_point"]
    chosen_point = data["chosen_point"]
    pts = data["points"][data["mask"]]
    x_span = float(pts[:, 0].max() - pts[:, 0].min())
    y_span = float(pts[:, 1].max() - pts[:, 1].min())
    min_x = float(pts[:, 0].min())
    min_y = float(pts[:, 1].min())
    x_rel = x_span / max(abs(min_x), 1e-12)
    y_rel = y_span / max(abs(min_y), 1e-12)
    print(
        f"{suffix}: Pareto span Δdex={x_span:.4f} "
        f"(x{x_rel:.2f} relative), inference time span={y_span:.4f}s "
        f"(x{y_rel:.2f} relative)"
    )
    best_err, best_time = float(best_point[0]), float(best_point[1])
    print(
        f"{suffix}: lowest-error trial Δdex={best_err:.4f}, inference time={best_time:.4f}s"
    )
    if chosen_point is None:
        print(
            f"{suffix}: chosen trial not available after filtering; skipping summary."
        )
        return
    chosen_err, chosen_time = float(chosen_point[0]), float(chosen_point[1])
    print(
        f"{suffix}: chosen trial Δdex={chosen_err:.4f}, inference time={chosen_time:.4f}s"
    )
    err_reduction = 1.0 - best_err / chosen_err if chosen_err > 0 else float("nan")
    time_ratio = best_time / chosen_time if chosen_time > 0 else float("nan")
    if math.isnan(err_reduction) or math.isnan(time_ratio):
        print(f"{suffix}: insufficient data to compute tradeoff summary.")
        return
    err_phrase = (
        f"{abs(err_reduction) * 100:.1f}% lower error"
        if err_reduction >= 0
        else f"{abs(err_reduction) * 100:.1f}% higher error"
    )
    ratio_phrase = (
        f"but {time_ratio:.2f}x higher inference time"
        if time_ratio >= 1
        else f"and {time_ratio:.2f}x lower inference time"
    )
    print(f"{suffix}: {err_phrase}, {ratio_phrase}.")


def _render_pareto_scatter(
    ax,
    data,
    show_xlabel: bool,
    show_ylabel: bool,
    title: str | None,
    hide_xticklabels: bool = False,
):
    pts = data["points"]
    mask = data["mask"]
    best_point = data["best_point"]
    chosen_point = data["chosen_point"]

    # Remove best point and chosen point from pareto points to avoid double-plotting
    if mask.sum() > 0:
        pareto_pts = pts[mask]
        pareto_pts = pareto_pts[
            ~np.all(pareto_pts == best_point, axis=1)
        ]  # remove best point
        if chosen_point is not None:
            pareto_pts = pareto_pts[
                ~np.all(pareto_pts == chosen_point, axis=1)
            ]  # remove chosen point
        # Recompute mask
        new_mask = np.array(
            [any(np.all(p == pp) for pp in pareto_pts) for p in pts], dtype=bool
        )
        mask = new_mask
    else:
        mask = np.array([False] * pts.shape[0], dtype=bool)

    ax.scatter(pts[:, 0], pts[:, 1], color=TRIAL_COLOR, alpha=0.7, label=None)
    ax.scatter(
        pts[mask, 0],
        pts[mask, 1],
        marker="o",
        color=PARETO_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
        label=None,
    )
    ax.scatter(
        best_point[0],
        best_point[1],
        marker="v",
        color=BEST_F1_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        label=None,
        zorder=3,
    )
    if chosen_point is not None:
        ax.scatter(
            chosen_point[0],
            chosen_point[1],
            marker="^",
            color=CHOSEN_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
            label=None,
            zorder=4,
        )

    ax.set_xlim(*data["xlim"])
    ax.set_ylim(*data["ylim"])
    if show_xlabel:
        ax.set_xlabel(r"LAE$_{99}$ [dex]")
    if show_ylabel:
        ax.set_ylabel("Inference Time [s]")
    if title:
        ax.set_title(title)
    ax.tick_params(labelbottom=not hide_xticklabels)
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def save_individual_pareto_plot(data, out_dir: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    _render_pareto_scatter(
        ax, data, show_xlabel=True, show_ylabel=True, title=data["suffix"]
    )
    ax.legend(handles=_pareto_legend_handles(), loc="best", ncol=2)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"pareto_front_{data['suffix']}.png"), dpi=300)
    plt.close(fig)
    print(
        f"Saved Pareto front plot for {data['suffix']}; "
        f"ignored last {data['ignore_last_n']} completed trials."
    )


def save_pareto_front_grid(datasets: list[dict], out_dir: str):
    if not datasets:
        return
    n_cols = 2
    n_rows = max(1, math.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False
    )

    for idx, data in enumerate(datasets):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        show_xlabel = row == n_rows - 1
        show_ylabel = col == 0
        proper_title = format_surrogate_title(data["suffix"])
        _render_pareto_scatter(
            ax,
            data,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            title=proper_title,
            hide_xticklabels=not show_xlabel,
        )
    for idx in range(len(datasets), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.tight_layout(rect=(0.01, 0.035, 0.99, 0.99))
    fig.legend(
        handles=_pareto_legend_handles(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, 0.0),
        fontsize=10,
    )
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "pareto_front_grid.png"), dpi=300)
    plt.close(fig)
    print(
        f"Saved combined Pareto front grid with {len(datasets)} subplot(s) "
        f"to {os.path.join(out_dir, 'pareto_front_grid.png')}."
    )


def save_relative_hv_grid(datasets: list[dict], out_dir: str):
    if not datasets:
        return
    n_cols = 2
    n_rows = max(1, math.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False
    )

    for idx, data in enumerate(datasets):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        show_xlabel = row == n_rows - 1
        show_ylabel = col == 0
        ax.plot(data["x"], data["y"], color="#1f77b4")
        ax.set_xlim(0, max(data["x"]) if len(data["x"]) else 1)
        ax.set_ylim(0.6, 1.03)
        ax.set_title(format_surrogate_title(data["suffix"]))
        if show_xlabel:
            ax.set_xlabel("Completed Trials")
        if show_ylabel:
            ax.set_ylabel("Normalized Hypervolume")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(True, linestyle="--", alpha=0.3)

    for idx in range(len(datasets), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "hypervolume_normalized_grid.png"), dpi=300)
    plt.close(fig)
    print(
        f"Saved combined normalized-hypervolume grid with {len(datasets)} subplot(s) "
        f"to {os.path.join(out_dir, 'hypervolume_normalized_grid.png')}."
    )


def save_relative_hv_combined(datasets: list[dict], out_dir: str):
    if len(datasets) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    color_palette = plt.cm.viridis(np.linspace(0, 0.95, len(SURROGATE_ORDER)))
    order_index = {name: idx for idx, name in enumerate(SURROGATE_ORDER)}

    def color_for(name: str):
        key = name.lower()
        idx = order_index.get(key)
        if idx is not None:
            return color_palette[idx]
        return plt.cm.tab10(0)

    ordered = sorted(
        datasets,
        key=lambda d: order_index.get(d["suffix"].lower(), len(SURROGATE_ORDER)),
    )

    for data in ordered:
        color = color_for(data["suffix"])
        ax.plot(
            data["x"],
            data["y"],
            label=format_surrogate_title(data["suffix"]),
            color=color,
        )
        if len(data["x"]):
            last_x = data["x"][-1]
            last_y = data["y"][-1]
            ax.vlines(
                last_x,
                max(0, last_y - 0.01),
                min(1.05, last_y + 0.01),
                colors=color,
                linewidth=2,
            )
    ax.set_xlim(0, max(max(d["x"]) for d in datasets if len(d["x"])) if datasets else 1)
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("Completed Trials")
    ax.set_ylabel("Normalized Hypervolume")
    # ax.set_title("Normalized Hypervolume")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "hypervolume_normalized_combined.png"), dpi=300)
    plt.close(fig)
    print(
        "Saved combined normalized-hypervolume line plot to "
        f"{os.path.join(out_dir, 'hypervolume_normalized_combined.png')}."
    )


def load_loss_history(model_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load loss histories from a saved .pth.
    If the checkpoint contains pickled Optuna storage (which may try to connect
    to a remote DB during unpickle), bound the timeout and fall back to skipping.
    """
    # Prefer the Tensor-only safe path if your PyTorch supports it (PyTorch >= 2.0)
    try:
        obj = torch.load(model_path, map_location="cpu", weights_only=True)
        # weights_only returns just state_dict; no custom 'attributes' available
        # We can’t recover losses from state_dict → skip gracefully.
        return None, None, 0
    except TypeError:
        # weights_only not supported → careful unpickle with timeout + catch
        pass

    try:
        obj = torch.load(model_path, map_location="cpu")
    except Exception as e:
        # Any error (including timeout from unpickling Optuna storage) → skip losses
        print(f"[warn] Could not safely load {model_path}: {e}. Skipping loss curves.")
        return None, None, 0

    attributes = obj.get("attributes", {}) if isinstance(obj, dict) else {}
    train_loss = (
        np.array(attributes.get("train_loss"))
        if attributes.get("train_loss") is not None
        else None
    )
    test_loss = (
        np.array(attributes.get("test_loss"))
        if attributes.get("test_loss") is not None
        else None
    )
    n_epochs = attributes.get(
        "n_epochs", len(train_loss) if train_loss is not None else 0
    )
    return train_loss, test_loss, n_epochs


def plot_losses(
    loss_histories, epochs, labels, title="Losses", save=False, out_dir=None, mode=""
):
    """
    Plot multiple loss trajectories on a log scale.
    """
    valid = [loss for loss in loss_histories if loss is not None and loss.size]
    if not valid:
        print("No valid loss arrays; skipping.")
        return
    max_len = max(len(loss) for loss in valid)
    # Determine y-limits excluding initial 2%
    mins, maxs = [], []
    for loss in valid:
        start = int(len(loss) * 0.02)
        vals = loss[start:][loss[start:] > 0]
        vals = vals[~np.isnan(vals)]  # Remove NaNs
        vals = vals[~np.isinf(vals)]  # Remove Infs
        if vals.size:
            mins.append(vals.min())
            maxs.append(vals.max())
    ymin = min(mins) if mins else 1e-8
    ymax = max(maxs) if maxs else 1.0
    # colors = plt.cm.magma(np.linspace(0.15, 0.85, len(loss_histories)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_histories)))

    plt.figure(figsize=(6, 4))
    for loss, lab in zip(loss_histories, labels):
        plt.plot(np.arange(len(loss)), loss, label=lab, color=colors[labels.index(lab)])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.xlim(0, max_len)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.legend()
    if save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"losses_{mode}.png"), dpi=300)
        print(f"Saved plot to {out_dir}")
    plt.close()


def get_best_trials(study: optuna.Study, top_n: int) -> list[int]:
    """
    Return trial numbers of best 'top_n' trials.
    Single-objective: lowest value.
    Multi-objective: Pareto front; if fewer than top_n, pad with closest-to-origin among the rest.
    """
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return []
    # Single-objective
    if len(study.directions) == 1:
        sorted_trials = sorted(completed, key=lambda t: t.value)
        return [t.number for t in sorted_trials[:top_n]]

    # Multi-objective
    def dominated(a, b):
        return all(bi <= ai for ai, bi in zip(a, b)) and any(
            bi < ai for ai, bi in zip(a, b)
        )

    # Pareto front
    pareto = [
        t
        for t in completed
        if not any(dominated(t.values, u.values) for u in completed if u != t)
    ]
    # Compute distances to origin for all trials
    dist_all = [(t.number, math.hypot(*t.values)) for t in completed]
    # Sort by distance
    dist_all.sort(key=lambda x: x[1])
    # Select top_n by prioritizing Pareto, then padding from sorted distances
    selected = []
    # First, add pareto points in any order (or by distance)
    pareto_dists = [(t.number, math.hypot(*t.values)) for t in pareto]
    pareto_dists.sort(key=lambda x: x[1])
    for num, _ in pareto_dists:
        if len(selected) < top_n:
            selected.append(num)
    # Then pad with nearest non-pareto
    if len(selected) < top_n:
        for num, _ in dist_all:
            if num in selected:
                continue
            selected.append(num)
            if len(selected) >= top_n:
                break
    return selected


def evaluate_tuning(
    study_prefix: str,
    top_n: int = 10,
    storage_name: str = "optuna_db",
    ignore_last_n: int = 10,
    chosen_indices: list[int] | None = None,
) -> None:
    """
    For all surrogate studies named '<study_prefix>_<surrogate>',
    plot the top_n test-loss trajectories.
    """
    # Load prefix config
    config_path = os.path.join("tuned", study_prefix, "optuna_config.yaml")
    if not os.path.exists(config_path):
        print(f"Config missing: {config_path}")
        return
    config = load_yaml_config(config_path)

    # Ensure Postgres server
    pg = config.get("postgres_config", {})
    try:
        psycopg2.connect(
            dbname="postgres",
            user=pg.get("user"),
            password=pg.get("password"),
            host=pg.get("host"),
            port=pg.get("port"),
            connect_timeout=5,
        ).close()
    except Exception:
        if str(pg.get("host", "localhost")) not in ("localhost", "127.0.0.1", "::1"):
            raise  # don’t try to start a local server if host isn't local
        pg_data = pg.get("data_dir", os.path.expanduser("~/postgres/data"))
        pg_ctl = os.path.join(
            pg.get("database_folder", os.path.expanduser("~/postgres")), "bin", "pg_ctl"
        )
        subprocess.run(
            [pg_ctl, "-D", pg_data, "-l", pg.get("log_file", "/tmp/pg.log"), "start"],
            check=True,
        )
        time.sleep(2)
    # Create DB if absent
    conn = psycopg2.connect(
        dbname="postgres",
        user=pg.get("user"),
        password=pg.get("password"),
        host=pg.get("host"),
        port=pg.get("port"),
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [study_prefix])
    if not cur.fetchone():
        cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(study_prefix)))
    cur.close()
    conn.close()

    # Build storage URL (hard-coded prefix)
    db_prefix = "postgresql://optuna_user:your_password@localhost:5432/"
    storage_url = f"{db_prefix}{storage_name}"

    # Discover all studies with this prefix
    from optuna.study import get_all_study_summaries

    summaries = get_all_study_summaries(storage=storage_url)
    study_names_all = [
        s.study_name for s in summaries if s.study_name.startswith(f"{study_prefix}_")
    ]
    if not study_names_all:
        print(f"No studies found with prefix '{study_prefix}_' in {storage_url}")
        return

        # Output directory: directly in main study folder
    save_dir = os.path.join("tuned", study_prefix)
    os.makedirs(save_dir, exist_ok=True)

    # Derive deterministic ordering from config surrogate list if available
    surrogate_entries = config.get("surrogates", [])
    surrogate_names = [
        str(entry.get("name"))
        for entry in surrogate_entries
        if isinstance(entry, dict) and entry.get("name")
    ]
    study_names_ordered = []
    used: set[str] = set()
    study_lookup = {name.lower(): name for name in study_names_all}
    if surrogate_names:
        for surrogate in surrogate_names:
            candidate = f"{study_prefix}_{surrogate}".lower()
            match = study_lookup.get(candidate)
            if match and match not in used:
                study_names_ordered.append(match)
                used.add(match)
    for name in sorted(study_names_all):
        if name not in used:
            study_names_ordered.append(name)
            used.add(name)
    study_names = study_names_ordered

    if chosen_indices is not None and len(chosen_indices) != len(study_names):
        print(
            "Provided chosen_indices length does not match number of studies; "
            "skipping chosen-trial highlighting."
        )
        chosen_indices = None

    pareto_datasets: list[dict] = []
    hypervolume_datasets: list[dict] = []

    # Loop over each surrogate study
    for idx, full_name in enumerate(study_names):
        suffix = full_name[len(study_prefix) + 1 :]
        print(f"--- Evaluating study {full_name} -> surrogate '{suffix}' ---")
        try:
            study = optuna.load_study(study_name=full_name, storage=storage_url)
        except KeyError:
            print(f"Could not load study '{full_name}'")
            continue

        # Compute hypervolume over time
        if len(study.directions) == 2:
            hvs, reference = compute_hypervolume_over_time(
                study, ignore_last_n=ignore_last_n
            )
            if hvs:
                # Normalize to final hypervolume for relative curve
                final_hv = hvs[-1]
                rel_hvs = [hv / final_hv if final_hv > 0 else 0 for hv in hvs]

                # Plot absolute and relative hypervolume
                plt.figure(figsize=(6, 4))
                plt.plot(np.arange(1, len(hvs) + 1), hvs, label="Hypervolume")
                plt.xlabel("Completed Trials")
                plt.ylabel("Hypervolume")
                plt.title(f"{suffix} Hypervolume over trials")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"hypervolume_{suffix}.png"), dpi=300
                )
                plt.close()

                plt.figure(figsize=(6, 4))
                plt.plot(
                    np.arange(1, len(rel_hvs) + 1),
                    rel_hvs,
                    label="Normalized Hypervolume",
                )
                plt.xlabel("Completed Trials")
                plt.ylabel("Normalized Hypervolume")
                plt.title(f"{suffix} Normalized Hypervolume")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"hypervolume_relative_{suffix}.png"),
                    dpi=300,
                )
                plt.close()
                max_x = max(1, len(rel_hvs) - max(0, ignore_last_n))
                hypervolume_datasets.append(
                    {
                        "suffix": suffix,
                        "x": np.arange(1, len(rel_hvs) + 1)[:max_x],
                        "y": rel_hvs[:max_x],
                    }
                )
                print(
                    f"Saved hypervolume plots for {suffix} (final HV={final_hv:.3e}); "
                    f"ignored last {ignore_last_n} trials in Pareto/HV."
                )
            else:
                print("No hypervolume computed (no complete trials).")
            chosen_trial = None
            if chosen_indices is not None:
                chosen_trial = chosen_indices[idx]
            plot_data = compute_pareto_plot_data(
                study,
                suffix,
                ignore_last_n=ignore_last_n,
                chosen_trial_number=chosen_trial,
            )
            if plot_data:
                save_individual_pareto_plot(plot_data, save_dir)
                pareto_datasets.append(plot_data)
                summarize_pareto_tradeoff(plot_data)
        else:
            print("Skipping hypervolume: study is not two-objective.")

        best = get_best_trials(study, top_n)
        if not best:
            print(f"No completed trials in {full_name}")
            continue

        # Plotting
        # Resolve surrogate models folder (case-insensitive)
        models_root = os.path.join("tuned", study_prefix, "models")
        if not os.path.isdir(models_root):
            print(f"Models root not found: {models_root}")
            continue
        available = [
            d
            for d in os.listdir(models_root)
            if os.path.isdir(os.path.join(models_root, d))
        ]
        match = next((d for d in available if d.lower() == suffix.lower()), None)
        if match is None:
            print(
                f"Models folder for surrogate '{suffix}' not found in {models_root}. Available: {available}"
            )
            continue

        # # Collect top trials' loss histories
        # records = []
        # epochs = None
        # for fname in os.listdir(models_dir):
        #     if not fname.endswith(".pth"):
        #         continue
        #     m = re.search(r"_(\d+)\.pth$", fname)
        #     if not m:
        #         continue
        #     tnum = int(m.group(1))
        #     if tnum not in best:
        #         continue
        #     path = os.path.join(models_dir, fname)
        #     train, test, ep = load_loss_history(path)
        #     if test is None:
        #         continue
        #     records.append((tnum, test))
        #     epochs = ep
        # if not records:
        #     print(f"No top-{top_n} trials for surrogate '{suffix}'")
        #     continue
        # records.sort(key=lambda x: best.index(x[0]))
        # tnums, losses = zip(*records)
        # labels = [f"Trial {n}" for n in tnums]
        # out_dir = save_dir
        # plot_losses(
        #     list(losses),
        #     epochs,
        #     labels,
        #     title=f"{suffix} Top-{top_n}",
        #     save=True,
        #     out_dir=out_dir,
        #     mode=suffix,
        # )

    save_pareto_front_grid(pareto_datasets, save_dir)
    save_relative_hv_grid(hypervolume_datasets, save_dir)
    save_relative_hv_combined(hypervolume_datasets, save_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot top-N tuning losses for all surrogate studies."
    )
    p.add_argument(
        "--study_name",
        type=str,
        default="primordial_final",
        help="Main study prefix (e.g. lvparams5)",
    )
    p.add_argument(
        "--storage_name",
        type=str,
        default="primordial_final",
        help="Main study prefix (e.g. lvparams5)",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top trials to plot per surrogate",
    )
    p.add_argument(
        "--ignore_last_n",
        type=int,
        default=10,
        help=(
            "Number of most-recent completed trials to exclude from Pareto/hypervolume"
        ),
    )
    p.add_argument(
        "--chosen_indices",
        type=str,
        # default="171,114,135,237",  # cloud_final
        # default="27,61,13,299",  # cloud_parametric_final
        # default="18,1,16,234",  # primordial_parametric_final
        default="196,107,31,243",  # primordial_final
        help=(
            "Comma-separated list of Optuna trial numbers chosen per study (order "
            "matches sorted study names)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    chosen_list = None
    if args.chosen_indices:
        chosen_list = []
        for item in args.chosen_indices.split(","):
            item = item.strip()
            if not item:
                continue
            chosen_list.append(int(item))
    evaluate_tuning(
        args.study_name,
        args.top_n,
        args.storage_name,
        ignore_last_n=args.ignore_last_n,
        chosen_indices=chosen_list,
    )


if __name__ == "__main__":
    nice_print("Starting evaluation of tuning results")
    main()
    nice_print("Done.")
