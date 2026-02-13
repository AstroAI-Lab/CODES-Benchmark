import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d

from codes.utils import batch_factor_to_float

from .bench_utils import format_time

# Utility functions for plotting


def save_plot(
    plt,
    filename: str,
    conf: dict,
    surr_name: str = "",
    dpi: int = 300,
    base_dir: str = "plots",  # Base directory for saving plots
    increase_count: bool = False,  # Whether to increase the count for existing filenames
    format: str = "jpg",  # Format for saving the plot
) -> None:
    """
    Save the plot to a file, creating necessary directories if they don't exist.

    Args:
        plt (matplotlib.pyplot): The plot object to save.
        filename (str): The desired filename for the plot.
        conf (dict): The configuration dictionary.
        surr_name (str): The name of the surrogate model.
        dpi (int): The resolution of the saved plot.
        base_dir (str, optional): The base directory where plots will be saved. Default is "plots".
        increase_count (bool, optional): Whether to increment the filename count if a file already exists. Default is True.
        format (str, optional): The format for saving the plot. Default is "png". Can be "png", "pdf", "svg", etc.

    Raises:
        ValueError: If the configuration dictionary does not contain the required keys.
    """
    if "training_id" not in conf:
        raise ValueError("Configuration dictionary must contain 'training_id'.")

    training_id = conf["training_id"]
    plot_dir = os.path.join(base_dir, training_id, surr_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    filename = f"{filename.split('.')[0]}.{format}"

    # Call save_plot_counter with increase_count option
    filepath = save_plot_counter(filename, plot_dir, increase_count=increase_count)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    if conf.get("verbose", False):
        print(f"Plot saved as: {filepath}")


def save_plot_counter(
    filename: str, directory: str, increase_count: bool = True
) -> str:
    """
    Save a plot with an incremented filename if a file with the same name already exists.

    Args:
        filename (str): The desired filename for the plot.
        directory (str): The directory to save the plot in.
        increase_count (bool, optional): Whether to increment the filename count if a file already exists. Default is True.

    Returns:
        str: The full path to the saved plot.
    """
    if not increase_count:
        filepath = os.path.join(directory, filename)
        return filepath

    base, ext = os.path.splitext(filename)
    counter = 1
    filepath = os.path.join(directory, filename)

    while os.path.exists(filepath):
        filepath = os.path.join(directory, f"{base}_{counter}{ext}")
        counter += 1

    return filepath


# Per-surrogate model plots


def plot_error_percentiles_over_time(
    surr_name: str,
    conf: dict,
    errors: np.ndarray,
    timesteps: np.ndarray,
    title: str,
    mode: str = "relative",  # "relative" or "deltadex"
    save: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot mean, median, and percentile error envelopes over time.

    - mode="relative":
        Treats `errors` as relative errors (0..∞).
        Plots bidirectional percentile bands (25-75, 5-95, 0.5-99.5).
        Y-axis is log-scaled.
    - mode="deltadex":
        Treats `errors` as log-space absolute errors (Δdex ≥ 0).
        Plots one-sided percentile bands (0-50, 0-90, 0-99).
        Y-axis is linear, starting at 0.

    Args:
        surr_name (str): Name of the surrogate model (used for saving).
        conf (dict): Configuration dictionary containing dataset and output settings.
        errors (np.ndarray): Error array of shape [N_samples, N_timesteps, N_quantities].
                             Values are either relative errors or Δdex depending on `mode`.
        timesteps (np.ndarray): Array of timesteps corresponding to the second axis of `errors`.
        title (str): Title for the plot.
        mode (str, optional): "relative" for relative errors, "deltadex" for log-space absolute errors.
                              Defaults to "relative".
        save (bool, optional): Whether to save the plot to disk. Defaults to False.
        show_title (bool, optional): Whether to show the plot title. Defaults to True.

    Returns:
        None
    """
    mean_ts = np.mean(errors, axis=(0, 2))
    median_ts = np.median(errors, axis=(0, 2))
    mean_val = mean_ts.mean()
    median_val = median_ts.mean()

    plt.figure(figsize=(6, 4))

    if mode == "relative":
        # bidirectional bands
        stats = {
            "50": (25, 75),
            "90": (5, 95),
            "99": (0.5, 99.5),
        }
        percentiles = {
            p: (
                np.percentile(errors, low, axis=(0, 2)),
                np.percentile(errors, high, axis=(0, 2)),
            )
            for p, (low, high) in stats.items()
        }

        plt.plot(
            timesteps,
            mean_ts,
            label=f"Mean Error\nMean={mean_val * 100:.2f}%",
            color="blue",
        )
        plt.plot(
            timesteps,
            median_ts,
            label=f"Median Error\nMedian={median_val * 100:.2f}%",
            color="red",
        )

        for p, (low_ts, high_ts) in percentiles.items():
            alpha = {"50": 0.45, "90": 0.4, "99": 0.15}[p]
            plt.fill_between(
                timesteps,
                low_ts,
                high_ts,
                color="grey",
                alpha=alpha,
                label=f"{p}th Percentile",
            )

        plt.yscale("log")
        plt.ylabel("Relative Error")
        filename = "accuracy_rel_errors_time.pdf"

    elif mode == "deltadex":
        # one-sided bands
        percentiles = {
            "50": np.percentile(errors, 50, axis=(0, 2)),
            "90": np.percentile(errors, 90, axis=(0, 2)),
            "99": np.percentile(errors, 99, axis=(0, 2)),
        }

        plt.plot(
            timesteps,
            mean_ts,
            label=f"Mean Δdex\nMean={mean_val:.3f}",
            color="blue",
        )
        plt.plot(
            timesteps,
            median_ts,
            label=f"Median Δdex\nMedian={median_val:.3f}",
            color="red",
        )

        for p, vals in percentiles.items():
            alpha = {"50": 0.45, "90": 0.35, "99": 0.2}[p]
            plt.fill_between(
                timesteps, 0, vals, alpha=alpha, color="grey", label=f"{p}th Percentile"
            )

        plt.ylabel(r"$\Delta dex$")
        plt.ylim(0, plt.ylim()[1])
        filename = "accuracy_delta_dex_time.pdf"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    plt.xlabel("Time")
    plt.xlim(timesteps[0], timesteps[-1])
    if conf.get("dataset", {}).get("log_timesteps", False):
        plt.xscale("log")
    if show_title:
        plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save:
        save_plot(plt, filename, conf, surr_name)

    plt.close()


def plot_dynamic_correlation(
    surr_name: str,
    conf: dict,
    gradients: np.ndarray,
    errors: np.ndarray,
    save: bool = False,
    show_title: bool = True,
):
    """
    Plot the correlation between the gradients of the data and the prediction errors.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        gradients (np.ndarray): The gradients of the data.
        errors (np.ndarray): The prediction errors.
        save (bool): Whether to save the plot.
        show_title (bool): Whether to show the title on the plot.
    """
    # Flatten the arrays for correlation plot
    gradients_flat = gradients.flatten()
    errors_flat = errors.flatten()

    # Scatter plot of gradients vs. errors
    plt.figure(figsize=(8, 4))
    plt.scatter(gradients_flat, errors_flat, alpha=0.5, s=5)
    plt.xlabel("Gradient of Data")
    plt.ylabel("Prediction Error")
    if show_title:
        plt.title("Correlation between Gradients and Prediction Errors")

    # Save the plot
    if save and conf:
        save_plot(plt, "gradient_error_correlation.png", conf, surr_name)

    plt.close()


def plot_generalization_errors(
    surr_name: str,
    conf: dict,
    metrics: np.ndarray,
    model_errors: np.ndarray,
    mode: str,
    save: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot the generalization errors of a model for various metrics.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        metrics (np.ndarray): The metrics (e.g., intervals, cutoffs, batch sizes, number of training samples).
        model_errors (np.ndarray): The model errors.
        mode (str): The mode of generalization ("interpolation", "extrapolation", "sparse", "batchsize").
        save (bool): Whether to save the plot.
        show_title (bool): Whether to show the title on the plot.

    Returns:
        None
    """
    if mode == "interpolation":
        xlabel = "Interpolation Interval"
        title = "Interpolation Errors"
        filename = "interpolation_errors.png"
    elif mode == "extrapolation":
        xlabel = "Extrapolation Cutoff"
        title = "Extrapolation Errors"
        filename = "extrapolation_errors.png"
    elif mode == "sparse":
        xlabel = "Number of Training Samples"
        title = "Sparse Training Errors"
        filename = "sparse_errors.png"
    elif mode == "batchsize":
        xlabel = "Elements per Batch"
        title = "Batch Size Training Errors"
        filename = "batchsize_errors.png"
    else:
        raise ValueError(
            "Invalid mode. Choose from 'interpolation', 'extrapolation', 'sparse', 'batchsize'."
        )

    plt.figure(figsize=(6, 4))
    plt.scatter(metrics, model_errors, label=surr_name, color="#3A1A5A")
    plt.xlabel(xlabel)
    if mode == "sparse" or mode == "batchsize":
        plt.xscale("log")
    plt.ylabel("Mean Absolute Error")
    plt.yscale("log")
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    if show_title:
        plt.title(title)
    plt.legend()

    if save and conf:
        save_plot(plt, filename, conf, surr_name)

    plt.close()


def plot_average_errors_over_time(
    surr_name: str,
    conf: dict,
    errors: np.ndarray,
    metrics: np.ndarray,
    timesteps: np.ndarray,
    mode: str,
    save: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot Δdex errors over time for different evaluation modes.

    Args:
        surr_name (str): Surrogate name.
        conf (dict): Config dictionary.
        errors (np.ndarray): Errors [N_metrics, n_timesteps].
        metrics (np.ndarray): Metrics [N_metrics].
        timesteps (np.ndarray): Timesteps.
        mode (str): One of 'interpolation', 'extrapolation', 'sparse', 'batchsize'.
    """
    plt.figure(figsize=(6, 4))

    labels = {
        "interpolation": "interval",
        "extrapolation": "cutoff",
        "sparse": "samples",
        "batchsize": "batch size",
    }
    if mode not in labels:
        raise ValueError("Invalid mode for error plotting.")

    colors = plt.cm.magma(np.linspace(0.15, 0.85, errors.shape[0]))

    if mode == "sparse":
        for i, metric in enumerate(metrics):
            label = f"{metric} {labels[mode]}"
            plt.plot(timesteps, errors[i], label=label, color=colors[i])
    else:
        for i, metric in enumerate(metrics):
            label = f"{labels[mode]} {metric}"
            plt.plot(timesteps, errors[i], label=label, color=colors[i])
            if mode == "extrapolation":
                cutoff_point = timesteps[metric - 1]
                plt.axvline(
                    x=cutoff_point, color=colors[i], linestyle="--", linewidth=0.8
                )
                plt.text(
                    cutoff_point,
                    errors[i, metric - 1],
                    f"{metric}",
                    color=colors[i],
                    va="bottom",
                    ha="right",
                )

    plt.xlabel("Time")
    plt.xlim(timesteps[0], timesteps[-1])
    plt.ylabel(r"Mean $\Delta dex$")
    plt.yscale("log")
    if conf.get("dataset", {}).get("log_timesteps", False):
        plt.xscale("log")
    title = f"Mean Δdex Errors over Time ({mode.capitalize()}, {surr_name})"
    filename = f"errors_over_time_{mode}.png"

    if show_title:
        plt.title(title)
    plt.legend()

    if save and conf:
        save_plot(plt, filename, conf, surr_name)

    plt.close()


def plot_example_mode_predictions(
    surr_name: str,
    conf: dict,
    preds_log: np.ndarray,
    preds_main_log: np.ndarray,
    targets_log: np.ndarray,
    timesteps: np.ndarray,
    metric: int,
    mode: str = "interpolation",
    example_idx: int = 0,
    num_quantities: int = 100,
    labels: list[str] | None = None,
    save: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot example predictions in log-space (Δdex) alongside ground truth targets
    for either interpolation or extrapolation mode.

    Predictions and targets are assumed to be in log10 space (leave_log=True).
    Axis labels and plotted values are consistent with this log representation.

    Args:
        surr_name (str): Name of the surrogate model.
        conf (dict): Configuration dictionary.
        preds_log (np.ndarray): Predictions in log-space of shape [N_samples, T, Q].
        preds_main_log (np.ndarray): Main model (reference) predictions in log-space of shape [N_samples, T, Q].
        targets_log (np.ndarray): Targets in log-space of shape [N_samples, T, Q].
        timesteps (np.ndarray): Array of timesteps.
        metric (int):
            - In interpolation mode: the training interval (e.g., 10 means every 10th timestep was used).
            - In extrapolation mode: the cutoff timestep index.
        mode (str, optional): Either "interpolation" or "extrapolation". Default is "interpolation".
        example_idx (int, optional): Index of the example to plot. Default is 0.
        num_quantities (int, optional): Maximum number of quantities to plot. Default is 100.
        labels (list[str], optional): Names of the quantities to display in legends.
        save (bool, optional): Whether to save the figure. Default is False.
        show_title (bool, optional): Whether to add a title to the figure. Default is True.

    Returns:
        None
    """
    num_quantities = min(preds_log.shape[2], num_quantities)
    quantities_per_plot = 10
    num_plots = int(np.ceil(num_quantities / quantities_per_plot))

    # colors = plt.cm.viridis(np.linspace(0, 0.95, preds_log.shape[2]))
    colors = get_custom_palette(quantities_per_plot)
    fig = plt.figure(figsize=(6, 4 * num_plots))
    gs = GridSpec(num_plots, 1, figure=fig)

    for plot_idx in range(num_plots):
        ax = fig.add_subplot(gs[plot_idx])
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, preds_log.shape[2])
        legend_lines = []

        for chem_idx in range(start_idx, end_idx):
            color = colors[chem_idx % quantities_per_plot]
            gt = targets_log[example_idx, :, chem_idx]
            pred = preds_log[example_idx, :, chem_idx]
            pred_main = preds_main_log[example_idx, :, chem_idx]
            (gt_line,) = ax.plot(timesteps, gt, "-", color=color)
            ax.plot(timesteps, pred, "--", color=color)
            ax.plot(timesteps, pred_main, ":", color=color)
            legend_lines.append(gt_line)

        if mode == "interpolation":
            used_timesteps = timesteps[::metric]
            for t in used_timesteps:
                ax.axvline(
                    x=t, color="gray", linestyle="dashed", linewidth=0.8, alpha=0.7
                )
        elif mode == "extrapolation":
            if metric < len(timesteps):
                cutoff = timesteps[metric]
                ax.axvline(
                    x=cutoff, color="red", linestyle="solid", linewidth=1.0, alpha=0.9
                )
            else:
                raise ValueError("Extrapolation metric exceeds timestep length.")

        ax.set_ylabel("log(Abundance)")
        ax.set_xlim(timesteps.min(), timesteps.max())
        if conf.get("dataset", {}).get("log_timesteps", False):
            ax.set_xscale("log")

        if labels is not None:
            legend_labels = labels[start_idx:end_idx]
            ax.legend(
                legend_lines,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                bbox_transform=ax.transAxes,
            )

    fig.text(0.5, 0.04, "Time", ha="center", va="center", fontsize=12)
    handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", label="Ground Truth"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Prediction"),
        plt.Line2D(
            [0], [0], color="black", linestyle=":", label="Main Model Prediction"
        ),
    ]
    y_pos = 0.95 - (0.06 / num_plots)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_pos),
        ncol=2,
        fontsize="small",
    )
    fig.align_ylabels()

    if mode == "interpolation":
        title = f"Interpolation Example Predictions ({surr_name})\nSample {example_idx}, Interval {metric}"
    else:
        title = f"Extrapolation Example Predictions ({surr_name})\nSample {example_idx}, Cutoff {metric}"

    if show_title:
        plt.suptitle(title, y=0.97)

    plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.92])
    if save and conf:
        filename = f"example_preds_{mode}.png"
        save_plot(plt, filename, conf, surr_name)
    plt.close()


def plot_example_iterative_predictions(
    surr_name: str,
    conf: dict,
    iterative_preds: np.ndarray,
    full_preds: np.ndarray,
    targets: np.ndarray,
    timesteps: np.ndarray,
    iter_interval: int,
    example_idx: int | None = None,
    num_quantities: int = 100,
    labels: list[str] | None = None,
    save: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot one sample's full iterative trajectory:
    ground truth vs. chained predictions, with retrigger lines.
    """
    # choose example if not given
    if example_idx is None:
        errors = np.mean(np.abs(iterative_preds - targets), axis=(1, 2))
        example_idx = int(np.argsort(np.abs(errors - np.median(errors)))[0])

    n_q = min(iterative_preds.shape[2], num_quantities)
    per_plot = min(10, n_q)

    n_plots = int(np.ceil(n_q / per_plot))
    # colors = plt.cm.viridis(np.linspace(0, 0.95, per_plot))
    colors = get_custom_palette(per_plot)

    fig = plt.figure(figsize=(6, 4 * n_plots))
    gs = GridSpec(n_plots, 1, figure=fig)

    for pi in range(n_plots):
        ax = fig.add_subplot(gs[pi])
        start, end = pi * per_plot, min((pi + 1) * per_plot, n_q)
        for qi in range(start, end):
            c = colors[qi % per_plot]
            gt = targets[example_idx, :, qi]
            pr = iterative_preds[example_idx, :, qi]
            ax.plot(timesteps, gt, "-", color=c)
            ax.plot(timesteps, pr, "--", color=c)
            init_pr = full_preds[example_idx, :, qi]
            ax.plot(timesteps, init_pr, ":", color=c)
        # retrigger lines
        for t in timesteps[::iter_interval]:
            ax.axvline(x=t, linestyle=":", linewidth=0.8, alpha=0.7)
        if conf.get("dataset", {}).get("log10_transform", True):
            ax.set_yscale("log")
        ax.set_xlim(timesteps.min(), timesteps.max())
        if conf["dataset"].get("log_timesteps", False):
            ax.set_xscale("log")
        ax.set_ylabel("Abundance")
        if labels is not None:
            legend_lines = [
                plt.Line2D([0], [0], color=colors[i % per_plot])
                for i in range(start, end)
            ]
            ax.legend(
                legend_lines,
                labels[start:end],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

    fig.text(0.5, 0.04, "Time", ha="center", va="center", fontsize=12)

    handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", label="Ground Truth"),
        plt.Line2D(
            [0], [0], color="black", linestyle="--", label="Iterative Prediction"
        ),
        plt.Line2D([0], [0], color="black", linestyle=":", label="Full Prediction"),
    ]
    pos = 0.95 - (0.06 / n_plots)
    fig.legend(
        handles,
        ["Ground Truth", "Iterative Prediction", "Full Prediction"],
        loc="upper center",
        bbox_to_anchor=(0.5, pos),
        ncol=2,
        fontsize="small",
    )
    fig.align_ylabels()

    if show_title:
        title = (
            f"Iterative Prediction Example ({surr_name})\n"
            f"Sample {example_idx}, Interval {iter_interval}"
        )
        plt.suptitle(title, y=0.97)

    plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.92])

    if save and conf:
        fname = "example_preds_iterative.png"
        save_plot(plt, fname, conf, surr_name)

    plt.close()


def plot_example_predictions_with_uncertainty(
    surr_name: str,
    conf: dict,
    log_mean: np.ndarray,
    log_std: np.ndarray,
    log_targets: np.ndarray,
    timesteps: np.ndarray,
    example_idx: int = 0,
    num_quantities: int = 100,
    labels: list[str] | None = None,
    save: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot example predictions with uncertainty in log10 space (dex).

    Args:
        surr_name (str): Name of the surrogate model.
        conf (dict): Configuration dictionary.
        log_mean (np.ndarray): Ensemble mean predictions in log10 space.
        log_std (np.ndarray): Ensemble standard deviation in log10 space.
        log_targets (np.ndarray): Ground truth targets in log10 space.
        timesteps (np.ndarray): Array of timesteps.
        example_idx (int): Index of the example to plot.
        num_quantities (int): Number of species/quantities to plot.
        labels (list, optional): Quantity labels.
        save (bool): Whether to save the figure.
        show_title (bool): Whether to display a title.
    """
    num_quantities = min(log_std.shape[2], num_quantities)
    quantities_per_plot = 10
    num_plots = int(np.ceil(num_quantities / quantities_per_plot))
    # colors = plt.cm.viridis(np.linspace(0, 0.95, quantities_per_plot))
    colors = get_custom_palette(quantities_per_plot)

    fig = plt.figure(figsize=(6, 4 * num_plots))
    gs = GridSpec(num_plots, 1, figure=fig)

    for plot_idx in range(num_plots):
        ax = fig.add_subplot(gs[plot_idx])
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, num_quantities)

        legend_lines = []
        for chem_idx in range(start_idx, end_idx):
            color = colors[chem_idx % quantities_per_plot]
            gt = log_targets[example_idx, :, chem_idx]
            mean = log_mean[example_idx, :, chem_idx]
            std = log_std[example_idx, :, chem_idx]

            (gt_line,) = ax.plot(timesteps, gt, "--", color=color)
            ax.plot(timesteps, mean, "-", color=color)
            legend_lines.append(gt_line)

            for sigma in [1, 2, 3]:
                lower = mean - sigma * std
                upper = mean + sigma * std
                ax.fill_between(timesteps, lower, upper, color=color, alpha=0.5 / sigma)

        ax.set_ylabel("log10(Abundance) [dex]")
        if labels is not None:
            ax.legend(
                legend_lines,
                labels[start_idx:end_idx],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                bbox_transform=ax.transAxes,
            )

        ax.set_xlim(timesteps.min(), timesteps.max())
        if conf.get("dataset", {}).get("log_timesteps", False):
            ax.set_xscale("log")

    fig.text(0.5, 0.04, "Time", ha="center", va="center", fontsize=12)
    handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", label="Ground Truth"),
        plt.Line2D(
            [0], [0], color="black", linestyle="-", label="Prediction (Ensemble Mean)"
        ),
    ]
    y_pos = 0.945 - (0.06 / num_plots)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_pos),
        ncol=2,
        fontsize="small",
    )

    if show_title:
        plt.suptitle(
            f"DeepEnsemble Predictions with Uncertainty for {surr_name}\n"
            f"Sample Index: {example_idx},  "
            + r"$\mu \pm (1,2,3)\,\sigma$ Intervals (log-space)",
            y=0.97,
        )

    plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.92])
    if save and conf:
        save_plot(plt, "uq_example_deepensemble_preds.png", conf, surr_name)
    plt.close()


def plot_average_uncertainty_over_time(
    surr_name: str,
    conf: dict,
    errors_time: np.ndarray,
    preds_std: np.ndarray,
    timesteps: np.ndarray,
    save: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot average predictive uncertainty and errors over time in log-space (dex).

    Args:
        surr_name (str): Name of the surrogate model.
        conf (dict): Configuration dictionary.
        errors_time (np.ndarray): Log-space prediction errors over time.
        preds_std (np.ndarray): Log-space ensemble standard deviation over time.
        timesteps (np.ndarray): Array of timesteps.
        save (bool): Whether to save the plot.
        show_title (bool): Whether to show a title.
    """
    mean_error = np.mean(errors_time)
    mean_uncertainty = np.mean(preds_std)

    plt.figure(figsize=(8, 4))
    plt.plot(
        timesteps,
        preds_std,
        label=f"Mean Uncertainty (avg: {mean_uncertainty:.2e})",
        color="#3A1A5A",
    )
    plt.plot(
        timesteps,
        errors_time,
        label=r"Mean Error $\Delta dex$" + f"(avg: {mean_error:.2e})",
        color="#DA5F4D",
    )

    plt.xlabel("Time")
    plt.ylabel(r"$\Delta dex$ / Log-Space Uncertainty")
    plt.xlim(timesteps[0], timesteps[-1])
    if conf.get("dataset", {}).get("log_timesteps", False):
        plt.xscale("log")
    if show_title:
        plt.title(r"Average Log-Space Uncertainty and Error ($\Delta dex$) Over Time")
    plt.legend()

    if save and conf:
        save_plot(plt, "uq_uncertainty_over_time.png", conf, surr_name)
    plt.close()


def plot_uncertainty_vs_errors(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    save: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot the correlation between predictive uncertainty and prediction errors.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        preds_std (np.ndarray): Standard deviation of predictions from the ensemble of models.
        errors (np.ndarray): Prediction errors.
        save (bool, optional): Whether to save the plot as a file.
        show_title (bool): Whether to show the title on the plot.
    """
    # Normalize the errors
    errors = errors / np.abs(errors).max()

    plt.figure(figsize=(8, 4))
    plt.scatter(preds_std.flatten(), errors.flatten(), alpha=0.5)
    plt.xlabel("Predictive Uncertainty")
    plt.ylabel("Prediction Error (Normalized)")
    if show_title:
        plt.title("Correlation between Predictive Uncertainty and Prediction Errors")

    if save and conf:
        save_plot(plt, "uncertainty_vs_errors.png", conf, surr_name)

    plt.close()


def plot_surr_losses(
    model,
    surr_name: str,
    conf: dict,
    timesteps: np.ndarray,
    show_title: bool = True,
) -> None:
    """
    Plot the training and test losses for the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        timesteps (np.ndarray): The timesteps array.
        show_title (bool): Whether to show the title on the plot.
    """
    training_id = conf["training_id"]

    def load_losses(model_identifier: str):
        model.load(training_id, surr_name, model_identifier=model_identifier)
        train_loss = model.train_loss
        test_loss = model.test_loss
        epochs = model.n_epochs
        return train_loss, test_loss, epochs

    # Main model losses
    main_train_loss, main_test_loss, epochs = load_losses(f"{surr_name.lower()}_main")
    plot_losses(
        (main_train_loss, main_test_loss),
        epochs,
        ("Train Loss", "Test Loss"),
        title="Main Model Losses",
        save=True,
        conf=conf,
        surr_name=surr_name,
        mode="main",
        show_title=show_title,
    )

    # Interpolation losses
    if conf.get("interpolation", {}).get("enabled", False):
        intervals = conf["interpolation"]["intervals"]
        interp_train_losses = [main_train_loss]
        interp_test_losses = [main_test_loss]
        for interval in intervals:
            train_loss, test_loss, epochs = load_losses(
                f"{surr_name.lower()}_interpolation_{interval}"
            )
            interp_train_losses.append(train_loss)
            interp_test_losses.append(test_loss)
        plot_losses(
            tuple(interp_test_losses),
            epochs,
            tuple(f"Interval {interval}" for interval in [1] + intervals),
            title="Interpolation Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="interpolation",
            show_title=show_title,
        )

    # Extrapolation losses
    if conf.get("extrapolation", {}).get("enabled", False):
        cutoffs = conf["extrapolation"]["cutoffs"]
        extra_train_losses = [main_train_loss]
        extra_test_losses = [main_test_loss]
        for cutoff in cutoffs:
            train_loss, test_loss, epochs = load_losses(
                f"{surr_name.lower()}_extrapolation_{cutoff}"
            )
            extra_train_losses.append(train_loss)
            extra_test_losses.append(test_loss)
        plot_losses(
            tuple(extra_test_losses),
            epochs,
            tuple(f"Cutoff {cutoff}" for cutoff in cutoffs + [len(timesteps)]),
            title="Extrapolation Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="extrapolation",
            show_title=show_title,
        )

    # Sparse losses
    if conf.get("sparse", {}).get("enabled", False):
        factors = conf["sparse"]["factors"]
        sparse_train_losses = [main_train_loss]
        sparse_test_losses = [main_test_loss]
        for factor in factors:
            train_loss, test_loss, epochs = load_losses(
                f"{surr_name.lower()}_sparse_{factor}"
            )
            sparse_train_losses.append(train_loss)
            sparse_test_losses.append(test_loss)
        plot_losses(
            tuple(sparse_test_losses),
            epochs,
            tuple(f"Factor {factor}" for factor in [1] + factors),
            title="Sparse Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="sparse",
            show_title=show_title,
        )

    # UQ losses
    if conf.get("uncertainty", {}).get("enabled", False):
        n_models = conf["uncertainty"]["ensemble_size"]
        uq_train_losses = [main_train_loss]
        uq_test_losses = [main_test_loss]
        for i in range(n_models - 1):
            train_loss, test_loss, epochs = load_losses(
                f"{surr_name.lower()}_UQ_{i + 1}"
            )
            uq_train_losses.append(train_loss)
            uq_test_losses.append(test_loss)
        plot_losses(
            tuple(uq_test_losses),
            epochs,
            tuple(f"Model {i}" for i in range(n_models)),
            title="UQ Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="UQ",
            show_title=show_title,
        )

    # Batchsize losses
    if conf.get("batch_scaling", {}).get("enabled", False):
        batch_factors = conf["batch_scaling"]["sizes"]
        batch_train_losses = []
        batch_test_losses = []
        batch_sizes = []
        surr_index = conf["surrogates"].index(surr_name)
        main_model_bs = conf["batch_size"][surr_index]
        for batch_factor in batch_factors:
            batch_factor = batch_factor_to_float(batch_factor)
            batch_size = int(main_model_bs * batch_factor)
            batch_sizes.append(batch_size)
            train_loss, test_loss, epochs = load_losses(
                f"{surr_name.lower()}_batchsize_{batch_size}"
            )
            batch_train_losses.append(train_loss)
            batch_test_losses.append(test_loss)
        plot_losses(
            tuple(batch_test_losses),
            epochs,
            tuple(f"Batch Size {batch_size}" for batch_size in batch_sizes),
            title="Batch Size Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="batchsize",
            show_title=show_title,
        )


def plot_error_distribution_per_quantity(
    surr_name: str,
    conf: dict,
    errors: np.ndarray,
    quantity_names: list[str] | None = None,
    num_quantities: int = 10,
    mode: str = "relative",  # "relative" or "deltadex"
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot the distribution of errors for each quantity as a smoothed histogram plot.

    - mode="relative":
        Errors are relative (0..∞).
        Histogram is plotted in log-space (x-axis log-scaled).
    - mode="deltadex":
        Errors are absolute log-space errors (Δdex ≥ 0).
        Histogram is plotted on linear scale.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        errors (np.ndarray): Errors array of shape [num_samples, num_timesteps, num_quantities].
        quantity_names (list, optional): List of quantity names for labeling the lines.
        num_quantities (int, optional): Number of quantities to plot. Default is 10.
        mode (str, optional): "relative" or "deltadex". Default is "relative".
        save (bool, optional): Whether to save the plot as a file.
        show_title (bool): Whether to show the title on the plot.
    """
    total_quantities = errors.shape[2]
    errors = errors.reshape(-1, total_quantities)

    num_quantities = min(num_quantities, 50)
    errors = errors[:, :num_quantities]
    quantity_names = (
        quantity_names[:num_quantities] if quantity_names is not None else None
    )

    quantities_per_plot = 10
    num_plots = int(np.ceil(num_quantities / quantities_per_plot))

    data_per_quantity = []
    for i in range(num_quantities):
        q_errors = errors[:, i]
        if np.isnan(q_errors).any():
            raise ValueError("Error values contain NaNs.")
        if mode == "relative":
            q_errors = q_errors[q_errors > 0]
            data_per_quantity.append(np.log10(q_errors))
        elif mode == "deltadex":
            q_errors = q_errors[q_errors >= 0]
            data_per_quantity.append(np.log10(q_errors))  # q_errors)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if mode == "relative":
        min_percentiles = [np.percentile(d, 1) for d in data_per_quantity if len(d) > 0]
        max_percentiles = [
            np.percentile(d, 99) for d in data_per_quantity if len(d) > 0
        ]
        global_min, global_max = np.min(min_percentiles), np.max(max_percentiles)
        x_min, x_max = np.floor(global_min), np.ceil(global_max)
        x_vals = np.linspace(x_min, x_max + 0.1, 100)
    else:  # deltadex
        # max_percentiles = [
        #     np.percentile(d, 98) for d in data_per_quantity if len(d) > 0
        # ]
        # global_max = np.max(max_percentiles)
        # x_min, x_max = 0.0, global_max
        min_percentiles = [np.percentile(d, 1) for d in data_per_quantity if len(d) > 0]
        max_percentiles = [
            np.percentile(d, 99) for d in data_per_quantity if len(d) > 0
        ]
        global_min, global_max = np.min(min_percentiles), np.max(max_percentiles)
        x_min, x_max = np.floor(global_min), np.ceil(global_max)
        x_vals = np.linspace(x_min, x_max + 0.1, 100)

    fig, axes = plt.subplots(
        num_plots,
        1,
        figsize=(8, 3 * num_plots) if num_plots > 1 else (10, 4),
        sharex=True,
    )
    if num_plots == 1:
        axes = [axes]

    colors = get_custom_palette(quantities_per_plot)

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, num_quantities)

        for i in range(start_idx, end_idx):
            d = data_per_quantity[i]
            if len(d) == 0:
                continue

            hist, bin_edges = np.histogram(d, bins=x_vals, density=True)
            smoothed_hist = gaussian_filter1d(hist, sigma=2)

            if mode == "relative":
                x_axis = 10 ** bin_edges[:-1]
            else:
                x_axis = 10 ** bin_edges[:-1]  # bin_edges[:-1]

            ax.plot(
                x_axis,
                smoothed_hist,
                label=(
                    quantity_names[i]
                    if quantity_names is not None and len(quantity_names) > i
                    else None
                ),
                color=colors[i % quantities_per_plot],
            )

        ax.set_ylabel("Smoothed Histogram Count")
        if quantity_names is not None:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    fig.align_ylabels()

    if mode == "relative":
        plt.xscale("log")
        plt.xlim(np.maximum(10**x_min, 1e-8), 10**x_max)
        plt.xlabel("Relative Error")
        filename = "accuracy_rel_error_per_quantity.png"
        title_str = f"Relative Error Distribution per Quantity ({surr_name})"
    else:
        plt.xscale("log")
        plt.xlim(np.maximum(10**x_min, 1e-8), 10**x_max)  # x_min, x_max)
        plt.xlabel(r"$\Delta dex$")
        filename = "accuracy_delta_dex_per_quantity.png"
        title_str = f"Absolute Log-Space Error Distribution per Quantity ({surr_name})"

    if show_title:
        if num_plots > 1:
            fig.suptitle(title_str)
        else:
            plt.title(title_str)

    if save and conf:
        save_plot(plt, filename, conf, surr_name)

    plt.close()


# Comparative plots from here on!


def plot_losses(
    loss_histories: tuple[np.array, ...],
    epochs: int,
    labels: tuple[str, ...],
    title: str = "Losses",
    save: bool = False,
    conf: Optional[dict] = None,
    surr_name: Optional[str] = None,
    mode: str = "main",
    percentage: float = 2.0,
    show_title: bool = True,
) -> None:
    """
    Plot the loss trajectories for multiple models on a single axis.

    Args:
        loss_histories (tuple[np.ndarray, ...]): Loss arrays for each model.
        epochs (int): Number of epochs in the associated training run.
        labels (tuple[str, ...]): Labels matching ``loss_histories``.
        title (str): Plot title.
        save (bool): Whether to save the figure.
        conf (dict | None): Configuration dictionary used for saving.
        surr_name (str | None): Surrogate identifier used in filenames.
        mode (str): Training mode name appended to the filename.
        percentage (float): Portion of early iterations ignored when computing y-limits.
        show_title (bool): Whether to draw the title.
    """

    # Determine start index based on percentage
    start_idx = int(len(loss_histories[0]) * (percentage / 100))

    # Handle NaN or inf values
    for loss in loss_histories:
        if loss is not None:
            if np.isinf(loss).any():
                loss[np.isinf(loss)] = 0.0
            if np.isnan(loss).any():
                loss[np.isnan(loss)]
            if np.isinf(loss).any() or np.isnan(loss).any():
                print(
                    "Warning: Loss array contains NaN or inf values. Replacing with 0."
                )

    # Determine min and max range where the losses are non-zero
    min_val = min(
        loss[start_idx:][loss[start_idx:] > 0].min()
        for loss in loss_histories
        if loss is not None
    )
    max_val = max(
        loss[start_idx:][loss[start_idx:] > 0].max()
        for loss in loss_histories
        if loss is not None
    )

    # Colormap
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(loss_histories)))

    num_epochs = epochs + 1

    epochs = np.linspace(0, epochs + 1, num=len(loss_histories[0]), endpoint=False)

    # Create the figure
    plt.figure(figsize=(6, 4))
    loss_plotted = False
    for loss, label in zip(loss_histories, labels):
        if loss is not None:
            plt.plot(epochs, loss, label=label, color=colors[labels.index(label)])
            loss_plotted = True

    plt.xlabel("Epoch")
    plt.xlim(0, num_epochs)
    plt.ylabel("Log-MAE")
    plt.yscale("log")
    plt.ylim(min_val, max_val)
    if show_title:
        plt.title(title)
    plt.legend()

    if not loss_plotted:
        plt.text(
            0.5,
            0.5,
            "No losses available",
            horizontalalignment="center",
            verticalalignment="center",
        )

    if save and conf and surr_name:
        save_plot(plt, "losses_" + mode.lower() + ".png", conf, surr_name)

    plt.close()


def plot_losses_dual_axis(
    train_loss: np.array,
    test_loss: np.array,
    labels: tuple[str, str] = ("Train Loss", "Test Loss"),
    title: str = "Losses",
    save: bool = False,
    conf: Optional[dict] = None,
    surr_name: Optional[str] = None,
    show_title: bool = True,
) -> None:
    """
    Plot training and validation losses on dual y-axes.

    Args:
        train_loss (np.ndarray): Training loss curve.
        test_loss (np.ndarray): Validation loss curve.
        labels (tuple[str, str]): Axis labels for train/test.
        title (str): Plot title.
        save (bool): Whether to save the figure.
        conf (dict | None): Configuration dictionary used for saving.
        surr_name (str | None): Surrogate identifier used in filenames.
        show_title (bool): Whether to draw the title.
    """

    # Colormap
    colors = plt.cm.magma(np.linspace(0.15, 0.85, 2))

    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    # Plot train loss
    ax1.plot(train_loss, label=labels[0], color=colors[0])
    ax1.set_xlabel("Epoch")
    ax1.set_xlim(0, len(train_loss))
    ax1.set_ylabel(labels[0], color=colors[0])
    ax1.set_yscale("log")

    # Plot test loss
    ax2.plot(test_loss, label=labels[1], color=colors[1])
    ax2.set_ylabel(labels[1], color=colors[1])
    ax2.set_yscale("log")

    if show_title:
        plt.title(title)

    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    # Save plot
    if save and conf and surr_name:
        save_plot(fig, "main_losses.png", conf, surr_name)

    plt.show()
    plt.close(fig)


def plot_loss_comparison(
    train_losses: tuple[np.ndarray, ...],
    test_losses: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot the training and test losses for different surrogate models.

    Args:
        train_losses (tuple): Tuple of training loss arrays for each surrogate model.
        test_losses (tuple): Tuple of test loss arrays for each surrogate model.
        labels (tuple): Tuple of labels for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.
        show_title (bool): Whether to show the title on the plot.

    Returns:
        None
    """
    plt.figure(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(train_losses)))
    max_epochs, min_val, max_val = 0, np.inf, 0

    for i, (train_loss, test_loss, label) in enumerate(
        zip(train_losses, test_losses, labels)
    ):
        # plt.plot(train_loss, label=f"{label} Train Loss", color=colors[i])
        epochs = np.linspace(0, len(train_loss) * 10, num=len(train_loss))
        plt.plot(epochs, train_loss, label=f"{label}", color=colors[i])
        plt.plot(
            epochs,
            test_loss,
            # label=f"{label} Test Loss",
            linestyle="--",
            color=colors[i],
        )
        num_epochs = int(np.max(epochs))
        max_epochs = max(max_epochs, num_epochs)

        # To determine the y range, exclude the first 2% of the values
        start_idx = int(len(test_loss) * 0.02)
        min_val = min(
            min_val, np.min(test_loss[start_idx:]), np.min(train_loss[start_idx:])
        )
        max_val = max(
            max_val, np.max(test_loss[start_idx:]), np.max(train_loss[start_idx:])
        )

    # Add additional legend entries for the train and test losses
    plt.plot([], linestyle="-", color="black", label="Train Loss")
    plt.plot([], linestyle="--", color="black", label="Test Loss")

    plt.xlabel("Epoch")
    plt.xlim(0, max_epochs)
    plt.ylabel("Log-MAE")
    plt.yscale("log")
    plt.ylim(min_val, max_val)
    if show_title:
        plt.title("Comparison of Training and Test Losses")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "losses_main_model.png", config)

    plt.close()


def plot_loss_comparison_equal(
    train_losses: tuple[np.ndarray, ...],
    test_losses: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot the test loss trajectories for different surrogate models on a single plot,
    after log-transforming and normalizing each trajectory. This makes it easier
    to see convergence behavior even when the losses span several orders of magnitude.
    Numeric y-axis labels are removed.

    Each loss trajectory is processed as follows:
    1. Log-transform the loss values.
    2. Normalize the log-transformed values to the range [0, 1].
    3. Plot the normalized trajectory on a normalized x-axis.

    Args:
        train_losses (tuple): Tuple of training loss arrays for each surrogate model.
        test_losses (tuple): Tuple of test loss arrays for each surrogate model.
        labels (tuple): Tuple of labels for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.
        show_title (bool): Whether to show the title on the plot.

    Returns:
        None
    """
    plt.figure(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(test_losses)))

    for i, (train_loss, test_loss, label) in enumerate(
        zip(train_losses, test_losses, labels)
    ):
        # Create a normalized x-axis for the current loss trajectory.
        x_axis = np.linspace(0, 1, len(test_loss))

        # Log-transform the test loss. Ensure no zero or negative values.
        # If zeros exist, add a small constant to avoid -inf.
        safe_test_loss = np.where(test_loss <= 0, 1e-12, test_loss)
        log_loss = np.log(safe_test_loss)

        # Normalize the log-transformed loss to span the full y-axis (0 to 1)
        # Use a small offset to move plots away from the x-axis
        loss_min = log_loss.min()
        loss_max = log_loss[0]
        if loss_max - loss_min > 0:
            norm_loss = (log_loss - loss_min + 0.05) / (loss_max - loss_min)
        else:
            norm_loss = np.zeros_like(log_loss)

        plt.plot(
            x_axis,
            norm_loss,
            label=label,
            linestyle="--",
            color=colors[i],
        )

    plt.xlabel("Normalized Training Duration")
    plt.xlim(0, 1)
    plt.ylabel("Normalized Log-MAE")
    plt.ylim(0, 1)
    # plt.yticks([])  # Remove numeric y-axis labels
    if show_title:
        plt.title("Comparison of Normalized Test Loss Trajectories (Log-transformed)")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "losses_main_model_equal.png", config)

    plt.close()


def plot_MAE_comparison(
    MAEs: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot the MAE for different surrogate models.

    Args:
        MAE (tuple): Tuple of accuracy arrays for each surrogate model.
        labels (tuple): Tuple of labels for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.
        show_title (bool): Whether to show the title on the plot.
    """
    plt.figure(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(MAEs)))

    for i, (accuracy, label) in enumerate(zip(MAEs, labels)):
        # Modify the label to include the final MAE
        final_MAE = accuracy[-1]
        label = f"{label} (final MAE: {final_MAE:.2e})"
        plt.plot(accuracy, label=label, color=colors[i])

    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.yscale("log")
    if show_title:
        plt.title("Comparison of Model Mean Absolute Errors")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "MAE_main_model.png", config)

    plt.close()


def plot_loss_comparison_train_duration(
    test_losses: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    train_durations: tuple[float, ...],
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot the test loss trajectories for different surrogate models over training duration.

    Args:
        test_losses (tuple): Tuple of test loss arrays for each surrogate model.
        labels (tuple): Tuple of labels for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.#
        show_title (bool): Whether to show the title on the plot.
    """
    plt.figure(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(test_losses)))
    min_val, max_val = np.inf, 0

    for i, (test_loss, label, train_duration) in enumerate(
        zip(test_losses, labels, train_durations)
    ):
        epoch_times = np.linspace(0, train_duration, len(test_loss))
        plt.plot(epoch_times, test_loss, label=label, color=colors[i])

        # To determine the y range, exclude the first 2% of the values
        start_idx = int(len(test_loss) * 0.02)
        min_val = min(min_val, np.min(test_loss[start_idx:]))
        max_val = max(max_val, np.max(test_loss[start_idx:]))

    plt.xlabel("Time (s)")
    plt.xlim(0, max(train_durations))
    plt.ylabel("Log-MAE")
    plt.yscale("log")
    plt.ylim(min_val, max_val)
    if show_title:
        plt.title("Comparison of Model Test Losses Over Training Duration")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "losses_main_model_duration.png", config)

    plt.close()


def plot_relative_errors(
    mean_errors: dict[str, np.ndarray],
    median_errors: dict[str, np.ndarray],
    timesteps: np.ndarray,
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot the relative errors over time for different surrogate models.

    Args:
        mean_errors (dict): Dictionary containing the mean relative errors for each surrogate model.
        median_errors (dict): Dictionary containing the median relative errors for each surrogate model.
        timesteps (np.ndarray): Array of timesteps.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.
        show_title (bool): Whether to show the title on the plot.

    Returns:
        None
    """
    plt.figure(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(mean_errors)))
    linestyles = ["-", "--"]

    for i, surrogate in enumerate(mean_errors.keys()):
        mean = np.mean(mean_errors[surrogate])
        mean_label = f"{surrogate}\nMean = {mean * 100:.2f}%"
        plt.plot(
            timesteps,
            mean_errors[surrogate],
            label=mean_label,
            color=colors[i],
            linestyle=linestyles[0],
        )
        median = np.mean(median_errors[surrogate])
        median_label = f"{surrogate}\nMedian = {median * 100:.2f}%"
        plt.plot(
            timesteps,
            median_errors[surrogate],
            label=median_label,
            color=colors[i],
            linestyle=linestyles[1],
        )

    plt.xlabel("Time")
    plt.xlim(timesteps[0], timesteps[-1])
    plt.ylabel("Relative Error")
    plt.yscale("log")
    if show_title:
        plt.title("Comparison of Relative Errors Over Time")
    if config.get("dataset", {}).get("log_timesteps", False):
        plt.xscale("log")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save and config:
        save_plot(plt, "accuracy_rel_errors_time_models.png", config)

    plt.close()


def plot_errors_over_time(
    mean_errors: dict[str, np.ndarray],
    median_errors: dict[str, np.ndarray],
    timesteps: np.ndarray,
    config: dict,
    save: bool = True,
    show_title: bool = True,
    mode: str = "relative",  # "relative", "deltadex", or "iterative"
    iter_interval: int | None = None,
) -> None:
    """
    Plot error trajectories for each surrogate model.

    Args:
        mean_errors (dict[str, np.ndarray]): Mean error curves keyed by surrogate name.
        median_errors (dict[str, np.ndarray]): Median error curves keyed by surrogate name.
        timesteps (np.ndarray): Timeline in the same shape as the error curves.
        config (dict): Benchmark configuration used for saving.
        save (bool): Whether to write the figure to disk.
        show_title (bool): Whether to display the figure title.
        mode (str): ``\"relative\"`` for percentage errors, ``\"deltadex\"`` for log-space errors,
            or ``\"iterative\"`` for chained Δdex errors with guide lines every ``iter_interval`` steps.
        iter_interval (int | None): Interval for the dashed guide lines when ``mode == "iterative"``.
    """
    plt.figure(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(mean_errors)))
    linestyles = ["-", "--"]

    # Support both dict inputs and array-like median_errors
    for i, surrogate in enumerate(mean_errors.keys()):
        mean_series = mean_errors[surrogate]
        median_series = median_errors[surrogate]

        mean_val = float(np.mean(mean_series))
        median_val = float(np.mean(median_series))

        if mode == "relative":
            mean_label = f"{surrogate}\nMean = {mean_val * 100:.2f}%"
            median_label = f"{surrogate}\nMedian = {median_val * 100:.2f}%"
        else:  # deltadex or iterative
            mean_label = f"{surrogate}\nMean = {mean_val:.3f} dex"
            median_label = f"{surrogate}\nMedian = {median_val:.3f} dex"

        plt.plot(
            timesteps,
            mean_series,
            label=mean_label,
            color=colors[i],
            linestyle=linestyles[0],
        )
        plt.plot(
            timesteps,
            median_series,
            label=median_label,
            color=colors[i],
            linestyle=linestyles[1],
        )

    plt.xlabel("Time")
    plt.xlim(timesteps[0], timesteps[-1])
    if mode == "relative":
        plt.ylabel("Relative Error")
        plt.yscale("log")
        fname = "accuracy_rel_errors_time_models.png"
        title = "Comparison of Relative Errors Over Time"
    elif mode == "deltadex":
        plt.ylabel(r"Log-MAE ($\Delta dex$)")
        fname = "accuracy_delta_dex_time.png"
        title = "Comparison of Δdex Errors Over Time"
    elif mode == "iterative":
        # Single backslash inside raw string to render the LaTeX Delta properly
        plt.ylabel(r"Log-MAE ($\Delta dex$)")
        plt.ylim(bottom=0, top=min(np.max(list(mean_errors.values())) * 1.1, 5))
        fname = "iterative_delta_dex_time.png"
        title = "Comparison of Δdex Errors Over Time for Iterative Predictions"
        # Add subtle dashed vertical lines at every n-th timestep if provided and valid
        if isinstance(iter_interval, int) and iter_interval > 0:
            # start at iter_interval to avoid drawing a line at the very first x-limit
            for idx in range(iter_interval, len(timesteps), iter_interval):
                x = timesteps[idx]
                plt.axvline(x=x, linestyle="--", color="gray", alpha=0.3, linewidth=0.8)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if config.get("dataset", {}).get("log_timesteps", False):
        plt.xscale("log")
    if show_title:
        plt.title(title)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save and config:
        save_plot(plt, fname, config)

    plt.close()


def plot_uncertainty_confidence(
    weighted_diffs: dict[str, np.ndarray],
    conf: dict,
    save: bool = True,
    percentile: float = 2,
    summary_stat: str = "mean",  # currently only "mean" is implemented
    show_title: bool = True,
) -> dict[str, float]:
    """
    Plot overconfidence/underconfidence statistics for each surrogate.

    Each ``weighted_diff`` array stores ``(predicted_uncertainty - abs(prediction - target)) / target``.
    Negative values indicate overconfidence (uncertainty too small), whereas positive values indicate
    underconfidence. The function:

    1. Selects catastrophic events in the lower ``percentile`` tail (overconfidence) and upper tail
       (underconfidence).
    2. Computes the mean and standard deviation of each tail.
    3. Draws grouped bars with error caps so the tails are easy to compare visually.

    Args:
        weighted_diffs (dict[str, np.ndarray]): Weighted-difference arrays keyed by surrogate name.
        conf (dict): Configuration dictionary used for saving.
        save (bool): Whether to save the figure.
        percentile (float): Tail percentile for defining catastrophic events.
        summary_stat (str): Currently unused hook for other aggregations (defaults to ``\"mean\"``).
        show_title (bool): Whether to draw the title.

    Returns:
        dict[str, float]: Net overconfidence/underconfidence score for each surrogate.
    """
    surrogate_names = list(weighted_diffs.keys())
    num_surrogates = len(surrogate_names)

    # Prepare lists to store summary statistics and their standard deviations.
    overconf_summary = []
    underconf_summary = []
    overconf_std = []
    underconf_std = []
    net_diff = {}  # Will store net difference for each surrogate

    # We'll assume all surrogates have the same number of predictions.
    # Use the first surrogate to compute the number of samples in each tail.
    first_surrogate = surrogate_names[0]
    total_samples = weighted_diffs[first_surrogate].size
    samples_per_tail = int(percentile / 100 * total_samples)

    for surrogate in surrogate_names:
        wd = weighted_diffs[surrogate].flatten()

        # Overconfidence: samples with weighted_diff below the lower threshold.
        lower_threshold = np.percentile(wd, percentile)
        over_mask = wd <= lower_threshold
        wd_over = wd[over_mask]
        # print(
        #     "Low tail: min=",
        #     wd_over.min(),
        #     "max=",
        #     wd_over.max(),
        #     "mean=",
        #     wd_over.mean(),
        #     "std=",
        #     wd_over.std(),
        # )

        # Underconfidence: samples with weighted_diff above the upper threshold.
        upper_threshold = np.percentile(wd, 100 - percentile)
        under_mask = wd >= upper_threshold
        wd_under = wd[under_mask]
        # print(
        #     "High tail: min=",
        #     wd_under.min(),
        #     "max=",
        #     wd_under.max(),
        #     "mean=",
        #     wd_under.mean(),
        #     "std=",
        #     wd_under.std(),
        # )

        # Compute summary statistic and standard deviation; using mean.
        over_mean = np.mean(wd_over) if wd_over.size > 0 else np.nan
        under_mean = np.mean(wd_under) if wd_under.size > 0 else np.nan
        over_std_val = np.std(wd_over) if wd_over.size > 0 else np.nan
        under_std_val = np.std(wd_under) if wd_under.size > 0 else np.nan

        # Express in percentage.
        over_pct = 100 * over_mean
        under_pct = 100 * under_mean
        over_std_pct = 100 * over_std_val
        under_std_pct = 100 * under_std_val

        overconf_summary.append(over_pct)
        underconf_summary.append(under_pct)
        overconf_std.append(over_std_pct)
        underconf_std.append(under_std_pct)

        # Net difference: simply add the two.
        net_diff[surrogate] = over_pct + under_pct

    # Create grouped bar chart.
    x = np.arange(num_surrogates)
    width = 0.35  # width of each bar

    fig, ax = plt.subplots(figsize=(7, 4))

    # Bars for overconfidence (expected to be negative), with error bars.
    bars_over = ax.bar(
        x - width / 2,
        overconf_summary,
        width,
        color="salmon",
        yerr=overconf_std,
        capsize=3,
        error_kw=dict(lw=0.5),
        label=f"Overconfidence ({percentile}st perc. mean)",
    )
    # Bars for underconfidence (expected to be positive), with error bars.
    bars_under = ax.bar(
        x + width / 2,
        underconf_summary,
        width,
        color="lightblue",
        yerr=underconf_std,
        capsize=3,
        error_kw=dict(lw=0.5),
        label=f"Underconfidence ({100 - percentile}th perc. mean)",
    )

    ax.set_ylabel(
        "Relative Difference Summary (%)\n((PU - |Error|) / Target)", fontsize=12
    )
    if show_title:
        title_str = (
            f"Over- and Underconfidence Summary (Metric: Mean ± Std Dev)\n"
            f"Samples per tail: {samples_per_tail}"
        )
        ax.set_title(title_str, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(surrogate_names)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.legend()

    # Use a fixed offset (in points) from the x-axis.
    # We'll use ax.get_xaxis_transform() so that y=0 corresponds to the x-axis.
    for bar in bars_over:
        x_center = bar.get_x() + bar.get_width() / 2
        # For overconfidence bars (negative), place label 3 points above x-axis.
        ax.annotate(
            f"{bar.get_height():.1f}%",
            xy=(x_center, 0),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )
    for bar in bars_under:
        x_center = bar.get_x() + bar.get_width() / 2
        # For underconfidence bars (positive), place label 3 points below x-axis.
        ax.annotate(
            f"{bar.get_height():.1f}%",
            xy=(x_center, 0),
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )

    # Check if the overall range of summary values spans more than two orders of magnitude.
    all_values = [abs(v) for v in overconf_summary + underconf_summary if v != 0]
    if all_values:
        ratio = max(all_values) / min(all_values)
        if ratio > 100:  # more than two orders of magnitude
            ax.set_yscale("symlog", linthresh=1)

    plt.tight_layout()

    if save and conf:
        fname = "uncertainty_confidence.png"
        save_plot(plt, fname, conf)

    plt.close()

    return net_diff


def plot_mean_deltadex_over_time_main_vs_ensemble(
    main_errors: dict[str, np.ndarray],
    ensemble_errors: dict[str, np.ndarray],
    timesteps: np.ndarray,
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot mean Δdex over time for each surrogate: main vs ensemble.

    Args:
        main_errors (dict): Main model Δdex arrays [N, T, Q] per surrogate.
        ensemble_errors (dict): Ensemble Δdex arrays [N, T, Q] per surrogate.
        timesteps (np.ndarray): Timesteps array.
        config (dict): Configuration dictionary.
        save (bool): Whether to save figure.
        show_title (bool): Whether to add a title.
    """
    plt.figure(figsize=(7, 4))
    names = list(ensemble_errors.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(names)))

    for i, name in enumerate(names):
        main_ts = np.mean(main_errors[name], axis=(0, 2))
        ens_ts = np.mean(ensemble_errors[name], axis=(0, 2))
        main_avg = main_ts.mean()
        ens_avg = ens_ts.mean()

        plt.plot(
            timesteps,
            ens_ts,
            color=colors[i],
            linestyle="-",
            label=f"{name} (Ensemble)\nMean={ens_avg:.3f} dex",
        )
        plt.plot(
            timesteps,
            main_ts,
            color=colors[i],
            linestyle="--",
            label=f"{name} (Main)\nMean={main_avg:.3f} dex",
        )

    plt.xlabel("Time")
    plt.ylabel(r"Log-MAE ($\Delta dex$)")
    plt.xlim(timesteps[0], timesteps[-1])
    if config.get("dataset", {}).get("log_timesteps", False):
        plt.xscale("log")
    if show_title:
        plt.title("Mean Δdex Over Time: Main vs Ensemble")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save and config:
        save_plot(plt, "uq_deltadex_main_vs_ensemble.png", config)
    plt.close()


def plot_uncertainty_over_time_comparison(
    uncertainties: dict[str, np.ndarray],
    absolute_errors: dict[str, np.ndarray],
    timesteps: np.ndarray,
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot log-space uncertainty and Δdex over time for multiple surrogates.

    Args:
        uncertainties (dict): Mean log-space std over time per surrogate (1σ time series).
        absolute_errors (dict): Δdex arrays [N, T, Q] per surrogate.
        timesteps (np.ndarray): Timesteps array.
        config (dict): Configuration dictionary.
        save (bool): Save figure.
        show_title (bool): Add title.
    """
    plt.figure(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(uncertainties)))

    for i, name in enumerate(uncertainties.keys()):
        uq_ts = uncertainties[name]
        err_ts = np.mean(absolute_errors[name], axis=(0, 2))
        uq_mean = uq_ts.mean()
        err_mean = err_ts.mean()

        plt.plot(
            timesteps,
            uq_ts,
            color=colors[i],
            linestyle="-",
            label=f"{name} Unc (1σ)\nMean={uq_mean:.3f}",
        )
        plt.plot(
            timesteps,
            err_ts,
            color=colors[i],
            linestyle="--",
            label=f"{name} Δdex\nMean={err_mean:.3f}",
        )

    plt.xlabel("Time")
    plt.ylabel("Log-Space Uncertainty / Log-Space MAE")
    plt.xlim(timesteps[0], timesteps[-1])
    if config.get("dataset", {}).get("log_timesteps", False):
        plt.xscale("log")
    if show_title:
        plt.title("Comparison of Log-Space Uncertainty and Log-MAE Over Time")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save and config:
        save_plot(plt, "uq_deltadex_vs_uncertainty.png", config)
    plt.close()


def inference_time_bar_plot(
    surrogates: list[str],
    means: list[float],
    stds: list[float],
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot the mean inference time with standard deviation for different surrogate models.

    Args:
        surrogates (List[str]): List of surrogate model names.
        means (List[float]): List of mean inference times for each surrogate model.
        stds (List[float]): List of standard deviation of inference times for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool, optional): Whether to save the plot. Defaults to True.
        show_title (bool): Whether to show the title on the plot.

    Returns:
        None
    """
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(surrogates)))
    ax.bar(
        surrogates, means, yerr=stds, capsize=5, alpha=0.7, color=colors, ecolor="black"
    )

    # Calculate the upper y-limit to provide space for text
    max_bar = max(means[i] + stds[i] for i in range(len(means)))
    # min_bar = min(means[i] - stds[i] for i in range(len(means)))
    ax.set_ylim(0, max_bar * 1.2)  # Set limits with some padding

    # Add inference time as text to the bars using the format_time function
    for i, (mean, std) in enumerate(zip(means, stds)):
        formatted_time = format_time(mean, std)
        ax.text(
            i,
            mean + std * 1.2,  # Position the text above the bar
            formatted_time,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Surrogate Model")
    ax.set_ylabel("Mean Inference Time per Run")
    if show_title:
        ax.set_title("Surrogate Mean Inference Time Comparison")

    if save:
        save_plot(plt, "timing_inference.png", config)

    plt.close()


def plot_generalization_error_comparison(
    surrogates: list[str],
    metrics_list: list[np.array],
    model_errors_list: list[np.array],
    xlabel: str,
    filename: str,
    config: dict,
    save: bool = True,
    xlog: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot the generalization errors of different surrogate models.

    Args:
        surrogates (list): List of surrogate model names.
        metrics_list (list[np.array]): List of numpy arrays containing the metrics for each surrogate model.
        model_errors_list (list[np.array]): List of numpy arrays containing the errors for each surrogate model.
        xlabel (str): Label for the x-axis.
        filename (str): Filename to save the plot.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.
        xlog (bool): Whether to use a log scale for the x-axis.
        show_title (bool): Whether to show the title on the plot.

    Returns:
        None
    """
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(surrogates)))

    plt.figure(figsize=(6, 4))
    for i, surrogate in enumerate(surrogates):
        plt.scatter(
            metrics_list[i], model_errors_list[i], label=surrogate, color=colors[i]
        )
        # Add thin lines of the same color to show the trend

        plt.plot(
            metrics_list[i],
            model_errors_list[i],
            color=colors[i],
            alpha=0.5,
            linestyle="-",
        )

    plt.xlabel(xlabel)
    if xlog:
        plt.xscale("log")
    plt.ylabel(r"Log-MAE($\Delta dex$)")
    if show_title:
        plt.title(f"Comparison of {xlabel} Errors " + r"($\Delta dex$)")
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, axis="y")
    plt.legend()

    if save:
        save_plot(plt, filename, config)

    plt.close()


def plot_uncertainty_heatmap(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    average_correlation: float,
    save: bool = True,
    cutoff_mass: float = 0.98,
    show_title: bool = True,
) -> tuple[float, float]:
    """
    Plot correlation between predictive log-space uncertainty and log-space errors (delta dex).

    Args:
        surr_name (str): Name of the surrogate model.
        conf (dict): Configuration dictionary.
        preds_std (np.ndarray): Log-space ensemble standard deviation.
        errors (np.ndarray): Log-space prediction errors.
        average_correlation (float): Correlation between log uncertainty and log error.
        save (bool): Whether to save the figure.
        cutoff_mass (float): Fraction of mass to keep in histogram.
        show_title (bool): Whether to show a title.

    Returns:
        tuple: (max histogram count, axis_max used for plotting).
    """
    plt.figure(figsize=(8, 4))

    heatmap_init, _, _ = np.histogram2d(preds_std.flatten(), errors.flatten(), bins=100)
    all_data = np.concatenate([preds_std.flatten(), errors.flatten()])
    axis_min = np.min(all_data)
    axis_max = np.percentile(all_data, cutoff_mass * 100)

    heatmap, _, _ = np.histogram2d(
        preds_std.flatten(),
        errors.flatten(),
        bins=100,
        range=[[axis_min, axis_max], [axis_min, axis_max]],
    )

    max_value = heatmap.max()
    heatmap[heatmap == 0] = 1

    plt.imshow(
        np.log10(heatmap.T),
        origin="lower",
        aspect="auto",
        extent=[axis_min, axis_max, axis_min, axis_max],
        cmap="inferno",
    )
    plt.colorbar(label=r"$\log_{10}$(Counts)")
    plt.xlabel("Log-Space Uncertainty")
    plt.ylabel(r"$\Delta dex$")

    if show_title:
        plt.title(
            r"Correlation Between Log-Space Uncertainty and Errors ($\Delta dex$)"
            + f"\nAverage Correlation: {average_correlation:.2f}"
        )

    plt.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color="white",
        linestyle="--",
        linewidth=1,
    )

    if save and conf:
        save_plot(plt, "uq_uncertainty_errors_correlation.png", conf, surr_name)
    plt.close()
    return max_value, axis_max


def plot_gradients_heatmap(
    surr_name: str,
    conf: dict,
    gradients: np.ndarray,
    errors_log: np.ndarray,
    average_correlation: float,
    save: bool = False,
    cutoff_mass: float = 0.98,
    show_title: bool = True,
) -> tuple[float, float, float]:
    """
    Plot correlation between gradients (normalized) and Δdex errors using a heatmap.

    Both gradients and errors are in log-space.
    Gradients are normalized, errors are absolute log differences (Δdex).

    Args:
        surr_name (str): Surrogate name.
        conf (dict): Config dictionary.
        gradients (np.ndarray): Normalized log-space gradients.
        errors_log (np.ndarray): Δdex errors.
        average_correlation (float): Mean correlation value.
        save (bool): Save plot.
        cutoff_mass (float): Fraction of mass to retain in axes.
        show_title (bool): Show title.

    Returns:
        (max_value, x_max, y_max): Histogram stats for reuse.
    """
    plt.figure(figsize=(8, 4))

    marginal_cutoff = np.sqrt(cutoff_mass)

    x_min = 0  # gradients normalized
    x_max = np.percentile(gradients.flatten(), 100 * marginal_cutoff)

    y_min = np.min(errors_log.flatten())
    y_max = np.percentile(errors_log.flatten(), 100 * marginal_cutoff)

    heatmap, _, _ = np.histogram2d(
        gradients.flatten(),
        errors_log.flatten(),
        bins=100,
        range=[[x_min, x_max], [y_min, y_max]],
    )

    max_value = heatmap.max()
    heatmap[heatmap == 0] = 1

    plt.imshow(
        np.log10(heatmap.T),
        origin="lower",
        aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
        cmap="inferno",
    )
    plt.colorbar(label=r"$\log_{10}$(Counts)")
    plt.xlabel("Absolute Gradient (Normalized)")
    plt.ylabel(r"Log-MAE ($\Delta dex$)")
    if show_title:
        plt.title(
            f"Correlation between Gradients and Δdex Errors\n"
            f"Average Correlation: {average_correlation:.2f}"
        )

    if save and conf:
        save_plot(plt, "gradient_error_heatmap.png", conf, surr_name)

    plt.close()
    return max_value, x_max, y_max


def plot_error_distribution_comparative(
    errors: dict[str, np.ndarray],
    conf: dict,
    save: bool = True,
    show_title: bool = True,
    mode: str = "relative",  # "relative", "deltadex", or "iterative"
) -> None:
    """
    Plot comparative error distributions for each surrogate model.

    Args:
        errors (dict): Model → array of errors [num_samples, num_timesteps, num_quantities].
        conf (dict): Configuration dictionary.
        save (bool): Whether to save the figure.
        show_title (bool): Whether to add a title.
        mode (str): "relative" (unitless %), "deltadex" (log-space abs. errors), or
            "iterative" (same as deltadex plotting, different title/filename for iterative context).
    """
    model_names = list(errors.keys())
    num_models = len(model_names)

    log_errors = []
    mean_errors = []
    median_errors = []

    for model_name in model_names:
        model_errors = errors[model_name].flatten()
        mean_errors.append(np.mean(model_errors))
        median_errors.append(np.median(model_errors))

        # transform for histogram plotting
        non_zero = model_errors[model_errors > 0]
        log_errors.append(np.log10(non_zero))

    min_percentiles = [np.percentile(err, 2) for err in log_errors if len(err) > 0]
    max_percentiles = [np.percentile(err, 98) for err in log_errors if len(err) > 0]
    global_min, global_max = np.min(min_percentiles), np.max(max_percentiles)

    x_min, x_max = np.floor(global_min), np.ceil(global_max)

    plt.figure(figsize=(8, 3))
    colors = plt.cm.viridis(np.linspace(0, 0.95, num_models))
    x_vals = np.linspace(x_min, x_max + 0.1, 100)

    for i, model_name in enumerate(model_names):
        hist, bin_edges = np.histogram(log_errors[i], bins=x_vals, density=True)
        smoothed = gaussian_filter1d(hist, sigma=2)

        plt.plot(
            10 ** bin_edges[:-1],
            smoothed,
            label=model_name,
            color=colors[i],
        )
        # add mean/median markers
        plt.axvline(
            x=mean_errors[i],
            color=colors[i],
            linestyle="--",
            label=f"{model_name} mean = {mean_errors[i]:.3g}",
        )
        plt.axvline(
            x=median_errors[i],
            color=colors[i],
            linestyle="-.",
            label=f"{model_name} median = {median_errors[i]:.3g}",
        )

    plt.xscale("log")
    plt.xlim(np.maximum(10**x_min, 1e-8), 10**x_max)

    if mode == "relative":
        xlabel = "Relative Error Magnitude"
        title = "Distribution of Surrogate Relative Errors"
        fname = "accuracy_error_dist_relative.png"
    elif mode == "deltadex":
        xlabel = r"Log-MAE ($\Delta dex$)"
        title = "Distribution of Surrogate Δdex Errors"
        fname = "accuracy_error_dist_deltadex.png"
    elif mode == "iterative":
        # Plot identical to deltadex but labeled for iterative evaluation context
        xlabel = r"Log-MAE ($\Delta dex$)"
        title = "Distribution of Surrogate Relative Errors for Iterative Prediction"
        fname = "iterative_error_dist_deltadex.png"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    plt.xlabel(xlabel)
    plt.ylabel("Smoothed Histogram Count")
    if show_title:
        plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save and conf:
        save_plot(plt, fname, conf)

    plt.close()


def plot_comparative_error_correlation_heatmaps(
    preds_std: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    avg_correlations: dict[str, float],
    axis_max: dict[str, float],
    max_count: dict[str, float],
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Comparative heatmaps of log-space uncertainty vs Δdex.

    Args:
        preds_std (dict): Log-space std arrays per surrogate.
        errors (dict): Δdex arrays per surrogate.
        avg_correlations (dict): Pearson r per surrogate (log-space).
        axis_max (dict): Axis maxima from per-surrogate plots.
        max_count (dict): Peak counts for normalization per surrogate.
        config (dict): Configuration dictionary.
        save (bool): Save figure.
        show_title (bool): Add title.
    """
    names = list(preds_std.keys())
    n = len(names)
    global_axis_max = max(axis_max.values())
    global_max_count = max(max_count.values())

    fig, axes = plt.subplots(n, 1, figsize=(8, 4.5 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, name in enumerate(names):
        ax = axes[i]
        u = preds_std[name].flatten()
        e = errors[name].flatten()

        H, _, _ = np.histogram2d(
            u, e, bins=100, range=[[0, global_axis_max], [0, global_axis_max]]
        )
        H[H == 0] = 1

        im = ax.imshow(
            np.log10(H.T),
            origin="lower",
            aspect="auto",
            extent=[0, global_axis_max, 0, global_axis_max],
            cmap="inferno",
            vmin=0,
            vmax=np.log10(global_max_count),
        )
        ax.plot(
            [0, global_axis_max],
            [0, global_axis_max],
            color="white",
            linestyle="--",
            linewidth=1,
        )
        ax.set_ylabel(r"$\Delta dex$")
        ax.set_title(f"{name}\nAvg Correlation: {avg_correlations[name]:.2f}")

    axes[-1].set_xlabel("Log-Space Uncertainty")

    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.08, right=0.9, hspace=0.2)
    fig.colorbar(im, cax=cbar_ax, label=r"$\log_{10}$(Counts)")

    if show_title:
        plt.suptitle("Uncertainty vs Δdex: Comparative Heatmaps", y=0.98)

    if save and config:
        save_plot(fig, "uq_heatmaps.png", config)
    plt.close()


def plot_catastrophic_detection_curves(
    errors_log: dict[str, np.ndarray],
    std_log: dict[str, np.ndarray],
    conf: dict,
    percentiles: tuple[float, ...] = (99.0, 90.0),
    flag_fractions: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.20, 0.30),
    save: bool = True,
    show_title: bool = True,
) -> dict[str, dict[float, dict[str, float]]]:
    """
    Plot recall vs. flagged fraction curves for catastrophic errors.

    For each surrogate and percentile threshold ``p`` we:

    - Treat samples with ``Δdex >= percentile(p)`` as catastrophic.
    - Sweep the requested ``flag_fractions`` to determine which proportion of points would be
      deferred based on predictive uncertainty.
    - Report both the catastrophic recall and the hypothetical MAE if flagged samples were replaced
      by exact solver outputs.

    Args:
        errors_log (dict[str, np.ndarray]): Log-space errors per surrogate.
        std_log (dict[str, np.ndarray]): Log-space predictive standard deviations.
        conf (dict): Configuration dictionary.
        percentiles (tuple[float, ...]): Catastrophic thresholds to evaluate.
        flag_fractions (tuple[float, ...]): Fractions of samples to flag at each step.
        save (bool): Whether to save the figure.
        show_title (bool): Whether to draw the title.

    Returns:
        dict[str, dict[float, dict[str, float]]]: Recall/threshold summary per surrogate.
    """
    n_rows = len(percentiles) + 1  # +1 for MAE improvement curves
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 4 * n_rows), sharex=True)

    if n_rows == 1:
        axes = [axes]

    names = list(errors_log.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(names)))
    summary: dict[str, dict[float, dict[str, float]]] = {}

    # --- Recall vs fraction flagged (per catastrophic percentile) ---
    for ax, perc in zip(axes[:-1], percentiles):
        for i, name in enumerate(names):
            e = errors_log[name].flatten()
            u = std_log[name].flatten()
            if e.size == 0 or u.size == 0:
                continue

            # catastrophic cutoff
            cat_thr = np.percentile(e, perc)
            is_cat = e >= cat_thr
            n_cat = int(is_cat.sum())
            if n_cat == 0:
                continue

            xs, ys = [], []
            for f in flag_fractions:
                if f <= 0.0:
                    xs.append(0.0)
                    ys.append(0.0)
                    recall = 0.0
                else:
                    unc_thr = np.percentile(u, 100.0 * (1.0 - float(f)))
                    flagged = u >= unc_thr
                    recall = (flagged & is_cat).sum() / n_cat if n_cat > 0 else 0.0
                    xs.append(100.0 * flagged.mean())
                    ys.append(100.0 * recall)

            ax.plot(
                xs,
                ys,
                marker="o",
                color=colors[i],
                label=f"{name} (thr ≈ {cat_thr:.3f} dex)",
            )

            # Store last-point summary
            summary.setdefault(name, {})[perc] = {
                "flag_fraction": xs[-1] / 100.0,
                "recall": ys[-1] / 100.0,
                "cat_threshold": cat_thr,
            }

        ax.set_ylabel("Catastrophic error recall (%)")
        ax.set_xlim(0, 100 * max(flag_fractions) * 1.05)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        if show_title:
            ax.set_title(
                f"Detection @ {perc}th percentile (Top {100 - perc:.0f}% Δdex)"
            )

    # MAE improvement plot
    ax_mae = axes[-1]
    for i, name in enumerate(names):
        e = errors_log[name].flatten()
        u = std_log[name].flatten()

        mae_curve = []
        xs, ys = [], []
        for f in flag_fractions:
            unc_thr = np.percentile(u, 100.0 * (1.0 - float(f))) if f > 0 else np.inf
            flagged = u >= unc_thr if f > 0 else np.zeros_like(u, dtype=bool)

            # replace flagged errors with 0 (numerical solver fallback)
            adjusted = e.copy()
            adjusted[flagged] = 0.0
            mae_val = adjusted.mean()

            xs.append(100.0 * flagged.mean())
            ys.append(mae_val)
            mae_curve.append((f, mae_val))

        ax_mae.plot(xs, ys, marker="o", color=colors[i], label=name)

        # store mae curve under percentile-independent key
        for perc in percentiles:
            summary[name][perc]["mae_curve"] = mae_curve

    ax_mae.set_xlabel("Fraction flagged by uncertainty (%)")
    ax_mae.set_ylabel("Log MAE (Δdex)")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend(loc="best")
    ax_mae.set_ylim(0, ax_mae.get_ylim()[1])
    ax_mae.set_title("Log MAE Improvement")

    if show_title:
        fig.suptitle(
            "Catastrophic Error Detection and UQ-based Performance Gain", y=0.99
        )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save and conf:
        save_plot(fig, "uq_catastrophic_detection.png", conf)
    plt.close(fig)

    return summary


def plot_comparative_gradient_heatmaps(
    gradients: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    avg_correlations: dict[str, float],
    max_grad: dict[str, float],
    max_err: dict[str, float],
    max_count: dict[str, float],
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Plot comparative heatmaps of correlation between gradient and prediction errors
    for multiple surrogate models.

    Args:
        gradients (dict[str, np.ndarray]): Dictionary of gradients from the ensemble of models.
        errors (dict[str, np.ndarray]): Dictionary of prediction errors.
        avg_correlations (dict[str, float]): Dictionary of average correlations between gradients and prediction errors.
        max_grad (dict[str, float]): Dictionary of maximum gradient values for axis scaling across models.
        max_err (dict[str, float]): Dictionary of maximum error values for axis scaling across models.
        max_count (dict[str, float]): Dictionary of maximum count values for heatmap normalization across models.
        config (dict): Configuration dictionary.
        save (bool, optional): Whether to save the plot. Defaults to True.
        show_title (bool): Whether to show the title on the plot.

    Returns:
        None
    """
    # Number of models
    model_names = list(gradients.keys())
    num_models = len(model_names)

    # Determine global max for axis scaling
    global_x_max = max(max_grad.values())
    global_y_max = max(max_err.values())

    # Determine global max for heatmap normalization
    global_max_count = max(max_count.values())

    # Create subplots, one row per model
    fig, axes = plt.subplots(
        num_models, 1, figsize=(7, 5 * num_models), sharex=False, sharey=False
    )

    # Adjust layout for shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        grad_model = gradients[model_name].flatten()
        errors_model = errors[model_name].flatten()
        avg_corr = avg_correlations[model_name]

        # Re-bin the data using the global axis limits
        heatmap, xedges, yedges = np.histogram2d(
            grad_model,
            errors_model,
            bins=100,
            range=[[0, global_x_max], [0, global_y_max]],
        )

        # Set all fields with 0 counts to 1 count for log scale plotting
        heatmap[heatmap == 0] = 1

        # Plot heatmap
        im = ax.imshow(
            np.log10(heatmap.T),
            origin="lower",
            aspect="auto",
            extent=[0, global_x_max, 0, global_y_max],
            cmap="inferno",
            vmin=0,
            vmax=np.log10(global_max_count),  # Normalize across all models
        )

        if i == num_models - 1:
            ax.set_xlabel("Absolute Gradient (Normalized)")
        ax.set_ylabel(r"$\Delta dex$ (log-space error)")

        ax.set_title(f"{model_name}\nAvg Correlation: {avg_corr:.2f}")

    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.08, right=0.9, hspace=0.2)

    # Shared colorbar, ensure that it is not too lengthy
    fig.colorbar(im, cax=cbar_ax, label=r"$\log_{10}$(Counts)")

    if show_title:
        plt.suptitle(
            "Comparative Gradient vs. Prediction Error Heatmaps Across Models", y=0.98
        )

    if save:
        save_plot(fig, "gradients_heatmaps.png", config)

    plt.show()
    plt.close()


def get_custom_palette(num_colors):
    """
    Returns a list of colors sampled from a custom color palette.

    Args:
        num_colors (int): The number of colors needed.

    Returns:
        list: A list of RGBA color tuples.
    """
    # Define your custom color palette as a list of hex color codes
    # custom_palette = [
    #     "#1f77b4",  # blue
    #     "#ff7f0e",  # orange
    #     "#2ca02c",  # green
    #     "#d62728",  # red
    #     "#9467bd",  # purple
    #     "#8c564b",  # brown
    #     "#e377c2",  # pink
    #     "#7f7f7f",  # gray
    #     "#bcbd22",  # yellow
    #     "#17becf",  # teal
    # ]

    custom_palette = [
        # Original palette
        # "#10002B",  # Darkest violet
        # "#3A1A5A",  # Dark purple (slightly brighter)
        # "#6A1C75",  # Deep magenta (slightly brighter)
        # "#8A3065",  # Magenta with a hint of red
        # "#B73779",  # Deep reddish-pink
        # "#DA5F4D",  # Dark red-orange
        # "#F48842",  # Bright orange
        # "#FBC200",  # Yellow-orange (slightly more yellow)
        # "#C0CD32",  # Yellow-green (more yellowish)
        # "#D2B48C",  # Tan (brownish)
        # Clean palette
        # "#1B3A4B",  # Deep teal-blue
        # "#005F73",  # Teal
        # "#0A9396",  # Bright cyan-teal
        # "#94D2BD",  # Soft aqua
        # "#E9D891",  # Sand (light neutral contrast)
        # "#EE9B00",  # Amber (warm accent)
        # "#CA6702",  # Burnt orange
        # "#BB3E03",  # Rust red
        # "#9B2226",  # Deep crimson
        # "#5C4D7D",  # Muted indigo (ties back to cool theme)
        # Distinct palette
        "#1B3A4B",  # Deep teal-blue
        "#006D77",  # Ocean teal
        "#83C5BE",  # Soft turquoise
        "#A7C957",  # Fresh green
        "#217E53",  # Jade green
        "#E9D891",  # Sand (light neutral contrast)
        "#E86241",  # Coral
        "#9B2226",  # Deep crimson
        "#7F33B2",  # Vivid purple
        "#9C7E56",  # Tan (brownish)
    ]

    # Create a custom colormap from the palette
    custom_cmap = ListedColormap(custom_palette)

    # Sample colors from your custom colormap
    colors = custom_cmap(np.linspace(0, 1, num_colors))

    return colors


def plot_all_generalization_errors(
    all_metrics: dict,
    config: dict,
    show_title: bool = True,
) -> None:
    """
    Function to make one comparative plot of the interpolation, extrapolation, sparse, and batch size errors.
    Only the modalities present in all_metrics will be plotted.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.
        show_title (bool): Whether to show the title on the plot.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())

    # Prepare modality data: create dictionaries mapping surrogate -> (metrics, errors)
    interp_data = {}
    extrap_data = {}
    sparse_data = {}
    batch_data = {}

    for surrogate in surrogates:
        if "interpolation" in all_metrics[surrogate]:
            interp_data[surrogate] = (
                all_metrics[surrogate]["interpolation"]["intervals"],
                all_metrics[surrogate]["interpolation"]["model_errors"],
            )
        if "extrapolation" in all_metrics[surrogate]:
            extrap_data[surrogate] = (
                all_metrics[surrogate]["extrapolation"]["cutoffs"],
                all_metrics[surrogate]["extrapolation"]["model_errors"],
            )
        if "sparse" in all_metrics[surrogate]:
            sparse_data[surrogate] = (
                all_metrics[surrogate]["sparse"]["n_train_samples"],
                all_metrics[surrogate]["sparse"]["model_errors"],
            )
        if "batch_size" in all_metrics[surrogate]:
            batch_data[surrogate] = (
                all_metrics[surrogate]["batch_size"]["batch_elements"],
                all_metrics[surrogate]["batch_size"]["model_errors"],
            )

    # Determine which modalities are available
    available_modalities = []
    if interp_data:
        available_modalities.append("interpolation")
    if extrap_data:
        available_modalities.append("extrapolation")
    if sparse_data:
        available_modalities.append("sparse")
    if batch_data:
        available_modalities.append("batch_size")

    # Define properties for each modality
    modality_props = {
        "interpolation": {
            "xlabel": "Interpolation Intervals",
            "title": "Interpolation",
            "xscale": None,
        },
        "extrapolation": {
            "xlabel": "Extrapolation Cutoffs",
            "title": "Extrapolation",
            "xscale": None,
        },
        "sparse": {
            "xlabel": "Number of Training Samples (log scale)",
            "title": "Sparse Training",
            "xscale": "log",
        },
        "batch_size": {
            "xlabel": "Elements per Batch",
            "title": "Batch Size",
            "xscale": "log",
        },
    }

    # Create a consistent color mapping for all surrogates
    cmap = plt.cm.viridis(np.linspace(0, 0.95, len(surrogates)))
    color_map = {s: cmap[i] for i, s in enumerate(surrogates)}

    # Create figure and subplots: width scales with number of modalities
    num_modalities = len(available_modalities)
    fig, axes = plt.subplots(
        1, num_modalities, figsize=(4 * num_modalities, 4.5), sharey=True
    )

    # If there's only one subplot, wrap axes in a list for uniform processing
    if num_modalities == 1:
        axes = [axes]

    # Helper: given a modality key, get the corresponding data dictionary.
    def get_modality_data(modality):
        if modality == "interpolation":
            return interp_data
        elif modality == "extrapolation":
            return extrap_data
        elif modality == "sparse":
            return sparse_data
        elif modality == "batch_size":
            return batch_data
        else:
            return {}

    # Loop over modalities and plot each in its corresponding axis.
    for ax, modality in zip(axes, available_modalities):
        mod_data = get_modality_data(modality)
        for surrogate, (x_data, y_data) in mod_data.items():
            # Scatter and line plot for each surrogate in this modality
            ax.scatter(x_data, y_data, label=surrogate, color=color_map[surrogate])
            ax.plot(
                x_data, y_data, color=color_map[surrogate], alpha=0.5, linestyle="-"
            )
        ax.set_xlabel(modality_props[modality]["xlabel"])
        ax.set_title(modality_props[modality]["title"])
        if modality_props[modality]["xscale"]:
            ax.set_xscale(modality_props[modality]["xscale"])
        # ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, axis="y")
        # For the sparse modality, if available, set x-ticks to the first surrogate's data for consistency.
        if (modality == "sparse" or modality == "batch_size") and mod_data:
            # Get all unique modality values from all surrogates
            all_values = np.unique(
                np.concatenate([x_data for x_data, _ in mod_data.values()])
            )
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            if len(all_values) > 5 and modality == "batch_size":
                ax.get_xaxis().set_major_formatter(
                    FuncFormatter(lambda val, pos: f"{val / 1e3:.1f}")
                )
                ax.set_xlabel(r"Elements per Batch $[\times 10^3]$")
                all_values = all_values[np.diff(all_values, prepend=0) > 1000]
            else:
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            ax.set_xticks(all_values)

    # Get current y-axis limits from the first subplot and ensure the upper limit does not exceed 1e3.
    y_min, y_max = axes[0].get_ylim()
    if y_max > 1e3:
        axes[0].set_ylim(y_min, 1e3)

    # Set ylabel on the leftmost subplot
    axes[0].set_ylabel("Mean Absolute Error")

    # Add legend to the first subplot only
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", fontsize="small")

    # Set overall title and adjust layout
    if show_title:
        fig.suptitle("Model Errors for Different Modalities", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    save_plot(fig, "errors_all_modalities.png", config)
    plt.close()


def rel_errors_and_uq(
    metrics: dict[str, dict],
    config: dict,
    save: bool = True,
    show_title: bool = True,
) -> None:
    """
    Create a figure with two subplots: relative errors over time and uncertainty over time for different surrogate models.

    Args:
        metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.
        show_title (bool): Whether to show the title on the plot.

    Returns:
        None
    """
    # Prepare data for relative errors plot
    mean_errors = {}
    median_errors = {}
    uncertainties = {}
    absolute_errors = {}
    timesteps = None

    for surrogate, surrogate_metrics in metrics.items():
        relative_error_model = surrogate_metrics["accuracy"].get("relative_errors")
        if relative_error_model is not None:
            mean_errors[surrogate] = np.mean(relative_error_model, axis=(0, 2))
            median_errors[surrogate] = np.median(relative_error_model, axis=(0, 2))

        uncertainties[surrogate] = surrogate_metrics["UQ"]["pred_uncertainty"]
        absolute_errors[surrogate] = surrogate_metrics["accuracy"]["absolute_errors"]
        timesteps = surrogate_metrics["timesteps"]

    # Prepare data for uncertainty plot
    pred_unc_time = {
        surrogate: np.mean(uncertainty, axis=(0, 2))
        for surrogate, uncertainty in uncertainties.items()
    }

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Relative Errors
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(mean_errors)))
    linestyles = ["-", "--"]

    for i, surrogate in enumerate(mean_errors.keys()):
        mean = np.mean(mean_errors[surrogate])
        mean_label = f"{surrogate} Mean={mean * 100:.2f} %"
        ax1.plot(
            timesteps,
            mean_errors[surrogate],
            label=mean_label,
            color=colors[i],
            linestyle=linestyles[0],
        )
        median = np.mean(median_errors[surrogate])
        median_label = f"{surrogate} Median={median * 100:.2f} %"
        ax1.plot(
            timesteps,
            median_errors[surrogate],
            label=median_label,
            color=colors[i],
            linestyle=linestyles[1],
        )

    ax1.set_xlabel("Time")
    ax1.set_xlim(timesteps[0], timesteps[-1])
    ax1.set_ylabel("Relative Error")
    ax1.set_yscale("log")
    ax1.set_title("Comparison of Relative Errors Over Time")
    ax1.legend(loc="best")

    # Plot 2: Uncertainty over Time
    for i, surrogate in enumerate(pred_unc_time.keys()):
        avg_uncertainty = np.mean(pred_unc_time[surrogate])
        uq_label = f"{surrogate} uncertainty (mean: {avg_uncertainty:.2e})"
        ax2.plot(
            timesteps,
            pred_unc_time[surrogate],
            label=uq_label,
            linestyle="-",
            color=colors[i],
        )

        abs_errors_time = np.mean(absolute_errors[surrogate], axis=(0, 2))
        abs_error_avg = np.mean(abs_errors_time)
        err_label = f"{surrogate} abs. error (mean: {abs_error_avg:.2e})"
        ax2.plot(
            timesteps,
            abs_errors_time,
            label=err_label,
            color=colors[i],
            linestyle="--",
        )

    ax2.set_xlabel("Time")
    ax2.set_xlim(timesteps[0], timesteps[-1])
    ax2.set_ylabel("Uncertainty/Absolute Error")
    if show_title:
        ax2.set_title("Comparison of Predictive Uncertainty Over Time")
    ax2.set_yscale("log")
    ax2.legend(loc="best")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Save plot if required
    if save and config:
        save_plot(fig, "combined_rel_errors_and_uq.png", config)

    plt.close()
