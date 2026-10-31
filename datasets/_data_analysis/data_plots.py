import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(1, "../..")

from tqdm import tqdm

from codes import save_plot


def plot_example_trajectories(
    dataset_name: str,
    data: np.ndarray,
    timesteps: np.ndarray,
    num_quantities: int = 10,
    labels: list[str] | None = None,
    save: bool = False,
    sample_idx: int = 0,
    log: bool = False,
    log_time: bool = False,
    quantities_per_plot: int = 6,
    show_title: bool = True,
) -> None:
    num_quantities = min(data.shape[2], num_quantities)
    data = data[sample_idx, :, :num_quantities]
    labels = (
        labels[:num_quantities]
        if labels is not None
        else [f"Quantity {i + 1}" for i in range(num_quantities)]
    )
    num_plots = math.ceil(num_quantities / quantities_per_plot)
    nrows, ncols = compute_subplot_layout(num_plots)

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), sharex=True)
    axes = np.array([axes]) if (nrows == 1 and ncols == 1) else axes.flatten()
    colors = plt.cm.viridis(np.linspace(0, 0.95, quantities_per_plot))

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start = plot_idx * quantities_per_plot
        end = min((plot_idx + 1) * quantities_per_plot, num_quantities)
        for chem_idx in range(start, end):
            label = (
                labels[chem_idx] if labels is not None else f"Quantity {chem_idx + 1}"
            )
            ax.plot(
                timesteps,
                data[:, chem_idx],
                "-",
                label=label,
                color=colors[chem_idx % quantities_per_plot],
            )
        ax.set_ylabel("log(Abundance)" if log else "Abundance")
        ax.set_xlim(timesteps[0], timesteps[-1])
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            borderaxespad=0.0,
            fontsize="small",
        )
        if log_time:
            ax.set_xscale("log")

    for ax in axes[num_plots:]:
        ax.set_visible(False)

    plt.xlabel("Time")
    if show_title:
        fig.suptitle(f"Example Trajectories for Dataset: {dataset_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        conf = {
            "training_id": dataset_name.lower(),
            "verbose": True,
        }

        save_plot(
            plt,
            "example_trajectories.jpg",
            conf,
            dpi=200,
            base_dir="datasets",
            increase_count=False,
        )
    plt.close()


def plot_example_trajectories_paper(
    dataset_name: str,
    data: np.ndarray,
    timesteps: np.ndarray,
    save: bool = False,
    sample_idx: int = 0,
    labels: list[str] | None = None,
) -> None:
    """
    Plot example trajectories for the dataset with two subplots, one showing 15 quantities and another showing the remaining.

    Args:
        dataset_name (str): The name of the dataset.
        data (np.ndarray): The data to plot.
        timesteps (np.ndarray): Timesteps array.
        save (bool, optional): Whether to save the plot as a file.
        sample_idx (int, optional): Index of the sample to plot.
        labels (list, optional): List of labels for the quantities.
    """
    # Ensure we are plotting the correct sample
    data = data[sample_idx]

    # Define the number of quantities per subplot
    total_quantities = data.shape[1]
    num_quantities_subplots = [
        total_quantities // 2,
        total_quantities - total_quantities // 2,
    ]

    # Ensure the labels list matches the number of quantities
    if labels is not None:
        assert len(labels) == total_quantities, (
            "Labels must match the number of quantities."
        )

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Define color palettes
    colors1 = plt.cm.viridis(
        np.linspace(0, 0.95, num_quantities_subplots[0])
    )  # First 15 quantities
    colors2 = plt.cm.viridis(
        np.linspace(0, 0.95, num_quantities_subplots[1])
    )  # Remaining 14 quantities

    # Plot first set of quantities on ax1
    for chem_idx in range(num_quantities_subplots[0]):
        color = colors1[chem_idx]
        gt = data[:, chem_idx]
        label = labels[chem_idx] if labels is not None else f"Chemical {chem_idx + 1}"
        ax1.plot(timesteps, gt, "-", color=color, label=label)

    # Plot second set of quantities on ax2
    for chem_idx in range(num_quantities_subplots[0], total_quantities):
        color = colors2[chem_idx - num_quantities_subplots[0]]
        gt = data[:, chem_idx]
        label = labels[chem_idx] if labels is not None else f"Chemical {chem_idx + 1}"
        ax2.plot(timesteps, gt, "-", color=color, label=label)

    # Set labels and title
    ax1.set_xlabel("Time")
    ax1.set_ylabel("log(Chemical Abundance)")
    ax2.set_xlabel("Time")

    # Remove individual subplot titles and add a global title
    fig.suptitle(r"Example Trajectories: $\tt{osu2008}$ Dataset", fontsize=16)
    # fig.subplots_adjust(wspace=0.05)  # Adjust space between subplots

    # Place the legend of the left plot between the two subplots
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(
        handles1,
        labels1,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
    )

    # Legend for the second plot can stay outside on the right if needed
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles2,
        labels2,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
    )

    # Add grid to both subplots
    ax1.grid(True)
    ax2.grid(True)

    # Set x-axis limits tightly
    for ax in [ax1, ax2]:
        ax.set_xlim(timesteps.min(), timesteps.max())

    # Remove tick marks but keep labels and gridlines
    ax1.tick_params(axis="both", which="both", length=0)
    ax2.tick_params(axis="both", which="both", length=0)

    # Adjust layout to make room for the legend between the subplots
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect parameter

    # Save the plot if required
    if save:
        # Create a dummy conf dictionary
        conf = {
            "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
            "verbose": True,
        }

        # Call save_plot with the dummy conf
        save_plot(
            plt,
            "example_trajectories_paper.png",
            conf,
            dpi=200,
            base_dir="datasets",
            increase_count=False,
        )

    plt.show()


def plot_example_trajectories_poster(
    dataset_name: str,
    data: np.ndarray,
    timesteps: np.ndarray,
    save: bool = False,
    sample_idx: int = 0,
    labels: list[str] | None = None,
) -> None:
    """
    Plot example trajectories for the dataset with two subplots, one showing 15 quantities and another showing the remaining.

    Args:
        dataset_name (str): The name of the dataset.
        data (np.ndarray): The data to plot.
        timesteps (np.ndarray): Timesteps array.
        save (bool, optional): Whether to save the plot as a file.
        sample_idx (int, optional): Index of the sample to plot.
        labels (list, optional): List of labels for the quantities.
    """
    # Ensure we are plotting the correct sample
    data = data[sample_idx]

    # Define the number of quantities per subplot
    total_quantities = data.shape[1]
    num_quantities_subplots = [
        total_quantities // 2,
        total_quantities - total_quantities // 2,
    ]

    # Ensure the labels list matches the number of quantities
    if labels is not None:
        assert len(labels) == total_quantities, (
            "Labels must match the number of quantities."
        )

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4), sharey=True
    )  # Reduced horizontal width

    # Define color palettes
    colors1 = plt.cm.viridis(
        np.linspace(0, 0.95, num_quantities_subplots[0])
    )  # First 15 quantities
    colors2 = plt.cm.viridis(
        np.linspace(0, 0.95, num_quantities_subplots[1])
    )  # Remaining 14 quantities

    # Plot first set of quantities on ax1
    for chem_idx in range(num_quantities_subplots[0]):
        color = colors1[chem_idx]
        gt = data[:, chem_idx]
        label = labels[chem_idx] if labels is not None else f"Chemical {chem_idx + 1}"
        ax1.plot(timesteps, gt, "-", color=color, label=label)

    # Plot second set of quantities on ax2
    for chem_idx in range(num_quantities_subplots[0], total_quantities):
        color = colors2[chem_idx - num_quantities_subplots[0]]
        gt = data[:, chem_idx]
        label = labels[chem_idx] if labels is not None else f"Chemical {chem_idx + 1}"
        ax2.plot(timesteps, gt, "-", color=color, label=label)

    # Set labels and title
    ax1.set_xlabel("Time", fontsize=12)  # Increased font size
    ax1.set_ylabel("log(Chemical Abundance)", fontsize=12)  # Increased font size
    ax2.set_xlabel("Time", fontsize=12)  # Increased font size

    # Remove individual subplot titles and add a global title
    fig.suptitle(
        r"Example Trajectories: $\tt{osu2008}$ Dataset", fontsize=14
    )  # Slightly larger title
    # fig.subplots_adjust(wspace=0.05)  # Adjust space between subplots

    # Place the legend of the left plot between the two subplots
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(
        handles1,
        labels1,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
        fontsize=10,  # Increased font size for legend
    )

    # Legend for the second plot can stay outside on the right if needed
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles2,
        labels2,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
        fontsize=10,  # Increased font size for legend
    )

    # Add grid to both subplots
    ax1.grid(True)
    ax2.grid(True)

    # Set x-axis limits tightly
    for ax in [ax1, ax2]:
        ax.set_xlim(timesteps.min(), timesteps.max())

    # Remove tick marks but keep labels and gridlines
    ax1.tick_params(
        axis="both", which="both", length=0, labelsize=10
    )  # Adjust tick label size
    ax2.tick_params(
        axis="both", which="both", length=0, labelsize=10
    )  # Adjust tick label size

    # Adjust layout to make room for the legend between the subplots
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect parameter

    # Save the plot if required
    if save:
        # Create a dummy conf dictionary
        conf = {
            "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
            "verbose": True,
        }

        # Call save_plot with the dummy conf
        save_plot(
            plt,
            "example_trajectories_poster.png",
            conf,
            dpi=200,
            base_dir="datasets",
            increase_count=False,
        )

    plt.show()


def plot_initial_conditions_distribution(
    dataset_name: str,
    train_data: np.ndarray,
    test_data: np.ndarray,
    quantity_names: list[str] | None = None,
    max_quantities: int = 10,
    spread: float = 0.01,
    quantities_per_plot: int = 10,
    log: bool = True,
    show_title: bool = True,
) -> None:
    if log:
        train_data = 10**train_data
        test_data = 10**test_data

    train_initial = train_data[:, 0, :max_quantities]
    test_initial = test_data[:, 0, :max_quantities]
    num_quantities = train_initial.shape[1]

    quantity_names = (
        quantity_names[:num_quantities]
        if quantity_names is not None
        else [f"Chemical {i + 1}" for i in range(num_quantities)]
    )
    num_plots = math.ceil(num_quantities / quantities_per_plot)
    nrows, ncols = compute_subplot_layout(num_plots)

    processed_train = []
    processed_test = []
    zero_counts_train = zero_counts_test = 0

    for i in range(num_quantities):
        train_cond = train_initial[:, i]
        test_cond = test_initial[:, i]
        if log:
            non_zero_train = train_cond > 0
            non_zero_test = test_cond > 0
            processed_train.append(np.log10(train_cond[non_zero_train]))
            processed_test.append(np.log10(test_cond[non_zero_test]))
            zero_counts_train += np.sum(~non_zero_train)
            zero_counts_test += np.sum(~non_zero_test)
        else:
            processed_train.append(train_cond)
            processed_test.append(test_cond)

    all_train = np.concatenate([cond for cond in processed_train if len(cond) > 0])
    all_test = np.concatenate([cond for cond in processed_test if len(cond) > 0])
    if len(all_train) == 0 and len(all_test) == 0:
        raise ValueError("No data available for plotting after filtering.")

    global_min = min(np.min(all_train), np.min(all_test))
    global_max = max(np.max(all_train), np.max(all_test))
    bins = np.linspace(global_min, global_max, 100)

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), sharex=True)
    axes = np.array([axes]) if (nrows == 1 and ncols == 1) else axes.flatten()
    colors = plt.cm.viridis(np.linspace(0, 0.95, quantities_per_plot))

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start = plot_idx * quantities_per_plot
        end = min((plot_idx + 1) * quantities_per_plot, num_quantities)
        for i in range(start, end):
            train_hist, _ = np.histogram(processed_train[i], bins=bins, density=True)
            train_smooth = gaussian_filter1d(train_hist, sigma=1)
            test_hist, _ = np.histogram(processed_test[i], bins=bins, density=True)
            test_smooth = gaussian_filter1d(test_hist, sigma=1)
            x = 10 ** bins[:-1] if log else bins[:-1]
            ax.plot(
                x,
                train_smooth,
                label=quantity_names[i],
                color=colors[i % quantities_per_plot],
            )
            ax.plot(x, test_smooth, "--", color=colors[i % quantities_per_plot])
        ax.set_ylabel("Density (PDF)")
        ax.set_ylim(0, 1.2 * max(train_smooth.max(), test_smooth.max()))
        ax.set_xscale("log" if log else "linear")
        ax.set_xlim(
            10**global_min if log else global_min, 10**global_max if log else global_max
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            borderaxespad=0.0,
            fontsize="small",
        )

    for ax in axes[num_plots:]:
        ax.set_visible(False)

    plt.xlabel("Initial Condition Value")
    if show_title:
        fig.suptitle(
            f"Initial Condition Distribution per Quantity \n (Dataset: {dataset_name}, {train_data.shape[0]} train samples, {test_data.shape[0]} test samples)"
        )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    conf = {
        "training_id": dataset_name.lower(),
        "verbose": True,
    }

    save_plot(
        plt,
        "IC_distribution.jpg",
        conf,
        dpi=200,
        base_dir="datasets",
        increase_count=False,
    )
    plt.close()


def compute_subplot_layout(num_plots: int) -> tuple[int, int]:
    """
    Compute the (nrows, ncols) layout for `num_plots` subplots such that:
      - The grid is as close to square as possible.
      - nrows >= ncols.
      - Minimal number of leftover grid cells (nrows * ncols - num_plots).
      - Among layouts with the same minimal leftover, prefer the one with the smallest difference between nrows and ncols.

    Args:
        num_plots (int): Total number of subplots required.

    Returns:
        Tuple[int, int]: A tuple containing (nrows, ncols).
    """
    if num_plots <= 0:
        raise ValueError("Number of plots must be a positive integer.")

    if num_plots <= 5:
        return (num_plots, 1)
    else:
        # Start searching from the ceiling of the square root of num_plots
        sqrt_n = math.sqrt(num_plots)
        ncols_start = int(math.floor(sqrt_n))
        ncols_end = int(math.ceil(sqrt_n)) + 1  # +1 to ensure coverage

        best_layout = None
        min_leftover = float("inf")
        min_diff = float("inf")

        # Iterate over possible number of columns near the square root
        for ncols in range(ncols_start, ncols_end + 1):
            if ncols < 1:
                continue  # Ensure at least one column

            nrows = math.ceil(num_plots / ncols)
            if nrows < ncols:
                continue  # Ensure nrows >= ncols

            leftover = (nrows * ncols) - num_plots
            diff = nrows - ncols

            # Update best_layout based on minimal leftover and then minimal difference
            if (leftover < min_leftover) or (
                leftover == min_leftover and diff < min_diff
            ):
                best_layout = (nrows, ncols)
                min_leftover = leftover
                min_diff = diff

        # If no layout found in the initial range, expand the search
        if not best_layout:
            for ncols in range(1, num_plots + 1):
                nrows = math.ceil(num_plots / ncols)
                if nrows < ncols:
                    continue

                leftover = (nrows * ncols) - num_plots
                diff = nrows - ncols

                if (leftover < min_leftover) or (
                    leftover == min_leftover and diff < min_diff
                ):
                    best_layout = (nrows, ncols)
                    min_leftover = leftover
                    min_diff = diff

        # Final fallback (should not be necessary, but for safety)
        if not best_layout:
            best_layout = (num_plots, 1)

        return best_layout


def plot_all_trajectories_and_gradients(
    dataset_name: str,
    train_data: np.ndarray,
    quantity_names: list[str] = None,
    timesteps: np.ndarray = None,
    max_quantities: int = 10,
    opacity: float = 0.01,
    quantities_per_plot: int = 6,
    max_trajectories: int = 20,
    log: bool = False,
    log_time: bool = False,
    show_title: bool = True,
) -> None:
    """
    Plot both the trajectories and gradients of each quantity in the train dataset over time,
    with individual sample trajectories and gradients shown with low opacity and Gaussian spread.

    Args:
        dataset_name (str): The name of the dataset (e.g., "osu2008").
        train_data (np.ndarray): Training dataset array of shape [n_samples, n_timesteps, n_quantities].
        quantity_names (list, optional): List of quantity names for labeling the lines.
        timesteps (np.ndarray, optional): Array of timesteps for the x-axis.
        max_quantities (int, optional): Maximum number of quantities to plot. Default is 10.
        opacity (float, optional): Opacity of individual trajectories and gradients. Default is 0.01.
        quantities_per_plot (int, optional): Number of quantities to plot per subplot. Default is 6.
        max_trajectories (int, optional): Maximum number of trajectories to plot. Default is 200.
        log (bool, optional): Whether the data is in log-space (only for axis labels). Default is False.
        log_time (bool, optional): Whether the time axis is in log-space. Default is False.
    """
    print(f"Plotting trajectories and gradients for dataset: {dataset_name}")
    # Limit the number of trajectories
    num_trajectories = train_data.shape[0]
    if num_trajectories > max_trajectories:
        factor = (train_data.shape[0] // max_trajectories) + 1
        train_data = train_data[::factor]
        print(
            f"Number of trajectories {num_trajectories} exceeds the limit of {max_trajectories}."
        )
        print(f"Slicing data to {train_data.shape[0]} trajectories (factor {factor}).")

    # Restrict to the first `max_quantities` along the last axis
    max_quantities_to_use = min(max_quantities, train_data.shape[2])
    train_data = train_data[:, :, :max_quantities_to_use]
    if quantity_names is not None:
        quantity_names = quantity_names[:max_quantities_to_use]

    # Compute subplot layout
    num_plots = math.ceil(max_quantities_to_use / quantities_per_plot)
    if num_plots > 10:
        print(f"Creating {num_plots} subplots. This may take a while.")
    nrows, ncols = compute_subplot_layout(num_plots)

    # Prepare data for trajectories and gradients
    avg_trajectories = np.mean(train_data, axis=0)  # [n_timesteps, n_quantities]
    gradients = np.gradient(
        train_data, axis=1
    )  # [n_samples, n_timesteps, n_quantities]
    avg_gradients = np.mean(gradients, axis=0)  # [n_timesteps, n_quantities]
    n_timesteps = train_data.shape[1]
    time = np.arange(n_timesteps) if timesteps is None else timesteps

    # Create figures and axes for both plots
    fig_traj, axes_traj = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 3 * nrows), sharex=True
    )
    fig_grad, axes_grad = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 3 * nrows), sharex=True
    )

    if nrows == 1 and ncols == 1:
        axes_traj = np.array([axes_traj])
        axes_grad = np.array([axes_grad])
    else:
        axes_traj = axes_traj.flatten()
        axes_grad = axes_grad.flatten()

    # Prepare color palette
    colors = plt.cm.viridis(np.linspace(0, 0.95, quantities_per_plot))

    progress = tqdm(range(num_plots), desc="Plotting", unit="plot")

    # Set formatter for y axis ticks

    for plot_idx in progress:
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, max_quantities_to_use)

        for i in range(start_idx, end_idx):
            color = colors[i % quantities_per_plot]
            label_name = (
                quantity_names[i]
                if quantity_names is not None and i < len(quantity_names)
                else f"Quantity {i + 1}"
            )

            # Trajectories Plot
            for sample_idx in range(train_data.shape[0]):
                axes_traj[plot_idx].plot(
                    time,
                    train_data[sample_idx, :, i],
                    color=color,
                    alpha=opacity,
                    linewidth=1,
                )
            axes_traj[plot_idx].plot(
                time, avg_trajectories[:, i], label=label_name, color=color, linewidth=1
            )

            # Gradients Plot
            for sample_idx in range(train_data.shape[0]):
                axes_grad[plot_idx].plot(
                    time,
                    gradients[sample_idx, :, i],
                    color=color,
                    alpha=opacity,
                    linewidth=1,
                )
            axes_grad[plot_idx].plot(
                time, avg_gradients[:, i], label=label_name, color=color, linewidth=1
            )

        # Set labels and legends for Trajectories
        if plot_idx == num_plots - 1:
            axes_traj[plot_idx].set_xlabel("Time")
        axes_traj[plot_idx].set_xlim(time[0], time[-1])
        if log_time:
            axes_traj[plot_idx].set_xscale("log")
        ylabel_traj = "log(Quantity)" if log else "Quantity"
        axes_traj[plot_idx].set_ylabel(ylabel_traj)
        if quantity_names is not None:
            axes_traj[plot_idx].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                fontsize="small",
            )

        # Set labels and legends for Gradients
        if plot_idx == num_plots - 1:
            axes_grad[plot_idx].set_xlabel("Time")
        if log_time:
            axes_grad[plot_idx].set_xscale("log")
        axes_grad[plot_idx].set_xlim(time[0], time[-1])
        axes_grad[plot_idx].set_ylabel("Gradient")
        if quantity_names is not None:
            axes_grad[plot_idx].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                fontsize="small",
            )

    # Hide any unused axes
    for ax in axes_traj[num_plots:]:
        ax.set_visible(False)
    for ax in axes_grad[num_plots:]:
        ax.set_visible(False)

    # Align y-axis labels
    fig_traj.align_ylabels(axes_traj)
    fig_grad.align_ylabels(axes_grad)

    # Set titles
    if show_title:
        fig_traj.suptitle(
            f"Overview of Trajectories per Quantity over Time \n (Dataset: {dataset_name})",
            fontsize="large",
        )
        fig_grad.suptitle(
            f"Overview of Gradients per Quantity over Time \n (Dataset: {dataset_name})",
            fontsize="large",
        )

    # Adjust layout
    fig_traj.tight_layout(rect=[0, 0, 1, 0.95])
    fig_grad.tight_layout(rect=[0, 0, 1, 0.95])

    conf = {
        "training_id": dataset_name.lower(),
        "verbose": True,
    }

    print("Saving trajectory plot...")

    # Save both plots
    save_plot(
        fig_traj,
        "all_trajectories.jpg",
        conf,
        dpi=300,
        base_dir="datasets",
        increase_count=False,
    )

    print("Saving gradient plot...")

    save_plot(
        fig_grad,
        "all_gradients.jpg",
        conf,
        dpi=300,
        base_dir="datasets",
        increase_count=False,
    )

    # Close figures
    plt.close(fig_traj)
    plt.close(fig_grad)


def debug_numerical_errors_plot(
    dataset_name: str,
    train_data: np.ndarray,
    quantity_names: list[str] | None = None,
    max_quantities: int = 10,
    threshold: float = 0.1,
    max_faulty: int = 10,
    quantities_per_plot: int = 10,
    show_title: bool = True,
) -> None:
    """
    Plot faulty trajectories with gradients exceeding a threshold.

    Args:
        dataset_name (str): The name of the dataset.
        train_data (np.ndarray): Dataset array of shape [n_samples, n_timesteps, n_quantities].
        quantity_names (list, optional): Names for each quantity.
        max_quantities (int, optional): Maximum number of quantities to plot. Default is 10.
        threshold (float, optional): Gradient threshold to define faulty trajectories. Default is 0.1.
        max_faulty (int, optional): Maximum number of faulty trajectories to plot. Default is 10.
        quantities_per_plot (int, optional): Number of quantities to plot per subplot. Only for faulty IC plot.
    """
    # Ensure max_faulty does not exceed 10
    max_faulty = min(max_faulty, 10)

    n_samples, n_timesteps, n_quantities = train_data.shape

    # Cap the number of quantities to plot
    num_quantities = min(max_quantities, n_quantities)
    train_data = train_data[:, :, :num_quantities]

    if quantity_names is not None:
        quantity_names = quantity_names[:num_quantities]
    else:
        quantity_names = [f"Quantity {i + 1}" for i in range(num_quantities)]

    # Identify faulty trajectories
    gradients = np.gradient(train_data, axis=1)
    faulty_mask = np.any(np.abs(gradients) > threshold, axis=(1, 2))
    faulty_indices = np.where(faulty_mask)[0]
    print(f"Detected {len(faulty_indices)} faulty trajectories")

    if faulty_indices.size == 0:
        print("No faulty trajectories found.")
        return

    plot_faulty_initial_conditions_distribution(
        dataset_name=dataset_name,
        train_data=train_data,
        faulty_indices=faulty_indices,
        quantity_names=quantity_names,
        max_quantities=max_quantities,
        log=True,  # Set to False if data is not in log-space
        quantities_per_plot=quantities_per_plot,
    )

    # Limit the number of faulty trajectories to plot
    faulty_indices = faulty_indices[:max_faulty]
    time = np.linspace(0, 1, n_timesteps)

    # Plot each faulty trajectory
    fig, axes = plt.subplots(
        len(faulty_indices), 1, figsize=(9, 4 * len(faulty_indices))
    )

    if len(faulty_indices) == 1:
        axes = [axes]

    for idx, sample_idx in enumerate(faulty_indices):
        ax = axes[idx]
        trajectory = train_data[sample_idx]
        traj_gradients = gradients[sample_idx]

        faulty_quantities = np.any(np.abs(traj_gradients) > threshold, axis=0)
        faulty_labels = [
            quantity_names[q]
            for q, is_faulty in enumerate(faulty_quantities)
            if is_faulty
        ]

        for q in range(num_quantities):
            ax.plot(time, trajectory[:, q], label=quantity_names[q], linewidth=1.5)
            if faulty_quantities[q]:
                spike_timesteps = np.where(np.abs(traj_gradients[:, q]) > threshold)[0]
                ax.scatter(
                    time[spike_timesteps],
                    trajectory[spike_timesteps, q],
                    color="red",
                    s=20,
                )

        ax.set_title(
            f"Sample {sample_idx}, Faulty Quantities: {', '.join(faulty_labels)}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Quantity")
        ax.legend(loc="upper right")

    plt.tight_layout()
    if show_title:
        fig.suptitle(
            f"Faulty Trajectories (Dataset: {dataset_name}, Max: {max_faulty})",
            fontsize=16,
        )

    # Saving the plot
    conf = {
        "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
        "verbose": True,
    }

    save_plot(
        plt,
        "faulty_trajectories.png",  # Save the plot with this name
        conf,
        dpi=200,
        base_dir="datasets",  # Base directory for saving the plot
        increase_count=False,
    )


def plot_faulty_initial_conditions_distribution(
    dataset_name: str,
    train_data: np.ndarray,
    faulty_indices: np.ndarray,
    quantity_names: list[str] | None = None,
    max_quantities: int = 10,
    log: bool = True,
    quantities_per_plot: int = 10,
    show_title: bool = True,
) -> None:
    """
    Plot the distribution of initial conditions (t=0) for each quantity from faulty trajectories.

    Args:
        dataset_name (str): The name of the dataset (e.g., "osu2008").
        train_data (np.ndarray): Training dataset array of shape [num_samples, num_timesteps, num_quantities].
        faulty_indices (np.ndarray): Indices of faulty trajectories within train_data.
        quantity_names (list, optional): List of quantity names for labeling the lines.
        max_quantities (int, optional): Maximum number of quantities to plot. Default is 10.
        log (bool, optional): Whether the data is in log-space and should be exponentiated (i.e., data = 10**data).
        quantities_per_plot (int, optional): Number of quantities to plot per subplot. Default is 10.
    """
    # If data is in log space, exponentiate it
    if log:
        train_data = 10**train_data

    # Extract initial conditions (t=0) for faulty trajectories
    faulty_data = train_data[faulty_indices, :, :]
    faulty_initial_conditions = faulty_data[:, 0, :]  # Extract t=0

    # Cap the number of quantities to plot at max_quantities
    num_quantities = min(max_quantities, faulty_initial_conditions.shape[1])
    faulty_initial_conditions = faulty_initial_conditions[:, :num_quantities]

    if quantity_names is not None:
        quantity_names = quantity_names[:num_quantities]
    else:
        quantity_names = [f"Chemical {i + 1}" for i in range(num_quantities)]

    # Split the quantities into groups of 10 for plotting
    num_plots = int(np.ceil(num_quantities / quantities_per_plot))

    # Initialize list to hold log-transformed non-zero initial conditions
    log_faulty_conditions = []
    zero_counts_faulty = 0

    # Transform data to log-space and filter out zeros
    for i in range(num_quantities):
        faulty_quantity_conditions = faulty_initial_conditions[:, i]

        non_zero_faulty_conditions = faulty_quantity_conditions[
            faulty_quantity_conditions > 0
        ]

        if log:
            log_faulty_conditions.append(np.log10(non_zero_faulty_conditions))
        else:
            log_faulty_conditions.append(non_zero_faulty_conditions)

        zero_counts_faulty += np.sum(faulty_quantity_conditions == 0)

    # Determine global min and max for histogram bins
    min_values = [np.min(cond) for cond in log_faulty_conditions if len(cond) > 0]
    max_values = [np.max(cond) for cond in log_faulty_conditions if len(cond) > 0]

    if not min_values or not max_values:
        print("No non-zero initial conditions to plot.")
        return

    global_min = np.min(min_values)
    global_max = np.max(max_values)

    # Set up the x-axis range to nearest whole numbers in log-space
    x_min = np.floor(global_min)
    x_max = np.ceil(global_max)

    # Define the x-axis range for plotting
    x_vals = np.linspace(x_min, x_max + 0.1, 100)

    # Create subplots with shared x-axis
    fig, axes = plt.subplots(num_plots, 1, figsize=(9, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one plot

    # Get custom color palette
    # Assuming get_custom_palette is defined elsewhere
    colors = plt.cm.viridis(np.linspace(0, 0.95, quantities_per_plot))

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, num_quantities)

        for i in range(start_idx, end_idx):
            # Compute histogram for faulty dataset
            faulty_hist, faulty_bin_edges = np.histogram(
                log_faulty_conditions[i], bins=x_vals, density=True
            )
            # Smooth the histogram with a Gaussian filter
            faulty_smoothed_hist = gaussian_filter1d(faulty_hist, sigma=1)

            # Plot the smoothed histogram for the faulty data
            ax.plot(
                10 ** faulty_bin_edges[:-1],  # Convert back to original scale
                faulty_smoothed_hist,
                label=quantity_names[i] if quantity_names is not None else None,
                color=colors[i % quantities_per_plot],
            )

        ax.set_yscale("linear")
        ax.set_ylabel("Density (PDF)")
        ax.set_xlim(10**x_min, 10**x_max)
        if quantity_names is not None:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(
                    1,
                    0.5,
                ),
                borderaxespad=0.0,
            )

    plt.xscale("log")  # Log scale for initial conditions magnitudes
    plt.xlabel("Initial Condition Magnitude")
    if show_title:
        fig.suptitle(
            f"Initial Condition Distribution per Chemical for Faulty Trajectories \n (Dataset: {dataset_name}, {len(faulty_indices)} faulty samples)"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot
    conf = {
        "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
        "verbose": True,
    }

    save_plot(
        plt,
        "Faulty_IC_distribution.png",
        conf,
        dpi=200,
        base_dir="datasets",
        increase_count=False,
    )

    plt.close()
