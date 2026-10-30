import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import yaml
from optuna.visualization import (
    plot_contour,
    plot_parallel_coordinate,
    plot_param_importances,
)


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate an Optuna study and its top models."
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="cloudparams",
        help="The name of the study to evaluate.",
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="Number of top models to evaluate."
    )
    return parser.parse_args()


def load_study_config(study_name: str) -> dict:
    """
    Load the YAML config used by the study (optuna_config.yaml).
    """

    config_path = os.path.join("tuned", "studies", study_name, "optuna_config.yaml")
    with open(config_path) as file:
        return yaml.safe_load(file)


def load_model_test_losses(model_path: str) -> np.ndarray:
    """
    Load the test losses from the model checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.

    Returns:
        np.ndarray: Test losses.
    """

    model_dict = torch.load(model_path, map_location=torch.device("cpu"))
    return model_dict["attributes"]["test_loss"]


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute the moving average of a 1D array.

    Args:
        data (np.ndarray): 1D array to compute the moving average.
        window_size (int): Size of the window for the moving average.

    Returns:
        np.ndarray: Moving average of the input data.

    Raises:
        ValueError: If the window size is not a positive integer.
    """

    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_test_losses(
    test_losses: list[np.ndarray],
    labels: list[str],
    study_name: str,
    window_size: int = 5,
) -> None:
    """
    Plot the test losses of the top models.

    Args:
        test_losses (list[np.ndarray]): List of test losses.
        labels (list[str]): List of labels for the test losses.
        study_name (str): Name of the study.
        window_size (int, optional): Size of the window for the moving average. Defaults to 5.
    """

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(test_losses)))

    for i, test_loss in enumerate(test_losses):
        avg_loss = moving_average(test_loss, window_size)
        plt.plot(
            np.arange(len(avg_loss)) + window_size - 1,
            avg_loss,
            label=f"Test Loss: {labels[i]}",
            color=colors[i],
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(f"Comparison of Test Losses for {study_name}")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plots_dir = os.path.join("tuned", "studies", study_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "test_losses_comparison.png"))
    plt.close()


def main():
    """
    Main function to evaluate an Optuna study and its top models.
    Usually, viewing the study database with Optuna Dashboard is more informative.
    """

    args = parse_arguments()
    study_name = args.study_name
    top_n = args.top_n

    # Load the existing study from its DB
    db_path = f"sqlite:///tuned/studies/{study_name}/{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=db_path)

    # Print best trial
    print(f"Best trial parameters for study '{study_name}':")
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Print total number of runs
    print(f"\nTotal number of runs: {len(study.trials)}")

    # Make sure plots directory exists
    plots_dir = os.path.join("tuned", "studies", study_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Optionally transform MSE to log10 for better plotting
    for trial in study.trials:
        if trial.value is not None:
            trial.value = np.log10(trial.value)

    # Save built-in plots
    plot_contour(study).write_html(os.path.join(plots_dir, "contour_plot.html"))
    plot_param_importances(study).write_html(
        os.path.join(plots_dir, "param_importances_plot.html")
    )
    plot_parallel_coordinate(study).write_html(
        os.path.join(plots_dir, "parallel_coordinate_plot.html")
    )

    # Evaluate top models
    config = load_study_config(study_name)
    surrogate_name = config["surrogate"]["name"]

    # Filter out None-value trials
    completed_trials = [t for t in study.trials if t.value is not None]
    best_trials = sorted(completed_trials, key=lambda t: t.value)[:top_n]

    test_losses = []
    labels = []

    # Models are stored:
    #   tuned/studies/<study_name>/models/<ClassName>/<model_name>.pth
    # where <model_name> might be something like "fullyconnected_3.pth"
    for trial in best_trials:
        model_filename = f"{surrogate_name.lower()}_{trial.number}.pth"
        model_path = os.path.join(
            "tuned",
            "studies",
            study_name,
            "models",
            surrogate_name,
            model_filename,
        )

        # If your class name differs from the config's "surrogate_name",
        # adjust accordingly.
        test_loss = load_model_test_losses(model_path)
        test_losses.append(test_loss)
        labels.append(f"Trial {trial.number} LR: {trial.params['learning_rate']:.2e}")

    # Plot test losses
    plot_test_losses(test_losses, labels, study_name, window_size=1)


if __name__ == "__main__":
    main()
