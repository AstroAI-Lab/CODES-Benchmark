"""
Utility functions that execute the different benchmark suites (accuracy, gradients,
timing, sparsity studies, etc.) for every surrogate model.

The functions in this module are invoked by `run_eval.py` and populate the metrics
dictionary that later feeds the comparative plots and CSV exports.
"""

import os
from contextlib import redirect_stdout
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr
from tabulate import tabulate
from torch.utils.data import DataLoader

from codes.utils import batch_factor_to_float, check_and_load_data

from .bench_plots import (  # plot_generalization_errors,; rel_errors_and_uq,; plot_uncertainty_confidence,
    inference_time_bar_plot,
    plot_all_generalization_errors,
    plot_average_errors_over_time,
    plot_average_uncertainty_over_time,
    plot_catastrophic_detection_curves,
    plot_comparative_error_correlation_heatmaps,
    plot_comparative_gradient_heatmaps,
    plot_error_distribution_comparative,
    plot_error_distribution_per_quantity,
    plot_error_percentiles_over_time,
    plot_errors_over_time,
    plot_example_iterative_predictions,
    plot_example_mode_predictions,
    plot_example_predictions_with_uncertainty,
    plot_generalization_error_comparison,
    plot_gradients_heatmap,
    plot_loss_comparison,
    plot_loss_comparison_equal,
    plot_loss_comparison_train_duration,
    plot_mean_deltadex_over_time_main_vs_ensemble,
    plot_surr_losses,
    plot_uncertainty_heatmap,
    plot_uncertainty_over_time_comparison,
)
from .bench_utils import (
    count_trainable_parameters,
    format_seconds,
    format_time,
    format_value,
    get_model_config,
    get_surrogate,
    make_comparison_csv,
    measure_inference_time,
    measure_memory_footprint,
    save_table_csv,
    write_metrics_to_yaml,
)

TITLE = True


def run_benchmark(surr_name: str, surrogate_class, conf: dict) -> dict[str, Any]:
    """
    Run benchmarks for a given surrogate model.

    Args:
        surr_name (str): The name of the surrogate model to benchmark.
        surrogate_class: The class of the surrogate model.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing all relevant metrics for the given model.
    """
    # Instantiate the model
    device = conf["devices"]
    device = device[0] if isinstance(device, list) else device

    # Determine the batch size
    surr_idx = conf["surrogates"].index(surr_name)
    if isinstance(conf["batch_size"], list):
        if len(conf["batch_size"]) != len(conf["surrogates"]):
            raise ValueError(
                "The number of provided batch sizes must match the number of surrogate models."
            )
        else:
            batch_size = conf["batch_size"][surr_idx]
    else:
        batch_size = conf["batch_size"]

    # Load full data and parameters
    dataset_conf = conf.get("dataset", {})
    (
        (train_data, test_data, val_data),
        (train_params, test_params, val_params),
        timesteps,
        n_train_samples,
        _,
        labels,
    ) = check_and_load_data(
        dataset_conf["name"],
        verbose=conf.get("verbose", False),
        log=dataset_conf.get("log10_transform", True),
        log_params=dataset_conf.get("log10_transform_params", True),
        normalisation_mode=dataset_conf.get("normalise", "minmax"),
        tolerance=dataset_conf.get("tolerance", 1e-25),
        per_species=dataset_conf.get("normalise_per_species", False),
    )

    model_config = get_model_config(surr_name, conf)
    n_timesteps = train_data.shape[1]
    n_quantities = train_data.shape[2]
    n_test_samples = n_timesteps * val_data.shape[0]
    n_params = train_params.shape[1] if train_params is not None else 0
    model = surrogate_class(
        device=device,
        n_quantities=n_quantities,
        n_timesteps=n_timesteps,
        n_parameters=n_params,
        config=model_config,
    )

    # Placeholder for metrics
    metrics = {}
    metrics["timesteps"] = timesteps
    metrics["n_params"] = n_params

    # Create dataloader for the validation data
    _, _, val_loader = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=val_data,
        timesteps=timesteps,
        batch_size=batch_size,
        shuffle=True,
        dataset_train_params=train_params,
        dataset_test_params=test_params,
        dataset_val_params=val_params,
        dummy_timesteps=True,
    )

    # Plot training losses
    if conf.get("losses", False):
        print("Loss plots...")
        plot_surr_losses(model, surr_name, conf, timesteps, show_title=TITLE)

    # Accuracy benchmark
    print("Running accuracy benchmark...")
    metrics["accuracy"] = evaluate_accuracy(
        model, surr_name, timesteps, val_loader, conf, labels
    )

    if conf.get("iterative", False):
        # Iterative training benchmark
        print("Running iterative training benchmark...")
        metrics["iterative"] = evaluate_iterative_predictions(
            model, surr_name, timesteps, val_loader, val_params, conf, labels
        )

    # Gradients benchmark
    if conf.get("gradients", False):
        print("Running gradients benchmark...")
        # For this benchmark, we can also use the main model
        metrics["gradients"] = evaluate_gradients(model, surr_name, val_loader, conf)

    # Timing benchmark
    if conf.get("timing", False):
        print("Running timing benchmark...")
        metrics["timing"] = time_inference(
            model, surr_name, val_loader, conf, n_test_samples
        )

    # Compute (resources) benchmark
    if conf.get("compute", False):
        print("Running compute benchmark...")
        metrics["compute"] = evaluate_compute(model, surr_name, val_loader, conf)

    # Interpolation benchmark
    if conf.get("interpolation", {}).get("enabled", False):
        print("Running interpolation benchmark...")
        metrics["interpolation"] = evaluate_interpolation(
            model, surr_name, val_loader, timesteps, conf, labels
        )

    # Extrapolation benchmark
    if conf.get("extrapolation", {}).get("enabled", False):
        print("Running extrapolation benchmark...")
        metrics["extrapolation"] = evaluate_extrapolation(
            model, surr_name, val_loader, timesteps, conf, labels
        )

    # Sparse data benchmark
    if conf.get("sparse", {}).get("enabled", False):
        print("Running sparse benchmark...")
        metrics["sparse"] = evaluate_sparse(
            model, surr_name, val_loader, timesteps, n_train_samples, conf
        )

    # Batch size benchmark
    if conf.get("batch_scaling", {}).get("enabled", False):
        print("Running batch size benchmark...")
        metrics["batch_size"] = evaluate_batchsize(
            model, surr_name, val_loader, timesteps, conf
        )

    # Uncertainty Quantification (UQ) benchmark
    if conf.get("uncertainty", {}).get("enabled", False):
        print("Running UQ benchmark...")
        metrics["UQ"] = evaluate_UQ(
            model, surr_name, val_loader, timesteps, conf, labels
        )

    # Write metrics to yaml
    write_metrics_to_yaml(surr_name, conf, metrics)

    return metrics


def evaluate_accuracy(
    model,
    surr_name: str,
    timesteps: np.ndarray,
    test_loader: DataLoader,
    conf: dict,
    labels: list | None = None,
) -> dict[str, Any]:
    """
    Evaluate the accuracy of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        timesteps (np.ndarray): The timesteps array.
        test_loader (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.
        labels (list, optional): The labels for the quantities.
        percentile (int, optional): The percentile for error metrics.

    Returns:
        dict: A dictionary containing accuracy metrics.
    """
    training_id = conf["training_id"]
    percentile = conf.get("error_percentile", 99)

    # Load the model
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")
    train_time = model.train_duration
    num_quantities = model.n_quantities
    model_index = conf["surrogates"].index(surr_name)
    n_epochs = conf["epochs"][model_index]

    # Obtain log-space predictions and targets
    preds, targets = model.predict(data_loader=test_loader, leave_log=True)
    preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()

    # Compute log-space error metrics
    absolute_errors_log = np.abs(preds - targets)
    root_mean_squared_error_log = np.sqrt(np.mean(absolute_errors_log**2))
    median_absolute_error_log = np.median(absolute_errors_log)
    mean_absolute_error_log = np.mean(absolute_errors_log)
    percentile_absolute_error_log = np.percentile(absolute_errors_log, percentile)

    # Obtain real-space predictions and targets
    preds, targets = model.predict(data_loader=test_loader, leave_log=False)
    preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()

    # Compute real-space error metrics
    absolute_errors = np.abs(preds - targets)
    root_mean_squared_error_real = np.sqrt(np.mean(absolute_errors**2))
    median_absolute_error_real = np.median(absolute_errors)
    mean_absolute_error_real = np.mean(absolute_errors)
    percentile_absolute_error_real = np.percentile(absolute_errors, percentile)

    # Additional real-space errors: Relative error
    relative_error_threshold = float(conf.get("relative_error_threshold", 0.0))
    relative_errors = np.abs(
        absolute_errors / np.maximum(np.abs(targets), relative_error_threshold)
    )
    median_relative_error = np.median(relative_errors)
    mean_relative_error = np.mean(relative_errors)
    percentile_relative_error = np.percentile(relative_errors, percentile)

    plot_error_percentiles_over_time(
        surr_name,
        conf,
        relative_errors,
        timesteps,
        title=f"Relative Errors over Time for {surr_name}",
        mode="relative",
        save=True,
        show_title=TITLE,
    )

    plot_error_percentiles_over_time(
        surr_name,
        conf,
        absolute_errors_log,
        timesteps,
        title=r"Absolute Log-Space Errors ($\Delta dex$) over Time for "
        + f"{surr_name}",
        mode="deltadex",
        save=True,
        show_title=TITLE,
    )

    plot_error_distribution_per_quantity(
        surr_name,
        conf,
        relative_errors,
        quantity_names=labels,
        num_quantities=num_quantities,
        mode="relative",
        save=True,
        show_title=TITLE,
    )

    plot_error_distribution_per_quantity(
        surr_name,
        conf,
        absolute_errors_log,
        quantity_names=labels,
        num_quantities=num_quantities,
        mode="deltadex",
        save=True,
        show_title=TITLE,
    )

    # Store metrics
    accuracy_metrics = {
        "root_mean_squared_error_log": root_mean_squared_error_log,
        "median_absolute_error_log": median_absolute_error_log,
        "mean_absolute_error_log": mean_absolute_error_log,
        "percentile_absolute_error_log": percentile_absolute_error_log,
        "root_mean_squared_error_real": root_mean_squared_error_real,
        "median_absolute_error_real": median_absolute_error_real,
        "mean_absolute_error_real": mean_absolute_error_real,
        "percentile_absolute_error_real": percentile_absolute_error_real,
        "median_relative_error": median_relative_error,
        "mean_relative_error": mean_relative_error,
        "percentile_relative_error": percentile_relative_error,
        "error_percentile": percentile,
        "main_model_training_time": train_time,
        "main_model_epochs": n_epochs,
        "absolute_errors": absolute_errors,
        "relative_errors": relative_errors,
        "absolute_errors_log": absolute_errors_log,
    }

    return accuracy_metrics


def evaluate_iterative_predictions(
    model,
    surr_name: str,
    timesteps: np.ndarray,
    val_loader: DataLoader,
    val_params: np.ndarray,
    conf: dict,
    labels: list | None = None,
) -> dict[str, Any]:
    """
    Measure how prediction errors accumulate when the surrogate is rolled forward in
    an auto-regressive fashion.

    The model is fed with its own previous output (instead of ground truth) and
    evaluated for each chunk of the trajectory. The same error metrics as
    :func:`evaluate_accuracy` are reported so you can compare one-shot predictions
    with long-horizon roll-outs.

    Args:
        model: Loaded surrogate instance.
        surr_name (str): Surrogate identifier used to locate checkpoints.
        timesteps (np.ndarray): Timeline associated with the validation data.
        val_loader (DataLoader): Validation loader that provides initial states.
        val_params (np.ndarray): Optional auxiliary parameters aligned with the data.
        conf (dict): Benchmark configuration.
        labels (list | None): Quantity names for labelling plots.

    Returns:
        dict[str, Any]: Error statistics for iterative predictions together with the
        generated trajectories.
    """
    # load trained model
    training_id = conf["training_id"]
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    # get full ground truth (targets) and ignore one-shot preds
    full_preds, targets = model.predict(
        data_loader=val_loader, leave_log=True, leave_norm=True
    )
    targets = targets.detach().cpu().numpy()
    n_samples, n_timesteps, n_quantities = targets.shape

    original_n_timesteps = model.n_timesteps

    # how many timesteps per chunk
    iter_interval = 10  # conf["iterative"]["interval"]
    # batch size same as in run_benchmark
    surr_idx = conf["surrogates"].index(surr_name)
    if isinstance(conf["batch_size"], list):
        batch_size = conf["batch_size"][surr_idx]
    else:
        batch_size = conf["batch_size"]

    # container for the piecewise predictions; seed t=0 with ground truth so errors
    # are computed only on actual predictions for t>=1 while keeping shape intact
    iterative_preds = np.zeros_like(targets)

    # number of chunks
    n_chunks = (n_timesteps + iter_interval - 1) // iter_interval

    T_min = float(timesteps[0])
    T_max = float(timesteps[-1])
    log_ts = bool(conf["dataset"].get("log_timesteps", False))
    log_ratio = np.log(T_max / T_min) if log_ts else None

    for i in range(n_chunks):
        start = i * iter_interval
        end = min(start + iter_interval, n_timesteps - 1)
        model.n_timesteps = end - start + 1

        init_state = targets[:, 0, :] if i == 0 else iterative_preds[:, start - 1, :]
        ds = np.zeros((n_samples, model.n_timesteps, n_quantities))
        ds[:, 0, :] = init_state

        real_chunk = timesteps[start : end + 1]
        # Step 1: subtract first real time in the chunk
        delta_real = real_chunk - real_chunk[0]

        if log_ts:
            # Step 2: map "T_min + delta_real" into dummy space
            shifted = T_min + delta_real
            u_chunk = np.log(shifted / T_min) / log_ratio
        else:
            shifted = T_min + delta_real
            u_chunk = (shifted - T_min) / (T_max - T_min)

        dt = u_chunk

        if not (np.all(dt >= 0) and dt[-1] <= 1.0):
            raise ValueError(f"Invalid dummy times for chunk {i}: {dt}")

        train_loader, _, _ = model.prepare_data(
            dataset_train=ds,
            dataset_test=None,
            dataset_val=None,
            timesteps=dt,
            batch_size=batch_size,
            shuffle=False,
            dataset_train_params=val_params,
            dataset_test_params=None,
            dataset_val_params=None,
            dummy_timesteps=False,
        )

        # predict this chunk and insert into the global array
        preds_chunk, _ = model.predict(
            data_loader=train_loader, leave_log=True, leave_norm=True
        )
        # We predict steps 1..(chunk_len-1) relative to the provided init state (index 0).
        # Map these to global indices [start+1 .. end] inclusively.
        if i == 0:
            iterative_preds[:, start : end + 1, :] = preds_chunk[:, : model.n_timesteps, :].detach().cpu().numpy()
        iterative_preds[:, start + 1 : end + 1, :] = (
            preds_chunk[:, 1 : model.n_timesteps, :].detach().cpu().numpy()
        )

    iterative_preds_log = model.denormalize(iterative_preds, leave_log=True)
    full_preds_log = model.denormalize(full_preds, leave_log=True)
    targets_log = model.denormalize(targets, leave_log=True)
    iterative_preds = model.denormalize(iterative_preds)
    full_preds_real = model.denormalize(full_preds.detach().cpu().numpy())
    targets = model.denormalize(targets)

    # compute error metrics
    errors = iterative_preds - targets
    abs_errors = np.abs(errors)
    mse = float(np.mean(errors**2))
    mae = float(np.mean(abs_errors))

    # compute log-space errors
    abs_errors_log = np.abs(iterative_preds_log - targets_log)
    rmse_log = float(np.mean(abs_errors_log**2))
    mae_log = float(np.mean(abs_errors_log))
    percentile = conf.get("error_percentile", 99)
    percentile_abs_error_log = float(np.percentile(abs_errors_log, percentile))

    errors = np.mean(np.abs(iterative_preds - targets), axis=(1, 2))
    example_idx = int(np.argsort(np.abs(errors - np.median(errors)))[0])

    # Restore original number of timesteps
    model.n_timesteps = original_n_timesteps

    plot_example_iterative_predictions(
        surr_name,
        conf,
        iterative_preds,
        full_preds_real,
        targets,
        timesteps,
        iter_interval=iter_interval,
        example_idx=example_idx,
        labels=labels,
        save=True,
        show_title=TITLE,
    )

    return {
        "root_mean_squared_error_log": rmse_log,
        "mean_absolute_error_log": mae_log,
        "percentile_absolute_error_log": percentile_abs_error_log,
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "absolute_errors": abs_errors,
        "absolute_errors_log": abs_errors_log,
        "iteration_interval": iter_interval,
    }


def evaluate_gradients(
    model,
    surr_name: str,
    test_loader: DataLoader,
    conf: dict,
    species_names: list = None,
) -> dict:
    """
    Evaluate the gradients of the surrogate model in log-space (Δdex).

    Predictions and targets are kept in log10 space (leave_log=True).
    Errors are computed as absolute log differences (Δdex).

    Args:
        model: Surrogate model instance.
        surr_name (str): Surrogate name.
        test_loader (DataLoader): Test data.
        conf (dict): Configuration dictionary.
        species_names (list, optional): Names of the species/quantities.

    Returns:
        dict: Gradient–error correlation metrics in log-space.
    """
    training_id = conf["training_id"]

    # Load the main model
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    # Predict in log-space (dex)
    preds_log, targets_log = model.predict(test_loader, leave_log=True)
    preds_log = preds_log.detach().cpu().numpy()
    targets_log = targets_log.detach().cpu().numpy()

    # Gradients of targets in log-space
    gradients = np.gradient(targets_log, axis=1)
    gradients = np.abs(gradients)
    gradients = gradients / gradients.max()  # normalize

    # Δdex errors
    prediction_errors = np.abs(preds_log - targets_log)

    # Correlation per species
    species_correlations = []
    for i in range(targets_log.shape[2]):
        grad_species = gradients[:, :, i].flatten()
        err_species = prediction_errors[:, :, i].flatten()
        corr, _ = pearsonr(grad_species, err_species)
        species_correlations.append(corr)

    # Average correlation
    avg_grad = gradients.mean(axis=2).flatten()
    avg_err = prediction_errors.mean(axis=2).flatten()
    avg_corr, _ = pearsonr(avg_grad, avg_err)

    # Heatmap of average correlation
    max_count, max_grad, max_err = plot_gradients_heatmap(
        surr_name,
        conf,
        gradients,
        prediction_errors,
        avg_corr,
        save=True,
        show_title=TITLE,
    )

    # Ensure species names
    if species_names is None:
        species_names = [f"quantity_{i}" for i in range(targets_log.shape[2])]
    species_correlations = dict(zip(species_names, species_correlations))

    return {
        "gradients": gradients,
        "species_correlations": species_correlations,
        "avg_correlation": avg_corr,
        "max_counts": max_count,
        "max_gradient": max_grad,
        "max_error": max_err,
    }


def time_inference(
    model,
    surr_name: str,
    test_loader: DataLoader,
    conf: dict,
    n_test_samples: int,
    n_runs: int = 5,
) -> dict[str, Any]:
    """
    Time the inference of the surrogate model (full version with metrics).

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.
        n_test_samples (int): The number of test samples.
        n_runs (int, optional): Number of times to run the inference for timing.

    Returns:
        dict: A dictionary containing timing metrics.
    """
    training_id = conf["training_id"]
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    inference_times = measure_inference_time(model, test_loader, n_runs=n_runs)

    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    return {
        "mean_inference_time_per_run": mean_inference_time,
        "std_inference_time_per_run": std_inference_time,
        "num_predictions": n_test_samples,
        "mean_inference_time_per_prediction": mean_inference_time / n_test_samples,
        "std_inference_time_per_prediction": std_inference_time / n_test_samples,
    }


def evaluate_compute(
    model, surr_name: str, test_loader: DataLoader, conf: dict
) -> dict[str, Any]:
    """
    Evaluate the computational resource requirements of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing model complexity metrics.
    """
    training_id = conf["training_id"]
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    device = torch.device(getattr(model, "device", torch.device("cpu")))
    memory_footprint = {}

    if torch.cuda.is_available():
        try:
            inputs = next(iter(test_loader))
        except StopIteration:
            inputs = None

        if inputs is None:
            print(
                "Skipping GPU memory profiling for compute evaluation "
                "(test loader provided no samples)."
            )
        else:
            try:
                memory_footprint, model = measure_memory_footprint(
                    model, inputs, device
                )
            except RuntimeError as exc:
                print(
                    "Skipping GPU memory profiling for compute evaluation "
                    f"(reason: {exc})."
                )
    else:
        print(
            "Skipping GPU memory profiling for compute evaluation "
            "(CUDA not available)."
        )

    # Count the number of trainable parameters
    num_params = count_trainable_parameters(model)

    # Store complexity metrics
    complexity_metrics = {
        "num_trainable_parameters": num_params,
        "memory_footprint": memory_footprint,
    }

    return complexity_metrics


def evaluate_interpolation(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate the interpolation performance of the surrogate model in log-space (Δdex).

    Predictions and targets are kept in log10 space (leave_log=True).
    Errors are computed as absolute log differences (Δdex).

    Args:
        model: Surrogate model instance.
        surr_name (str): Name of the surrogate.
        test_loader (DataLoader): DataLoader with test data.
        timesteps (np.ndarray): Timesteps array.
        conf (dict): Configuration dictionary.
        labels (list, optional): Labels for the predicted quantities.

    Returns:
        dict: Interpolation metrics in log-space.
    """
    training_id = conf["training_id"]
    intervals = conf["interpolation"]["intervals"]
    intervals = np.sort(np.array(intervals, dtype=int))
    intervals = intervals[intervals > 1]
    intervals = np.insert(intervals, 0, 1)

    interpolation_metrics = {}
    errors = np.zeros((len(intervals), len(timesteps)))

    for interval in intervals:
        model_id = (
            f"{surr_name.lower()}_main"
            if interval == 1
            else f"{surr_name.lower()}_interpolation_{interval}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)

        # log-space predictions
        preds_log, targets_log = model.predict(test_loader, leave_log=True)
        preds_log = preds_log.detach().cpu().numpy()
        targets_log = targets_log.detach().cpu().numpy()

        # Δdex errors
        abs_errors = np.abs(preds_log - targets_log)
        mean_abs_errors_time = np.mean(abs_errors, axis=(0, 2))
        errors[intervals == interval] = mean_abs_errors_time

        mean_abs_error = np.mean(mean_abs_errors_time)
        interpolation_metrics[f"interval {interval}"] = {"MAE_log": mean_abs_error}

        if interval == 1:
            preds_main = preds_log

        if interval == intervals[-1]:
            preds_last = preds_log
            MAE_last = np.mean(abs_errors, axis=(1, 2))
            sample_idx = np.argmin(np.abs(MAE_last - np.median(MAE_last)))
            targets_last = targets_log

    # Collect metrics for plotting
    model_errors = np.array(
        [metric["MAE_log"] for metric in interpolation_metrics.values()]
    )
    interpolation_metrics["model_errors"] = model_errors
    interpolation_metrics["intervals"] = intervals

    # Plots
    plot_average_errors_over_time(
        surr_name,
        conf,
        errors,
        intervals,
        timesteps,
        mode="interpolation",
        save=True,
        show_title=TITLE,
    )
    plot_example_mode_predictions(
        surr_name,
        conf,
        preds_last,
        preds_main,
        targets_last,
        timesteps,
        metric=intervals[-1],
        labels=labels,
        mode="interpolation",
        example_idx=sample_idx,
        save=True,
        show_title=TITLE,
    )

    return interpolation_metrics


def evaluate_extrapolation(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate the extrapolation performance of the surrogate model in log-space (Δdex).

    Predictions and targets are kept in log10 space (leave_log=True).
    Errors are computed as absolute log differences (Δdex).

    Args:
        model: Surrogate model instance.
        surr_name (str): Name of the surrogate.
        test_loader (DataLoader): DataLoader with test data.
        timesteps (np.ndarray): Timesteps array.
        conf (dict): Configuration dictionary.
        labels (list, optional): Labels for the predicted quantities.

    Returns:
        dict: Extrapolation metrics in log-space.
    """
    training_id = conf["training_id"]
    cutoffs = conf["extrapolation"]["cutoffs"]
    cutoffs = np.sort(np.array(cutoffs, dtype=int))
    max_cut = len(timesteps)
    cutoffs = cutoffs[cutoffs < max_cut]
    cutoffs = np.append(cutoffs, max_cut)

    extrapolation_metrics = {}
    errors = np.zeros((len(cutoffs), len(timesteps)))

    for cutoff in cutoffs:
        model_id = (
            f"{surr_name.lower()}_main"
            if cutoff == max_cut
            else f"{surr_name.lower()}_extrapolation_{cutoff}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)

        preds_log, targets_log = model.predict(test_loader, leave_log=True)
        preds_log, targets_log = (
            preds_log.detach().cpu().numpy(),
            targets_log.detach().cpu().numpy(),
        )

        abs_errors = np.abs(preds_log - targets_log)  # Δdex
        mean_abs_errors_time = np.mean(abs_errors, axis=(0, 2))
        errors[cutoffs == cutoff] = mean_abs_errors_time
        mean_abs_error = np.mean(mean_abs_errors_time)

        extrapolation_metrics[f"cutoff {cutoff}"] = {"MAE_log": mean_abs_error}

        if cutoff == cutoffs[0]:
            preds_first = preds_log
            sample_MAE = np.mean(abs_errors, axis=(1, 2))
            sample_idx = np.argmin(np.abs(sample_MAE - np.median(sample_MAE)))
            targets_first = targets_log

        if cutoff == cutoffs[-1]:
            preds_main = preds_log

    extrapolation_metrics["model_errors"] = np.array(
        [m["MAE_log"] for m in extrapolation_metrics.values()]
    )
    extrapolation_metrics["cutoffs"] = cutoffs

    plot_average_errors_over_time(
        surr_name,
        conf,
        errors,
        cutoffs,
        timesteps,
        mode="extrapolation",
        save=True,
        show_title=TITLE,
    )
    plot_example_mode_predictions(
        surr_name,
        conf,
        preds_first,
        preds_main,
        targets_first,
        timesteps,
        metric=cutoffs[0],
        labels=labels,
        mode="extrapolation",
        example_idx=sample_idx,
        save=True,
        show_title=TITLE,
    )

    return extrapolation_metrics


def evaluate_sparse(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    n_train_samples: int,
    conf: dict,
) -> dict[str, Any]:
    """
    Evaluate the sparse-data training performance of the surrogate model in log-space (Δdex).

    Predictions and targets are kept in log10 space (leave_log=True).
    Errors are computed as absolute log differences (Δdex).

    Args:
        model: Surrogate model instance.
        surr_name (str): Name of the surrogate.
        test_loader (DataLoader): DataLoader with test data.
        timesteps (np.ndarray): Timesteps array.
        n_train_samples (int): Number of training samples in the full dataset.
        conf (dict): Configuration dictionary.

    Returns:
        dict: Sparse training metrics in log-space.
    """
    training_id = conf["training_id"]
    factors = conf["sparse"]["factors"]
    factors = np.sort(np.array(factors, dtype=int))
    factors = factors[factors > 1]
    factors = np.insert(factors, 0, 1)

    sparse_metrics = {}
    errors = np.zeros((len(factors), len(timesteps)))

    maes = []
    samples = []

    for factor in factors:
        model_id = (
            f"{surr_name.lower()}_main"
            if factor == 1
            else f"{surr_name.lower()}_sparse_{factor}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)

        preds_log, targets_log = model.predict(test_loader, leave_log=True)
        preds_log, targets_log = (
            preds_log.detach().cpu().numpy(),
            targets_log.detach().cpu().numpy(),
        )

        abs_errors = np.abs(preds_log - targets_log)
        mean_abs_errors_time = np.mean(abs_errors, axis=(0, 2))
        errors[factors == factor] = mean_abs_errors_time
        mean_abs_error = np.mean(mean_abs_errors_time)

        train_samples = n_train_samples // factor

        # store per-factor metrics
        sparse_metrics[f"factor {factor}"] = {
            "MAE_log": mean_abs_error,
            "n_train_samples": train_samples,
        }

        maes.append(mean_abs_error)
        samples.append(train_samples)

    # now add clean summary arrays
    sparse_metrics["model_errors"] = np.array(maes)
    sparse_metrics["n_train_samples"] = np.array(samples)

    plot_average_errors_over_time(
        surr_name,
        conf,
        errors,
        sparse_metrics["n_train_samples"],
        timesteps,
        mode="sparse",
        save=True,
        show_title=TITLE,
    )

    return sparse_metrics


def evaluate_batchsize(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
) -> dict[str, Any]:
    """
    Evaluate the batch-size scaling performance of the surrogate model in log-space (Δdex).

    Predictions and targets are kept in log10 space (leave_log=True).
    Errors are computed as absolute log differences (Δdex).

    Args:
        model: Surrogate model instance.
        surr_name (str): Name of the surrogate.
        test_loader (DataLoader): DataLoader with test data.
        timesteps (np.ndarray): Timesteps array.
        conf (dict): Configuration dictionary.

    Returns:
        dict: Batch-size scaling metrics in log-space.
    """
    training_id = conf["training_id"]
    batch_factors = conf["batch_scaling"]["sizes"].copy()
    batch_metrics = {}

    model_idx = conf["surrogates"].index(surr_name)
    main_batch_size = conf["batch_size"][model_idx]

    batch_sizes = [
        int(main_batch_size * batch_factor_to_float(bf)) for bf in batch_factors
    ]
    if main_batch_size not in batch_sizes:
        batch_sizes.append(main_batch_size)
        batch_sizes = sorted(batch_sizes)

    errors = np.zeros((len(batch_sizes), len(timesteps)))

    for i, batch_size in enumerate(batch_sizes):
        model_id = (
            f"{surr_name.lower()}_main"
            if batch_size == main_batch_size
            else f"{surr_name.lower()}_batchsize_{batch_size}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)

        # infer elements per batch from a dummy forward pass
        dummy_inputs = next(iter(test_loader))
        dummy_outputs, _ = model(dummy_inputs)
        if dummy_outputs.ndim == 2:
            batch_elements = batch_size
        elif dummy_outputs.ndim == 3:
            batch_elements = batch_size * model.n_timesteps
        else:
            raise ValueError(
                "Unexpected model output shape. This eval function may have to be updated."
            )

        preds_log, targets_log = model.predict(test_loader, leave_log=True)
        preds_log, targets_log = (
            preds_log.detach().cpu().numpy(),
            targets_log.detach().cpu().numpy(),
        )

        abs_errors = np.abs(preds_log - targets_log)
        mean_abs_errors_time = np.mean(abs_errors, axis=(0, 2))
        errors[i] = mean_abs_errors_time
        mean_abs_error = np.mean(mean_abs_errors_time)

        batch_metrics[f"batch_size {batch_size}"] = {
            "MAE_log": mean_abs_error,
            "batch_elements": batch_elements,
        }

    # after the loop
    batch_metrics["model_errors"] = np.array(
        [v["MAE_log"] for k, v in batch_metrics.items() if k.startswith("batch_size")]
    )
    batch_metrics["batch_elements"] = np.array(
        [
            v["batch_elements"]
            for k, v in batch_metrics.items()
            if k.startswith("batch_size")
        ]
    )

    plot_average_errors_over_time(
        surr_name,
        conf,
        errors,
        batch_metrics["batch_elements"],  # use elements, not nominal batch size
        timesteps,
        mode="batchsize",
        save=True,
        show_title=TITLE,
    )

    return batch_metrics


def evaluate_UQ(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate the uncertainty quantification (UQ) performance of the surrogate model.

    Predictions and targets are kept in log10 space (leave_log=True).
    All UQ metrics are computed in log space (Δdex).

    Args:
        model: The surrogate model instance.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): DataLoader with the test data.
        timesteps (np.ndarray): Array of timesteps.
        conf (dict): Configuration dictionary.
        labels (list, optional): Labels for the predicted quantities.

    Returns:
        dict: Dictionary containing log-space UQ metrics and arrays.
    """
    training_id = conf["training_id"]
    n_models = conf["uncertainty"]["ensemble_size"]

    all_log_preds = []
    for i in range(n_models):
        model_id = (
            f"{surr_name.lower()}_main" if i == 0 else f"{surr_name.lower()}_UQ_{i}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)
        preds_log, targets_log = model.predict(test_loader, leave_log=True)
        preds_log = preds_log.detach().cpu().numpy()
        targets_log = targets_log.detach().cpu().numpy()
        all_log_preds.append(preds_log)

    all_log_preds = np.array(all_log_preds)

    log_mean = np.mean(all_log_preds, axis=0)
    log_std = np.std(all_log_preds, axis=0, ddof=1)

    # log-space absolute errors (Δdex)
    log_errors = np.abs(log_mean - targets_log)

    # summaries
    mae_log = np.mean(log_errors)  # mean Δdex
    medae_log = np.median(log_errors)  # median Δdex
    pae99_log = np.percentile(log_errors, 99)  # 99th percentile Δdex
    avg_uncertainty_log = np.mean(log_std)  # mean predictive log-σ
    log_errors_time = np.mean(log_errors, axis=(0, 2))
    log_std_time = np.mean(log_std, axis=(0, 2))
    avg_correlation, _ = pearsonr(log_errors.flatten(), log_std.flatten())

    # "relative" in log-space == Δdex
    rel_errors = log_errors
    weighted_diff = (log_std - log_errors) / np.maximum(log_errors, 1e-12)

    errs = np.mean(log_errors, axis=(1, 2))
    example_idx = int(np.argsort(np.abs(errs - np.median(errs)))[0])

    # plots
    plot_example_predictions_with_uncertainty(
        surr_name,
        conf,
        log_mean,
        log_std,
        targets_log,
        timesteps,
        example_idx=example_idx,
        save=True,
        labels=labels,
        show_title=True,
    )
    plot_average_uncertainty_over_time(
        surr_name,
        conf,
        log_errors_time,
        log_std_time,
        timesteps,
        save=True,
        show_title=True,
    )
    max_counts, axis_max = plot_uncertainty_heatmap(
        surr_name,
        conf,
        log_std,
        log_errors,
        avg_correlation,
        save=True,
        show_title=True,
    )

    uq_metrics = {
        "average_uncertainty_log": avg_uncertainty_log,
        "MAE_log": mae_log,
        "median_absolute_error_log": medae_log,
        "percentile_absolute_error_log": pae99_log,
        "correlation_metrics_log": avg_correlation,
        "pred_uncertainty_log": log_std,
        "absolute_errors_log": log_errors,
        "relative_errors_log": rel_errors,
        "weighted_diff_log": weighted_diff,
        "max_counts": max_counts,
        "axis_max": axis_max,
        "targets_log": targets_log,
    }
    return uq_metrics


def compare_models(metrics: dict, config: dict):
    """
    Aggregate surrogate-level metrics into comparison plots and CSV summaries.

    Args:
        metrics (dict): Output of :func:`run_benchmark` for each surrogate.
        config (dict): Benchmark configuration that determines which comparisons to run.

    The function is intentionally side-effect heavy: it creates plots, writes tables,
    and prints status messages. Nothing is returned; all artifacts are stored under
    `plots/<training_id>` and `results/<training_id>`.
    """

    print("Making comparative plots... \n")

    # Compare errors
    compare_errors(metrics, config)
    if config.get("losses", False):
        compare_main_losses(metrics, config)

    if config.get("iterative", False):
        compare_iterative(metrics, config)

    if config.get("gradients", False):
        compare_gradients(metrics, config)

    # Compare inference time
    if config.get("timing", False):
        compare_inference_time(metrics, config)

    # Compare interpolation errors
    if config.get("interpolation", {}).get("enabled", False):
        compare_interpolation(metrics, config)

    # Compare extrapolation errors
    if config.get("extrapolation", {}).get("enabled", False):
        compare_extrapolation(metrics, config)

    # Compare sparse training errors
    if config.get("sparse", {}).get("enabled", False):
        compare_sparse(metrics, config)

    if (
        config.get("interpolation", {}).get("enabled", False)
        and config.get("extrapolation", {}).get("enabled", False)
        and config.get("sparse", {}).get("enabled", False)
    ):
        plot_all_generalization_errors(metrics, config, show_title=TITLE)

    # Compare batch size training errors
    if config.get("batch_scaling", {}).get("enabled", False):
        compare_batchsize(metrics, config)

    # Compare UQ metrics
    if config.get("uncertainty", {}).get("enabled", False):
        compare_UQ(metrics, config)
        # rel_errors_and_uq(metrics, config)

    tabular_comparison(metrics, config)


def compare_main_losses(metrics: dict, config: dict) -> None:
    """
    Compare the training and test losses of the main models for different surrogate models.

    Args:
        metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    train_losses = []
    test_losses = []
    train_durations = []
    labels = []
    device = config["devices"]
    device = device[0] if isinstance(device, list) else device

    for surr_name, _ in metrics.items():
        training_id = config["training_id"]
        surrogate_class = get_surrogate(surr_name)
        n_timesteps = metrics[surr_name]["timesteps"].shape[0]
        n_quantities = metrics[surr_name]["accuracy"]["absolute_errors"].shape[2]
        n_params = metrics[surr_name]["n_params"]
        model_config = get_model_config(surr_name, config)
        model = surrogate_class(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
            n_parameters=n_params,
            config=model_config,
        )

        def load_losses(model_identifier: str):
            model.load(training_id, surr_name, model_identifier=model_identifier)
            return model.train_loss, model.test_loss

        # Load main model losses
        main_train_loss, main_test_loss = load_losses(f"{surr_name.lower()}_main")
        train_losses.append(main_train_loss)
        test_losses.append(main_test_loss)
        labels.append(surr_name)
        train_durations.append(model.train_duration)

    # Plot the comparison of main model losses
    plot_loss_comparison(
        tuple(train_losses), tuple(test_losses), tuple(labels), config, show_title=TITLE
    )
    plot_loss_comparison_equal(
        tuple(train_losses), tuple(test_losses), tuple(labels), config, show_title=TITLE
    )
    # plot_MAE_comparison(MAE, labels, config)
    plot_loss_comparison_train_duration(
        test_losses, labels, train_durations, config, show_title=TITLE
    )


def compare_errors(metrics: dict[str, dict], config: dict) -> None:
    """
    Compare relative errors and Δdex errors over time for different surrogate models.

    Args:
        metrics (dict): Benchmark metrics for each surrogate.
        config (dict): Configuration dictionary.
    """
    rel_errors, log_errors = {}, {}
    mean_rel, median_rel = {}, {}
    mean_log, median_log = {}, {}

    for surrogate, surrogate_metrics in metrics.items():
        re = surrogate_metrics["accuracy"].get("relative_errors")
        de = surrogate_metrics["accuracy"].get("absolute_errors_log")
        if re is not None:
            rel_errors[surrogate] = re
            mean_rel[surrogate] = np.mean(re, axis=(0, 2))
            median_rel[surrogate] = np.median(re, axis=(0, 2))
            timesteps = surrogate_metrics["timesteps"]
        if de is not None:
            log_errors[surrogate] = de
            mean_log[surrogate] = np.mean(de, axis=(0, 2))
            median_log[surrogate] = np.median(de, axis=(0, 2))

    if rel_errors:
        plot_errors_over_time(mean_rel, median_rel, timesteps, config, mode="relative")
        plot_error_distribution_comparative(rel_errors, config, mode="relative")

    if log_errors:
        plot_errors_over_time(mean_log, median_log, timesteps, config, mode="deltadex")
        plot_error_distribution_comparative(log_errors, config, mode="deltadex")


def compare_iterative(metrics: dict[str, dict], config: dict) -> None:
    """
    Compare the iterative prediction errors of different surrogate models.

    Args:
        metrics (dict[str, dict]): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    iterative_errors = {}
    mean_iterative_errors = {}
    median_iterative_errors = {}

    for surrogate, surrogate_metrics in metrics.items():
        if "iterative" in surrogate_metrics:
            iterative_errors[surrogate] = surrogate_metrics["iterative"][
                "absolute_errors_log"
            ]
            mean_iterative_errors[surrogate] = np.mean(
                iterative_errors[surrogate], axis=(0, 2)
            )
            median_iterative_errors[surrogate] = np.median(
                iterative_errors[surrogate], axis=(0, 2)
            )

    plot_errors_over_time(
        mean_iterative_errors,
        median_iterative_errors,
        surrogate_metrics["timesteps"],
        config,
        mode="iterative",
        iter_interval=surrogate_metrics["iterative"]["iteration_interval"],
    )
    plot_error_distribution_comparative(iterative_errors, config, mode="iterative")


def compare_inference_time(
    metrics: dict[str, dict], config: dict, save: bool = True
) -> None:
    """
    Compare the mean inference time of different surrogate models.

    Args:
        metrics (dict[str, dict]): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool, optional): Whether to save the plot. Defaults to True.

    Returns:
        None
    """
    mean_inference_times = {}
    std_inference_times = {}

    for surrogate, surrogate_metrics in metrics.items():
        if "timing" in surrogate_metrics:
            mean_time = surrogate_metrics["timing"].get("mean_inference_time_per_run")
            std_time = surrogate_metrics["timing"].get("std_inference_time_per_run")
            if mean_time is not None and std_time is not None:
                mean_inference_times[surrogate] = mean_time
                std_inference_times[surrogate] = std_time

    surrogates = list(mean_inference_times.keys())
    means = list(mean_inference_times.values())
    stds = list(std_inference_times.values())

    inference_time_bar_plot(surrogates, means, stds, config, save, show_title=TITLE)


def compare_gradients(metrics: dict, config: dict) -> None:
    """
    Compare the gradients of different surrogate models.

    Args:
        metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    gradients = {}
    abs_errors = {}
    max_grads = {}
    max_errors = {}
    max_counts = {}
    corrs = {}

    for surrogate, surrogate_metrics in metrics.items():
        gradients[surrogate] = surrogate_metrics["gradients"]["gradients"]
        abs_errors[surrogate] = surrogate_metrics["accuracy"]["absolute_errors_log"]
        max_grads[surrogate] = surrogate_metrics["gradients"]["max_gradient"]
        max_errors[surrogate] = surrogate_metrics["gradients"]["max_error"]
        max_counts[surrogate] = surrogate_metrics["gradients"]["max_counts"]
        corrs[surrogate] = surrogate_metrics["gradients"]["avg_correlation"]

    plot_comparative_gradient_heatmaps(
        gradients,
        abs_errors,
        corrs,
        max_grads,
        max_errors,
        max_counts,
        config,
        show_title=TITLE,
    )


def compare_interpolation(all_metrics: dict, config: dict) -> None:
    """
    Compare the interpolation errors of different surrogate models.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())
    intervals = []
    model_errors = []

    for surrogate in surrogates:
        if "interpolation" in all_metrics[surrogate]:
            intervals.append(all_metrics[surrogate]["interpolation"]["intervals"])
            model_errors.append(all_metrics[surrogate]["interpolation"]["model_errors"])

    plot_generalization_error_comparison(
        surrogates,
        intervals,
        model_errors,
        "Interpolation Interval",
        "errors_interpolation.png",
        config,
        show_title=TITLE,
    )


def compare_extrapolation(all_metrics: dict, config: dict) -> None:
    """
    Compare the extrapolation errors of different surrogate models.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())
    cutoffs = []
    model_errors = []

    for surrogate in surrogates:
        if "extrapolation" in all_metrics[surrogate]:
            cutoffs.append(all_metrics[surrogate]["extrapolation"]["cutoffs"])
            model_errors.append(all_metrics[surrogate]["extrapolation"]["model_errors"])

    plot_generalization_error_comparison(
        surrogates,
        cutoffs,
        model_errors,
        "Extrapolation Cutoff",
        "errors_extrapolation.png",
        config,
        show_title=TITLE,
    )


def compare_sparse(all_metrics: dict, config: dict) -> None:
    """
    Compare the sparse training errors of different surrogate models.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())
    n_train_samples = []
    model_errors = []

    for surrogate in surrogates:
        if "sparse" in all_metrics[surrogate]:
            n_train_samples.append(all_metrics[surrogate]["sparse"]["n_train_samples"])
            model_errors.append(all_metrics[surrogate]["sparse"]["model_errors"])

    plot_generalization_error_comparison(
        surrogates,
        n_train_samples,
        model_errors,
        "Number of Training Samples",
        "errors_sparse.png",
        config,
        xlog=True,
        show_title=TITLE,
    )


def compare_batchsize(all_metrics: dict, config: dict) -> None:
    """
    Compare the batch size training errors of different surrogate models.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())
    batch_sizes = []
    model_errors = []

    for surrogate in surrogates:
        if "batch_size" in all_metrics[surrogate]:
            batch_sizes.append(all_metrics[surrogate]["batch_size"]["batch_elements"])
            model_errors.append(all_metrics[surrogate]["batch_size"]["model_errors"])

    plot_generalization_error_comparison(
        surrogates,
        batch_sizes,
        model_errors,
        "Batch Size",
        "errors_batch_size.png",
        config,
        xlog=True,
        show_title=TITLE,
    )


def compare_UQ(all_metrics: dict, config: dict) -> None:
    """
    Compare log-space UQ across surrogates, focusing on:
        - Ensemble vs Main errors (Δdex) over time
        - Correlation between log-space uncertainty and errors
        - Catastrophic-error detection from uncertainty thresholds

    Args:
        all_metrics (dict): Benchmark metrics per surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    ensemble_std = {}
    ensemble_std_time = {}
    ensemble_errors = {}
    main_errors = {}
    axis_max = {}
    max_counts = {}
    corrs = {}
    timesteps = None

    for surrogate, sm in all_metrics.items():
        timesteps = sm["timesteps"]
        uq = sm["UQ"]

        # log-space quantities from evaluate_UQ
        ensemble_std[surrogate] = uq["pred_uncertainty_log"]
        ensemble_std_time[surrogate] = np.mean(uq["pred_uncertainty_log"], axis=(0, 2))
        ensemble_errors[surrogate] = uq["absolute_errors_log"]
        corrs[surrogate] = uq["correlation_metrics_log"]
        axis_max[surrogate] = uq["axis_max"]
        max_counts[surrogate] = uq["max_counts"]

        # main model log-space errors from accuracy section
        main_errors[surrogate] = sm["accuracy"]["absolute_errors_log"]

    # Q1: main vs ensemble Δdex (mean over samples/quantities) as time series
    plot_mean_deltadex_over_time_main_vs_ensemble(
        main_errors, ensemble_errors, timesteps, config, save=True, show_title=True
    )

    # Q2a: uncertainty vs error over time (means)
    plot_uncertainty_over_time_comparison(
        ensemble_std_time,
        ensemble_errors,
        timesteps,
        config,
        save=True,
        show_title=True,
    )

    # Q2b: comparative heatmaps of uncertainty vs error
    plot_comparative_error_correlation_heatmaps(
        ensemble_std,
        ensemble_errors,
        corrs,
        axis_max,
        max_counts,
        config,
        save=True,
        show_title=True,
    )

    # Q3: catastrophic detection curves (recall vs fraction flagged)
    plot_catastrophic_detection_curves(
        ensemble_errors,
        ensemble_std,
        config,
        flag_fractions=(0, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        save=True,
        show_title=True,
    )


def tabular_comparison(all_metrics: dict, config: dict) -> None:
    """
    Compare the metrics of different surrogate models in a tabular format.
    Prints a table to the CLI, saves the table into a text file, and saves a CSV file with all metrics.
    Also saves a CSV file with only the metrics that appear in the CLI table.
    """
    print("The results are in! Here is a summary of the benchmark metrics:\n")

    model_names = list(all_metrics.keys())
    headers = ["Metric"] + model_names
    rows = []
    percentile = all_metrics[model_names[0]]["accuracy"]["error_percentile"]

    # Unified metric config
    metric_config = {
        "rmse": {
            "label": "RMSE",
            "path": ("accuracy", "root_mean_squared_error_real"),
            "fmt": lambda v: format_value(v),
            "highlight": "min",
            "required": True,
        },
        "mae": {
            "label": "MAE",
            "path": ("accuracy", "mean_absolute_error_real"),
            "fmt": lambda v: format_value(v),
            "highlight": "min",
            "required": True,
        },
        "medae": {
            "label": "Median AE",
            "path": ("accuracy", "median_absolute_error_real"),
            "fmt": lambda v: format_value(v),
            "highlight": "min",
            "required": True,
        },
        "pae": {
            "label": f"{percentile}th Perc. AE",
            "path": ("accuracy", "percentile_absolute_error_real"),
            "fmt": lambda v: format_value(v),
            "highlight": "min",
            "required": True,
        },
        "rmselog": {
            "label": "RMSE (log)",
            "path": ("accuracy", "root_mean_squared_error_log"),
            "fmt": lambda v: format_value(v, "dex"),
            "highlight": "min",
            "required": True,
        },
        "maelog": {
            "label": "MAE (log)",
            "path": ("accuracy", "mean_absolute_error_log"),
            "fmt": lambda v: format_value(v, "dex"),
            "highlight": "min",
            "required": True,
        },
        "medae_log": {
            "label": "Median AE (log)",
            "path": ("accuracy", "median_absolute_error_log"),
            "fmt": lambda v: format_value(v, "dex"),
            "highlight": "min",
            "required": True,
        },
        "paelog": {
            "label": f"{percentile}th Perc. AE (log)",
            "path": ("accuracy", "percentile_absolute_error_log"),
            "fmt": lambda v: format_value(v, "dex"),
            "highlight": "min",
            "required": True,
        },
        "mre": {
            "label": "MRE",
            "path": ("accuracy", "mean_relative_error"),
            "fmt": lambda v: format_value(v * 100, suffix="%"),
            "highlight": "min",
            "required": True,
        },
        "medre": {
            "label": "Median RE",
            "path": ("accuracy", "median_relative_error"),
            "fmt": lambda v: format_value(v * 100, suffix="%"),
            "highlight": "min",
            "required": True,
        },
        "pre": {
            "label": f"{percentile}th Percentile RE",
            "path": ("accuracy", "percentile_relative_error"),
            "fmt": lambda v: format_value(v * 100, suffix="%"),
            "highlight": "min",
            "required": True,
        },
        "epochs": {
            "label": "Epochs",
            "path": ("accuracy", "main_model_epochs"),
            "fmt": str,
            "highlight": None,
            "required": True,
        },
        "train_time": {
            "label": "Train Time (hh:mm:ss)",
            "path": ("accuracy", "main_model_training_time"),
            "fmt": lambda v: format_seconds(int(v)),
            "highlight": "min",
            "required": True,
        },
        # Optional metrics
        "inference_time": {
            "label": "Inference Times",
            "path": ("timing", "mean_inference_time_per_run"),
            "fmt": None,  # handled separately
            "highlight": "min",
            "required": False,
            "group": "timing",
        },
        "iterative_mae": {
            "label": " Iterative MAE (log)",
            "path": ("iterative", "mean_absolute_error_log"),
            "fmt": lambda v: format_value(v, "dex"),
            "highlight": "min",
            "required": False,
            "group": "iterative",
        },
        "gradient_pcc": {
            "label": "Gradient-Error PCC",
            "path": ("gradients", "avg_correlation"),
            "fmt": lambda v: format_value(v),
            "highlight": None,
            "required": False,
            "group": "gradients",
        },
        "num_params": {
            "label": "# Trainable Params",
            "path": ("compute", "num_trainable_parameters"),
            "fmt": str,
            "highlight": "min",
            "required": False,
            "group": "compute",
        },
        "model_mem": {
            "label": "Model Memory (MB)",
            "path": ("compute", "memory_footprint", "model_memory"),
            "fmt": lambda v: format_value(v / 1024**2, "MB"),
            "highlight": "min",
            "required": False,
            "group": "compute",
        },
        "forward_mem": {
            "label": "Forward Pass Memory (MB)",
            "path": ("compute", "memory_footprint", "forward_memory_nograd"),
            "fmt": lambda v: format_value(v / 1024**2, "MB"),
            "highlight": "min",
            "required": False,
            "group": "compute",
        },
        "avg_uncertainty_log": {
            "label": "Mean Log Uncertainty",
            "path": ("UQ", "average_uncertainty_log"),
            "fmt": lambda v: format_value(v, "dex"),
            "highlight": None,
            "required": False,
            "group": "uncertainty",
        },
        "uq_mae_log": {
            "label": "DE MAE",
            "path": ("UQ", "MAE_log"),
            "fmt": lambda v: format_value(v, "dex"),
            "highlight": "min",
            "required": False,
            "group": "uncertainty",
        },
        "uq_medae_log": {
            "label": "DE Median AE",
            "path": ("UQ", "median_absolute_error_log"),
            "fmt": lambda v: format_value(v, "dex"),
            "highlight": "min",
            "required": False,
            "group": "uncertainty",
        },
        "uq_pae99_log": {
            "label": "DE 99th Perc. AE",
            "path": ("UQ", "percentile_absolute_error_log"),
            "fmt": lambda v: format_value(v, "dex"),
            "highlight": "min",
            "required": False,
            "group": "uncertainty",
        },
        "uq_corr_log": {
            "label": "UQ-Error PCC",
            "path": ("UQ", "correlation_metrics_log"),
            "fmt": lambda v: format_value(v),
            "highlight": "max",
            "required": False,
            "group": "uncertainty",
        },
    }

    # Build table
    for name, cfg in metric_config.items():
        # Skip metrics based on config flags
        group = cfg.get("group")
        if not cfg["required"]:
            if group and not config.get(group, False):
                continue
            # For UQ we have nested config
            if group == "uncertainty" and not config.get("uncertainty", {}).get(
                "enabled", False
            ):
                continue

        # Collect values
        path = cfg["path"]
        values = []
        for metrics in all_metrics.values():
            try:
                v = metrics
                for p in path:
                    v = v[p]
                values.append(v)
            except KeyError:
                values.append(None)

        # Skip entirely if missing
        if all(v is None for v in values):
            continue

        # Determine best index if highlight is enabled
        best_idx = None
        if cfg["highlight"] == "min":
            best_idx = int(np.argmin(values))
        elif cfg["highlight"] == "max":
            best_idx = int(np.argmax(values))
        elif cfg["highlight"] == "minabs":
            best_idx = int(np.argmin(np.abs(values)))

        # Special-case formatting for inference times (mean ± std)
        if name == "inference_time":
            means = [
                m["timing"]["mean_inference_time_per_run"] for m in all_metrics.values()
            ]
            stds = [
                m["timing"]["std_inference_time_per_run"] for m in all_metrics.values()
            ]
            row = [cfg["label"]]
            for i, (mean, std) in enumerate(zip(means, stds)):
                s = format_time(mean, std)
                if best_idx is not None and i == best_idx:
                    s = f"* {s} *"
                row.append(s)
            rows.append(row)
            continue

        # Standard row
        row = [cfg["label"]]
        for i, v in enumerate(values):
            if v is None:
                row.append("—")
                continue
            s = cfg["fmt"](v) if cfg["fmt"] else str(v)
            if best_idx is not None and i == best_idx:
                s = f"* {s} *"
            row.append(s)
        rows.append(row)

    # Print and save
    table = tabulate(rows, headers, tablefmt="simple_grid")
    print(table)
    print()

    txt_path = f"results/{config['training_id']}/metrics_table.txt"
    with open(txt_path, "w") as f:
        with redirect_stdout(f):
            print(table)

    make_comparison_csv(metrics=all_metrics, config=config)
    save_table_csv(headers, rows, config)
