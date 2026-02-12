import csv
import importlib.util
import inspect
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from codes.surrogates import SurrogateModel, surrogate_classes
from codes.utils import read_yaml_config


def check_surrogate(surrogate: str, conf: dict) -> None:
    """
    Check whether the required models for the benchmark are present in the expected directories.

    Args:
        surrogate (str): The name of the surrogate model to check.
        conf (dict): The configuration dictionary.

    Raises:
        FileNotFoundError: If any required models are missing.
    """
    training_id = conf["training_id"]
    base_dir = os.getcwd()
    base_dir = os.path.join(base_dir, "trained", training_id, surrogate)

    required_models = get_required_models_list(surrogate, conf)

    for model_name in required_models:
        model_path = os.path.join(base_dir, f"{model_name}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Required model {model_name} for surrogate {surrogate} not found in {base_dir}"
            )

    print(f"All required models for surrogate {surrogate} are present.")


def check_benchmark(conf: dict) -> None:
    """
    Check whether there are any configuration issues with the benchmark.

    Args:
        conf (dict): The configuration dictionary.

    Raises:
        FileNotFoundError: If the training ID directory is missing or if the .yaml file is missing.
        ValueError: If the configuration is missing required keys or the values do not match the training configuration.
    """
    # Check whether cuda devices have been selected and if they are available
    for device in conf.get("devices", []):
        if "cuda" in device:
            if not torch.cuda.is_available():
                raise ValueError(
                    "You have selected at least one cuda device, but CUDA is not available. Please adjust the device settings in config.yaml."
                )
    # Check for the training directory and load the training configuration
    print("\nChecking benchmark configuration...")
    training_id = conf.get("training_id")
    if not training_id:
        raise ValueError("Configuration must include a 'training_id'.")

    trained_dir = os.path.join(os.getcwd(), "trained", training_id)
    if not os.path.exists(trained_dir):
        raise FileNotFoundError(f"Training ID directory {training_id} not found.")

    yaml_file = os.path.join(trained_dir, "config.yaml")
    if not os.path.isfile(yaml_file):
        raise FileNotFoundError(
            f"Training configuration file not found in directory {trained_dir}."
        )

    training_conf = read_yaml_config(yaml_file)

    # Check Surrogates
    training_surrogates = set(training_conf.get("surrogates", []))
    benchmark_surrogates = set(conf.get("surrogates", []))
    if not benchmark_surrogates.issubset(training_surrogates):
        raise ValueError(
            "Benchmark configuration includes surrogates that were not in the training configuration."
        )

    # Check Batch Size
    if "batch_size" in conf:
        training_batch_size = training_conf.get("batch_size", [])
        benchmark_batch_size = conf.get("batch_size", [])

        # Check if batch sizes correspond to the correct surrogates
        for i, surrogate in enumerate(conf.get("surrogates", [])):
            if surrogate in training_conf["surrogates"]:
                index = training_conf["surrogates"].index(surrogate)
                if training_batch_size[index] != benchmark_batch_size[i]:
                    print(
                        f"Warning: Batch size for surrogate '{surrogate}' has changed from {training_batch_size[index]} to {benchmark_batch_size[i]}."
                    )
                    # Get user input to confirm the change
                    user_input = input("Do you want to continue? (y/n): ")
                    if user_input.lower() == "y":
                        print(f"Continuing with batch size {benchmark_batch_size[i]}.")
                    else:
                        print("Exiting...")
                        exit()

    # Check Dataset Settings
    training_dataset = training_conf.get("dataset", {})
    benchmark_dataset = conf.get("dataset", {})

    # Check if any dataset keys or values do not match
    for key, training_value in training_dataset.items():
        benchmark_value = benchmark_dataset.get(key)
        if benchmark_value != training_value:
            raise ValueError(
                f"Dataset setting '{key}' does not match between training and benchmark configurations. "
                f"Training value: {training_value}, Benchmark value: {benchmark_value}."
            )

    # Check if there are any additional keys in the benchmark dataset not present in training
    for key in benchmark_dataset.keys():
        if key not in training_dataset:
            raise ValueError(
                f"Additional dataset setting '{key}' found in benchmark configuration that is not present in training configuration."
            )

    # Check Modalities (Interpolation, Extrapolation, Sparse, Batch Scaling, Uncertainty)
    modalities = [
        "interpolation",
        "extrapolation",
        "sparse",
        "batch_scaling",
        "uncertainty",
    ]
    for modality in modalities:
        training_modality = training_conf.get(modality, {})
        benchmark_modality = conf.get(modality, {})

        # Check if enabled state has changed incorrectly
        if benchmark_modality.get("enabled", False) and not training_modality.get(
            "enabled", False
        ):
            raise ValueError(
                f"Modality '{modality}' is enabled in benchmark but was not enabled in training."
            )

        # Check values within each modality
        if training_modality.get("enabled", False):
            for key, value in benchmark_modality.items():
                if key == "enabled":
                    continue
                if key not in training_modality:
                    raise ValueError(
                        f"Benchmark configuration provides a value for '{key}' in '{modality}' not present in training."
                    )
                if isinstance(value, list):
                    if not set(value).issubset(set(training_modality.get(key, []))):
                        raise ValueError(
                            f"Benchmark configuration provides values for '{key}' in '{modality}' not trained for."
                        )
                else:
                    if modality == "uncertainty" and key == "ensemble_size":
                        if value > training_modality.get(key, value):
                            raise ValueError(
                                f"Benchmark ensemble_size for '{modality}' cannot be larger than in training."
                            )
                    else:
                        if value != training_modality.get(key):
                            raise ValueError(
                                f"Benchmark configuration value for '{key}' in '{modality}' does not match training configuration."
                            )

    print("Configuration check passed successfully.")


def get_required_models_list(surrogate: str, conf: dict) -> list:
    """
    Generate a list of required models based on the configuration settings.

    Args:
        surrogate (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.

    Returns:
        list: A list of required model names.
    """
    required_models = []
    required_models.append(f"{surrogate.lower()}_main.pth")

    # Gradients does not require a separate model
    if conf.get("gradients", False):
        pass

    if conf.get("interpolation", {}).get("enabled", False):
        intervals = conf["interpolation"]["intervals"]
        required_models.extend(
            [
                f"{surrogate.lower()}_interpolation_{interval}.pth"
                for interval in intervals
            ]
        )

    if conf.get("extrapolation", {}).get("enabled", False):
        cutoffs = conf["extrapolation"]["cutoffs"]
        required_models.extend(
            [f"{surrogate.lower()}_extrapolation_{cutoff}.pth" for cutoff in cutoffs]
        )

    if conf.get("sparse", {}).get("enabled", False):
        factors = conf["sparse"]["factors"]
        required_models.extend(
            [f"{surrogate.lower()}_sparse_{factor}.pth" for factor in factors]
        )

    if conf.get("uncertainty", {}).get("enabled", False):
        n_models = conf["uncertainty"]["ensemble_size"]
        required_models.extend(
            [f"{surrogate.lower()}_UQ_{i + 1}.pth" for i in range(n_models - 1)]
        )

    return required_models


def load_model(
    model, training_id: str, surr_name: str, model_identifier: str
) -> torch.nn.Module:
    """
    Load a trained surrogate model.

    Args:
        model: Instance of the surrogate model class.
        training_id (str): The training identifier.
        surr_name (str): The name of the surrogate model.
        model_identifier (str): The identifier of the model (e.g., 'main').

    Returns:
        The loaded surrogate model.
    """
    statedict_path = os.path.join(
        "trained", training_id, surr_name, f"{model_identifier}.pth"
    )
    model.load_state_dict(torch.load(statedict_path))
    model.eval()
    return model


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory_footprint(
    model: torch.nn.Module, inputs: tuple, device: torch.device
) -> dict:
    """
    Measure peak GPU memory usage for forward/backward passes.

    Args:
        model (torch.nn.Module): Model to profile.
        inputs (tuple): Sample batch used for the forward pass.

    Returns:
        tuple[dict, torch.nn.Module]: Dictionary with ``model_memory``, ``forward_memory``,
        ``backward_memory``, and ``forward_memory_nograd`` plus the model (moved back to its
        original device).
    """
    # Determine the target device
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("CUDA device required for memory profiling.")

    model.to("cpu")

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before_load = torch.cuda.memory_allocated(device)

    model.to(device)
    torch.cuda.synchronize(device)
    peak_after_load = torch.cuda.max_memory_allocated(device)
    model_memory = peak_after_load - before_load

    # Prepare inputs: move them to the target device
    if isinstance(inputs, (list, tuple)):
        inputs = tuple((i.to(device) if i is not None else i) for i in inputs)
    else:
        inputs = inputs.to(device)

    # --- Forward pass with gradients ---
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before_forward = torch.cuda.memory_allocated(device)

    preds, targets = model(inputs=inputs)
    torch.cuda.synchronize(device)
    forward_peak = torch.cuda.max_memory_allocated(device)
    forward_memory = forward_peak - before_forward

    # --- Backward pass ---
    loss = (preds - targets).sum()  # Example loss computation
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before_backward = torch.cuda.memory_allocated(device)

    loss.backward()
    torch.cuda.synchronize(device)
    backward_peak = torch.cuda.max_memory_allocated(device)
    backward_memory = backward_peak - before_backward

    # --- Forward pass without gradients ---
    model.zero_grad()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before_forward_nograd = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        preds, targets = model(inputs=inputs)
    torch.cuda.synchronize(device)
    forward_nograd_peak = torch.cuda.max_memory_allocated(device)
    forward_memory_nograd = forward_nograd_peak - before_forward_nograd

    memory_usage = {
        "model_memory": model_memory,
        "forward_memory": forward_memory,
        "backward_memory": backward_memory,
        "forward_memory_nograd": forward_memory_nograd,
    }

    return memory_usage, model


def convert_to_standard_types(data):
    """
    Recursively convert data to standard types that can be serialized to YAML.

    Args:
        data: The data to convert.

    Returns:
        The converted data.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {k: convert_to_standard_types(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_to_standard_types(i) for i in data]
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        return str(data)


def discard_numpy_entries(d: dict) -> dict:
    """
    Recursively remove dictionary entries that contain NumPy arrays.

    Args:
        d (dict): The input dictionary.

    Returns:
        dict: A new dictionary without entries containing NumPy arrays.
    """
    if not isinstance(d, dict):
        return d

    clean_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            cleaned_value = discard_numpy_entries(value)
            if cleaned_value:  # Only add non-empty dictionaries
                clean_dict[key] = cleaned_value
        elif not isinstance(value, np.ndarray):
            clean_dict[key] = value

    return clean_dict


def clean_metrics(metrics: dict, conf: dict) -> dict:
    """
    Clean the metrics dictionary to remove problematic entries.

    Args:
        metrics (dict): The benchmark metrics.
        conf (dict): The configuration dictionary.

    Returns:
        dict: The cleaned metrics dictionary.
    """

    # Make a deep copy of the metrics
    write_metrics = deepcopy(metrics)

    # Remove problematic entries
    write_metrics.pop("timesteps", None)
    write_metrics["accuracy"].pop("absolute_errors", None)
    write_metrics["accuracy"].pop("relative_errors", None)
    write_metrics["accuracy"].pop("absolute_errors_log", None)

    if conf.get("iterative", False):
        write_metrics["iterative"].pop("absolute_errors", None)
        write_metrics["iterative"].pop("absolute_errors_log", None)
    if conf.get("gradients", False):
        write_metrics["gradients"].pop("gradients", None)
        write_metrics["gradients"].pop("max_counts", None)
        write_metrics["gradients"].pop("max_gradient", None)
        write_metrics["gradients"].pop("max_error", None)
        write_metrics["gradients"].pop("errors_log", None)
    if conf.get("interpolation", {}).get("enabled", False):
        write_metrics["interpolation"].pop("model_errors", None)
        write_metrics["interpolation"].pop("intervals", None)
    if conf.get("extrapolation", {}).get("enabled", False):
        write_metrics["extrapolation"].pop("model_errors", None)
        write_metrics["extrapolation"].pop("cutoffs", None)
    if conf.get("sparse", {}).get("enabled", False):
        write_metrics["sparse"].pop("model_errors", None)
        write_metrics["sparse"].pop("n_train_samples", None)
    if conf.get("uncertainty", {}).get("enabled", False):
        write_metrics["UQ"].pop("pred_uncertainty_log", None)
        write_metrics["UQ"].pop("max_counts", None)
        write_metrics["UQ"].pop("axis_max", None)
        write_metrics["UQ"].pop("absolute_errors_log", None)
        write_metrics["UQ"].pop("relative_errors_log", None)
        write_metrics["UQ"].pop("weighted_diff_log", None)
        write_metrics["UQ"].pop("targets_log", None)

    return write_metrics


def write_metrics_to_yaml(surr_name: str, conf: dict, metrics: dict) -> None:
    """
    Write the benchmark metrics to a YAML file.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        metrics (dict): The benchmark metrics.
    """
    # Clean the metrics
    write_metrics = clean_metrics(metrics, conf)

    # Convert metrics to standard types
    write_metrics = convert_to_standard_types(write_metrics)

    # Make results directory
    os.makedirs(f"results/{conf['training_id']}", exist_ok=True)

    with open(
        f"results/{conf['training_id']}/{surr_name.lower()}_metrics.yaml",
        mode="w",
        encoding="utf-8",
    ) as f:
        yaml.dump(write_metrics, f, sort_keys=False)


def get_surrogate(surrogate_name: str) -> SurrogateModel | None:
    """
    Check if the surrogate model exists.

    Args:
        surrogate_name (str): The name of the surrogate model.

    Returns:
        SurrogateModel | None: The surrogate model class if it exists, otherwise None.
    """
    for surrogate in surrogate_classes:
        if surrogate_name == surrogate.__name__:
            return surrogate

    return None


def format_time(mean_time, std_time):
    """
    Format mean and std time consistently in ns, µs, ms, or s.

    Args:
        mean_time: The mean time.
        std_time: The standard deviation of the time.

    Returns:
        str: The formatted time string.
    """
    if mean_time < 1e-6:
        # Both in ns
        return f"{mean_time * 1e9:.2f} ns ± {std_time * 1e9:.2f} ns"
    elif mean_time < 1e-3:
        # Both in µs
        return f"{mean_time * 1e6:.2f} µs ± {std_time * 1e6:.2f} µs"
    elif mean_time < 1:
        # Both in ms
        return f"{mean_time * 1e3:.2f} ms ± {std_time * 1e3:.2f} ms"
    else:
        # Both in s
        return f"{mean_time:.2f} s ± {std_time:.2f} s"


def format_seconds(seconds: int) -> str:
    """
    Format a duration given in seconds as hh:mm:ss.

    Args:
        seconds (int): The duration in seconds.

    Returns:
        str: The formatted duration string.
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def format_value(v: float, suffix: str = "") -> str:
    """
    Format a float with ~3 significant digits, in fixed or scientific notation.
    Optionally append a suffix (e.g. 'dex', 'MB').

    Args:
        v (float): Value to format.
        suffix (str): Unit/suffix to append, e.g. 'dex'. Ignored if as_percent=True.

    Returns:
        str: Formatted string with optional suffix or percent.
    """

    if v == 0 or not math.isfinite(v):
        base = "0" if v == 0 else str(v)
    else:
        av = abs(v)
        if 1e-2 <= av < 1e4:
            e = math.floor(math.log10(av))
            dec = max(0, 3 - 1 - e)
            base = f"{v:.{dec}f}".rstrip("0").rstrip(".")
        else:
            base = f"{v:.2e}"

    return f"{base} {suffix}".strip()


def flatten_dict(d: dict, parent_key: str = "", sep: str = " - ") -> dict:
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key string.
        sep (str): The separator between keys.

    Returns:
        dict: Flattened dictionary with composite keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_dict_to_scientific_notation(d: dict, precision: int = 8) -> dict:
    """
    Convert all numerical values in a dictionary to scientific notation.

    Args:
        d (dict): The input dictionary.

    Returns:
        dict: The dictionary with numerical values in scientific notation.
    """
    return {
        k: f"{v:.{precision}e}" if isinstance(v, (int, float)) else v
        for k, v in d.items()
    }


def make_comparison_csv(metrics: dict, config: dict) -> None:
    """
    Generate a CSV file comparing metrics for different surrogate models.

    Args:
        metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """

    # Clean the metrics
    for surr_name in metrics.keys():
        metrics[surr_name] = clean_metrics(metrics[surr_name], config)

    # cleaned_metrics = convert_to_standard_types(metrics)

    # Flatten the metrics dictionary for each surrogate model
    flattened_metrics = {
        surr_name: flatten_dict(met) for surr_name, met in metrics.items()
    }

    # Convert all numerical values to scientific notation
    for surr_name in flattened_metrics.keys():
        flattened_metrics[surr_name] = convert_dict_to_scientific_notation(
            flattened_metrics[surr_name]
        )

    # Get all unique keys across all models (i.e., all possible metric categories)
    all_keys = set()
    for surr_name, flat_met in flattened_metrics.items():
        all_keys.update(flat_met.keys())

    # Convert the set of keys to a sorted list
    all_keys = sorted(all_keys)

    # Prepare the CSV file path
    csv_file_path = f"results/{config['training_id']}/all_metrics.csv"
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Write to the CSV file
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        header = ["Category"] + list(metrics.keys())
        writer.writerow(header)

        # Write each row with the values corresponding to each model
        for key in all_keys:
            row = [key]
            for surr_name in metrics.keys():
                value = flattened_metrics[surr_name].get(
                    key, "N/A"
                )  # Use 'N/A' if key is missing
                row.append(value)
            writer.writerow(row)

    if config.get("verbose", False):
        print(f"Comparison CSV file saved at {csv_file_path}")


def save_table_csv(headers: list, rows: list, config: dict) -> None:
    """
    Save the CLI table (headers and rows) to a CSV file.
    This version strips out any formatting (like asterisks) from the table cells.

    Args:
        headers (list): The list of header names.
        rows (list): The list of rows, where each row is a list of string values.
        config (dict): Configuration dictionary that contains 'training_id'.

    Returns:
        None
    """
    # Convert each cell to a string and remove asterisks
    cleaned_rows = [
        [str(cell).replace("*", "").strip() for cell in row] for row in rows
    ]

    csv_path = f"results/{config['training_id']}/metrics_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(cleaned_rows)
    print(f"CLI table metrics saved to {csv_path}")


def get_model_config(surr_name: str, config: dict) -> dict:
    """
    Get the model configuration for a specific surrogate model from the dataset folder.
    Returns an empty dictionary if config["dataset"]["use_optimal_params"] is False,
    or if no configuration file is found in the dataset folder.

    Args:
        surr_name (str): The name of the surrogate model.
        config (dict): The configuration dictionary.

    Returns:
        dict: The model configuration dictionary.
    """
    if not config["dataset"].get("use_optimal_params", True):
        return {}

    dataset_name = config["dataset"]["name"].lower()
    dataset_folder = f"datasets/{dataset_name}"
    config_file = f"{dataset_folder}/surrogates_config.py"

    if os.path.exists(config_file):
        spec = importlib.util.spec_from_file_location("config_module", config_file)
        if spec is None:
            raise ImportError(f"Failed to import config module from {config_file}")
        config_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(f"Failed to load config module from {config_file}")
        spec.loader.exec_module(config_module)

        # Look for the dataclass matching the surr_name + 'Config'
        config_class = None
        target_class_name = f"{surr_name}config".lower()
        for name, obj in inspect.getmembers(config_module, inspect.isclass):
            if (
                hasattr(obj, "__dataclass_fields__")
                and name.lower() == target_class_name
            ):
                config_class = obj
                break

        if config_class:
            # Instantiate the dataclass and convert it to a dictionary
            config_instance = config_class()
            model_config = asdict(config_instance)
        else:
            model_config = {}
    else:
        model_config = {}

    return model_config


def measure_inference_time(
    model,
    test_loader: DataLoader,
    n_runs: int = 5,
) -> list[float]:
    """
    Measure total inference time over a DataLoader across multiple runs.

    Args:
        model: Model instance with a `.forward()` method.
        test_loader (DataLoader): Loader with test data.
        n_runs (int): Number of repeated runs for averaging.

    Returns:
        list[float]: List of total inference times per run (in seconds).
    """
    inference_times = []
    for _ in range(n_runs):
        total_time = 0
        with torch.inference_mode():
            for inputs in test_loader:
                start_time = time.perf_counter()
                _, _ = model.forward(inputs)
                end_time = time.perf_counter()
                total_time += end_time - start_time
        inference_times.append(total_time)
    return inference_times
