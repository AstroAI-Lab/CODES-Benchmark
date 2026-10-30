import functools
import json
import os
import random
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor
from tqdm import tqdm


def read_yaml_config(config_path):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Parse for "None" strings
    config = parse_for_none(config)
    return config


def get_version_info() -> dict:
    """
    The CODES version and, in a git checkout, the commit and whether the checkout has
    uncommitted changes. Stored with trained models and results, so that they can be
    traced back to the code that produced them.

    Returns:
        dict: "codes_version", "git_commit" and "git_dirty" (None outside a checkout).
    """
    from codes import __version__

    info = {"codes_version": __version__, "git_commit": None, "git_dirty": None}
    repo = Path(__file__).resolve().parents[2]
    if not (repo / ".git").exists():
        return info
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        info["git_dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def time_execution(func):
    """
    Decorator to time the execution of a function and store the duration
    as an attribute of the function.

    For methods, the duration is also stored on the instance as
    `<method name>_duration` (e.g. `model.fit_duration`). The attribute on the
    function is shared by all instances, so it is unreliable when several
    instances run concurrently.

    Args:
        func (callable): The function to be timed.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        wrapper.duration = duration
        if args and hasattr(args[0], "__dict__"):
            setattr(args[0], f"{func.__name__}_duration", duration)
        # tqdm.write(f"{func.__name__} executed in {wrapper.duration:.2f} seconds.")
        return result

    wrapper.duration = None
    return wrapper


def create_model_dir(
    base_dir: str = ".", subfolder: str = "trained", unique_id: str = ""
) -> str:
    """
    Create a directory based on a unique identifier inside a specified subfolder of the base directory.

    Args:
        base_dir (str): The base directory where the subfolder and unique directory will be created.
        subfolder (str): The subfolder inside the base directory to include before the unique directory.
        unique_id (str): A unique identifier to be included in the directory name.

    Returns:
        str: The path of the created unique directory within the specified subfolder.
    """
    full_path = os.path.join(base_dir, subfolder, unique_id)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)

    return full_path


def load_and_save_config(config_path: str = "config.yaml", save: bool = True) -> dict:
    """
    Load configuration from a YAML file and save a copy to the specified directory.

    Args:
        config_path (str): The path to the configuration YAML file.
        save (bool): Whether to save a copy of the configuration file. Default is True.

    Returns:
        dict: The loaded configuration dictionary.
    """
    # Load the configuration
    config = read_yaml_config(config_path)

    if save:
        # Get training ID from the config
        training_id = config["training_id"]

        # Create the directory if it does not exist
        save_dir = os.path.join("trained", training_id)
        os.makedirs(save_dir, exist_ok=True)

        # Copy the config file to the directory
        config_save_path = os.path.join(save_dir, "config.yaml")
        shutil.copyfile(config_path, config_save_path)

    return config


def parse_for_none(dictionary: dict) -> dict:
    """
    Parse a dictionary and replace all strings "None" with None.
    If value is a dictionary, recursively parse the dictionary.

    Args:
        dictionary (dict): The dictionary to parse.

    Returns:
        dictionary (dict): The parsed dictionary.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = parse_for_none(value)
        elif value == "None":
            dictionary[key] = None
    return dictionary


def set_random_seeds(seed: int, device: str) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The random seed to set.
        device (str): The device to use for training, e.g., 'cuda:0'
    """
    # Set the device explicitly in case of GPU:
    if "cuda" in device:
        torch.cuda.device(torch.device(device))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nice_print(message: str, width: int = 80) -> None:
    """
    Print a message in a nicely formatted way with a fixed width.

    Args:
        message (str): The message to print.
        width (int): The width of the printed box. Default is 80.
    """
    # Calculate padding
    padding = (width - len(message) - 2) // 2
    padding_left = padding
    padding_right = padding

    # If message length is odd, add one more space to the right
    if (width - len(message)) % 2 != 0:
        padding_right += 1

    border = "-" * width
    print(
        f"\n{border}\n|{' ' * padding_left}{message}{' ' * padding_right}|\n{border}\n"
    )


def make_description(mode: str, device: str, metric: str, surrogate_name: str) -> str:
    """
    Create a formatted description for the progress bar that ensures consistent alignment.

    Args:
        mode (str): The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
        device (str): The device to use for training (e.g., 'cuda:0').
        metric (str): The specific metric for the mode (e.g., interval, cutoff, factor, batch size).
        surrogate_name (str): The name of the surrogate model.

    Returns:
        str: A formatted description string for the progress bar.
    """
    surrogate_name = surrogate_name.ljust(14)
    mode = mode.ljust(13)
    metric = metric.ljust(2)
    device = device.ljust(6) if device == "" else f"({device})".ljust(8)

    description = f"{surrogate_name} {mode} {metric} {device}"
    return description


def get_progress_bar(tasks: list) -> tqdm:
    """
    Create a progress bar with a specific description.

    Args:
        tasks (list): The list of tasks to be executed.

    Returns:
        tqdm: The created progress bar.
    """
    overall_progress_bar = tqdm(
        total=len(tasks),
        desc=make_description("", "", "", "Overall Progress"),
        position=0,
        leave=True,
        bar_format="{l_bar}{bar} | {n_fmt:>3}/{total_fmt} models trained [elapsed time: {elapsed}]",
    )
    return overall_progress_bar


def make_shuffle_generator() -> torch.Generator:
    """
    Create a random number generator for shuffling a dataset, seeded from the global
    torch RNG.

    Each loader owns its generator, so its shuffling order depends only on the seed set
    before the loader was created, not on other threads drawing from the global RNG
    while models are trained in parallel.

    Returns:
        torch.Generator: The seeded generator.
    """
    generator = torch.Generator()
    generator.manual_seed(int(torch.randint(0, 2**62, (1,)).item()))
    return generator


def worker_init_fn(worker_id):
    """
    Initialize the random seed for each worker in PyTorch DataLoader.

    Args:
        worker_id (int): The worker ID.
    """
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32 - 1
    np.random.seed(np_seed)


def save_task_list(tasks: list, filepath: str) -> None:
    """
    Save a list of tasks to a JSON file. The file is replaced atomically, so an
    interrupted run never leaves a truncated task list behind.

    Args:
        tasks (list): The list of tasks to save.
        filepath (str): The path to the JSON file.
    """
    tmp_path = f"{filepath}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f)
    os.replace(tmp_path, filepath)


def load_task_list(filepath: str | None) -> list:
    """
    Load a list of tasks from a JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        list: The loaded list of tasks
    """
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            tasks = json.load(f)
        return tasks
    else:
        return []


def check_training_status(config: dict) -> tuple[str, bool]:
    """
    Check if the training is already completed by looking for a completion marker file.
    If the training is not complete, compare the configurations and ask for a confirmation if there are differences.

    Args:
        config (dict): The configuration dictionary.
    Returns:
        str: The path to the task list file.
        bool: Whether to copy the configuration file.
    """
    training_id = config["training_id"]
    task_list_filepath = os.path.join(f"trained/{training_id}/train_tasks.json")
    completion_marker_filepath = os.path.join(f"trained/{training_id}/completed.txt")

    # Check if training is already complete
    if os.path.exists(completion_marker_filepath):
        print()
        print("Training is already completed. Exiting. \n")
        sys.exit()
    else:
        # Check if the configuration is different from the saved configuration
        saved_config_path = os.path.join(f"trained/{training_id}/config.yaml")
        if not os.path.exists(saved_config_path):
            return task_list_filepath, True

        saved_config = read_yaml_config(saved_config_path)

        # Check if cuda is available if the device is set to cuda
        for device in config["devices"]:
            if "cuda" in device and not torch.cuda.is_available():
                raise ValueError(
                    "You have selected at least one cuda device, but CUDA is not available. Please adjust the device settings in config.yaml."
                )

        # Check if the configurations are the same
        errors = []
        for key, value in config.items():
            if key not in saved_config:
                errors.append(f"Key '{key}' not found in saved configuration.")
            elif value != saved_config[key]:
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in saved_config[key]:
                            errors.append(
                                f"Key '{sub_key}' not found in '{key}' of saved configuration."
                            )
                        elif sub_value != saved_config[key][sub_key]:
                            errors.append(
                                f"Value of '{sub_key}' in '{key}' is different from saved configuration:\n"
                                f"    Previous: {saved_config[key][sub_key]}, current: {sub_value}"
                            )
                elif key == "devices":
                    print(
                        "Note: Devices differ from saved configuration: \n",
                        f"    Previous: {saved_config[key]}, current: {value}",
                    )
                else:
                    errors.append(
                        f"Value of '{key}' is different from saved configuration:\n"
                        f"    Previous: {saved_config[key]}, current: {value}"
                    )

        print()
        if len(errors) > 0:
            print("The current configuration differs from the saved configuration:")
            for error in errors:
                print(f"  - {error}")
            print(
                "You can overwrite the saved configuration or resume the training with the previous configuration."
            )
            confirmation = input("Overwrite? [y/n]: ")
            if confirmation.lower() == "y":
                print("Overwriting the saved configuration.")
                if os.path.exists(task_list_filepath):
                    os.remove(task_list_filepath)
                copy_config = True
            else:
                print("Continuing training with the previous configuration.")
                nice_print("Resuming training.")
                copy_config = False
        else:
            nice_print("Resuming training.")
            copy_config = True

    return task_list_filepath, copy_config


def determine_batch_size(config, surr_idx, mode, metric):
    """
    Determine the appropriate batch size based on the config, surrogate index, mode, and metric.

    Args:
        config (dict): The configuration dictionary.
        surr_idx (int): Index of the surrogate model in the config.
        mode (str): The benchmark mode (e.g., "main", "batch_size").
        metric (int): Metric used for determining the batch size in "batch_size" mode.

    Returns:
        int: The determined batch size.

    Raises:
        ValueError: If the number of batch sizes does not match the number of surrogates.
    """
    batch_size_config = config["batch_size"]
    if isinstance(batch_size_config, list):
        if len(batch_size_config) != len(config["surrogates"]):
            raise ValueError(
                "The number of provided batch sizes must match the number of surrogate models."
            )
        batch_size = batch_size_config[surr_idx]
    else:
        batch_size = batch_size_config

    if mode == "batchsize":
        # The metric is the batch size factor, so we multiply the base batch size by the metric
        batch_size = int(batch_size * metric)

    return batch_size


def batch_factor_to_float(batch_factor: str | int | float) -> float:
    """
    Convert a batch factor to a float value.

    Args:
        batch_factor (str | int | float): The batch factor to convert.

    Returns:
        float: The converted batch factor as a float.
    """
    try:
        if isinstance(batch_factor, float):
            return batch_factor
        elif isinstance(batch_factor, int):
            return float(batch_factor)
        elif isinstance(batch_factor, str):
            # This should mean that the user specified a fraction like "1/2"
            if "/" in batch_factor:
                return float(Fraction(batch_factor))
            else:
                return float(batch_factor)
    except ValueError as e:
        raise ValueError(
            f"Invalid batch factor: {batch_factor}. Must be a float, int, or a valid fraction string."
        ) from e


def parse_hyperparameters(hyperparams: dict) -> dict:
    for key, value in hyperparams.items():
        if isinstance(value, np.ndarray):
            hyperparams[key] = value.tolist()
        elif isinstance(value, Tensor):
            hyperparams[key] = value.cpu().detach().numpy().tolist()
        elif isinstance(value, (np.number, np.bool_)):
            # Handle numpy scalars (like numpy.float32, numpy.int64, etc.)
            hyperparams[key] = value.item()
        elif isinstance(value, dict):
            hyperparams[key] = parse_hyperparameters(value)
        elif isinstance(value, (list, tuple)):
            # Recursively handle lists and tuples that might contain numpy objects
            hyperparams[key] = [
                (
                    item.item()
                    if isinstance(item, (np.number, np.bool_))
                    else (
                        item.tolist()
                        if isinstance(item, np.ndarray)
                        else (
                            item.cpu().detach().numpy().tolist()
                            if isinstance(item, Tensor)
                            else item
                        )
                    )
                )
                for item in value
            ]
        elif hasattr(value, "item") and callable(getattr(value, "item")):
            # Catch-all for any object with an .item() method (like numpy scalars)
            try:
                hyperparams[key] = value.item()
            except (ValueError, TypeError):
                # If .item() fails, convert to string as fallback
                hyperparams[key] = str(value)
        elif not isinstance(value, (str, int, float, bool, type(None))):
            # Convert any remaining non-serializable objects to string
            hyperparams[key] = str(value)
    return hyperparams
