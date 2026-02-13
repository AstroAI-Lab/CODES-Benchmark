import os
import threading
from queue import Queue
from threading import Thread

from tqdm import tqdm

from codes.benchmark.bench_utils import get_model_config, get_surrogate
from codes.utils import (
    batch_factor_to_float,
    check_and_load_data,
    determine_batch_size,
    get_data_subset,
    get_progress_bar,
    load_and_save_config,
    load_task_list,
    make_description,
    save_task_list,
    set_random_seeds,
)


class DummyLock:
    def acquire(self):
        pass

    def release(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def train_and_save_model(
    surr_name: str,
    mode: str,
    metric: int,
    training_id: str,
    seed: int | None = None,
    epochs: int | None = None,
    device: str = "cpu",
    position: int = 1,
    threadlock: threading.Lock = DummyLock(),
):
    """
    Train and save a model for a specific benchmark mode.

    Args:
        surr_name (str): The name of the surrogate model.
        mode (str): The benchmark mode.
        metric (int): The metric for the benchmark mode.
        training_id (str): The training ID for the current training session.
        seed (int, optional): Random seed for training.
        epochs (int, optional): Number of training epochs.
        device (str, optional): Device to run training on.
        position (int, optional): Model position in the task list.
        threadlock (threading.Lock, optional): Lock for deterministic threading.
    """
    config_path = f"trained/{training_id}/config.yaml"
    config = load_and_save_config(config_path, save=False)

    # Load full data and parameters
    (
        (train_data, test_data, _),
        (train_params, test_params, _),
        timesteps,
        _,
        data_info,
        _,
    ) = check_and_load_data(
        config["dataset"]["name"],
        verbose=config.get("verbose", False),
        log=config["dataset"].get("log10_transform", True),
        log_params=config["dataset"].get("log10_transform_params", True),
        normalisation_mode=config["dataset"].get("normalise", "minmax"),
        tolerance=config["dataset"].get("tolerance", None),
        per_species=config["dataset"].get("normalise_per_species", False),
    )

    # Get the appropriate data subset
    (train_data, test_data), (train_params, test_params), timesteps = get_data_subset(
        (train_data, test_data),
        timesteps,
        mode,
        metric,
        (train_params, test_params),
        config["dataset"].get("subset_factor", 1),
    )

    _, n_timesteps, n_quantities = train_data.shape
    n_params = train_params.shape[1] if train_params is not None else 0

    surrogate_class = get_surrogate(surr_name)
    model_config = get_model_config(surr_name, config)

    with threadlock:
        set_random_seeds(seed, device)
        model = surrogate_class(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
            n_parameters=n_params,
            config=model_config,
            training_id=config["training_id"],
        )
    model.normalisation = data_info
    model.checkpointing = config.get("checkpoint", False)
    surr_idx = config["surrogates"].index(surr_name)

    batch_size = determine_batch_size(config, surr_idx, mode, metric)

    with threadlock:
        set_random_seeds(seed, device)
        # Pass the parameter subsets (if any) to prepare_data.
        train_loader, test_loader, _ = model.prepare_data(
            dataset_train=train_data,
            dataset_test=test_data,
            dataset_val=None,
            timesteps=timesteps,
            batch_size=batch_size,
            shuffle=True,
            dataset_train_params=train_params,
            dataset_test_params=test_params,
            dummy_timesteps=True,
        )

    description = make_description(mode, device, str(metric), surr_name)

    model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        position=position,
        description=description,
    )

    metric = batch_size if mode == "batchsize" else metric
    model_name = f"{surr_name.lower()}_{mode}_{str(metric)}".strip("_")
    model_name = model_name.replace("__", "_")
    base_dir = os.path.join(os.getcwd(), "trained")
    model.save(
        model_name=model_name,
        training_id=config["training_id"],
        base_dir=base_dir,
    )


def create_task_list_for_surrogate(config, surr_name: str) -> list:
    """
    Creates a list of training tasks for a specific surrogate model based on the
    configuration file.

    Args:
        config (dict): The configuration dictionary taken from the config file.
        surr_name (str): The name of the surrogate model.

    Returns:
        list: A list of training tasks for the surrogate model.
    """
    tasks = []
    seed = config.get("seed", 42)
    surr_idx = config["surrogates"].index(surr_name)
    id = config["training_id"]
    epochs = (
        config["epochs"][surr_idx]
        if isinstance(config["epochs"], list)
        else config["epochs"]
    )

    tasks.append((surr_name, "main", "", id, seed, epochs))

    interpolation_conf = config.get("interpolation", {})
    if interpolation_conf.get("enabled", False):
        mode = "interpolation"
        for interval in interpolation_conf["intervals"]:
            tasks.append((surr_name, mode, interval, id, seed + interval, epochs))

    extrapolation_conf = config.get("extrapolation", {})
    if extrapolation_conf.get("enabled", False):
        mode = "extrapolation"
        for cutoff in extrapolation_conf["cutoffs"]:
            tasks.append((surr_name, mode, cutoff, id, seed + cutoff, epochs))

    sparse_conf = config.get("sparse", {})
    if sparse_conf.get("enabled", False):
        for factor in sparse_conf["factors"]:
            tasks.append((surr_name, "sparse", factor, id, seed + factor, epochs))

    uncertainty_conf = config.get("uncertainty", {})
    if uncertainty_conf.get("enabled", False):
        n_models = uncertainty_conf["ensemble_size"]
        for i in range(n_models - 1):
            tasks.append((surr_name, "UQ", i + 1, id, seed + i, epochs))

    batch_scaling_conf = config.get("batch_scaling", {})
    if batch_scaling_conf.get("enabled", False):
        mode = "batchsize"
        for bf in batch_scaling_conf["sizes"]:
            bf_index = batch_scaling_conf["sizes"].index(bf)
            bf = batch_factor_to_float(bf)
            tasks.append((surr_name, mode, float(bf), id, seed + bf_index, epochs))

    return tasks


def worker(
    task_queue: Queue,
    device: str,
    device_idx: int,
    overall_progress_bar: tqdm,
    task_list_filepath: str,
    errors_encountered: list[bool],
    threadlock: threading.Lock,
):
    """
    Worker function to process tasks from the task queue on the given device.

    Args:
        task_queue (Queue): The in-memory queue containing the training tasks.
        device (str): The device to use for training.
        device_idx (int): The index of the device in the device list.
        overall_progress_bar (tqdm): The overall progress bar for the training.
        task_list_filepath (str): The filepath to the JSON task list.
        errors_encountered (list[bool]): A shared mutable flag array indicating if an error has occurred
                                         (True if at least one task failed).
        threadlock (threading.Lock): A lock to prevent threading issues with PyTorch.
    """
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()  # Remove the task from the in-memory queue
            train_and_save_model(
                *task, device=device, position=device_idx + 1, threadlock=threadlock
            )

            # Mark that we have successfully processed this task
            task_queue.task_done()
            overall_progress_bar.update(1)
            current_list = load_task_list(task_list_filepath)
            task_as_list = list(task)

            try:
                current_list.remove(task_as_list)
            except ValueError:
                pass

            save_task_list(current_list, task_list_filepath)

        except Exception as e:
            tqdm.write(f"Exception for task {task[:3]}: {e}")
            # Mark this task as "done" for the queue, so the loop can move on
            task_queue.task_done()
            overall_progress_bar.update(1)

            # Flag that at least one task has failed
            errors_encountered[0] = True


def parallel_training(tasks, device_list, task_list_filepath: str):
    """
    Execute the queued training tasks across multiple devices using worker threads.

    Args:
        tasks (list[tuple]): Output of :func:`create_task_list_for_surrogate`.
        device_list (list[str]): Devices allocated to training (e.g. ["cuda:0", "cuda:1"]).
        task_list_filepath (str): Path to the persisted JSON task list that tracks progress.

    Returns:
        float: Elapsed wall-clock time reported by the shared progress bar.
    """
    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)

    errors_encountered = [False]
    overall_progress_bar = get_progress_bar(tasks)

    threads = []
    threadlock = threading.Lock()
    for i, device in enumerate(device_list):
        thread = Thread(
            target=worker,
            args=(
                task_queue,
                device,
                i,
                overall_progress_bar,
                task_list_filepath,
                errors_encountered,
                threadlock,
            ),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    overall_progress_bar.close()
    elapsed_time = overall_progress_bar.format_dict["elapsed"]

    # If no errors, mark training done & remove tasks file
    if not errors_encountered[0]:
        with open(
            os.path.join(os.path.dirname(task_list_filepath), "completed.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("Training completed")
        os.remove(task_list_filepath)
    else:
        tqdm.write(
            "Some tasks failed. The task list file was NOT removed.\n"
            "Please fix the error(s) and re-run to complete the remaining tasks."
        )

    return elapsed_time


def sequential_training(tasks, device_list, task_list_filepath: str):
    """
    Run all training tasks sequentially on a single device.

    Args:
        tasks (list[tuple]): Task specification tuples generated from the config.
        device_list (list[str]): Contains exactly one element (typically \"cpu\" or a single CUDA id).
        task_list_filepath (str): Path to the JSON file used to resume interrupted runs.

    Returns:
        float: Total elapsed time once all tasks finish.
    """
    overall_progress_bar = get_progress_bar(tasks)
    errors_encountered = False
    device = device_list[0]

    for task in tasks:
        try:
            train_and_save_model(*task, device=device)
            overall_progress_bar.update(1)

            # Only remove *this* task from JSON if success
            current_list = load_task_list(task_list_filepath)
            task_as_list = list(task)
            try:
                current_list.remove(task_as_list)
            except ValueError:
                pass
            save_task_list(current_list, task_list_filepath)

        except Exception as e:
            tqdm.write(f"Exception for task {task[:3]}: {e}")
            overall_progress_bar.update(1)
            errors_encountered = True

    elapsed_time = overall_progress_bar.format_dict["elapsed"]
    overall_progress_bar.close()

    if not errors_encountered:
        with open(
            os.path.join(os.path.dirname(task_list_filepath), "completed.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("Training completed")
        os.remove(task_list_filepath)
    else:
        tqdm.write(
            "Some tasks failed. The task list file was NOT removed.\n"
            "Please fix the error(s) and re-run to complete the remaining tasks."
        )

    return elapsed_time
