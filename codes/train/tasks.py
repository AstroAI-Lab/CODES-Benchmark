"""Run individual training tasks independently of their scheduler."""

import os
from contextlib import nullcontext

import torch

from codes.benchmark.bench_utils import get_model_config, get_surrogate
from codes.utils import (
    check_and_load_data,
    determine_batch_size,
    get_data_subset,
    load_and_save_config,
    make_description,
    set_random_seeds,
)


def train_and_save_model(
    surr_name: str,
    mode: str,
    metric: int,
    training_id: str,
    seed: int | None = None,
    epochs: int | None = None,
    device: str = "cpu",
    position: int = 1,
    threadlock=None,
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
        threadlock (optional): Legacy lock used around seeded initialisation.
    """
    surrogate_class = get_surrogate(surr_name)
    if surrogate_class is None:
        raise ValueError(f"Unknown surrogate model: {surr_name!r}")

    lock = threadlock if threadlock is not None else nullcontext()
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

    model_config = get_model_config(surr_name, config)

    with lock:
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

    with lock:
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


def validate_execution_settings(device, cpu_threads):
    """Validate worker settings without selecting a CUDA context.

    CUDA indices are logical indices within the process's visible devices. CPU
    threads must fit within both its affinity mask and any Slurm allocation.
    """
    if (
        isinstance(cpu_threads, bool)
        or not isinstance(cpu_threads, int)
        or cpu_threads < 1
    ):
        raise ValueError("cpu_threads must be a positive integer")

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            available_cpus = len(affinity(0))
        except OSError:
            available_cpus = os.cpu_count()
    else:
        available_cpus = os.cpu_count()
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            slurm_limit = int(slurm_cpus)
            if slurm_limit < 1:
                raise ValueError
        except ValueError as exc:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer") from exc
        available_cpus = (
            min(available_cpus, slurm_limit)
            if available_cpus is not None
            else slurm_limit
        )
    if available_cpus is not None and cpu_threads > available_cpus:
        raise ValueError(
            f"cpu_threads ({cpu_threads}) exceeds available CPUs ({available_cpus})"
        )

    try:
        selected_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training device: {device!r}") from exc
    if selected_device.type not in ("cpu", "cuda", "mps") or (
        selected_device.type == "cpu" and selected_device.index is not None
    ):
        raise ValueError(f"Unsupported training device: {device!r}")

    # Set before checking CUDA, which may initialise the CUDA runtime.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if selected_device.type == "cuda":
        index = selected_device.index if selected_device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {device!r} is not available among visible devices"
            )
    elif selected_device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(f"MPS device {device!r} is not available")
    return selected_device


def execute_training_task(task, *, device="cpu", position=0, cpu_threads=1):
    """Configure a worker process and train and save exactly one model.

    ``task`` contains ``(surrogate, mode, metric, training_id, seed, epochs)``,
    as produced by ``create_task_list_for_surrogate``. The saved configuration
    is read from ``trained/<training_id>/config.yaml``. ``device`` is a logical
    device string within the worker's allocation; ``cpu_threads`` sets PyTorch's
    intra-operation thread count.

    Local spawned processes and future external workers can both call this
    function. It propagates errors and does not update pending-task files or
    completion markers; those belong to the scheduler invoking it.
    """
    selected_device = validate_execution_settings(device, cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(cpu_threads)
    if selected_device.type == "cuda":
        index = selected_device.index if selected_device.index is not None else 0
        torch.cuda.set_device(index)

    return train_and_save_model(*task, device=device, position=position)
