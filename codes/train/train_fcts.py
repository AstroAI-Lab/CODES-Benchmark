import contextlib
import multiprocessing as mp
import os
import signal
import threading
import traceback
from multiprocessing.connection import wait

from tqdm import tqdm

from codes.train.tasks import execute_training_task, validate_execution_settings
from codes.train.tasks import train_and_save_model as train_and_save_model
from codes.utils import (
    batch_factor_to_float,
    get_progress_bar,
    load_task_list,
    save_task_list,
)


class TrainingRunError(RuntimeError):
    """One or more local training tasks did not complete successfully."""


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
        # Member k uses seed + k; the main model (seed) is member 0.
        for i in range(n_models - 1):
            tasks.append((surr_name, "UQ", i + 1, id, seed + i + 1, epochs))

    batch_scaling_conf = config.get("batch_scaling", {})
    if batch_scaling_conf.get("enabled", False):
        mode = "batchsize"
        for bf in batch_scaling_conf["sizes"]:
            bf_index = batch_scaling_conf["sizes"].index(bf)
            bf = batch_factor_to_float(bf)
            tasks.append((surr_name, mode, float(bf), id, seed + bf_index, epochs))

    return tasks


def _run_task_child(task, device, position, cpu_threads, connection):
    """Run exactly one task and send only a small status message to the parent."""
    try:
        execute_training_task(
            task, device=device, position=position, cpu_threads=cpu_threads
        )
        connection.send((True, ""))
    except BaseException:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            connection.send((False, traceback.format_exc()[-4000:]))
        raise
    finally:
        connection.close()


def _stop_children(active):
    for state in active.values():
        process = state["process"]
        if process.pid is not None and process.is_alive():
            process.terminate()
    for state in active.values():
        process = state["process"]
        if process.pid is not None:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        state["connection"].close()
        state["sender"].close()
        # A process that could not be killed must not make cleanup block forever.
        with contextlib.suppress(ValueError):
            process.close()


def _run_training(tasks, devices, task_list_filepath, cpu_threads):
    tasks = [list(task) for task in tasks]
    devices = list(devices)
    if not devices:
        raise ValueError("At least one training device is required")
    # Preserve entries outside this invocation, including repeated identical tasks.
    pending = load_task_list(task_list_filepath)
    unmatched = list(pending)
    for task in tasks:
        if task in unmatched:
            unmatched.remove(task)
        else:
            pending.append(task)
    save_task_list(pending, task_list_filepath)
    if pending:
        with contextlib.suppress(FileNotFoundError):
            os.remove(
                os.path.join(os.path.dirname(task_list_filepath), "completed.txt")
            )

    for device in devices:
        validate_execution_settings(device, cpu_threads)

    context = mp.get_context("spawn")
    progress = get_progress_bar(tasks)
    active = {}
    available = list(range(len(devices)))
    next_task = 0
    failures = []
    original_handlers = {}
    signal_state = {"defer": False, "pending": None, "cleanup": False}

    def handle_signal(signum, frame):
        if signal_state["cleanup"]:
            return
        if signal_state["defer"]:
            signal_state["pending"] = signal_state["pending"] or (signum, frame)
            return
        signal_state["cleanup"] = True
        if signum == signal.SIGTERM:
            raise SystemExit(128 + signum)
        previous = original_handlers[signal.SIGINT]
        if previous == signal.SIG_IGN:
            signal_state["cleanup"] = False
        elif previous == signal.SIG_DFL:
            signal.default_int_handler(signum, frame)
        else:
            previous(signum, frame)
            signal_state["cleanup"] = False

    # Deferring signals while starting a process ensures it is registered before
    # cleanup can run, even if an interruption arrives inside Process.start().
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)
    try:
        while next_task < len(tasks) or active:
            while available and next_task < len(tasks):
                slot = available.pop(0)
                task = tasks[next_task]
                next_task += 1
                signal_state["defer"] = True
                try:
                    receiver, sender = context.Pipe(duplex=False)
                    process = context.Process(
                        target=_run_task_child,
                        args=(
                            tuple(task),
                            devices[slot],
                            slot + 1,
                            cpu_threads,
                            sender,
                        ),
                    )
                    active[slot] = {
                        "process": process,
                        "connection": receiver,
                        "sender": sender,
                        "task": task,
                        "message": None,
                        "pipe_open": True,
                    }
                    try:
                        process.start()
                    except Exception as exc:
                        # If start partially succeeded, leave it registered for cleanup.
                        if process.pid is not None:
                            raise
                        del active[slot]
                        receiver.close()
                        sender.close()
                        process.close()
                        failures.append((task, f"Could not start process: {exc}"))
                        progress.update(1)
                        available.append(slot)
                    else:
                        sender.close()
                finally:
                    signal_state["defer"] = False
                    if signal_state["pending"] is not None:
                        signum, frame = signal_state["pending"]
                        signal_state["pending"] = None
                        handle_signal(signum, frame)

            if not active:
                continue
            handles = [state["process"].sentinel for state in active.values()]
            handles += [
                state["connection"] for state in active.values() if state["pipe_open"]
            ]
            ready = wait(handles)
            for slot, state in list(active.items()):
                conn = state["connection"]
                process = state["process"]
                if state["pipe_open"] and (conn in ready or process.sentinel in ready):
                    try:
                        if conn.poll():
                            state["message"] = conn.recv()
                    except (EOFError, OSError):
                        state["pipe_open"] = False
                    if process.sentinel in ready or state["message"] is not None:
                        state["pipe_open"] = False
                if process.sentinel not in ready:
                    continue
                process.join()
                message = state["message"]
                if process.exitcode == 0 and message is not None and message[0] is True:
                    new_pending = list(pending)
                    new_pending.remove(state["task"])
                    save_task_list(new_pending, task_list_filepath)
                    pending = new_pending
                else:
                    detail = (
                        message[1]
                        if message and not message[0]
                        else "No success acknowledgement"
                    )
                    failures.append(
                        (state["task"], f"{detail} (exit code {process.exitcode})")
                    )
                progress.update(1)
                del active[slot]
                conn.close()
                process.close()
                available.append(slot)
        elapsed = progress.format_dict["elapsed"]
        if failures:
            for task, detail in failures:
                tqdm.write(f"Exception for task {tuple(task[:3])}:\n{detail}")
            raise TrainingRunError(
                f"{len(failures)} training task(s) failed; remaining tasks were saved to {task_list_filepath}"
            )
        if not pending:
            with open(
                os.path.join(os.path.dirname(task_list_filepath), "completed.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("Training completed")
            os.remove(task_list_filepath)
        return elapsed
    finally:
        signal_state["cleanup"] = True
        try:
            _stop_children(active)
        finally:
            try:
                progress.close()
            finally:
                for signum, handler in original_handlers.items():
                    signal.signal(signum, handler)


def parallel_training(tasks, device_list, task_list_filepath: str, *, cpu_threads=1):
    """Run each task in a fresh process, with one process per device slot."""
    return _run_training(tasks, device_list, task_list_filepath, cpu_threads)


def sequential_training(tasks, device_list, task_list_filepath: str, *, cpu_threads=1):
    """Run each task in a fresh process using the first device slot."""
    return _run_training(tasks, device_list[:1], task_list_filepath, cpu_threads)
