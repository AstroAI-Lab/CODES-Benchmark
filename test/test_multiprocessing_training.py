"""Process isolation, failure recovery and worker configuration."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from codes.surrogates.FCNN.fcnn import FullyConnected
from codes.train import train_fcts
from codes.train.tasks import execute_training_task
from codes.utils import load_task_list, save_task_list


def fake_child(task, device, position, cpu_threads, connection):
    """Importable spawn target; each task writes its own independent evidence."""
    import random

    import numpy as np
    import torch

    kind, _, metric, directory, seed, _ = task
    root = Path(directory)
    start = time.monotonic()
    (root / f"{metric}.started").write_text(str(os.getpid()))
    if kind == "hard":
        os._exit(17)
    if kind == "ack-crash":
        connection.send((True, ""))
        connection.close()
        os._exit(18)
    if kind == "barrier":
        deadline = time.monotonic() + 20
        while len(list(root.glob("*.started"))) < 2:
            if time.monotonic() >= deadline:
                raise TimeoutError("parallel worker did not start")
            time.sleep(0.02)
    if kind == "failure":
        connection.send((False, "deliberate failure"))
        connection.close()
        raise RuntimeError("deliberate failure")
    if kind == "slow":
        time.sleep(30)
    else:
        time.sleep(0.8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = {
        "pid": os.getpid(),
        "start": start,
        "end": time.monotonic(),
        "random": [random.random(), float(np.random.rand()), float(torch.rand(1))],
    }
    (root / f"{metric}.json").write_text(json.dumps(data))
    connection.send((True, ""))
    connection.close()


def task(root, metric, kind="ok", seed=13):
    return (kind, "main", metric, str(root), seed, 1)


def run_with_fake(
    monkeypatch, root, tasks, devices, runner=train_fcts.parallel_training
):
    monkeypatch.setattr(train_fcts, "_run_task_child", fake_child)
    taskfile = root / "tasks.json"
    save_task_list([list(t) for t in tasks], str(taskfile))
    runner(tasks, devices, str(taskfile))
    return taskfile


def test_spawn_is_fresh_bounded_and_reproducible(monkeypatch, tmp_path):
    parallel = tmp_path / "parallel"
    sequential = tmp_path / "sequential"
    parallel.mkdir()
    sequential.mkdir()
    tasks = [
        task(parallel, i, kind="barrier" if i < 2 else "ok", seed=10 + i)
        for i in range(4)
    ]
    taskfile = run_with_fake(monkeypatch, parallel, tasks, ["cpu", "cpu"])
    assert not taskfile.exists()
    assert (parallel / "completed.txt").exists()
    results = [json.loads((parallel / f"{i}.json").read_text()) for i in range(4)]
    assert len({r["pid"] for r in results}) == 4
    assert all(r["pid"] != os.getpid() for r in results)
    # Every start/end interval is a spawned task's active lifetime.
    events = sorted(
        [(r["start"], 1) for r in results] + [(r["end"], -1) for r in results]
    )
    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    assert peak <= 2
    assert peak == 2
    serial_tasks = [task(sequential, i, seed=10 + i) for i in range(4)]
    run_with_fake(
        monkeypatch, sequential, serial_tasks, ["cpu"], train_fcts.sequential_training
    )
    serial = [json.loads((sequential / f"{i}.json").read_text()) for i in range(4)]
    assert [r["random"] for r in results] == [r["random"] for r in serial]
    assert len({tuple(r["random"]) for r in results}) == 4


def test_failed_and_crashed_tasks_remain_for_resume(monkeypatch, tmp_path):
    tasks = [
        task(tmp_path, 0),
        task(tmp_path, 1, "failure"),
        task(tmp_path, 2, "hard"),
        task(tmp_path, 3),
        task(tmp_path, 4, "ack-crash"),
    ]

    def timed_out(signum, frame):
        raise TimeoutError("crashed worker left the runner hanging")

    old_handler = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(40)
    try:
        with pytest.raises(train_fcts.TrainingRunError):
            run_with_fake(monkeypatch, tmp_path, tasks, ["cpu", "cpu"])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    assert load_task_list(str(tmp_path / "tasks.json")) == [
        list(tasks[1]),
        list(tasks[2]),
        list(tasks[4]),
    ]
    assert (tmp_path / "0.json").exists() and (tmp_path / "3.json").exists()
    assert not (tmp_path / "completed.txt").exists()


def test_executor_error_is_reported_and_task_is_retained(tmp_path, capsys):
    failed_task = ("UnknownSurrogate", "main", "", "missing_run", 42, 1)
    taskfile = tmp_path / "tasks.json"
    with pytest.raises(train_fcts.TrainingRunError):
        train_fcts.sequential_training([failed_task], ["cpu"], str(taskfile))
    assert "Unknown surrogate model" in capsys.readouterr().out
    assert load_task_list(str(taskfile)) == [list(failed_task)]
    assert not (tmp_path / "completed.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signal handling")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_joins_children_and_keeps_pending(tmp_path, sig):
    tasks = [task(tmp_path, i, "slow") for i in range(3)]
    taskfile = tmp_path / "tasks.json"
    save_task_list([list(t) for t in tasks], str(taskfile))
    script = """
import sys
from test.test_multiprocessing_training import fake_child
from codes.train import train_fcts
from codes.utils import load_task_list
train_fcts._run_task_child = fake_child
train_fcts.parallel_training(load_task_list(sys.argv[1]), ["cpu", "cpu"], sys.argv[1])
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(taskfile)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while (
            len(list(tmp_path.glob("*.started"))) < 2
            and proc.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        starts = list(tmp_path.glob("*.started"))
        assert len(starts) == 2, "workers did not start"
        pids = [int(p.read_text()) for p in starts]
        proc.send_signal(sig)
        assert proc.wait(timeout=15) != 0
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert load_task_list(str(taskfile)) == [list(t) for t in tasks]
        assert not (tmp_path / "completed.txt").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_invalid_cpu_threads(threads):
    with pytest.raises(ValueError, match="cpu_threads"):
        execute_training_task(("x",), cpu_threads=threads)


def test_cpu_allocation_and_visible_cuda_validation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(ValueError, match="available CPUs"):
        execute_training_task(("x",), cpu_threads=3)
    with (
        patch("codes.train.tasks.torch.cuda.is_available", return_value=True),
        patch("codes.train.tasks.torch.cuda.device_count", return_value=1),
        patch("codes.train.tasks.torch.cuda.set_device") as set_device,
    ):
        with pytest.raises(ValueError, match="visible devices"):
            execute_training_task(("x",), device="cuda:1")
        set_device.assert_not_called()


def test_cli_propagates_failed_training_without_success_message(
    monkeypatch, tmp_path, capsys
):
    from argparse import Namespace

    import run_training

    config = {
        "dataset": {"name": "unused"},
        "training_id": "test",
        "devices": ["cpu"],
        "surrogates": ["A"],
    }
    taskfile = tmp_path / "tasks.json"
    monkeypatch.setattr(run_training, "load_and_save_config", lambda **kwargs: config)
    monkeypatch.setattr(run_training, "download_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_training, "check_training_status", lambda config: (str(taskfile), True)
    )
    monkeypatch.setattr(
        run_training,
        "create_task_list_for_surrogate",
        lambda *args: [task(tmp_path, 0)],
    )

    def fail(*args, **kwargs):
        raise train_fcts.TrainingRunError("worker failed")

    monkeypatch.setattr(run_training, "sequential_training", fail)
    with pytest.raises(train_fcts.TrainingRunError, match="worker failed"):
        run_training.main(Namespace(config="unused"))
    assert "Training completed" not in capsys.readouterr().out


# The subclass is registered only inside the spawned wrapper, keeping the test
# module's import from changing the global model registry used by other tests.
class StochasticFullyConnected(FullyConnected):
    pass


def stochastic_child(task, device, position, cpu_threads, connection):
    from codes.surrogates.AbstractSurrogate import AbstractSurrogateModel
    from codes.train.train_fcts import _run_task_child

    AbstractSurrogateModel.register(StochasticFullyConnected)
    _run_task_child(task, device, position, cpu_threads, connection)


def test_real_stochastic_training_matches_across_schedules(monkeypatch, tmp_path):
    import random

    import numpy as np
    import torch
    import yaml

    from codes.utils.data_utils import create_dataset

    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(9)
    data = rng.uniform(0.5, 2.0, (8, 8, 2)).astype(np.float32)
    create_dataset(
        "toy",
        (data[:4], data[4:6], data[6:]),
        timesteps=np.linspace(0, 1, 8),
        labels=["a", "b"],
    )
    Path("datasets/toy/surrogates_config.py").write_text("""
from dataclasses import dataclass, field
from torch import nn
@dataclass
class FullyConnectedConfig:
    hidden_size: int = 8
    num_hidden_layers: int = 2
    activation: nn.Module = field(default_factory=lambda: nn.Dropout(0.3))
""")
    monkeypatch.setattr(train_fcts, "_run_task_child", stochastic_child)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    for run, devices, runner in [
        ("serial", ["cpu"], train_fcts.sequential_training),
        ("parallel", ["cpu", "cpu"], train_fcts.parallel_training),
    ]:
        directory = Path("trained") / run
        directory.mkdir(parents=True)
        config = {
            "training_id": run,
            "surrogates": ["StochasticFullyConnected"],
            "batch_size": 2,
            "checkpoint": True,
            "dataset": {
                "name": "toy",
                "log10_transform": False,
                "normalise": "minmax",
                "use_optimal_params": True,
            },
        }
        (directory / "config.yaml").write_text(yaml.safe_dump(config))
        tasks = [
            ("StochasticFullyConnected", "main", "", run, 42, 2),
            ("StochasticFullyConnected", "UQ", 1, run, 43, 2),
        ]
        taskfile = directory / "tasks.json"
        save_task_list([list(t) for t in tasks], str(taskfile))
        runner(tasks, devices, str(taskfile))
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0] and np.array_equal(
        current_numpy[1], numpy_state[1]
    )
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)

    def weights(run, suffix):
        path = (
            Path("trained")
            / run
            / "StochasticFullyConnected"
            / f"stochasticfullyconnected_{suffix}.pth"
        )
        return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

    for suffix in ("main", "UQ_1"):
        left, right = weights("serial", suffix), weights("parallel", suffix)
        assert left.keys() == right.keys()
        assert all(torch.equal(left[key], right[key]) for key in left)
    main, member = weights("serial", "main"), weights("serial", "UQ_1")
    assert any(not torch.equal(main[key], member[key]) for key in main)


def test_unrelated_pending_task_survives_success(monkeypatch, tmp_path):
    monkeypatch.setattr(train_fcts, "_run_task_child", fake_child)
    scheduled, unrelated = task(tmp_path, 0), task(tmp_path, 99)
    taskfile = tmp_path / "tasks.json"
    save_task_list([list(scheduled), list(unrelated)], str(taskfile))
    train_fcts.sequential_training([scheduled], ["cpu"], str(taskfile))
    assert load_task_list(str(taskfile)) == [list(unrelated)]
    assert not (tmp_path / "completed.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signal handling")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_during_process_start_joins_child(tmp_path, sig):
    tasks = [task(tmp_path, 0, "slow"), task(tmp_path, 1, "slow")]
    taskfile = tmp_path / "tasks.json"
    pidfile = tmp_path / "child.pid"
    save_task_list([list(t) for t in tasks], str(taskfile))
    script = """
import multiprocessing as mp
import os, signal, sys
from pathlib import Path
from test.test_multiprocessing_training import fake_child
from codes.train import train_fcts
from codes.utils import load_task_list
train_fcts._run_task_child = fake_child
process_class = type(mp.get_context("spawn").Process())
original_start = process_class.start
def interrupted_start(self):
    original_start(self)
    Path(sys.argv[2]).write_text(str(self.pid))
    os.kill(os.getpid(), int(sys.argv[3]))
process_class.start = interrupted_start
train_fcts.parallel_training(load_task_list(sys.argv[1]), ["cpu", "cpu"], sys.argv[1])
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(taskfile), str(pidfile), str(sig)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    assert proc.returncode != 0
    assert pidfile.exists(), "signal was not delivered during process start"
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    assert load_task_list(str(taskfile)) == [list(t) for t in tasks]
    assert not (tmp_path / "completed.txt").exists()


def test_cli_finalizes_existing_empty_task_list(monkeypatch, tmp_path):
    from argparse import Namespace

    import run_training

    config = {
        "dataset": {"name": "unused"},
        "training_id": "test",
        "devices": ["cpu"],
        "surrogates": ["A"],
    }
    taskfile = tmp_path / "tasks.json"
    save_task_list([], str(taskfile))
    monkeypatch.setattr(run_training, "load_and_save_config", lambda **kwargs: config)
    monkeypatch.setattr(run_training, "download_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_training, "check_training_status", lambda config: (str(taskfile), False)
    )

    def unexpected_regeneration(*args):
        raise AssertionError("empty existing task list was regenerated")

    monkeypatch.setattr(
        run_training, "create_task_list_for_surrogate", unexpected_regeneration
    )
    run_training.main(Namespace(config="unused"))
    assert not taskfile.exists()
    assert (tmp_path / "completed.txt").exists()


def test_failed_process_start_preserves_task_and_runs_sibling(monkeypatch, tmp_path):
    import multiprocessing as mp

    context = mp.get_context("spawn")

    class FailingFirstStart:
        def __init__(self):
            self.calls = 0

        def Pipe(self, *args, **kwargs):
            return context.Pipe(*args, **kwargs)

        def Process(self, *args, **kwargs):
            process = context.Process(*args, **kwargs)
            self.calls += 1
            if self.calls == 1:

                def fail_start():
                    raise OSError("deliberate start failure")

                process.start = fail_start
            return process

    fake_context = FailingFirstStart()
    monkeypatch.setattr(train_fcts.mp, "get_context", lambda name: fake_context)
    monkeypatch.setattr(train_fcts, "_run_task_child", fake_child)
    tasks = [task(tmp_path, 0), task(tmp_path, 1)]
    taskfile = tmp_path / "tasks.json"
    save_task_list([list(t) for t in tasks], str(taskfile))
    with pytest.raises(train_fcts.TrainingRunError):
        train_fcts.sequential_training(tasks, ["cpu"], str(taskfile))
    assert load_task_list(str(taskfile)) == [list(tasks[0])]
    assert (tmp_path / "1.json").exists()
    assert not (tmp_path / "completed.txt").exists()


def test_executor_configures_runtime_and_selects_visible_cuda(monkeypatch):
    import torch

    from codes.train import tasks as task_module

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(task_module.os, "cpu_count", lambda: 4)
    if hasattr(task_module.os, "sched_getaffinity"):
        monkeypatch.setattr(task_module.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    original_threads = torch.get_num_threads()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    observed = []
    monkeypatch.setattr(
        task_module,
        "train_and_save_model",
        lambda *args, **kwargs: observed.append(
            (
                torch.get_num_threads(),
                torch.are_deterministic_algorithms_enabled(),
                kwargs,
            )
        ),
    )
    try:
        execute_training_task(("example",), cpu_threads=2)
        assert observed == [(2, True, {"device": "cpu", "position": 0})]
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "set_device") as select,
        ):
            execute_training_task(("example",), device="cuda:1", position=3)
            select.assert_called_once_with(1)
            assert observed[-1] == (1, True, {"device": "cuda:1", "position": 3})
    finally:
        torch.set_num_threads(original_threads)
        torch.use_deterministic_algorithms(original_deterministic)
