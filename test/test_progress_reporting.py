"""Regression coverage for process-safe training progress output."""

import re

import pytest

from codes.train import train_fcts
from codes.utils import load_task_list


@pytest.mark.parametrize("mode", ["auto", "plain", "off"])
def test_real_worker_failure_is_visible_without_terminal_controls(
    tmp_path, capsys, mode
):
    """Redirected output stays readable and failures remain resumable in every mode."""
    task = ("UnknownSurrogate", "main", "", "missing_run", 42, 1)
    taskfile = tmp_path / "tasks.json"
    with pytest.raises(train_fcts.TrainingRunError):
        train_fcts.sequential_training([task], ["cpu"], str(taskfile), progress=mode)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Unknown surrogate model" in output
    assert "\r" not in output
    assert not re.search(r"\x1b\[[0-?]*[ -/]*[@-~]", output)
    assert load_task_list(str(taskfile)) == [list(task)]
    assert not (tmp_path / "completed.txt").exists()


def noisy_child(task, device, position, cpu_threads, connection):
    """Use the real child wrapper with a deterministic, importable task body."""
    import sys
    import warnings

    from codes.train import train_fcts as child_training

    def noisy_task(*args, **kwargs):
        print("worker stdout marker")
        print("worker stderr marker", file=sys.stderr)
        warnings.warn("worker warning marker", UserWarning)
        raise RuntimeError("worker failure marker")

    child_training.execute_training_task = noisy_task
    child_training._run_task_child(task, device, position, cpu_threads, connection)


def test_off_saves_worker_output_and_surfaces_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(train_fcts, "_run_task_child", noisy_child)
    task = ("Example", "main", "", "run", 42, 1)
    taskfile = tmp_path / "tasks.json"
    with pytest.raises(train_fcts.TrainingRunError):
        train_fcts.sequential_training([task], ["cpu"], str(taskfile), progress="off")
    output = capsys.readouterr()
    assert "worker failure marker" in output.out + output.err
    assert load_task_list(str(taskfile)) == [list(task)]
    logs = list((tmp_path / "task_logs").glob("*"))
    assert logs
    saved = "\n".join(path.read_text() for path in logs if path.is_file())
    assert "worker stdout marker" in saved
    assert "worker stderr marker" in saved
    assert "worker warning marker" in saved


def test_worker_reporting_restores_context_and_bounds_output():
    import sys

    from codes.train.progress import current_reporter, worker_reporting

    messages = []
    original_streams = (sys.stdout, sys.stderr)
    with pytest.raises(RuntimeError, match="stop"), worker_reporting(messages.append):
        print("x" * 20000)
        print("stderr marker", file=sys.stderr)
        raise RuntimeError("stop")
    assert (sys.stdout, sys.stderr) == original_streams
    assert current_reporter() is None
    chunks = [payload for kind, payload in messages if kind == "log"]
    assert chunks and max(map(len, chunks)) <= 4096
    output = "".join(chunks)
    assert "x" * 20000 in output
    assert "stderr marker" in output


def test_epoch_events_are_throttled_but_keep_first_and_final(monkeypatch):
    from codes.train import progress

    monkeypatch.setattr(progress.time, "monotonic", lambda: 100.0)
    messages = []
    with progress.worker_reporting(messages.append):
        bar = progress.EventProgress(range(100), desc="training")
        for _ in bar:
            bar.set_postfix(loss=0.5)
        bar.close()
    events = [payload for kind, payload in messages if kind == "progress"]
    assert events[0]["completed"] in (0, 1)
    assert events[-1]["completed"] == 100
    assert events[-1]["metrics"]["loss"] == 0.5
    assert len(events) <= 5


def test_plain_display_throttles_and_prints_final_epoch(monkeypatch):
    import io

    from codes.train import progress

    monkeypatch.setattr(progress.time, "monotonic", lambda: 100.0)
    stream = io.StringIO()
    display = progress.ProgressDisplay(1, 1, mode="plain", stream=stream)
    display.start(0, ("Model", "main"), "cpu")
    for epoch in range(1, 101):
        display.event(0, {"completed": epoch, "total": 100, "metrics": {}})
    display.finish(0, True)
    display.close()
    output = stream.getvalue()
    assert "1/100" in output and "100/100" in output
    assert len(output.splitlines()) <= 5
    assert "\r" not in output and "\x1b" not in output


def test_auto_requires_both_streams_to_be_ttys(monkeypatch):
    import io

    from codes.train import progress

    class Stream(io.StringIO):
        def __init__(self, tty):
            super().__init__()
            self.tty = tty

        def isatty(self):
            return self.tty

    for stdout_tty, stderr_tty, expected in [
        (True, True, "bars"),
        (False, True, "plain"),
        (True, False, "plain"),
        (False, False, "plain"),
    ]:
        stdout, stderr = Stream(stdout_tty), Stream(stderr_tty)
        with monkeypatch.context() as patch:
            patch.setattr(progress.sys, "stdout", stdout)
            display = progress.ProgressDisplay(1, 1, mode="auto", stream=stderr)
            assert display.mode == expected
            display.close()


def test_bars_have_distinct_stable_positions(monkeypatch):
    from codes.train import progress

    positions = []

    class FakeBar:
        def __init__(self, **kwargs):
            positions.append(kwargs["position"])

        def close(self):
            pass

        def update(self, count):
            pass

    monkeypatch.setattr(progress, "tqdm", FakeBar)
    display = progress.ProgressDisplay(2, 2, mode="bars")
    display.start(0, ("A",), "cpu")
    display.start(1, ("B",), "cpu")
    display.finish(0, True)
    display.finish(1, True)
    display.close()
    assert positions == [0, 1, 2]


def test_worker_reporter_restores_outer_context_after_error():
    from codes.train.progress import current_reporter, worker_reporting

    outer, inner = [], []
    with worker_reporting(outer.append):
        original = current_reporter()
        with pytest.raises(RuntimeError), worker_reporting(inner.append):
            print("inner message")
            raise RuntimeError("stop")
        assert current_reporter() == original
        print("outer message")
    assert "inner message" in "".join(value for kind, value in inner if kind == "log")
    assert "outer message" in "".join(value for kind, value in outer if kind == "log")


def test_parallel_worker_output_is_attributed_to_each_task(tmp_path):
    import io

    from codes.train.progress import ProgressDisplay

    stream = io.StringIO()
    display = ProgressDisplay(2, 2, mode="plain", stream=stream, log_dir=tmp_path)
    display.start(0, ("Alpha", "main"), "cpu")
    display.start(1, ("Beta", "main"), "cpu")
    display.log(0, "alpha warning\n")
    display.log(1, "beta warning\n")
    display.finish(0, True)
    display.finish(1, True)
    display.close()
    lines = stream.getvalue().splitlines()
    assert any("Alpha" in line and "alpha warning" in line for line in lines)
    assert any("Beta" in line and "beta warning" in line for line in lines)
    logs = [path.read_text() for path in tmp_path.glob("*.log")]
    assert any("alpha warning" in log and "beta warning" not in log for log in logs)
    assert any("beta warning" in log and "alpha warning" not in log for log in logs)
