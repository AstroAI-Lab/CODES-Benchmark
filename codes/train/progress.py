"""Progress messages from training workers and their parent-owned display."""

import hashlib
import io
import logging
import re
import sys
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sized
from contextlib import contextmanager, redirect_stderr, redirect_stdout, suppress
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

_reporter = ContextVar("training_progress_reporter", default=None)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def current_reporter() -> Callable[[Any], None] | None:
    """Return the current worker's message callback, if reporting is enabled."""
    return _reporter.get()


def _send(callback: Callable[[Any], None], message: Any) -> None:
    # A parent that has exited should not turn a successful fit into a failure.
    with suppress(BrokenPipeError, EOFError, OSError):
        callback(message)


class _LogStream(io.TextIOBase):
    """Send complete lines or bounded chunks, retaining only a small partial line."""

    def __init__(self, callback: Callable[[Any], None], chunk_size: int = 4096) -> None:
        self.callback = callback
        self.chunk_size = chunk_size
        self.buffered = ""

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("write() argument must be str")
        size = len(text)
        # Process slices to avoid retaining arbitrarily long writes.
        for offset in range(0, size, self.chunk_size):
            self.buffered += text[offset : offset + self.chunk_size]
            while self.buffered:
                newline = self.buffered.find("\n")
                if 0 <= newline < self.chunk_size:
                    end = newline + 1
                elif len(self.buffered) >= self.chunk_size:
                    end = self.chunk_size
                else:
                    break
                _send(self.callback, ("log", self.buffered[:end]))
                self.buffered = self.buffered[end:]
        return size

    def flush(self) -> None:
        if self.buffered:
            _send(self.callback, ("log", self.buffered))
            self.buffered = ""


@contextmanager
def worker_reporting(send: Callable[[Any], None]) -> Generator[None, None, None]:
    """Route a worker's progress and Python output to its parent process."""
    token = _reporter.set(send)
    stream = _LogStream(send)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    # Existing logging handlers hold their original streams after redirect_stderr.
    changed = []
    loggers = [logging.getLogger()] + [
        logger
        for logger in logging.root.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    for logger in loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream in (
                original_stdout,
                original_stderr,
            ):
                changed.append((handler, handler.stream))
                handler.stream = stream
    try:
        with redirect_stdout(stream), redirect_stderr(stream):
            yield
    finally:
        stream.flush()
        for handler, old_stream in changed:
            handler.stream = old_stream
        _reporter.reset(token)


class EventProgress:
    """Small tqdm-compatible iterable that reports completed epochs to the parent."""

    def __init__(self, iterable: Any, desc: str = "") -> None:
        self.iterable = iterable
        self.total = len(iterable)
        self.desc = desc
        self.n = 0
        self.postfix = {}
        self._last_sent = 0.0
        self._send = current_reporter()

    def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if self._send and (force or now - self._last_sent >= 1.0):
            _send(
                self._send,
                (
                    "progress",
                    {
                        "completed": self.n,
                        "total": self.total,
                        "description": self.desc,
                        "metrics": dict(self.postfix),
                    },
                ),
            )
            self._last_sent = now

    def __iter__(self) -> Iterator[Any]:
        for item in self.iterable:
            yield item
            self.update(1)

    def update(self, n: int = 1) -> None:
        self.n += n
        self._emit(force=self.n == 1 or self.n >= self.total)

    def set_postfix(
        self,
        ordered_dict: Mapping[str, Any] | None = None,
        refresh: bool = True,
        **kwargs: Any,
    ) -> None:
        self.postfix = dict(ordered_dict or {})
        self.postfix.update(kwargs)
        self._emit(force=self._last_sent == 0.0)

    def close(self) -> None:
        pass


class ProgressDisplay:
    """Render worker events centrally, with stable bars or plain log lines."""

    def __init__(
        self,
        tasks: int | Sized,
        slots: int | Sized,
        mode: str = "auto",
        stream: TextIO | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        if mode not in ("auto", "bars", "plain", "off"):
            raise ValueError(f"Unknown progress mode: {mode}")
        self.stream = sys.stderr if stream is None else stream
        if mode == "auto":

            def tty(s: TextIO) -> bool:
                return bool(getattr(s, "isatty", lambda: False)())

            mode = "bars" if tty(self.stream) and tty(sys.stdout) else "plain"
        self.mode = mode
        self.total = tasks if isinstance(tasks, int) else len(tasks)
        self.slots = slots if isinstance(slots, int) else len(slots)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._started = time.monotonic()
        self._active = {}
        self._count = 0
        self._completed = 0
        self._last_print = {}
        self._bars = {}
        self._overall = None
        if mode == "bars":
            self._overall = tqdm(
                total=self.total, desc="Tasks", position=0, file=self.stream
            )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def _line(self, text: str) -> None:
        text = _ANSI.sub("", str(text)).replace("\r", "")
        if self.mode == "bars":
            tqdm.write(text, file=self.stream)
        else:
            self.stream.write(text + "\n")
            self.stream.flush()

    def start(self, slot: int, task: Any, device: str) -> None:
        self._count += 1
        name = (
            " / ".join(map(str, task[:3]))
            if isinstance(task, (tuple, list))
            else str(task)
        )
        log = None
        if self.log_dir is not None:
            digest = hashlib.sha256(repr(task).encode("utf-8")).hexdigest()[:12]
            log = (self.log_dir / f"task-{digest}.log").open("a", encoding="utf-8")
            log.write(
                f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Started {name} on {device}\n"
            )
            log.flush()
        self._active[slot] = (name, device, log)
        self._last_print.pop(slot, None)
        if self.mode == "bars":
            self._bars[slot] = tqdm(
                total=0,
                desc=name,
                position=int(slot) + 1,
                leave=False,
                file=self.stream,
            )
        elif self.mode == "plain":
            self._line(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Started {name} on {device}"
            )

    def event(self, slot: int, payload: Mapping[str, Any]) -> None:
        if slot not in self._active:
            return
        completed = payload.get("completed", 0)
        total = payload.get("total", 0)
        metrics = payload.get("metrics", {})
        detail = " ".join(f"{key}={value}" for key, value in metrics.items())
        log = self._active[slot][2]
        if log is not None:
            log.write(
                f"Epoch {completed}/{total}" + (f" {detail}" if detail else "") + "\n"
            )
            log.flush()
        if self.mode == "bars":
            bar = self._bars[slot]
            bar.total = total
            bar.n = completed
            if payload.get("description"):
                bar.set_description_str(payload["description"], refresh=False)
            bar.set_postfix(metrics, refresh=False)
            bar.refresh()
        elif self.mode == "plain":
            now = time.monotonic()
            if (
                slot not in self._last_print
                or now - self._last_print[slot] >= 30
                or (total and completed >= total)
            ):
                name = self._active[slot][0]
                self._line(
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {name}: {completed}/{total}"
                    + (f" {detail}" if detail else "")
                )
                self._last_print[slot] = now

    def log(self, slot: int, text: str) -> None:
        if slot not in self._active:
            return
        log = self._active[slot][2]
        clean = _ANSI.sub("", text).replace("\r", "")
        if log is not None:
            log.write(clean)
            log.flush()
        # Worker output may contain warnings and errors; do not hide it in off mode.
        if clean:
            label = self._active[slot][0]
            for line in clean.splitlines(keepends=True):
                if self.mode == "bars":
                    tqdm.write(f"[{label}] {line.rstrip()}", file=self.stream)
                else:
                    self.stream.write(f"[{label}] {line}")
                    self.stream.flush()

    def finish(self, slot: int, success: bool, detail: str = "") -> None:
        active = self._active.pop(slot, None)
        if active is None:
            return
        self._completed += 1
        name, _, log = active
        if log is not None:
            status = "Completed" if success else "Failed"
            log.write(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {status} {name}"
                + (f": {detail}" if detail else "")
                + "\n"
            )
            log.close()
        if self.mode == "bars":
            self._bars.pop(slot).close()
            self._overall.update(1)
            if not success:
                self._line(f"Failed {name}: {detail}")
        elif self.mode == "plain":
            status = "Completed" if success else "Failed"
            self._line(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {status} {name}"
                + f" — {self._completed}/{self.total} tasks finished"
                + (f": {detail}" if detail else "")
            )
        elif not success:
            self._line(f"Failed {name}: {detail}")

    def close(self) -> None:
        for _, _, log in self._active.values():
            if log is not None:
                log.close()
        self._active.clear()
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()
        if self._overall is not None:
            self._overall.close()
