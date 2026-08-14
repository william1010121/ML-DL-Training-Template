"""Ephemeral progress reporting for terminals and detached logs."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import IO, Literal, TypeVar

from tqdm import tqdm

ProgressMode = Literal["auto", "plain", "off"]
_Item = TypeVar("_Item")


class ProgressReporter:
    """Render dynamic TTY bars or throttled append-only progress on stderr."""

    def __init__(
        self,
        mode: ProgressMode = "auto",
        *,
        stream: IO[str] | None = None,
        rank: int = 0,
        clock: Callable[[], float] = time.monotonic,
        percent_interval: int = 5,
        time_interval_seconds: float = 30.0,
    ) -> None:
        self.stream = stream or sys.stderr
        self.clock = clock
        self.percent_interval = percent_interval
        self.time_interval_seconds = time_interval_seconds
        if rank != 0 or mode == "off":
            self.mode: Literal["tty", "plain", "off"] = "off"
        elif mode == "plain":
            self.mode = "plain"
        else:
            self.mode = "tty" if self.stream.isatty() else "plain"

    def _write(self, message: str) -> None:
        if self.mode == "off":
            return
        try:
            if self.mode == "tty":
                tqdm.write(message, file=self.stream)
            else:
                self.stream.write(message + "\n")
                self.stream.flush()
        except Exception:
            # Progress is presentation only and must never change the workload result.
            self.mode = "off"

    @contextmanager
    def stage(self, name: str, *, epoch: int | None = None) -> Iterator[None]:
        """Report an unbounded stage and preserve the original exception."""

        label = f"{name} epoch={epoch}" if epoch is not None else name
        started = self.clock()
        self._write(f"[progress] {label}: started")
        try:
            yield
        except BaseException:
            self._write(f"[progress] {label}: failed elapsed={self.clock() - started:.1f}s")
            raise
        else:
            self._write(f"[progress] {label}: completed elapsed={self.clock() - started:.1f}s")

    def iterate(
        self,
        iterable: Iterable[_Item],
        *,
        total: int | None,
        description: str,
        unit: str = "item",
        position: int = 0,
        leave: bool = True,
    ) -> Iterator[_Item]:
        """Wrap an iterable without persisting progress as run evidence."""

        if self.mode == "off":
            yield from iterable
            return
        task = ProgressTask(
            self,
            total=total,
            description=description,
            unit=unit,
            position=position,
            leave=leave,
        )
        try:
            for count, item in enumerate(iterable, start=1):
                yield item
                task.update(count)
        except BaseException:
            task.close(failed=True)
            raise
        else:
            task.close(failed=False)

    def _plain_message(
        self,
        description: str,
        count: int,
        total: int | None,
        unit: str,
        started: float,
    ) -> str:
        elapsed = self.clock() - started
        if total is not None and total > 0:
            percent = min(100, int(count * 100 / total))
            return (
                f"[progress] {description}: {count}/{total} {unit} "
                f"({percent}%) elapsed={elapsed:.1f}s"
            )
        return f"[progress] {description}: {count} {unit} elapsed={elapsed:.1f}s"

    @contextmanager
    def task(
        self, *, total: int | None, description: str, unit: str = "item"
    ) -> Iterator[ProgressTask]:
        """Create a manually updated task for polling and byte streams."""

        task = ProgressTask(
            self, total=total, description=description, unit=unit, position=0, leave=True
        )
        try:
            yield task
        except BaseException:
            task.close(failed=True)
            raise
        else:
            task.close(failed=False)


class ProgressTask:
    """One reporter-owned task updated with absolute completed units."""

    def __init__(
        self,
        reporter: ProgressReporter,
        *,
        total: int | None,
        description: str,
        unit: str,
        position: int,
        leave: bool,
    ) -> None:
        self.reporter = reporter
        self.total = total
        self.description = description
        self.unit = unit
        self.started = reporter.clock()
        self.last_reported_at = self.started
        self.last_percent = 0
        self.completed = 0
        self.closed = False
        self.bar = None
        if reporter.mode == "tty":
            try:
                self.bar = tqdm(
                    total=total,
                    desc=description,
                    unit=unit,
                    file=reporter.stream,
                    position=position,
                    leave=leave,
                    dynamic_ncols=True,
                )
            except Exception:
                reporter.mode = "off"
        if reporter.mode == "plain":
            reporter._write(reporter._plain_message(description, 0, total, unit, self.started))

    def update(self, completed: int) -> None:
        if self.closed or self.reporter.mode == "off":
            return
        completed = max(self.completed, completed)
        if self.total is not None:
            completed = min(completed, self.total)
        if self.bar is not None:
            try:
                self.bar.update(completed - self.completed)
            except Exception:
                self.reporter.mode = "off"
            self.completed = completed
            return
        self.completed = completed
        now = self.reporter.clock()
        percent = (
            int(completed * 100 / self.total) if self.total is not None and self.total > 0 else None
        )
        percent_due = (
            percent is not None
            and percent >= self.last_percent + self.reporter.percent_interval
        )
        time_due = now - self.last_reported_at >= self.reporter.time_interval_seconds
        if percent_due or time_due:
            self.reporter._write(
                self.reporter._plain_message(
                    self.description, completed, self.total, self.unit, self.started
                )
            )
            self.last_reported_at = now
            if percent is not None:
                self.last_percent = percent - (percent % self.reporter.percent_interval)

    def close(self, *, failed: bool) -> None:
        if self.closed:
            return
        self.closed = True
        if self.bar is not None:
            try:
                self.bar.close()
            except Exception:
                self.reporter.mode = "off"
        if self.reporter.mode == "off":
            return
        state = "failed" if failed else "completed"
        self.reporter._write(
            f"[progress] {self.description}: {state} {self.completed} {self.unit} "
            f"elapsed={self.reporter.clock() - self.started:.1f}s"
        )
