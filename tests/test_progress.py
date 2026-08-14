from __future__ import annotations

import io
from pathlib import Path

import pytest

from mltrain.contracts import RunContext
from mltrain.progress import ProgressReporter


class _Stream(io.StringIO):
    def __init__(self, *, tty: bool = False) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


class _BrokenStream(_Stream):
    def write(self, value: str) -> int:
        del value
        raise OSError("presentation unavailable")


def test_auto_selects_plain_for_detached_log_and_throttles_by_five_percent() -> None:
    stream = _Stream()
    reporter = ProgressReporter("auto", stream=stream, time_interval_seconds=999)

    assert list(
        reporter.iterate(range(20), total=20, description="train", unit="batch")
    ) == list(range(20))

    output = stream.getvalue()
    assert reporter.mode == "plain"
    assert "train: 0/20 batch (0%)" in output
    assert "train: 1/20 batch (5%)" in output
    assert "train: completed 20 batch" in output
    assert "\x1b" not in output


def test_plain_unknown_total_reports_after_time_interval() -> None:
    now = [0.0]
    stream = _Stream()
    reporter = ProgressReporter(
        "plain", stream=stream, clock=lambda: now[0], time_interval_seconds=30
    )

    def values() -> object:
        for value in range(2):
            now[0] += 31
            yield value

    assert list(reporter.iterate(values(), total=None, description="hash", unit="file")) == [0, 1]
    assert "hash: 1 file elapsed=31.0s" in stream.getvalue()


def test_off_and_non_primary_rank_are_silent() -> None:
    for reporter in (
        ProgressReporter("off", stream=_Stream()),
        ProgressReporter("plain", stream=_Stream(), rank=1),
    ):
        assert list(reporter.iterate(range(2), total=2, description="quiet")) == [0, 1]
        assert reporter.stream.getvalue() == ""


def test_stage_records_failure_and_preserves_exception() -> None:
    stream = _Stream()
    reporter = ProgressReporter("plain", stream=stream)

    with pytest.raises(ValueError, match="original"), reporter.stage("model/build"):
        raise ValueError("original")

    assert "model/build: started" in stream.getvalue()
    assert "model/build: failed" in stream.getvalue()


def test_renderer_failure_does_not_change_workload() -> None:
    reporter = ProgressReporter("plain", stream=_BrokenStream())

    with reporter.stage("setup"):
        values = list(reporter.iterate(range(3), total=3, description="work"))

    assert values == [0, 1, 2]
    assert reporter.mode == "off"


def test_run_context_exposes_project_neutral_progress(tmp_path: Path) -> None:
    stream = _Stream()
    context = RunContext(
        run_id="run",
        run_dir=tmp_path,
        command=["train"],
    )
    context.attach_progress_sink(ProgressReporter("plain", stream=stream))

    with context.progress_stage("data/loaders"):
        assert list(
            context.progress_iter(range(2), total=2, description="loader", unit="batch")
        ) == [0, 1]

    assert "data/loaders: completed" in stream.getvalue()
    assert "loader: completed 2 batch" in stream.getvalue()


def test_auto_uses_tty_renderer_without_writing_stdout() -> None:
    stream = _Stream(tty=True)
    reporter = ProgressReporter("auto", stream=stream)

    assert list(reporter.iterate(range(1), total=1, description="epoch")) == [0]
    assert reporter.mode == "tty"
    assert "epoch" in stream.getvalue()
