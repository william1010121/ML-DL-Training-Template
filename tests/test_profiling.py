from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mltrain.contracts import RunContext
from mltrain.profiling import RunProfiler, validate_profile_evidence


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


class _Process:
    def cpu_percent(self, interval: float | None = None) -> float:
        del interval
        return 37.5

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=123_456)


def _system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mltrain.profiling.psutil.Process", _Process)
    monkeypatch.setattr("mltrain.profiling.psutil.cpu_percent", lambda interval=None: 62.5)
    monkeypatch.setattr(
        "mltrain.profiling.psutil.virtual_memory",
        lambda: SimpleNamespace(
            used=400,
            available=600,
            percent=40.0,
        ),
    )


def test_disabled_run_context_stage_is_a_noop(tmp_path: Path) -> None:
    context = RunContext(run_id="run", run_dir=tmp_path, command=["mltrain", "train"])

    with context.profile_stage("epoch/train", epoch=1):
        pass

    assert not (tmp_path / "profile").exists()


def test_stage_timer_uses_monotonic_clock_and_records_failure(tmp_path: Path) -> None:
    clock = _Clock()
    profiler = RunProfiler(
        tmp_path,
        rank=0,
        is_primary=True,
        device="cpu",
        interval_seconds=1.0,
        monotonic=clock,
    )
    profiler.profile_dir.mkdir()
    profiler.files[0].touch()
    profiler.files[1].touch()

    with profiler.stage("epoch/train", epoch=2):
        clock.value += 1.25
    with pytest.raises(ValueError, match="boom"), profiler.stage(
        "epoch/validation", epoch=2
    ):
        clock.value += 0.5
        raise ValueError("boom")
    profiler.stop()

    records = [json.loads(line) for line in profiler.files[0].read_text().splitlines()]
    assert [record["status"] for record in records] == ["succeeded", "failed"]
    assert records[0]["duration_seconds"] == pytest.approx(1.25)
    assert records[1]["duration_seconds"] == pytest.approx(0.5)
    summary = json.loads(profiler.files[2].read_text())
    assert summary["stage_statistics"]["epoch/train"]["mean"] == pytest.approx(1.25)


def test_primary_sampler_records_process_system_and_each_nvidia_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _system(monkeypatch)
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=(
            "0, GPU-first, 85, 1024, 24576, 210.5\n"
            "1, GPU-second, 40, 512, 24576, [N/A]\n"
        ),
        stderr="",
    )
    monkeypatch.setattr("mltrain.profiling.subprocess.run", lambda *_args, **_kwargs: completed)
    profiler = RunProfiler(
        tmp_path,
        rank=0,
        is_primary=True,
        device="cuda",
        interval_seconds=60.0,
    )

    profiler.start()
    with profiler.stage("model/build"):
        pass
    profiler.stop()

    resource = json.loads(profiler.files[1].read_text().splitlines()[0])
    assert resource["process"] == {"cpu_percent": 37.5, "rss_bytes": 123_456}
    assert resource["system"]["cpu_percent"] == 62.5
    assert [gpu["uuid"] for gpu in resource["gpu"]["devices"]] == [
        "GPU-first",
        "GPU-second",
    ]
    assert resource["gpu"]["devices"][1]["power_watts"] is None
    assert profiler.evidence()["status"] == "completed"


def test_cuda_sampler_failure_is_degraded_but_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _system(monkeypatch)

    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("mltrain.profiling.subprocess.run", missing)
    profiler = RunProfiler(
        tmp_path,
        rank=0,
        is_primary=True,
        device="cuda",
        interval_seconds=60.0,
    )

    profiler.start()
    with profiler.stage("lifecycle/setup"):
        pass
    profiler.stop()

    resource = json.loads(profiler.files[1].read_text().splitlines()[0])
    assert resource["gpu"] == {"devices": [], "status": "unavailable"}
    assert profiler.evidence()["status"] == "degraded"


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", "not_requested"), ("mps", "unsupported")],
)
def test_non_nvidia_devices_do_not_report_fake_gpu_utilization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected: str,
) -> None:
    _system(monkeypatch)
    monkeypatch.setattr(
        "mltrain.profiling.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("nvidia-smi must not run"),
    )
    profiler = RunProfiler(
        tmp_path / device,
        rank=0,
        is_primary=True,
        device=device,
        interval_seconds=60.0,
    )

    profiler.start()
    profiler.stop()

    resource = json.loads(profiler.files[1].read_text().splitlines()[0])
    assert resource["gpu"] == {"devices": [], "status": expected}
    assert profiler.degraded is False


def test_rank_specific_files_do_not_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _system(monkeypatch)
    primary = RunProfiler(
        tmp_path,
        rank=0,
        is_primary=True,
        device="cpu",
        interval_seconds=60.0,
    )
    secondary = RunProfiler(
        tmp_path,
        rank=1,
        is_primary=False,
        device="cpu",
        interval_seconds=60.0,
    )

    primary.start()
    secondary.start()
    primary.stop()
    secondary.stop()

    assert primary.files[0] != secondary.files[0]
    secondary_resource = json.loads(secondary.files[1].read_text().splitlines()[0])
    assert "system" not in secondary_resource
    assert secondary_resource["gpu"]["status"] == "not_collected_non_primary"


def test_profile_validation_rejects_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _system(monkeypatch)
    profiler = RunProfiler(
        tmp_path,
        rank=0,
        is_primary=True,
        device="cpu",
        interval_seconds=60.0,
    )
    profiler.start()
    with profiler.stage("lifecycle/setup"):
        pass
    profiler.stop()
    manifest = {"profiling": profiler.evidence()}

    valid = validate_profile_evidence(
        tmp_path,
        manifest,
        enabled=True,
        interval_seconds=60.0,
    )
    assert all(valid.values())

    profiler.files[0].write_text("{}\n", encoding="utf-8")
    tampered = validate_profile_evidence(
        tmp_path,
        manifest,
        enabled=True,
        interval_seconds=60.0,
    )
    assert tampered["profiling_hashes"] is False
    assert tampered["profiling_stages"] is False
