"""Low-overhead, run-local stage timing and resource sampling."""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from mltrain.config import sha256_file

PROFILE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _finite(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total": sum(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


class RunProfiler:
    """One profiler per process/rank; only rank zero samples shared resources."""

    def __init__(
        self,
        run_dir: Path,
        *,
        rank: int,
        is_primary: bool,
        device: str,
        interval_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.profile_dir = run_dir / "profile"
        self.rank = rank
        self.is_primary = is_primary
        self.device = device
        self.interval_seconds = interval_seconds
        self._monotonic = monotonic
        self._origin = monotonic()
        self._stages_path = self.profile_dir / f"stages.rank-{rank:03d}.jsonl"
        self._resources_path = self.profile_dir / f"resources.rank-{rank:03d}.jsonl"
        self._summary_path = self.profile_dir / f"summary.rank-{rank:03d}.json"
        self._stages: list[dict[str, Any]] = []
        self._resources: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.degraded = False
        self.notes: list[str] = []
        self._process = psutil.Process()

    @property
    def files(self) -> tuple[Path, Path, Path]:
        return self._stages_path, self._resources_path, self._summary_path

    def _degrade(self, note: str) -> None:
        with self._lock:
            self.degraded = True
            if note not in self.notes:
                self.notes.append(note)

    def start(self) -> None:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._stages_path.touch(exist_ok=False)
        self._resources_path.touch(exist_ok=False)
        self._process.cpu_percent(interval=None)
        self._sample_once()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name=f"mltrain-profiler-rank-{self.rank}",
            daemon=True,
        )
        self._thread.start()

    def _sample_loop(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                self._sample_once()
        except Exception:
            self._degrade("resource sampler failed")

    def _gpu_sample(self) -> dict[str, Any]:
        if not self.is_primary:
            return {"status": "not_collected_non_primary", "devices": []}
        if self.device == "cpu":
            return {"status": "not_requested", "devices": []}
        if self.device == "mps":
            return {"status": "unsupported", "devices": []}
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=min(2.0, max(0.2, self.interval_seconds * 0.8)),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            self._degrade("NVIDIA GPU sampler unavailable")
            return {"status": "unavailable", "devices": []}
        if result.returncode != 0:
            self._degrade("NVIDIA GPU sampler failed")
            return {"status": "unavailable", "devices": []}
        devices: list[dict[str, Any]] = []
        try:
            for line in result.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 6:
                    raise ValueError
                index, uuid, utilization, used, total, power = fields
                devices.append(
                    {
                        "index": int(index),
                        "uuid": uuid,
                        "utilization_percent": float(utilization),
                        "memory_used_mib": float(used),
                        "memory_total_mib": float(total),
                        "power_watts": None if power in {"N/A", "[N/A]"} else float(power),
                    }
                )
        except ValueError:
            self._degrade("NVIDIA GPU sampler returned invalid data")
            return {"status": "unavailable", "devices": []}
        if not devices:
            self._degrade("NVIDIA GPU sampler returned no devices")
            return {"status": "unavailable", "devices": []}
        return {"status": "available", "devices": devices}

    def _sample_once(self) -> None:
        try:
            memory = self._process.memory_info()
            record: dict[str, Any] = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "timestamp": _utc_now(),
                "elapsed_seconds": self._monotonic() - self._origin,
                "rank": self.rank,
                "process": {
                    "cpu_percent": self._process.cpu_percent(interval=None),
                    "rss_bytes": memory.rss,
                },
            }
            if self.is_primary:
                system_memory = psutil.virtual_memory()
                record["system"] = {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_used_bytes": system_memory.used,
                    "memory_available_bytes": system_memory.available,
                    "memory_percent": system_memory.percent,
                }
            record["gpu"] = self._gpu_sample()
            self._append(self._resources_path, record)
            with self._lock:
                self._resources.append(record)
        except Exception:
            self._degrade("resource sample could not be recorded")

    @staticmethod
    def _append(path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    @contextmanager
    def stage(self, name: str, *, epoch: int | None = None) -> Iterator[None]:
        started = self._monotonic()
        started_at = _utc_now()
        status = "succeeded"
        try:
            yield
        except BaseException as error:
            status = "failed" if isinstance(error, Exception) else "interrupted"
            raise
        finally:
            record: dict[str, Any] = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "timestamp": started_at,
                "elapsed_seconds": started - self._origin,
                "rank": self.rank,
                "stage": name,
                "epoch": epoch,
                "duration_seconds": self._monotonic() - started,
                "status": status,
            }
            try:
                self._append(self._stages_path, record)
                with self._lock:
                    self._stages.append(record)
            except Exception:
                self._degrade("stage timing could not be recorded")

    def _summary(self) -> dict[str, Any]:
        with self._lock:
            stages = list(self._stages)
            resources = list(self._resources)
            notes = list(self.notes)
        durations: defaultdict[str, list[float]] = defaultdict(list)
        for record in stages:
            value = _finite(record.get("duration_seconds"))
            if value is not None:
                durations[str(record["stage"])].append(value)

        metrics: defaultdict[str, list[float]] = defaultdict(list)
        for record in resources:
            process = record.get("process", {})
            system = record.get("system", {})
            for key, value in (
                ("process.cpu_percent", process.get("cpu_percent")),
                ("process.rss_bytes", process.get("rss_bytes")),
                ("system.cpu_percent", system.get("cpu_percent")),
                ("system.memory_used_bytes", system.get("memory_used_bytes")),
            ):
                number = _finite(value)
                if number is not None:
                    metrics[key].append(number)
            gpu = record.get("gpu", {})
            for device in gpu.get("devices", []):
                index = device.get("index")
                for field in ("utilization_percent", "memory_used_mib", "power_watts"):
                    number = _finite(device.get(field))
                    if number is not None:
                        metrics[f"gpu.{index}.{field}"].append(number)

        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "rank": self.rank,
            "sample_interval_seconds": self.interval_seconds,
            "degraded": self.degraded,
            "notes": notes,
            "stage_statistics": {
                name: _stats(values) for name, values in sorted(durations.items())
            },
            "resource_statistics": {
                name: _stats(values) for name, values in sorted(metrics.items())
            },
            "sample_count": len(resources),
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))
            if self._thread.is_alive():
                self._degrade("resource sampler did not stop")
        try:
            self._summary_path.write_text(
                json.dumps(self._summary(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            self._degrade("profile summary could not be written")

    def evidence(self) -> dict[str, Any]:
        files = {
            path.relative_to(self.profile_dir).as_posix(): sha256_file(path)
            for path in self.files
            if path.is_file()
        }
        return {
            "enabled": True,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "sample_interval_seconds": self.interval_seconds,
            "status": "degraded" if self.degraded else "completed",
            "degraded": self.degraded,
            "files": files,
        }


def _json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("profile JSONL record must be an object")
        records.append(value)
    return records


def validate_profile_evidence(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    enabled: bool,
    interval_seconds: float,
    rank: int = 0,
) -> dict[str, bool]:
    """Validate the primary rank's bounded profiling evidence."""

    state = manifest.get("profiling")
    if not isinstance(state, Mapping):
        # Runs created before profiling existed remain valid when their config uses
        # the backwards-compatible disabled default.
        return {"profiling_manifest": not enabled}
    checks = {
        "profiling_manifest": state.get("enabled") is enabled,
        "profiling_interval": state.get("sample_interval_seconds") == interval_seconds,
    }
    if not enabled:
        checks.update(
            {
                "profiling_disabled_status": state.get("status") == "disabled",
                "profiling_disabled_files": state.get("files") == {},
            }
        )
        return checks

    profile_dir = run_dir / "profile"
    names = {
        f"stages.rank-{rank:03d}.jsonl",
        f"resources.rank-{rank:03d}.jsonl",
        f"summary.rank-{rank:03d}.json",
    }
    files = state.get("files")
    checks["profiling_completed"] = (
        state.get("status") == "completed" and state.get("degraded") is False
    )
    checks["profiling_directory"] = (
        profile_dir.is_dir()
        and not profile_dir.is_symlink()
        and profile_dir.resolve() == profile_dir
    )
    checks["profiling_file_set"] = isinstance(files, Mapping) and set(files) == names
    if not checks["profiling_directory"] or not isinstance(files, Mapping):
        return checks

    safe = True
    hashes = True
    for name in names:
        path = profile_dir / name
        safe = safe and path.is_file() and not path.is_symlink()
        hashes = hashes and safe and files.get(name) == sha256_file(path)
    checks["profiling_files_safe"] = safe
    checks["profiling_hashes"] = hashes
    if not safe:
        return checks

    try:
        stages = _json_lines(profile_dir / f"stages.rank-{rank:03d}.jsonl")
        resources = _json_lines(profile_dir / f"resources.rank-{rank:03d}.jsonl")
        summary = json.loads(
            (profile_dir / f"summary.rank-{rank:03d}.json").read_text(encoding="utf-8")
        )
        checks["profiling_stages"] = bool(stages) and all(
            record.get("schema_version") == PROFILE_SCHEMA_VERSION
            and record.get("rank") == rank
            and isinstance(record.get("stage"), str)
            and _finite(record.get("duration_seconds")) is not None
            and record.get("status") in {"succeeded", "failed", "interrupted"}
            for record in stages
        )
        checks["profiling_resources"] = bool(resources) and all(
            record.get("schema_version") == PROFILE_SCHEMA_VERSION
            and record.get("rank") == rank
            and isinstance(record.get("process"), Mapping)
            and isinstance(record.get("gpu"), Mapping)
            for record in resources
        )
        checks["profiling_summary"] = (
            isinstance(summary, dict)
            and summary.get("schema_version") == PROFILE_SCHEMA_VERSION
            and summary.get("rank") == rank
            and summary.get("sample_count") == len(resources)
            and summary.get("degraded") is False
            and isinstance(summary.get("stage_statistics"), dict)
            and isinstance(summary.get("resource_statistics"), dict)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        checks.update(
            {
                "profiling_stages": False,
                "profiling_resources": False,
                "profiling_summary": False,
            }
        )
    return checks
