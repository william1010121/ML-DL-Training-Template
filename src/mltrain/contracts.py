from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrimaryMetric(StrictModel):
    name: str = Field(min_length=1)
    direction: Literal["minimize", "maximize"]


class ExperimentIdentity(StrictModel):
    id: str = Field(pattern=r"^exp-\d{3}$")
    research_line: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    goal: str = Field(min_length=1)
    primary_metric: PrimaryMetric


class Reproducibility(StrictModel):
    seed: int = Field(ge=0)
    mode: Literal["strict", "performance"] = "strict"


class Runtime(StrictModel):
    device: Literal["cpu", "cuda", "mps"]
    strategy: Literal["single", "ddp"] = "single"
    world_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def strategy_matches_world_size(self) -> Runtime:
        if self.strategy == "single" and self.world_size != 1:
            raise ValueError("single strategy requires world_size=1")
        if self.strategy == "ddp" and self.world_size < 2:
            raise ValueError("ddp strategy requires world_size>=2")
        return self


class TrackingConfig(StrictModel):
    backend: str = "none"


class ProfilingConfig(StrictModel):
    enabled: bool = False
    sample_interval_seconds: float = Field(default=1.0, ge=0.2, le=60.0)


class OutputConfig(StrictModel):
    root: Path = Path("runs")

    @model_validator(mode="after")
    def root_is_canonical(self) -> OutputConfig:
        if self.root != Path("runs"):
            raise ValueError("output.root must be exactly 'runs' (a repository-relative path)")
        return self


class ExperimentConfig(StrictModel):
    """Generic envelope. Project config models must override task-specific mappings."""

    schema_version: int = Field(default=1, ge=1)
    experiment: ExperimentIdentity
    reproducibility: Reproducibility
    runtime: Runtime
    data: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    validation: dict[str, Any]
    tracking: TrackingConfig
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    output: OutputConfig


class RunResult(StrictModel):
    primary_metric_name: str = Field(min_length=1)
    primary_metric: float
    metrics: dict[str, float] = Field(default_factory=dict)
    checkpoint: str | None = None
    data_sha256: str | None = None
    model_sha256: str | None = None
    tracking_degraded: bool = False


class ValidationResult(StrictModel):
    passed: bool = True
    checks: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ProfileSink(Protocol):
    def stage(self, name: str, *, epoch: int | None = None) -> AbstractContextManager[None]: ...


_ProgressItem = TypeVar("_ProgressItem")


class ProgressSink(Protocol):
    def stage(self, name: str, *, epoch: int | None = None) -> AbstractContextManager[None]: ...

    def iterate(
        self,
        iterable: Iterable[_ProgressItem],
        *,
        total: int | None,
        description: str,
        unit: str = "item",
        position: int = 0,
        leave: bool = True,
    ) -> Iterator[_ProgressItem]: ...


class RunContext(StrictModel):
    run_id: str
    run_dir: Path
    command: list[str]
    rank: int = 0
    world_size: int = 1
    is_primary: bool = True
    _profile_sink: ProfileSink | None = PrivateAttr(default=None)
    _progress_sink: ProgressSink | None = PrivateAttr(default=None)

    def attach_profile_sink(self, sink: ProfileSink) -> None:
        """Attach the run-local profiler without serializing runtime state."""
        self._profile_sink = sink

    def attach_progress_sink(self, sink: ProgressSink) -> None:
        """Attach ephemeral progress output without serializing runtime state."""
        self._progress_sink = sink

    def profile_stage(
        self, name: str, *, epoch: int | None = None
    ) -> AbstractContextManager[None]:
        """Time a stable stage name, or become a no-op when profiling is disabled."""
        if self._profile_sink is None:
            return nullcontext()
        return self._profile_sink.stage(name, epoch=epoch)

    def progress_stage(
        self, name: str, *, epoch: int | None = None
    ) -> AbstractContextManager[None]:
        if self._progress_sink is None:
            return nullcontext()
        return self._progress_sink.stage(name, epoch=epoch)

    def progress_iter(
        self,
        iterable: Iterable[_ProgressItem],
        *,
        total: int | None,
        description: str,
        unit: str = "item",
        position: int = 0,
        leave: bool = True,
    ) -> Iterator[_ProgressItem]:
        if self._progress_sink is None:
            return iter(iterable)
        return self._progress_sink.iterate(
            iterable,
            total=total,
            description=description,
            unit=unit,
            position=position,
            leave=leave,
        )

    def log_metrics(self, step: int, metrics: Mapping[str, float]) -> None:
        """Append canonical metrics. Only rank zero writes shared evidence."""
        if not self.is_primary:
            return
        record = {"step": step, **{key: float(value) for key, value in metrics.items()}}
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


@runtime_checkable
class ProjectAdapter(Protocol):
    config_model: type[ExperimentConfig]

    def train(
        self, config: ExperimentConfig, context: RunContext
    ) -> RunResult | Mapping[str, Any]: ...

    def evaluate(
        self, config: ExperimentConfig, checkpoint: Path, context: RunContext
    ) -> RunResult | Mapping[str, Any]: ...

    def validate(
        self, config: ExperimentConfig, run_dir: Path
    ) -> ValidationResult | Mapping[str, Any]: ...
