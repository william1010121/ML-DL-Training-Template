"""Strict configuration schema for the bundled MNIST reference project."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from mltrain.contracts import (
    ExperimentConfig,
    ExperimentIdentity,
    PrimaryMetric,
    Reproducibility,
    Runtime,
    StrictModel,
)
from mltrain.contracts import OutputConfig as CoreOutputConfig
from mltrain.contracts import TrackingConfig as CoreTrackingConfig


class PrimaryMetricConfig(PrimaryMetric):
    name: Literal["validation/loss"]
    direction: Literal["minimize"]


class ExperimentIdentityConfig(ExperimentIdentity):
    id: str = Field(pattern=r"^exp-\d{3}$")
    research_line: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    goal: str = Field(min_length=1)
    primary_metric: PrimaryMetricConfig


class ReproducibilityConfig(Reproducibility):
    seed: int = Field(ge=0, le=2**32 - 1)
    mode: Literal["strict", "performance"]


class RuntimeConfig(Runtime):
    device: Literal["cpu", "cuda", "mps"]
    strategy: Literal["single", "ddp"]
    world_size: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_strategy(self) -> RuntimeConfig:
        if self.strategy == "single" and self.world_size != 1:
            raise ValueError("single strategy requires world_size=1")
        if self.strategy == "ddp" and self.world_size < 2:
            raise ValueError("ddp strategy requires world_size>=2")
        if self.strategy == "ddp" and self.device == "mps":
            raise ValueError("MPS does not support the DDP reference implementation")
        return self


class DataConfig(StrictModel):
    root: Path
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    validation_size: int = Field(gt=0, lt=60_000)

    @field_validator("root")
    @classmethod
    def require_relative_root(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("data.root must be relative to the repository")
        return value


class ModelConfig(StrictModel):
    name: Literal["mnist_cnn"]
    in_channels: Literal[1]
    num_classes: Literal[10]
    hidden_channels: int = Field(gt=0)


class OptimizerConfig(StrictModel):
    name: Literal["adamw"]
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


class TrainingConfig(StrictModel):
    epochs: int = Field(gt=0)
    optimizer: OptimizerConfig


class ValidationConfig(StrictModel):
    batch_size: int = Field(gt=0)


class TrackingConfig(CoreTrackingConfig):
    backend: Literal["noop"]


class OutputConfig(CoreOutputConfig):
    root: Path

    @field_validator("root")
    @classmethod
    def require_relative_root(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("output.root must be relative to the repository")
        return value


class MNISTConfig(ExperimentConfig):
    """Complete, standalone experiment configuration for MNIST."""

    schema_version: Literal[1]
    experiment: ExperimentIdentityConfig
    reproducibility: ReproducibilityConfig
    runtime: RuntimeConfig
    # Pydantic supports narrowing these generic mapping fields in a model subclass.
    data: DataConfig  # type: ignore[assignment]
    model: ModelConfig  # type: ignore[assignment]
    training: TrainingConfig  # type: ignore[assignment]
    validation: ValidationConfig  # type: ignore[assignment]
    tracking: TrackingConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validation_split_is_exact_under_ddp(self) -> MNISTConfig:
        if (
            self.runtime.strategy == "ddp"
            and self.data.validation_size % self.runtime.world_size != 0
        ):
            raise ValueError("DDP requires validation_size divisible by world_size")
        return self
