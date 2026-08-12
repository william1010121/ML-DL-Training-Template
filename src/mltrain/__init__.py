"""Stable orchestration and evidence contracts for training projects."""

from mltrain.contracts import (
    ExperimentConfig,
    ProjectAdapter,
    RunContext,
    RunResult,
    ValidationResult,
)

__all__ = [
    "ExperimentConfig",
    "ProjectAdapter",
    "RunContext",
    "RunResult",
    "ValidationResult",
]

__version__ = "0.1.0"
