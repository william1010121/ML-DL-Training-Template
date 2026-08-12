"""The single project adapter loaded by the stable :mod:`mltrain` core."""

from __future__ import annotations

from pathlib import Path

from ml_training_template.config import MNISTConfig
from mltrain.contracts import RunContext, RunResult, ValidationResult


class MNISTProjectAdapter:
    """Bridge the generic training lifecycle to the MNIST implementation."""

    config_model = MNISTConfig

    def train(self, config: MNISTConfig, context: RunContext) -> RunResult:
        from ml_training_template.training.mnist import train_mnist

        return train_mnist(config, context)

    def evaluate(
        self,
        config: MNISTConfig,
        checkpoint: Path,
        context: RunContext,
    ) -> RunResult:
        from ml_training_template.training.mnist import evaluate_mnist

        return evaluate_mnist(config, checkpoint, context)

    def validate(self, config: MNISTConfig, run_dir: Path) -> ValidationResult:
        from ml_training_template.validate.mnist import validate_mnist_run

        return validate_mnist_run(config, run_dir)


adapter = MNISTProjectAdapter()
