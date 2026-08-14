#!/usr/bin/env python3
"""Load a real MNIST batch through the project and run one CPU optimizer step."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import cast

import torch
from torch import nn

from ml_training_template.config import MNISTConfig
from ml_training_template.data.mnist import build_train_validation_loaders
from ml_training_template.model.mnist import build_model
from ml_training_template.project import adapter
from mltrain.config import load_config
from mltrain.profiling import RunProfiler
from mltrain.progress import ProgressReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mnist-baseline/exp-004.yml"),
        help="profiled CPU experiment config (default: exp-004)",
    )
    parser.add_argument("--progress", choices=("auto", "plain", "off"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = cast(MNISTConfig, load_config(args.config, adapter))
    if config.runtime.device != "cpu" or config.runtime.strategy != "single":
        raise ValueError("smoke config must request single-process CPU execution")
    if not config.profiling.enabled:
        raise ValueError("smoke config must enable profiling")
    smoke_config = config.model_copy(
        update={"data": config.data.model_copy(update={"num_workers": 0})}
    )
    progress = ProgressReporter(args.progress)

    with tempfile.TemporaryDirectory(prefix="mltrain-profile-smoke-") as temporary:
        profiler = RunProfiler(
            Path(temporary),
            rank=0,
            is_primary=True,
            device="cpu",
            interval_seconds=config.profiling.sample_interval_seconds,
        )
        profiler.start()
        try:
            torch.manual_seed(smoke_config.reproducibility.seed)
            with progress.stage("data/loaders"), profiler.stage("data/loaders"):
                train_loader, _, _ = build_train_validation_loaders(smoke_config)
                with progress.task(
                    total=1, description="MNIST CPU smoke", unit="batch"
                ) as task:
                    inputs, targets = next(iter(train_loader))
                    task.update(1)
            if inputs.ndim != 4 or inputs.shape[1:] != (1, 28, 28):
                raise RuntimeError(f"unexpected MNIST batch shape: {tuple(inputs.shape)}")

            with progress.stage("model/build"), profiler.stage("model/build"):
                model = build_model(
                    in_channels=smoke_config.model.in_channels,
                    num_classes=smoke_config.model.num_classes,
                    hidden_channels=smoke_config.model.hidden_channels,
                )
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=smoke_config.training.optimizer.learning_rate,
                    weight_decay=smoke_config.training.optimizer.weight_decay,
                )
            with progress.stage("epoch/train", epoch=1), profiler.stage(
                "epoch/train", epoch=1
            ):
                before = model.classifier.weight.detach().clone()
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("real MNIST CPU loss is not finite")
                loss.backward()
                optimizer.step()
            if torch.equal(before, model.classifier.weight.detach()):
                raise RuntimeError("real MNIST CPU optimizer step did not update parameters")
        finally:
            profiler.stop()
        summary = json.loads(profiler.files[2].read_text(encoding="utf-8"))
        if summary["sample_count"] < 1 or "epoch/train" not in summary["stage_statistics"]:
            raise RuntimeError("CPU profiling smoke did not produce complete evidence")

    print(
        f"Real MNIST CPU step passed "
        f"(batch={inputs.shape[0]}, loss={loss.item():.6f}, profile_samples="
        f"{summary['sample_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
