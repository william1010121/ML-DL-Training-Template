"""Runnable MNIST train and evaluation hooks, including single-node DDP."""

from __future__ import annotations

import hashlib
import os
import random
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml_training_template.config import MNISTConfig
    from mltrain.contracts import RunContext, RunResult


from ml_training_template.evidence import sha256_file, sha256_tree

_MAX_EVALUATION_CHECKPOINT_BYTES = 100 * 1024 * 1024


def _archive_checkpoint(source: Path, destination: Path) -> str:
    """Copy a bounded regular checkpoint without following source symlinks."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        info = os.fstat(descriptor)
        size_ok = 0 < info.st_size <= _MAX_EVALUATION_CHECKPOINT_BYTES
        if not stat.S_ISREG(info.st_mode) or not size_ok:
            raise ValueError("evaluation checkpoint must be a non-empty regular file <= 100 MiB")
        with (
            os.fdopen(descriptor, "rb", closefd=False) as input_stream,
            destination.open("xb") as output_stream,
        ):
            source_digest = hashlib.sha256()
            remaining = info.st_size
            while remaining:
                chunk = input_stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError("evaluation checkpoint changed while being archived")
                output_stream.write(chunk)
                source_digest.update(chunk)
                remaining -= len(chunk)
    finally:
        os.close(descriptor)
    source_sha256 = source_digest.hexdigest()
    archived_sha256 = sha256_file(destination)
    if archived_sha256 != source_sha256:
        destination.unlink(missing_ok=True)
        raise OSError("archived evaluation checkpoint failed checksum verification")
    return archived_sha256


def _set_reproducibility(config: MNISTConfig) -> None:
    import torch

    seed = config.reproducibility.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    strict = config.reproducibility.mode == "strict"
    torch.use_deterministic_algorithms(strict)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not strict


def _resolve_device(config: MNISTConfig, *, local_rank: int = 0) -> Any:
    import torch

    requested = config.runtime.device
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device=cuda, but CUDA is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} has no matching CUDA device "
                f"({torch.cuda.device_count()} visible)"
            )
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("runtime.device=mps, but MPS is unavailable")
        return torch.device("mps")
    raise AssertionError(f"unhandled device: {requested}")


@contextmanager
def _distributed_runtime(config: MNISTConfig) -> Iterator[tuple[int, int, Any]]:
    import torch.distributed as dist

    if config.runtime.strategy == "single":
        yield 0, 1, _resolve_device(config)
        return

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != config.runtime.world_size:
        raise RuntimeError(
            f"Config world_size={config.runtime.world_size}, but torchrun WORLD_SIZE={world_size}"
        )
    if rank < 0 or local_rank < 0:
        raise RuntimeError("DDP requires RANK and LOCAL_RANK from torchrun")

    device = _resolve_device(config, local_rank=local_rank)
    backend = "nccl" if config.runtime.device == "cuda" else "gloo"
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend=backend, init_method="env://")
    try:
        yield rank, world_size, device
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def _model_from_config(config: MNISTConfig) -> Any:
    from ml_training_template.model.mnist import build_model

    return build_model(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        hidden_channels=config.model.hidden_channels,
    )


def _reduce_totals(
    loss_sum: float,
    correct: int,
    count: int,
    device: Any,
) -> tuple[float, int, int]:
    import torch
    import torch.distributed as dist

    totals = torch.tensor(
        [loss_sum, float(correct), float(count)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0].item()), int(totals[1].item()), int(totals[2].item())


def _run_epoch(
    model: Any,
    loader: Any,
    device: Any,
    *,
    optimizer: Any | None,
    context: RunContext | None = None,
    description: str = "batches",
) -> dict[str, float]:
    import torch
    from torch import nn

    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    count = 0

    batches = (
        context.progress_iter(
            loader,
            total=len(loader),
            description=description,
            unit="batch",
            position=1,
            leave=False,
        )
        if context is not None
        else iter(loader)
    )
    with torch.set_grad_enabled(training):
        for inputs, targets in batches:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            if optimizer is not None:
                (loss / targets.numel()).backward()
                optimizer.step()
            loss_sum += float(loss.detach().item())
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            count += int(targets.numel())

    loss_sum, correct, count = _reduce_totals(loss_sum, correct, count, device)
    if count == 0:
        raise RuntimeError("data loader produced no samples")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def train_mnist(config: MNISTConfig, context: RunContext) -> RunResult:
    """Train the reference model; the dataset must already exist locally."""

    import torch
    from torch.nn.parallel import DistributedDataParallel

    from ml_training_template.data.mnist import build_train_validation_loaders, resolve_data_root
    from ml_training_template.tracking import NoOpTracker, create_tracker
    from mltrain.contracts import RunResult

    run_dir = context.run_dir
    _set_reproducibility(config)
    tracker = create_tracker(config.tracking.backend) if context.is_primary else NoOpTracker()
    try:
        if context.is_primary:
            tracker.log_params(config.model_dump(mode="json"))

        with _distributed_runtime(config) as (rank, world_size, device):
            with context.progress_stage("data/loaders"), context.profile_stage("data/loaders"):
                train_loader, validation_loader, train_sampler = build_train_validation_loaders(
                    config,
                    rank=rank,
                    world_size=world_size,
                )
            with context.progress_stage("model/build"), context.profile_stage("model/build"):
                raw_model = _model_from_config(config).to(device)
                model = raw_model
                if config.runtime.strategy == "ddp":
                    model = DistributedDataParallel(
                        raw_model,
                        device_ids=[device.index] if device.type == "cuda" else None,
                    )
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.training.optimizer.learning_rate,
                    weight_decay=config.training.optimizer.weight_decay,
                )

            best_metrics: dict[str, float] | None = None
            checkpoint = run_dir / "checkpoint.pt"
            epochs = context.progress_iter(
                range(1, config.training.epochs + 1),
                total=config.training.epochs,
                description="training epochs",
                unit="epoch",
                position=0,
                leave=True,
            )
            for epoch in epochs:
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                with context.progress_stage("epoch/train", epoch=epoch), context.profile_stage(
                    "epoch/train", epoch=epoch
                ):
                    train_metrics = _run_epoch(
                        model,
                        train_loader,
                        device,
                        optimizer=optimizer,
                        context=context,
                        description=f"train {epoch}/{config.training.epochs}",
                    )
                with context.progress_stage(
                    "epoch/validation", epoch=epoch
                ), context.profile_stage("epoch/validation", epoch=epoch):
                    validation_metrics = _run_epoch(
                        model,
                        validation_loader,
                        device,
                        optimizer=None,
                        context=context,
                        description=f"validation {epoch}/{config.training.epochs}",
                    )
                epoch_metrics = {
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "validation/loss": validation_metrics["loss"],
                    "validation/accuracy": validation_metrics["accuracy"],
                }
                if context.is_primary:
                    context.log_metrics(epoch, epoch_metrics)
                    tracker.log_metrics(epoch_metrics, step=epoch)
                if (
                    best_metrics is None
                    or epoch_metrics["validation/loss"] < best_metrics["validation/loss"]
                ):
                    best_metrics = {**epoch_metrics, "best/epoch": float(epoch)}
                    if context.is_primary:
                        with context.progress_stage(
                            "checkpoint/write", epoch=epoch
                        ), context.profile_stage("checkpoint/write", epoch=epoch):
                            torch.save(raw_model.state_dict(), checkpoint)

            if best_metrics is None:
                raise RuntimeError("training completed without an epoch result")

            checkpoint_name = None
            data_sha256 = None
            model_sha256 = None
            if context.is_primary:
                with context.progress_stage("provenance/hash"), context.profile_stage(
                    "provenance/hash"
                ):
                    checkpoint_name = checkpoint.name
                    data_sha256 = sha256_tree(resolve_data_root(config.data.root) / "MNIST")
                    model_sha256 = sha256_file(checkpoint)
                    tracker.log_artifact(checkpoint, name="checkpoint")
        if context.is_primary:
            tracker.finish(status="completed")
        return RunResult(
            primary_metric_name="validation/loss",
            primary_metric=best_metrics["validation/loss"],
            metrics=best_metrics,
            checkpoint=checkpoint_name,
            data_sha256=data_sha256,
            model_sha256=model_sha256,
            tracking_degraded=tracker.degraded,
        )
    except Exception:
        if context.is_primary:
            tracker.finish(status="failed")
        raise


def evaluate_mnist(config: MNISTConfig, checkpoint: Path, context: RunContext) -> RunResult:
    """Evaluate a checkpoint on the untouched official MNIST test set."""

    import torch

    from ml_training_template.data.mnist import build_test_loader, resolve_data_root
    from ml_training_template.paths import portable_reference
    from ml_training_template.tracking import NoOpTracker, create_tracker
    from mltrain.contracts import RunResult

    if config.runtime.strategy != "single":
        raise ValueError("evaluate uses a single process; set runtime.strategy=single")
    tracker = create_tracker(config.tracking.backend) if context.is_primary else NoOpTracker()
    try:
        with context.progress_stage("checkpoint/load"), context.profile_stage("checkpoint/load"):
            if checkpoint.is_symlink():
                raise ValueError("evaluation checkpoint must not be a symlink")
            archived_checkpoint = context.run_dir / "checkpoint.pt"
            archived_sha256 = _archive_checkpoint(checkpoint, archived_checkpoint)
        _set_reproducibility(config)
        device = _resolve_device(config)
        with context.progress_stage("model/build"), context.profile_stage("model/build"):
            model = _model_from_config(config).to(device)
            state = torch.load(archived_checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
        with context.progress_stage("data/loaders"), context.profile_stage("data/loaders"):
            test_loader = build_test_loader(config)
        with context.progress_stage("evaluate/test"), context.profile_stage("evaluate/test"):
            metrics = _run_epoch(
                model,
                test_loader,
                device,
                optimizer=None,
                context=context,
                description="evaluate test",
            )
        recorded = {"test/loss": metrics["loss"], "test/accuracy": metrics["accuracy"]}
        if context.is_primary:
            tracker.log_params(config.model_dump(mode="json"))
            context.log_metrics(0, recorded)
            tracker.log_metrics(recorded, step=0)
        with context.progress_stage("provenance/hash"), context.profile_stage("provenance/hash"):
            data_sha256 = sha256_tree(resolve_data_root(config.data.root) / "MNIST")
            model_sha256 = archived_sha256
        if context.is_primary:
            tracker.finish(status="completed")
        return RunResult(
            primary_metric_name="test/loss",
            primary_metric=metrics["loss"],
            metrics=recorded,
            checkpoint=portable_reference(archived_checkpoint),
            data_sha256=data_sha256,
            model_sha256=model_sha256,
            tracking_degraded=tracker.degraded,
        )
    except Exception:
        if context.is_primary:
            tracker.finish(status="failed")
        raise
