from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from pydantic import ValidationError
from torch.utils.data import DataLoader, TensorDataset

from ml_training_template.config import MNISTConfig
from ml_training_template.data.mnist import (
    build_train_validation_loaders,
    deterministic_split,
)
from ml_training_template.evidence import sha256_file, sha256_tree
from ml_training_template.model.mnist import MNISTCNN
from ml_training_template.paths import portable_reference
from ml_training_template.tracking import ResilientTracker
from ml_training_template.training import mnist as training
from ml_training_template.validate.mnist import validate_mnist_run
from mltrain.contracts import RunContext


def _raw_config() -> dict[str, Any]:
    value = yaml.safe_load(Path("configs/mnist-baseline/exp-001.yml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _config(**runtime: Any) -> MNISTConfig:
    raw = _raw_config()
    raw["training"]["epochs"] = 2
    raw["data"]["num_workers"] = 0
    raw["runtime"].update(runtime)
    return MNISTConfig.model_validate(raw)


def test_cpu_forward_backward_uses_only_synthetic_data() -> None:
    torch.manual_seed(7)
    model = MNISTCNN(hidden_channels=4)
    inputs = torch.randn(4, 1, 28, 28)
    targets = torch.tensor([0, 1, 2, 3])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.features[0].weight.detach().clone()

    metrics = training._run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer)

    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert not torch.equal(before, model.features[0].weight.detach())


def test_deterministic_split_preserves_exact_indices() -> None:
    dataset = list(range(20))
    first_train, first_validation = deterministic_split(dataset, validation_size=5, seed=42)
    next_train, next_validation = deterministic_split(dataset, validation_size=5, seed=42)

    assert first_train.indices == next_train.indices
    assert first_validation.indices == next_validation.indices
    assert len(first_train) == 15
    assert len(first_validation) == 5
    assert set(first_train.indices).isdisjoint(first_validation.indices)


def test_training_loader_never_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchvision.datasets

    downloads: list[bool] = []

    class MissingMNIST:
        def __init__(self, *_: Any, download: bool, **__: Any) -> None:
            downloads.append(download)
            raise RuntimeError("missing")

    monkeypatch.setattr(torchvision.datasets, "MNIST", MissingMNIST)
    monkeypatch.setattr(
        "ml_training_template.data.mnist.resolve_data_root",
        lambda _: tmp_path / "datasets",
    )

    with pytest.raises(FileNotFoundError, match="Download it explicitly"):
        build_train_validation_loaders(_config())
    assert downloads == [False]


def test_requested_cuda_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        training._resolve_device(_config(device="cuda"))


def test_ddp_rejects_a_padded_validation_split() -> None:
    raw = _raw_config()
    raw["runtime"] = {"device": "cuda", "strategy": "ddp", "world_size": 3}
    raw["data"]["validation_size"] = 5_000
    with pytest.raises(ValidationError, match="validation_size divisible by world_size"):
        MNISTConfig.model_validate(raw)


class _BrokenTracker:
    degraded = False

    def log_params(self, params: Any) -> None:
        del params
        raise RuntimeError("token=do-not-expose")

    def log_metrics(self, metrics: Any, *, step: int) -> None:
        del metrics, step
        raise RuntimeError("token=do-not-expose")

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        del path, name
        raise RuntimeError("token=do-not-expose")

    def finish(self, *, status: str) -> None:
        del status
        raise RuntimeError("token=do-not-expose")


def test_resilient_tracker_suppresses_provider_error_details() -> None:
    tracker = ResilientTracker(_BrokenTracker())
    with pytest.warns(RuntimeWarning) as captured:
        tracker.log_params({"seed": 42})
        tracker.log_metrics({"validation/loss": 1.0}, step=1)
        tracker.log_artifact(Path("checkpoint.pt"))
        tracker.finish(status="failed")

    assert tracker.degraded
    assert len(captured) == 4
    assert all("do-not-expose" not in str(item.message) for item in captured)


class _RecordingTracker:
    degraded = False

    def __init__(self) -> None:
        self.finished: list[str] = []

    def log_params(self, params: Any) -> None:
        del params

    def log_metrics(self, metrics: Any, *, step: int) -> None:
        del metrics, step

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        del path, name

    def finish(self, *, status: str) -> None:
        self.finished.append(status)


def _context(run_dir: Path, *, primary: bool = True) -> RunContext:
    run_dir.mkdir(parents=True)
    return RunContext(
        run_id="test-run",
        run_dir=run_dir,
        command=["test"],
        is_primary=primary,
    )


def test_train_returns_best_validation_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _RecordingTracker()
    monkeypatch.setattr("ml_training_template.tracking.create_tracker", lambda _: tracker)
    monkeypatch.setattr(
        "ml_training_template.data.mnist.build_train_validation_loaders",
        lambda *_args, **_kwargs: (object(), object(), None),
    )
    dataset_root = tmp_path / "datasets"
    (dataset_root / "MNIST").mkdir(parents=True)
    (dataset_root / "MNIST" / "identity").write_bytes(b"mnist")
    monkeypatch.setattr(
        "ml_training_template.data.mnist.resolve_data_root",
        lambda _: dataset_root,
    )
    sequence = iter(
        [
            {"loss": 0.8, "accuracy": 0.7},
            {"loss": 0.4, "accuracy": 0.8},
            {"loss": 0.6, "accuracy": 0.8},
            {"loss": 0.7, "accuracy": 0.7},
        ]
    )
    monkeypatch.setattr(training, "_run_epoch", lambda *_args, **_kwargs: next(sequence))

    result = training.train_mnist(_config(), _context(tmp_path / "run"))

    assert result.primary_metric == 0.4
    assert result.metrics["validation/loss"] == 0.4
    assert result.metrics["best/epoch"] == 1.0
    assert result.checkpoint == "checkpoint.pt"
    assert (tmp_path / "run" / "checkpoint.pt").is_file()
    assert tracker.finished == ["completed"]


def test_tracker_finishes_failed_and_is_not_created_on_secondary_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _RecordingTracker()
    calls = 0

    def factory(_: str) -> _RecordingTracker:
        nonlocal calls
        calls += 1
        return tracker

    monkeypatch.setattr("ml_training_template.tracking.create_tracker", factory)
    monkeypatch.setattr(
        "ml_training_template.data.mnist.build_train_validation_loaders",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("training failed")),
    )
    with pytest.raises(RuntimeError, match="training failed"):
        training.train_mnist(_config(), _context(tmp_path / "primary"))
    assert tracker.finished == ["failed"]
    assert calls == 1

    with pytest.raises(RuntimeError, match="training failed"):
        training.train_mnist(
            _config(),
            _context(tmp_path / "secondary", primary=False),
        )
    assert calls == 1


@pytest.mark.parametrize(
    ("kind", "metric_name", "checkpoint"),
    [
        ("train", "validation/loss", "checkpoint.pt"),
        ("evaluate", "test/loss", "runs/source/checkpoint.pt"),
    ],
)
def test_project_validator_understands_train_and_evaluate_results(
    tmp_path: Path,
    kind: str,
    metric_name: str,
    checkpoint: str,
) -> None:
    dataset_root = tmp_path / "datasets"
    mnist_root = dataset_root / "MNIST"
    mnist_root.mkdir(parents=True)
    (mnist_root / "identity").write_bytes(b"mnist")
    config = _config()
    # The validator permits absolute paths only in this isolated constructed object.
    config = config.model_copy(
        update={"data": config.data.model_copy(update={"root": dataset_root})}
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"kind": kind}), encoding="utf-8")
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "primary_metric": 0.25,
                "primary_metric_name": metric_name,
                "metrics": {metric_name: 0.25},
                "checkpoint": checkpoint,
                "data_sha256": sha256_tree(mnist_root),
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"weights")

    validation = validate_mnist_run(config, tmp_path)

    assert validation.passed, validation.notes


def test_project_validator_detects_mnist_dataset_tampering(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"
    mnist_root = dataset_root / "MNIST"
    mnist_root.mkdir(parents=True)
    identity = mnist_root / "identity"
    identity.write_bytes(b"original")
    original_hash = sha256_tree(mnist_root)
    config = _config().model_copy(
        update={"data": _config().data.model_copy(update={"root": dataset_root})}
    )
    (tmp_path / "checkpoint.pt").write_bytes(b"weights")
    (tmp_path / "manifest.json").write_text(json.dumps({"kind": "train"}), encoding="utf-8")
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "primary_metric_name": "validation/loss",
                "primary_metric": 0.25,
                "metrics": {"validation/loss": 0.25},
                "checkpoint": "checkpoint.pt",
                "data_sha256": original_hash,
            }
        ),
        encoding="utf-8",
    )
    identity.write_bytes(b"tampered")

    validation = validate_mnist_run(config, tmp_path)

    assert not validation.passed
    assert validation.checks["dataset_authenticity"] is False


def test_checkpoint_references_are_portable(tmp_path: Path) -> None:
    outside = tmp_path / "private" / "checkpoint.pt"
    assert portable_reference(outside.resolve()) == "checkpoint.pt"


def test_evaluate_archives_checkpoint_and_uses_test_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pt"
    torch.save(MNISTCNN(hidden_channels=32).state_dict(), source)
    run_dir = tmp_path / "evaluate-run"
    context = _context(run_dir)
    dataset_root = tmp_path / "datasets"
    (dataset_root / "MNIST").mkdir(parents=True)
    (dataset_root / "MNIST" / "identity").write_bytes(b"mnist")
    monkeypatch.setattr(
        "ml_training_template.data.mnist.resolve_data_root",
        lambda _: dataset_root,
    )
    monkeypatch.setattr(
        "ml_training_template.data.mnist.build_test_loader",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        training,
        "_run_epoch",
        lambda *_args, **_kwargs: {"loss": 0.2, "accuracy": 0.9},
    )

    result = training.evaluate_mnist(_config(), source, context)

    archived = run_dir / "checkpoint.pt"
    assert archived.read_bytes() == source.read_bytes()
    assert result.primary_metric_name == "test/loss"
    assert result.primary_metric == 0.2
    assert result.metrics["test/loss"] == 0.2
    assert result.checkpoint == "checkpoint.pt"
    assert result.model_sha256 == sha256_file(archived)


def test_evaluate_refuses_symlink_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"weights")
    link = tmp_path / "link.pt"
    link.symlink_to(source)

    with pytest.raises(ValueError, match="must not be a symlink"):
        training.evaluate_mnist(_config(), link, _context(tmp_path / "evaluate"))
