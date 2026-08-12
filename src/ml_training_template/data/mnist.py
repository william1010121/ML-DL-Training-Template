"""MNIST download and deterministic data-loader construction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ml_training_template.config import MNISTConfig


def download_mnist(root: str | Path = "datasets") -> None:
    """Explicitly download the official MNIST train and test files."""

    from torchvision.datasets import MNIST  # type: ignore[import-untyped]

    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    MNIST(root=destination, train=True, download=True)
    MNIST(root=destination, train=False, download=True)


def resolve_data_root(root: Path) -> Path:
    """Resolve a config path against the repository, independent of the caller's CWD."""

    from ml_training_template.paths import resolve_repository_path

    return resolve_repository_path(root)


def deterministic_split(
    dataset: Any,
    *,
    validation_size: int,
    seed: int,
) -> tuple[Any, Any]:
    """Split a dataset reproducibly while keeping the official test set untouched."""

    import torch
    from torch.utils.data import random_split

    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    split = random_split(dataset, [train_size, validation_size], generator=generator)
    return cast(tuple[Any, Any], split)


def _mnist_dataset(root: Path, *, train: bool) -> Any:
    from torchvision import transforms  # type: ignore[import-untyped]
    from torchvision.datasets import MNIST

    try:
        return MNIST(
            root=root,
            train=train,
            transform=transforms.ToTensor(),
            download=False,
        )
    except RuntimeError as error:
        raise FileNotFoundError(
            f"MNIST is not available under {root}. Download it explicitly before "
            "training (for example: `uv run python scripts/download_mnist.py "
            "--root datasets`)."
        ) from error


def build_train_validation_loaders(
    config: MNISTConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Any, Any, Any | None]:
    """Build a seeded train/validation split and optional distributed sampler."""

    import torch
    from torch.utils.data import DataLoader, DistributedSampler

    dataset = _mnist_dataset(resolve_data_root(config.data.root), train=True)
    train_dataset, validation_dataset = deterministic_split(
        dataset,
        validation_size=config.data.validation_size,
        seed=config.reproducibility.seed,
    )
    generator = torch.Generator().manual_seed(config.reproducibility.seed)

    train_sampler = None
    validation_sampler = None
    if config.runtime.strategy == "ddp":
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.reproducibility.seed,
        )
        validation_sampler = DistributedSampler(
            validation_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    common = {
        "num_workers": config.data.num_workers,
        "pin_memory": config.runtime.device == "cuda",
        "persistent_workers": config.data.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation.batch_size,
        shuffle=False,
        sampler=validation_sampler,
        **common,
    )
    return train_loader, validation_loader, train_sampler


def build_test_loader(config: MNISTConfig) -> Any:
    """Build the official test loader without downloading data implicitly."""

    from torch.utils.data import DataLoader

    dataset = _mnist_dataset(resolve_data_root(config.data.root), train=False)
    return DataLoader(
        dataset,
        batch_size=config.validation.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.runtime.device == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )
