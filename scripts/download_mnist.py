#!/usr/bin/env python3
"""Explicitly download and verify the MNIST dataset used by the example."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets"),
        help="Dataset root (default: datasets)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required; run `uv sync --locked --extra cpu` first"
        ) from exc

    train = MNIST(root=root, train=True, download=True)
    test = MNIST(root=root, train=False, download=True)
    if len(train) != 60_000 or len(test) != 10_000:
        raise RuntimeError(
            f"unexpected MNIST sizes: train={len(train)}, test={len(test)}"
        )

    print(f"MNIST ready at {root} (train={len(train)}, test={len(test)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
