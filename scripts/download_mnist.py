#!/usr/bin/env python3
"""Explicitly download and verify the MNIST dataset used by the example."""

from __future__ import annotations

import argparse
from pathlib import Path

from mltrain.progress import ProgressReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets"),
        help="Dataset root (default: datasets)",
    )
    parser.add_argument("--progress", choices=("auto", "plain", "off"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(args.progress)

    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required; run `uv sync --locked --extra cpu` first"
        ) from exc

    with progress.stage("MNIST training files"):
        train = MNIST(root=root, train=True, download=True)
    with progress.stage("MNIST test files"):
        test = MNIST(root=root, train=False, download=True)
    if len(train) != 60_000 or len(test) != 10_000:
        raise RuntimeError(
            f"unexpected MNIST sizes: train={len(train)}, test={len(test)}"
        )

    print(f"MNIST ready at {root} (train={len(train)}, test={len(test)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
