"""Project evidence hashing shared by runtime and semantic validation."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def sha256_tree(path: Path) -> str:
    """Hash file names and contents in a deterministic, symlink-free directory tree."""

    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"evidence dataset directory is missing or unsafe: {path}")
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"evidence dataset directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise ValueError(f"evidence dataset contains a symlink: {item}")
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()
