from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from mltrain.config import sha256_file


def _git(root: Path, *args: str) -> str | None:
    process = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if process.returncode != 0:
        return None
    return process.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes | None:
    process = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    return process.stdout if process.returncode == 0 else None


def source_state(root: Path) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    commit_exists = bool(
        commit and _git(root, "cat-file", "-e", f"{commit}^{{commit}}") == ""
    )
    git_healthy = status is not None and commit_exists
    dirty = not git_healthy or bool(status)
    fingerprint = None
    if dirty:
        digest = hashlib.sha256()
        digest.update((status if status is not None else "git-status-failed").encode())
        digest.update(b"\0")
        digest.update(_git_bytes(root, "diff", "--binary", "HEAD") or b"")
        digest.update(b"\0")
        untracked = _git_bytes(root, "ls-files", "--others", "--exclude-standard", "-z")
        for raw_name in sorted((untracked or b"").split(b"\0")):
            if not raw_name:
                continue
            digest.update(raw_name)
            digest.update(b"\0")
            path = root / raw_name.decode(errors="surrogateescape")
            if path.is_symlink():
                digest.update(os.readlink(path).encode(errors="surrogateescape"))
            elif path.is_file():
                digest.update(sha256_file(path).encode())
            digest.update(b"\0")
        fingerprint = digest.hexdigest()
    return {
        "commit": commit if git_healthy else None,
        "dirty": dirty,
        "dirty_fingerprint": fingerprint,
    }


def commit_exists(root: Path, commit: object) -> bool:
    return isinstance(commit, str) and _git(root, "cat-file", "-e", f"{commit}^{{commit}}") == ""


def hash_path(path: Path) -> str | None:
    """Content hash for a file or a deterministic directory tree."""
    if not path.exists():
        return None
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if item.is_symlink():
            continue
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("ml-dl-training-template", "torch", "torchvision", "pydantic", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_state(root: Path) -> dict[str, Any]:
    lockfile = root / "uv.lock"
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "lockfile_sha256": sha256_file(lockfile) if lockfile.is_file() else None,
        "container": os.environ.get("MLTRAIN_CONTAINER", "native"),
        "image_digest": os.environ.get("MLTRAIN_IMAGE_DIGEST"),
        "oci_digest": os.environ.get("MLTRAIN_OCI_DIGEST"),
        "sif_sha256": os.environ.get("MLTRAIN_SIF_SHA256"),
    }
    encoded = json.dumps(details, sort_keys=True, separators=(",", ":")).encode()
    details["environment_sha256"] = hashlib.sha256(encoded).hexdigest()
    return details


def model_config_hash(model: Any) -> str:
    encoded = json.dumps(model, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    if not cleaned or cleaned.lower() in {"none", "default"}:
        raise ValueError("invalid shared DDP run id")
    return cleaned[:80]


def validate_run_id(value: object) -> str:
    if not isinstance(value, str) or safe_run_id(value) != value:
        raise ValueError("manifest run_id is not canonical")
    return value
