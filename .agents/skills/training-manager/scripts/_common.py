from __future__ import annotations

import difflib
import hashlib
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PACKAGE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
EXPERIMENT_RE = re.compile(r"^exp-(\d{3})\.ya?ml$")


def repo_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        raise SystemExit(f"not a template repository: {root}/pyproject.toml is missing")
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"required file is missing: {path}") from error
    except yaml.YAMLError as error:
        raise SystemExit(f"invalid YAML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected a YAML mapping in {path}")
    return value


def dump_yaml(value: dict[str, Any]) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def show_diff(path: Path, before: str, after: str) -> None:
    if before == after:
        return
    print(
        "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=str(path),
                tofile=str(path),
            )
        ),
        end="",
    )


def create_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite: {path}") from error


def reject_symlinks(root: Path) -> None:
    """Reject tracked-tree symlinks before any filesystem mutation."""
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(
            part in {".git", ".venv", "datasets", "checkpoints", "runs"}
            for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise SystemExit(f"refusing to operate on a repository containing symlink: {path}")


def apply_transaction(
    root: Path,
    *,
    new_files: dict[Path, str],
    replacements: dict[Path, str],
) -> None:
    """Commit prepared text files and restore originals if any replacement fails."""
    for path in new_files:
        if path.exists():
            raise SystemExit(f"refusing to overwrite: {path}")
    originals: dict[Path, bytes] = {}
    for path in replacements:
        if not path.is_file():
            raise SystemExit(f"replacement target is missing: {path}")
        originals[path] = path.read_bytes()

    created: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".mltrain-transaction-", dir=root) as temporary:
        stage = Path(temporary)
        staged: dict[Path, Path] = {}
        for index, (path, content) in enumerate((*new_files.items(), *replacements.items())):
            staged_path = stage / str(index)
            staged_path.write_text(content, encoding="utf-8")
            staged[path] = staged_path
        try:
            for path in new_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged[path], path)
                created.append(path)
            for path in replacements:
                os.replace(staged[path], path)
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            for path, original_bytes in originals.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                restore = stage / f"restore-{len(staged)}"
                restore.write_bytes(original_bytes)
                os.replace(restore, path)
            for parent in sorted(
                {path.parent for path in new_files},
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                if parent != root:
                    with suppress(OSError):
                        parent.rmdir()
            raise


def registry_lines(registry: dict[str, Any]) -> dict[str, Any]:
    lines = registry.get("research_lines")
    if not isinstance(lines, dict):
        raise SystemExit("configs/registry.yml must contain a research_lines mapping")
    return lines


def insert_research_row(document: str, row: str) -> str:
    marker = "\n## Global Decisions"
    if marker not in document:
        raise SystemExit("configs/research.md is missing the '## Global Decisions' heading")
    head, tail = document.split(marker, 1)
    return f"{head.rstrip()}\n{row}\n{marker}{tail}"
