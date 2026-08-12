"""Repository-relative path helpers shared by project implementations."""

from __future__ import annotations

from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    """Find the template repository without importing non-contract mltrain internals."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("could not find repository root (pyproject.toml + configs)")


def resolve_repository_path(path: Path) -> Path:
    return path if path.is_absolute() else repository_root() / path


def portable_reference(path: Path) -> str:
    """Return a host-independent checkpoint reference without parent traversal."""

    candidate = path
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(repository_root())
        except ValueError:
            candidate = Path(candidate.name)
    if ".." in candidate.parts:
        candidate = Path(candidate.name)
    return candidate.as_posix()
