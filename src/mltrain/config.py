from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from mltrain.contracts import ExperimentConfig, ProjectAdapter


def repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("could not find repository root (pyproject.toml + configs)")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=True, allow_unicode=True)


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error


def load_config(path: Path, adapter: ProjectAdapter) -> ExperimentConfig:
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    try:
        return adapter.config_model.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"config validation failed for {path}:\n{error}") from error


def canonical_config_path(root: Path, config: ExperimentConfig) -> Path:
    configs_root = root / "configs"
    if configs_root.resolve() != configs_root:
        raise ValueError("configs/ must not be a symlink")
    return configs_root / config.experiment.research_line / f"{config.experiment.id}.yml"


def load_registry(root: Path) -> dict[str, Any]:
    path = root / "configs" / "registry.yml"
    value = load_yaml(path)
    if not isinstance(value, dict):
        raise ValueError("configs/registry.yml must be a mapping")
    return value


def registry_entry(
    registry: dict[str, Any], line: str, experiment: str
) -> dict[str, Any]:
    try:
        entry = registry["research_lines"][line]["experiments"][experiment]
    except (KeyError, TypeError) as error:
        raise ValueError(f"experiment is not registered: {line}/{experiment}") from error
    if not isinstance(entry, dict):
        raise ValueError(f"invalid registry entry: {line}/{experiment}")
    return entry


def validate_config_registration(
    root: Path, path: Path, config: ExperimentConfig
) -> dict[str, Any]:
    expected = canonical_config_path(root, config)
    actual = path.resolve()
    if actual != expected:
        raise ValueError(f"config path must be {expected.relative_to(root)}")
    registry = load_registry(root)
    entry = registry_entry(
        registry, config.experiment.research_line, config.experiment.id
    )
    expected_relative = expected.relative_to(root).as_posix()
    if entry.get("config") != expected_relative:
        raise ValueError(
            f"registry config mismatch for {config.experiment.research_line}/"
            f"{config.experiment.id}"
        )
    locked_hash = entry.get("config_sha256")
    if locked_hash is not None and locked_hash != sha256_file(expected):
        raise ValueError(f"registered config hash has drifted: {expected_relative}")
    return entry


def adapter_spec(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    try:
        value = project["tool"]["mltrain"]["adapter"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("missing [tool.mltrain].adapter in pyproject.toml") from error
    if not isinstance(value, str) or ":" not in value:
        raise RuntimeError("[tool.mltrain].adapter must be 'module.path:attribute'")
    return value
