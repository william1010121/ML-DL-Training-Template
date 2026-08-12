#!/usr/bin/env python3
"""Validate the repository's research and clean-file contracts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

EXPERIMENT_FILE = re.compile(r"^exp-[0-9]{3}\.ya?ml$")
EXPERIMENT_ID = re.compile(r"^exp-[0-9]{3}$")
MAX_TRACKED_SIZE = 10 * 1024 * 1024
REQUIRED_PATHS = (
    "AGENTS.md",
    "README.md",
    "README.zh-TW.md",
    "pyproject.toml",
    "configs/research.md",
    "configs/registry.yml",
    ".agents/skills/training-manager/SKILL.md",
    ".agents/skills/add-experiment-logging/SKILL.md",
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(
        rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
        rb"[ \t]*[:=][ \t]*['\"]?[A-Za-z0-9_./+\-=]{12,}"
    ),
)
PROMOTED_FILES = {
    "checksums.sha256",
    "manifest.json",
    "resolved_config.yml",
    "result.json",
    "summary.json",
    "validation.json",
}


def _yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _candidate_files(root: Path) -> list[Path]:
    """Return tracked and unignored untracked files without following ignored runs."""
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]
    return [root / item.decode() for item in result.stdout.split(b"\0") if item]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry_paths(registry: Any) -> set[str]:
    return {
        value
        for value in _walk(registry)
        if isinstance(value, str)
        and re.fullmatch(r"configs/[a-z][a-z0-9-]*/exp-[0-9]{3}\.ya?ml", value)
    }


def _registry_entries(registry: Any) -> Iterable[Mapping[str, Any]]:
    for value in _walk(registry):
        if isinstance(value, Mapping) and isinstance(value.get("config"), str):
            yield value


def _safe_evidence_path(value: Any, prefix: tuple[str, ...]) -> bool:
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.parts[: len(prefix)] == prefix
        and len(path.parts) > len(prefix)
    )


def _validate_promoted_artifact(
    root: Path,
    artifact_path: Path,
    *,
    line: str,
    experiment: str,
    entry: Mapping[str, Any],
    errors: list[str],
) -> None:
    label = f"{line}/{experiment}"
    actual = {path.name for path in artifact_path.iterdir()}
    if actual != PROMOTED_FILES:
        errors.append(
            f"promoted artifact file set is invalid: {label}: "
            f"expected {sorted(PROMOTED_FILES)}, got {sorted(actual)}"
        )
        return
    checksums: dict[str, str] = {}
    for value in (artifact_path / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        parts = value.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            errors.append(f"promoted artifact checksum syntax is invalid: {label}")
            return
        digest, name = parts
        if name in checksums or name not in PROMOTED_FILES - {"checksums.sha256"}:
            errors.append(f"promoted artifact checksum member is invalid: {label}: {name}")
            return
        checksums[name] = digest
    expected_members = PROMOTED_FILES - {"checksums.sha256"}
    if set(checksums) != expected_members:
        errors.append(f"promoted artifact checksum set is incomplete: {label}")
    for name, digest in checksums.items():
        if _sha256(artifact_path / name) != digest:
            errors.append(f"promoted artifact checksum mismatch: {label}: {name}")

    try:
        summary = _yaml(artifact_path / "summary.json")
        manifest = _yaml(artifact_path / "manifest.json")
        result = _yaml(artifact_path / "result.json")
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"promoted artifact metadata is invalid: {label}: {exc}")
        return
    if not all(isinstance(value, Mapping) for value in (summary, manifest, result)):
        errors.append(f"promoted artifact metadata must be mappings: {label}")
        return
    assert isinstance(summary, Mapping) and isinstance(manifest, Mapping)
    assert isinstance(result, Mapping)
    source = manifest.get("source")
    config = manifest.get("config")
    primary_metric = summary.get("primary_metric")
    expected = {
        "experiment": experiment,
        "research_line": line,
        "source_commit": source.get("commit") if isinstance(source, Mapping) else None,
        "config_sha256": entry.get("config_sha256"),
        "result_sha256": _sha256(artifact_path / "result.json"),
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            errors.append(f"promoted artifact summary {field} is inconsistent: {label}")
    run_pointer = entry.get("completed_run")
    if (
        not isinstance(run_pointer, str)
        or PurePosixPath(run_pointer).name != artifact_path.name
        or manifest.get("run_id") != artifact_path.name
    ):
        errors.append(f"promoted artifact run identity is inconsistent: {label}")
    if isinstance(config, Mapping) and config.get("source_sha256") != entry.get("config_sha256"):
        errors.append(f"promoted artifact manifest config is inconsistent: {label}")
    if not isinstance(primary_metric, Mapping) or (
        primary_metric.get("name") != result.get("primary_metric_name")
        or primary_metric.get("value") != result.get("primary_metric")
    ):
        errors.append(f"promoted artifact primary result is inconsistent: {label}")
    research_path = root / "configs" / line / "research.md"
    research_row = next(
        (
            value
            for value in research_path.read_text(encoding="utf-8").splitlines()
            if value.startswith(f"| {experiment} |")
        ),
        None,
    )
    if research_row is None:
        errors.append(f"promoted artifact has no research result row: {label}")
    else:
        cells = [cell.strip() for cell in research_row.split("|")[1:-1]]
        metric_text = None
        if isinstance(primary_metric, Mapping):
            name = primary_metric.get("name")
            value = primary_metric.get("value")
            if isinstance(name, str) and isinstance(value, (int, float)):
                metric_text = f"{name} = {float(value):.6g}"
        research_identity = (
            len(cells) == 5
            and cells[1] == f"`{summary.get('source_commit')}`"
            and cells[2] == metric_text
            and cells[3] == summary.get("decision")
            and cells[4] == "promoted"
        )
        if not research_identity:
            errors.append(f"promoted artifact does not match research result: {label}")


def _validate_registry_shape(registry: Any, errors: list[str]) -> None:
    if not isinstance(registry, Mapping) or registry.get("schema_version") != 1:
        errors.append("registry schema_version must be 1")
        return
    lines = registry.get("research_lines")
    if not isinstance(lines, Mapping):
        errors.append("registry research_lines must be a mapping")
        return
    for line_name, line in lines.items():
        if not isinstance(line_name, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", line_name):
            errors.append(f"invalid registry research line: {line_name!r}")
            continue
        if not isinstance(line, Mapping) or not isinstance(line.get("experiments"), Mapping):
            errors.append(f"registry line {line_name!r} has no experiments mapping")
            continue
        for experiment_id, entry in line["experiments"].items():
            prefix = f"registry {line_name}/{experiment_id}"
            if not isinstance(experiment_id, str) or not EXPERIMENT_ID.fullmatch(experiment_id):
                errors.append(f"{prefix}: invalid experiment ID")
            if not isinstance(entry, Mapping):
                errors.append(f"{prefix}: entry must be a mapping")
                continue
            expected = f"configs/{line_name}/{experiment_id}.yml"
            if entry.get("config") != expected:
                errors.append(f"{prefix}: config must be {expected}")
            status = entry.get("status")
            if status not in {"planned", "exploratory", "completed", "promoted"}:
                errors.append(f"{prefix}: invalid status {status!r}")
            config_hash = entry.get("config_sha256")
            completed_run = entry.get("completed_run")
            promoted_artifact = entry.get("promoted_artifact")
            if config_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", str(config_hash)):
                errors.append(f"{prefix}: config_sha256 must be a lowercase SHA-256")
            if status == "planned" and any((config_hash, completed_run, promoted_artifact)):
                errors.append(f"{prefix}: planned entry must not contain run evidence")
            if status == "exploratory" and not (config_hash and completed_run):
                errors.append(f"{prefix}: exploratory entry requires config hash and run pointer")
            if status == "completed" and not (config_hash and completed_run):
                errors.append(f"{prefix}: completed entry requires config hash and completed run")
            if status == "promoted" and not (config_hash and completed_run and promoted_artifact):
                errors.append(f"{prefix}: promoted entry requires hash, run, and artifact")
            if completed_run and not _safe_evidence_path(
                completed_run, ("runs", line_name, str(experiment_id))
            ):
                errors.append(f"{prefix}: completed_run must stay in its canonical runs directory")
            if promoted_artifact:
                expected_prefix = ("artifacts", line_name, str(experiment_id))
                if not _safe_evidence_path(promoted_artifact, expected_prefix):
                    errors.append(
                        f"{prefix}: promoted_artifact must stay in its canonical artifact directory"
                    )


def _load_adapter_model(root: Path) -> type[Any] | None:
    with (root / "pyproject.toml").open("rb") as stream:
        document = tomllib.load(stream)
    try:
        spec = document["tool"]["mltrain"]["adapter"]
    except (KeyError, TypeError):
        return None
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("[tool.mltrain].adapter must be module:attribute")
    module_name, attribute = spec.split(":", 1)
    source_path = str(root / "src")
    sys.path.insert(0, source_path)
    try:
        adapter = getattr(importlib.import_module(module_name), attribute)
    finally:
        sys.path.remove(source_path)
    model = getattr(adapter, "config_model", None)
    if not isinstance(model, type) or not hasattr(model, "model_validate"):
        raise ValueError("configured adapter has no Pydantic config_model")
    return model


def _validate_research(root: Path, errors: list[str]) -> None:
    configs_root = root / "configs"
    config_files = sorted(
        path
        for path in configs_root.glob("*/*")
        if path.is_file() and EXPERIMENT_FILE.fullmatch(path.name)
    )
    try:
        registry = _yaml(configs_root / "registry.yml")
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"cannot read configs/registry.yml: {exc}")
        return

    _validate_registry_shape(registry, errors)
    try:
        adapter_model = _load_adapter_model(root)
    except (ImportError, AttributeError, OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        errors.append(f"cannot load project config adapter: {exc}")
        adapter_model = None
    if config_files and adapter_model is None:
        errors.append("experiment configs require [tool.mltrain].adapter")

    registered = _registry_paths(registry)
    actual = {path.relative_to(root).as_posix() for path in config_files}
    for missing in sorted(actual - registered):
        errors.append(f"experiment config is not registered: {missing}")
    for stale in sorted(registered - actual):
        errors.append(f"registry points to missing config: {stale}")

    for line_dir in sorted({path.parent for path in config_files}):
        research = line_dir / "research.md"
        if not research.is_file():
            errors.append(f"research line has no research.md: {line_dir.relative_to(root)}")

    for path in config_files:
        try:
            config = _yaml(path)
        except yaml.YAMLError as exc:
            errors.append(f"invalid YAML in {path.relative_to(root)}: {exc}")
            continue
        if not isinstance(config, Mapping):
            errors.append(f"experiment config must be a mapping: {path.relative_to(root)}")
            continue
        meta = config.get("experiment")
        if not isinstance(meta, Mapping):
            errors.append(f"missing experiment mapping: {path.relative_to(root)}")
            continue
        expected_id = path.stem
        if meta.get("id") != expected_id:
            errors.append(f"experiment.id must be {expected_id!r}: {path.relative_to(root)}")
        line = meta.get("research_line", meta.get("line"))
        if line != path.parent.name:
            errors.append(
                f"experiment research line must be {path.parent.name!r}: {path.relative_to(root)}"
            )
        if adapter_model is not None:
            try:
                adapter_model.model_validate(config)
            except Exception as exc:
                errors.append(f"project schema rejected {path.relative_to(root)}: {exc}")

    for entry in _registry_entries(registry):
        expected_hash = entry.get("config_sha256")
        if not expected_hash:
            continue
        path = root / str(entry["config"])
        if path.is_file() and _sha256(path) != expected_hash:
            errors.append(f"locked experiment config changed: {entry['config']}")

    if isinstance(registry, Mapping):
        for line_name, line in registry.get("research_lines", {}).items():
            if not isinstance(line, Mapping):
                continue
            for experiment_id, entry in line.get("experiments", {}).items():
                if not isinstance(entry, Mapping):
                    continue
                artifact = entry.get("promoted_artifact")
                if artifact:
                    artifact_path = root / str(artifact)
                    artifact_root = root / "artifacts"
                    unsafe = (
                        not artifact_path.is_dir()
                        or artifact_path.is_symlink()
                        or not artifact_path.resolve().is_relative_to(artifact_root.resolve())
                        or any(
                            parent.is_symlink()
                            for parent in artifact_path.parents
                            if parent != root
                        )
                    )
                    if unsafe:
                        errors.append(
                            "promoted artifact is missing or unsafe: "
                            f"{line_name}/{experiment_id}: {artifact}"
                        )
                    else:
                        _validate_promoted_artifact(
                            root,
                            artifact_path,
                            line=str(line_name),
                            experiment=str(experiment_id),
                            entry=entry,
                            errors=errors,
                        )


def _validate_research_documents(root: Path, errors: list[str]) -> None:
    top = root / "configs/research.md"
    if top.is_file():
        text = top.read_text(encoding="utf-8")
        for heading in ("## Goal", "## Research Lines", "## Global Decisions"):
            if heading not in text:
                errors.append(f"configs/research.md is missing {heading}")
        registry = _yaml(root / "configs/registry.yml")
        lines = registry.get("research_lines", {}) if isinstance(registry, Mapping) else {}
        index_rows = {
            match.group("line"): (match.group("status"), match.group("evidence"))
            for match in re.finditer(
                r"^\| \[(?P<line>[a-z][a-z0-9-]*)\]\([^\n)]+/research\.md\) \|"
                r" [^|]+ \| (?P<status>[^|]+) \| (?P<evidence>[^|]+) \|$",
                text,
                re.MULTILINE,
            )
        }
        indexed = set(index_rows)
        if isinstance(lines, Mapping):
            for line_name in sorted(set(lines) - indexed):
                errors.append(f"research line is missing from project index: {line_name}")
            for line_name in sorted(indexed - set(lines)):
                errors.append(f"project research index has unregistered line: {line_name}")
            for line_name, line in lines.items():
                if not isinstance(line, Mapping) or line_name not in index_rows:
                    continue
                entries = line.get("experiments", {})
                if not isinstance(entries, Mapping):
                    continue
                statuses = {
                    str(entry.get("status"))
                    for entry in entries.values()
                    if isinstance(entry, Mapping) and entry.get("status") != "planned"
                }
                status, evidence = index_rows[str(line_name)]
                if not statuses:
                    if status != "Planned" or evidence != "—":
                        errors.append(
                            "project research index should show "
                            f"{line_name} as Planned with no evidence"
                        )
                    continue
                match = re.fullmatch(r"`(exp-[0-9]{3})` @ `[0-9a-f]{8}`", evidence)
                if status.lower() not in statuses or match is None:
                    errors.append(f"project research index evidence is inconsistent: {line_name}")
                    continue
                selected = entries.get(match.group(1))
                if not isinstance(selected, Mapping) or selected.get("status") != status.lower():
                    errors.append(
                        f"project research index points to mismatched evidence: {line_name}"
                    )
    for path in sorted((root / "configs").glob("*/research.md")):
        text = path.read_text(encoding="utf-8")
        for heading in ("## Goal", "## Results"):
            if heading not in text:
                errors.append(f"{path.relative_to(root)} is missing {heading}")
        result_rows = {
            match.group(1): match.group(2)
            for match in re.finditer(
                r"^\| (exp-[0-9]{3}) \|.*\| (planned|exploratory|completed|promoted) \|$",
                text,
                re.MULTILINE,
            )
        }
        registry = _yaml(root / "configs/registry.yml")
        line_name = path.parent.name
        try:
            entries = registry["research_lines"][line_name]["experiments"]
        except (KeyError, TypeError):
            entries = {}
        if isinstance(entries, Mapping):
            expected_results = {
                experiment_id: entry.get("status")
                for experiment_id, entry in entries.items()
                if isinstance(entry, Mapping) and entry.get("status") != "planned"
            }
            for experiment_id, status in expected_results.items():
                if result_rows.get(str(experiment_id)) != status:
                    errors.append(
                        f"research result does not match registry: {line_name}/{experiment_id}"
                    )
            for experiment_id in set(result_rows) - set(expected_results):
                errors.append(f"research has unregistered result: {line_name}/{experiment_id}")


def _validate_skills(root: Path, errors: list[str]) -> None:
    for name in ("training-manager", "add-experiment-logging"):
        path = root / ".agents/skills" / name / "SKILL.md"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "[TODO" in text:
            errors.append(f"unfinished TODO in {path.relative_to(root)}")
        if not text.startswith("---\n") or f"name: {name}\n" not in text:
            errors.append(f"invalid skill frontmatter in {path.relative_to(root)}")
        manifest = path.parent / "agents/openai.yaml"
        try:
            value = _yaml(manifest)
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"invalid skill manifest {manifest.relative_to(root)}: {exc}")
            continue
        if not isinstance(value, Mapping):
            errors.append(f"skill manifest must be a mapping: {manifest.relative_to(root)}")
            continue
        interface = value.get("interface")
        policy = value.get("policy")
        if not isinstance(interface, Mapping):
            errors.append(f"skill manifest has no interface: {manifest.relative_to(root)}")
            continue
        for field in ("display_name", "short_description", "default_prompt"):
            if not isinstance(interface.get(field), str) or not interface[field].strip():
                errors.append(
                    f"skill manifest interface.{field} is required: {manifest.relative_to(root)}"
                )
        description = interface.get("short_description", "")
        if isinstance(description, str) and not 25 <= len(description) <= 64:
            errors.append(
                "skill short_description must be 25-64 characters: "
                f"{manifest.relative_to(root)}"
            )
        prompt = interface.get("default_prompt", "")
        if isinstance(prompt, str) and f"${name}" not in prompt:
            errors.append(
                f"skill default_prompt must mention ${name}: {manifest.relative_to(root)}"
            )
        if not isinstance(policy, Mapping) or not isinstance(
            policy.get("allow_implicit_invocation"), bool
        ):
            errors.append(
                "skill policy.allow_implicit_invocation must be boolean: "
                f"{manifest.relative_to(root)}"
            )


def _validate_hygiene(root: Path, errors: list[str]) -> None:
    for path in _candidate_files(root):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            errors.append(f"environment secret file must not be committed: {relative}")
        size = path.stat().st_size
        if size > MAX_TRACKED_SIZE:
            errors.append(f"file exceeds 10 MiB limit: {relative}")
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                errors.append(f"possible secret in {relative}")
                break

    for path in sorted((root / "configs").glob("**/*.y*ml")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if re.search(r"(?:^|[\s:'\"])(?:/Users/|/home/|[A-Za-z]:\\\\Users\\\\)", text):
            errors.append(f"host-specific absolute path in {path.relative_to(root)}")


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")
    if not (root / "configs/registry.yml").is_file():
        return errors
    _validate_research(root, errors)
    _validate_research_documents(root, errors)
    _validate_skills(root, errors)
    _validate_hygiene(root, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    errors = validate(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Repository contract is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
