#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import sys
import tomllib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    EXPERIMENT_RE,
    SLUG_RE,
    apply_transaction,
    dump_yaml,
    load_yaml,
    registry_lines,
    reject_symlinks,
    repo_root,
    show_diff,
)


def validate_with_project_adapter(root: Path, config: dict[str, Any]) -> None:
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    try:
        spec = project["tool"]["mltrain"]["adapter"]
    except (KeyError, TypeError) as error:
        raise SystemExit(
            "cannot create an experiment until [tool.mltrain].adapter is configured"
        ) from error
    if not isinstance(spec, str) or ":" not in spec:
        raise SystemExit("[tool.mltrain].adapter must be 'module.path:attribute'")
    module_name, attribute = spec.split(":", 1)
    source_root = str(root / "src")
    sys.path.insert(0, source_root)
    try:
        module = importlib.import_module(module_name)
        adapter = getattr(module, attribute)
        config_model = adapter.config_model
        config_model.model_validate(config)
    except (ImportError, AttributeError) as error:
        raise SystemExit(f"cannot load project adapter {spec}: {error}") from error
    except Exception as error:
        raise SystemExit(f"candidate config failed project schema validation:\n{error}") from error
    finally:
        if sys.path and sys.path[0] == source_root:
            sys.path.pop(0)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Create and register the next experiment.")
    value.add_argument("research_line", help="existing research-line slug")
    source = value.add_mutually_exclusive_group()
    source.add_argument(
        "--config", help="complete candidate YAML; required for the first experiment"
    )
    source.add_argument("--from", dest="from_config", help="prior experiment YAML to copy")
    value.add_argument("--goal", help="replace experiment.goal in the copied config")
    value.add_argument("--root", default=".", help="repository root (default: current directory)")
    value.add_argument("--apply", action="store_true", help="write changes; default is dry-run")
    return value


def experiment_numbers(line_dir: Path, experiments: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for path in line_dir.glob("exp-*.yml"):
        match = EXPERIMENT_RE.fullmatch(path.name)
        if match:
            values.append(int(match.group(1)))
    for name in experiments:
        match = EXPERIMENT_RE.fullmatch(f"{name}.yml")
        if match:
            values.append(int(match.group(1)))
    return values


def main() -> int:
    args = parser().parse_args()
    if not SLUG_RE.fullmatch(args.research_line):
        raise SystemExit("research line must be a lowercase hyphenated slug")

    root = repo_root(args.root)
    reject_symlinks(root)
    line_dir = root / "configs" / args.research_line
    if not (line_dir / "research.md").is_file():
        raise SystemExit(f"research line is missing research.md: {line_dir}")
    registry_path = root / "configs" / "registry.yml"
    registry = load_yaml(registry_path)
    lines = registry_lines(registry)
    line = lines.get(args.research_line)
    if not isinstance(line, dict):
        raise SystemExit(f"research line is not registered: {args.research_line}")
    experiments = line.get("experiments")
    if not isinstance(experiments, dict):
        raise SystemExit(f"registry experiments must be a mapping: {args.research_line}")

    numbers = experiment_numbers(line_dir, experiments)
    next_number = max(numbers, default=0) + 1
    if next_number > 999:
        raise SystemExit("experiment numbering exhausted at exp-999")
    experiment_id = f"exp-{next_number:03d}"
    destination = line_dir / f"{experiment_id}.yml"
    if destination.exists() or experiment_id in experiments:
        raise SystemExit(f"refusing to overwrite existing experiment: {experiment_id}")

    if args.config:
        unresolved_source = Path(args.config).expanduser()
        if unresolved_source.is_symlink():
            raise SystemExit(f"refusing symlink config source: {unresolved_source}")
        source_path = unresolved_source.resolve()
    elif args.from_config:
        unresolved_source = root / args.from_config
        if unresolved_source.is_symlink():
            raise SystemExit(f"refusing symlink config source: {unresolved_source}")
        source_path = unresolved_source.resolve()
    elif numbers:
        latest = max(numbers)
        source_path = line_dir / f"exp-{latest:03d}.yml"
    else:
        raise SystemExit("the first experiment requires --config with a complete YAML file")
    config = copy.deepcopy(load_yaml(source_path))
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise SystemExit(f"{source_path} must contain an experiment mapping")
    experiment["id"] = experiment_id
    experiment["research_line"] = args.research_line
    if args.goal is not None:
        goal = " ".join(args.goal.split())
        if not goal:
            raise SystemExit("goal must not be empty")
        experiment["goal"] = goal
    if not isinstance(experiment.get("goal"), str) or not experiment["goal"].strip():
        raise SystemExit("experiment.goal must be a non-empty string")
    validate_with_project_adapter(root, config)

    config_content = dump_yaml(config)
    experiments[experiment_id] = {
        "config": f"configs/{args.research_line}/{experiment_id}.yml",
        "status": "planned",
        "config_sha256": None,
        "completed_run": None,
        "promoted_artifact": None,
    }
    registry_before = registry_path.read_text(encoding="utf-8")
    registry_after = dump_yaml(registry)

    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: create {args.research_line}/{experiment_id}")
    print(f"Source: {source_path}")
    show_diff(destination, "", config_content)
    show_diff(registry_path, registry_before, registry_after)
    if not args.apply:
        print("No files changed. Rerun with --apply after reviewing the diff.")
        return 0

    apply_transaction(
        root,
        new_files={destination: config_content},
        replacements={registry_path: registry_after},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
