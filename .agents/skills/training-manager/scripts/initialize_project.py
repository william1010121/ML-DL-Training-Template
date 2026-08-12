#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import PACKAGE_RE, dump_yaml, reject_symlinks, repo_root, show_diff

TEXT_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml"}
SKIP_PARTS = {".git", ".venv", "datasets", "checkpoints", "runs"}
MNIST_PATHS = (
    "configs/mnist-baseline",
    "artifacts/mnist-baseline",
    "scripts/download_mnist.py",
    "scripts/smoke_cpu.py",
    "scripts/check_readme_sync.py",
    "tests/test_config_contract.py",
    "tests/test_skill_scripts.py",
    "tests/test_mnist_runtime.py",
    "tests/test_mnist.py",
    "tests/test_mnist_data.py",
    "tests/test_mnist_smoke.py",
    "scripts/__pycache__",
    "tests/__pycache__",
)
REQUIRED_TEMPLATE_PATHS = (
    "README.md",
    "README.zh-TW.md",
    "uv.lock",
    "configs/mnist-baseline/research.md",
    "configs/mnist-baseline/exp-001.yml",
    "configs/mnist-baseline/exp-002.yml",
    "configs/mnist-baseline/exp-003.yml",
    "scripts/download_mnist.py",
    "scripts/smoke_cpu.py",
    "scripts/check_readme_sync.py",
    "tests/test_config_contract.py",
    "tests/test_skill_scripts.py",
    "src/ml_training_template/config.py",
    "src/ml_training_template/data/mnist.py",
    "src/ml_training_template/model/mnist.py",
    "src/ml_training_template/training/mnist.py",
    "src/ml_training_template/validate/mnist.py",
    "src/ml_training_template/project.py",
)
MNIST_PACKAGE_PATHS = (
    "config.py",
    "data/mnist.py",
    "model/mnist.py",
    "training/mnist.py",
    "validate/mnist.py",
    "project.py",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Rename the project package and remove the MNIST reference implementation."
    )
    value.add_argument("--project-name", required=True, help="human-readable project name")
    value.add_argument("--package-name", required=True, help="Python import package name")
    value.add_argument("--root", default=".", help="repository root (default: current directory)")
    value.add_argument("--apply", action="store_true", help="write changes; default is dry-run")
    return value


def distribution_name(project_name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")
    if not value:
        raise SystemExit("project name must contain letters or digits")
    return value


def project_identity(pyproject_path: Path) -> tuple[str, str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("name"), str):
        raise SystemExit("pyproject.toml must contain project.name")
    mltrain = data.get("tool", {}).get("mltrain", {})
    adapter = mltrain.get("adapter") if isinstance(mltrain, dict) else None
    if not isinstance(adapter, str) or ":" not in adapter:
        raise SystemExit("pyproject.toml must contain tool.mltrain.adapter before initialization")
    old_package = adapter.split(":", 1)[0].split(".", 1)[0]
    if not PACKAGE_RE.fullmatch(old_package):
        raise SystemExit(f"cannot derive the current package from adapter: {adapter}")
    return project["name"], old_package


def git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise SystemExit(f"Git preflight failed: {detail.strip()}") from error
    return result.stdout


def preflight_template(root: Path) -> set[str]:
    top = Path(git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != root:
        raise SystemExit(f"--root must be the Git repository root: {top}")
    git(root, "rev-parse", "--verify", "HEAD")
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SystemExit("initialization requires a clean Git tree; commit or stash every change")
    tracked_output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout
    tracked = {value.decode() for value in tracked_output.split(b"\0") if value}
    missing = [value for value in REQUIRED_TEMPLATE_PATHS if value not in tracked]
    if missing:
        raise SystemExit("not the untouched template; missing tracked files: " + ", ".join(missing))
    for relative in tracked:
        path = root / relative
        if path.is_symlink():
            raise SystemExit(f"refusing tracked symlink: {relative}")
    return tracked


def rewrite_pyproject(
    content: str, old_distribution: str, new_distribution: str, old_package: str, new_package: str
) -> str:
    content = content.replace(old_package, new_package)
    lines = content.splitlines(keepends=True)
    section = ""
    output: list[str] = []
    found_name = False
    removed_adapter = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        if section == "[project]" and re.match(r"^\s*name\s*=", line):
            ending = "\n" if line.endswith("\n") else ""
            output.append(f'name = "{new_distribution}"{ending}')
            found_name = True
            continue
        if section == "[tool.mltrain]" and re.match(r"^\s*adapter\s*=", line):
            removed_adapter = True
            continue
        output.append(line)
    if not found_name or not removed_adapter:
        raise SystemExit("could not update project.name and remove tool.mltrain.adapter")
    return "".join(output).replace(old_distribution, new_distribution)


def candidate_text_files(root: Path) -> list[Path]:
    values: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if relative.parts[:3] == (".agents", "skills", "training-manager"):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            values.append(path)
    return sorted(values)


def initialized_readme(project_name: str, package_name: str, *, chinese: bool) -> str:
    if chinese:
        lines = [
            f"# {project_name}",
            "",
            "這個 repository 已由 ML/DL Training Template 初始化。",
            "通用的 `mltrain` 治理核心、研究紀錄與 local-first evidence contract 仍然保留。",
            "範例任務已移除。",
            "",
            "## 下一步",
            "",
            f"1. 在 `src/{package_name}/` 實作 data、model、training 與 validate。",
            f"2. 實作 `{package_name}.project:adapter`, 再於 `[tool.mltrain]` 設定 `adapter`。",
            "3. 執行 `uv lock` 與 `uv sync --extra cpu`。",
            "4. 用 training-manager 建立 research line 與通過 project schema 的 config。",
            "5. 執行測試後再開始訓練。沒有 validated evidence 不更新支援聲明。",
            "",
            "完整 contract 請見 `AGENTS.md` 與 `.agents/skills/training-manager/`。",
        ]
        return "\n".join(lines) + "\n"
    lines = [
        f"# {project_name}",
        "",
        "This repository was initialized from the ML/DL Training Template.",
        "The stable `mltrain` governance core and local-first evidence contract remain.",
        "The example task has been removed.",
        "",
        "## Next steps",
        "",
        f"1. Implement data, model, training, and validation under `src/{package_name}/`.",
        f"2. Implement `{package_name}.project:adapter` and configure `[tool.mltrain]`.",
        "3. Run `uv lock` and `uv sync --extra cpu`.",
        "4. Use training-manager to create a research line and a schema-valid config.",
        "5. Run tests before training; require validated evidence for support claims.",
        "",
        "See `AGENTS.md` and `.agents/skills/training-manager/` for the contract.",
    ]
    return "\n".join(lines) + "\n"


def initialized_ci() -> str:
    return """name: CI

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  quality:
    runs-on: ubuntu-24.04
    strategy:
      fail-fast: false
      matrix:
        python: ["3.11", "3.12", "3.13"]
    env:
      UV_PYTHON: ${{ matrix.python }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          version: "0.12.0"
          enable-cache: true
      - run: uv python install "${UV_PYTHON}"
      - run: uv sync --locked --extra cpu --group dev
      - run: uv run --no-sync ruff check .
      - run: uv run --no-sync mypy src
      - run: uv run --no-sync pytest
      - run: uv run --no-sync python scripts/validate_repo.py
      - run: uv run --no-sync python scripts/validate_skills.py
"""


def topmost(paths: set[Path]) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(paths, key=lambda value: len(value.parts)):
        if not any(parent == path or parent in path.parents for parent in selected):
            selected.append(path)
    return selected


def copy_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def main() -> int:
    args = parser().parse_args()
    project_name = " ".join(args.project_name.split())
    if not project_name:
        raise SystemExit("project name must not be empty")
    if not PACKAGE_RE.fullmatch(args.package_name):
        raise SystemExit("package name must be lowercase snake_case and start with a letter")

    requested_root = Path(args.root).expanduser()
    if requested_root.is_symlink():
        raise SystemExit(f"refusing symlink repository root: {requested_root}")
    root = repo_root(args.root)
    reject_symlinks(root)
    tracked = preflight_template(root)
    pyproject_path = root / "pyproject.toml"
    old_distribution, old_package = project_identity(pyproject_path)
    if old_distribution != "ml-dl-training-template" or old_package != "ml_training_template":
        raise SystemExit(
            "initializer only accepts the untouched ml-dl-training-template identity"
        )
    new_distribution = distribution_name(project_name)
    if old_package == args.package_name:
        raise SystemExit("new package name must differ from the template package name")
    old_package_dir = root / "src" / old_package
    new_package_dir = root / "src" / args.package_name
    if not old_package_dir.is_dir():
        raise SystemExit(f"current package directory is missing: {old_package_dir}")
    if new_package_dir.exists():
        raise SystemExit(f"refusing to overwrite package directory: {new_package_dir}")

    removal_candidates = [root / value for value in MNIST_PATHS]
    known_mnist_tests = {
        root / value for value in MNIST_PATHS if value.startswith("tests/test_mnist")
    }
    unexpected_mnist_tests = sorted(
        path for path in (root / "tests").glob("test_mnist*.py") if path not in known_mnist_tests
    )
    if unexpected_mnist_tests:
        names = ", ".join(path.relative_to(root).as_posix() for path in unexpected_mnist_tests)
        raise SystemExit(
            "not the untouched template; refusing unexpected MNIST-named tests: " + names
        )
    removal_candidates.extend(sorted(old_package_dir.rglob("__pycache__")))
    removal_candidates.extend(old_package_dir / value for value in MNIST_PACKAGE_PATHS)
    removals = list(dict.fromkeys(removal_candidates))
    existing_removals = [path for path in removals if path.exists()]
    for path in existing_removals:
        relative = path.relative_to(root).as_posix()
        covered = relative in tracked or any(value.startswith(f"{relative}/") for value in tracked)
        if not covered and path.name != "__pycache__":
            raise SystemExit(f"refusing to delete untracked template content: {relative}")
    lock_path = root / "uv.lock"

    rewrites: dict[Path, tuple[str, str]] = {}
    for path in candidate_text_files(root):
        if path == pyproject_path or any(
            path == target or target in path.parents for target in removals
        ):
            continue
        before = path.read_text(encoding="utf-8")
        after = before.replace(old_package, args.package_name).replace(
            old_distribution, new_distribution
        )
        after = after.replace("ML-DL-Training-Template", project_name)
        if before != after:
            rewrites[path] = (before, after)

    pyproject_before = pyproject_path.read_text(encoding="utf-8")
    pyproject_after = rewrite_pyproject(
        pyproject_before,
        old_distribution,
        new_distribution,
        old_package,
        args.package_name,
    )
    rewrites[pyproject_path] = (pyproject_before, pyproject_after)

    research_path = root / "configs" / "research.md"
    research_before = research_path.read_text(encoding="utf-8") if research_path.exists() else ""
    research_after = (
        "# Project Research\n\n## Goal\n\n"
        f"Define and validate the research goal for {project_name}.\n\n"
        "## Research Lines\n\n"
        "| Research line | Goal | Status | Latest evidence |\n"
        "| --- | --- | --- | --- |\n\n"
        "## Global Decisions\n\n"
        "| Commit | Decision | Reason |\n"
        "| --- | --- | --- |\n"
    )
    rewrites[research_path] = (research_before, research_after)
    registry_path = root / "configs" / "registry.yml"
    registry_before = registry_path.read_text(encoding="utf-8") if registry_path.exists() else ""
    registry_after = dump_yaml({"schema_version": 1, "research_lines": {}})
    rewrites[registry_path] = (registry_before, registry_after)
    readme_path = root / "README.md"
    readme_zh_path = root / "README.zh-TW.md"
    rewrites[readme_path] = (
        readme_path.read_text(encoding="utf-8"),
        initialized_readme(project_name, args.package_name, chinese=False),
    )
    rewrites[readme_zh_path] = (
        readme_zh_path.read_text(encoding="utf-8"),
        initialized_readme(project_name, args.package_name, chinese=True),
    )
    docker_run = root / "scripts" / "docker-run"
    if docker_run.is_file():
        before = docker_run.read_text(encoding="utf-8")
        after = re.sub(
            r"configs/mnist-baseline/exp-00[123]\.yml",
            "configs/RESEARCH_LINE/exp-001.yml",
            before,
        )
        rewrites[docker_run] = (before, after)
    ci_path = root / ".github" / "workflows" / "ci.yml"
    if ci_path.is_file():
        rewrites[ci_path] = (ci_path.read_text(encoding="utf-8"), initialized_ci())

    clean_initializers = {
        old_package_dir / "__init__.py": '"""Project-specific implementation package."""\n',
        old_package_dir / "data" / "__init__.py": '"""Project-specific data interfaces."""\n',
        old_package_dir / "model" / "__init__.py": '"""Project-specific model interfaces."""\n',
        old_package_dir / "training" / "__init__.py": (
            '"""Project-specific training implementations."""\n'
        ),
        old_package_dir / "validate" / "__init__.py": (
            '"""Project-specific validation hooks."""\n'
        ),
    }
    for path, after in clean_initializers.items():
        if path.is_file():
            rewrites[path] = (path.read_text(encoding="utf-8"), after)

    for path, (_, after) in rewrites.items():
        if re.search(r"mnist|download_mnist", after, re.IGNORECASE):
            raise SystemExit(f"unhandled example reference remains in {path.relative_to(root)}")

    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: initialize {project_name}")
    print(f"Rename: src/{old_package} -> src/{args.package_name}")
    for path in existing_removals:
        print(f"Remove MNIST example: {path.relative_to(root)}")
    print("Remove stale lockfile: uv.lock")
    for path, (before, after) in rewrites.items():
        show_diff(path, before, after)
    if not args.apply:
        print("No files changed. Rerun with --apply after reviewing the diff.")
        return 0

    affected = set(existing_removals) | set(rewrites) | {old_package_dir, lock_path}
    backup_targets = topmost({path for path in affected if path.exists()})
    with tempfile.TemporaryDirectory(prefix="mltrain-initialize-") as temporary:
        backup_root = Path(temporary)
        backups: list[tuple[Path, Path]] = []
        for index, path in enumerate(backup_targets):
            backup = backup_root / str(index)
            copy_path(path, backup)
            backups.append((path, backup))
        try:
            for path in existing_removals:
                remove_path(path)
            remove_path(lock_path)
            old_package_dir.rename(new_package_dir)
            for original_path, (_, after) in rewrites.items():
                path = original_path
                if old_package_dir == original_path or old_package_dir in original_path.parents:
                    path = new_package_dir / original_path.relative_to(old_package_dir)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary_file = path.with_name(f".{path.name}.initialize.tmp")
                temporary_file.write_text(after, encoding="utf-8")
                os.replace(temporary_file, path)
        except BaseException:
            remove_path(new_package_dir)
            for path, _ in backups:
                remove_path(path)
            for path, backup in backups:
                copy_path(backup, path)
            raise
    print(
        "Initialization complete. Implement a ProjectAdapter, restore tool.mltrain.adapter, "
        "then run uv lock; the stale template uv.lock was removed intentionally."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
