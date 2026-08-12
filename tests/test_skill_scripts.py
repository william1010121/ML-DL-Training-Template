from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / ".agents/skills/training-manager/scripts"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    _write(
        tmp_path / "pyproject.toml",
        "[project]\nname = 'test-project'\nversion = '0.1.0'\n\n"
        "[tool.mltrain]\nadapter = 'test_project.project:adapter'\n",
    )
    _write(tmp_path / "src/test_project/__init__.py", "")
    _write(
        tmp_path / "src/test_project/project.py",
        "from mltrain.contracts import ExperimentConfig\n\n"
        "class Adapter:\n    config_model = ExperimentConfig\n\n"
        "adapter = Adapter()\n",
    )
    _write(
        tmp_path / "configs/research.md",
        "# Project Research\n\n## Goal\n\nTest.\n\n## Research Lines\n\n"
        "| Research line | Goal | Status | Latest evidence |\n"
        "| --- | --- | --- | --- |\n\n## Global Decisions\n\n"
        "| Commit | Decision | Reason |\n| --- | --- | --- |\n",
    )
    _write(
        tmp_path / "configs/registry.yml",
        yaml.safe_dump({"schema_version": 1, "research_lines": {}}, sort_keys=False),
    )
    return tmp_path


def _run(script: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SKILL_SCRIPTS / script), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def test_new_research_line_is_dry_run_by_default(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    before = (root / "configs/research.md").read_text(encoding="utf-8")

    result = _run(
        "new_research_line.py",
        "baseline",
        "--goal",
        "Establish the baseline",
        "--root",
        str(root),
    )

    assert "DRY-RUN" in result.stdout
    assert not (root / "configs/baseline").exists()
    assert (root / "configs/research.md").read_text(encoding="utf-8") == before
    assert yaml.safe_load((root / "configs/registry.yml").read_text())["research_lines"] == {}


def test_new_research_line_applies_once_and_refuses_overwrite(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    args = (
        "baseline",
        "--goal",
        "Establish the baseline",
        "--root",
        str(root),
        "--apply",
    )

    _run("new_research_line.py", *args)
    second = _run("new_research_line.py", *args, check=False)

    assert (root / "configs/baseline/research.md").is_file()
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr


def test_new_experiment_uses_next_id_and_never_overwrites(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _run(
        "new_research_line.py",
        "baseline",
        "--goal",
        "Establish the baseline",
        "--root",
        str(root),
        "--apply",
    )
    candidate = root / "candidate.yml"
    _write(
        candidate,
        yaml.safe_dump(
                {
                    "schema_version": 1,
                "experiment": {
                    "id": "placeholder",
                    "research_line": "placeholder",
                    "goal": "First run",
                        "primary_metric": {"name": "validation/loss", "direction": "minimize"},
                    },
                    "reproducibility": {"seed": 42, "mode": "strict"},
                    "runtime": {"device": "cpu", "strategy": "single", "world_size": 1},
                    "data": {},
                    "model": {},
                    "training": {},
                    "validation": {},
                    "tracking": {"backend": "none"},
                    "output": {"root": "runs"},
                },
            sort_keys=False,
        ),
    )

    _run(
        "new_experiment.py",
        "baseline",
        "--config",
        str(candidate),
        "--root",
        str(root),
        "--apply",
    )
    first = root / "configs/baseline/exp-001.yml"
    first_before = first.read_bytes()
    _run(
        "new_experiment.py",
        "baseline",
        "--from",
        "configs/baseline/exp-001.yml",
        "--goal",
        "Second run",
        "--root",
        str(root),
        "--apply",
    )

    second = yaml.safe_load((root / "configs/baseline/exp-002.yml").read_text())
    assert first.read_bytes() == first_before
    assert second["experiment"]["id"] == "exp-002"
    assert second["experiment"]["goal"] == "Second run"
    registry = yaml.safe_load((root / "configs/registry.yml").read_text())
    assert list(registry["research_lines"]["baseline"]["experiments"]) == [
        "exp-001",
        "exp-002",
    ]


def test_initializer_removes_the_reference_project_without_stale_task_contracts(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "initialized"
    shutil.copytree(
        ROOT,
        copied,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", ".pytest_cache", "datasets", "runs"
        ),
    )
    subprocess.run(["git", "init", "-q"], cwd=copied, check=True)
    subprocess.run(["git", "add", "."], cwd=copied, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Template Test",
            "-c",
            "user.email=template-test@example.invalid",
            "commit",
            "-qm",
            "template fixture",
        ],
        cwd=copied,
        check=True,
    )

    result = _run(
        "initialize_project.py",
        "--project-name",
        "Fresh Research Project",
        "--package-name",
        "fresh_project",
        "--root",
        str(copied),
        "--apply",
    )

    assert result.returncode == 0
    assert not (copied / "src/ml_training_template").exists()
    assert (copied / "src/fresh_project/data/__init__.py").is_file()
    assert not (copied / "configs/mnist-baseline").exists()
    assert not list((copied / "tests").glob("*mnist*"))
    inspected = [
        path
        for base in (copied / "src", copied / "tests", copied / "scripts", copied / "configs")
        for path in base.rglob("*")
        if path.is_file() and ".agents" not in path.parts
    ]
    stale = [
        path.relative_to(copied).as_posix()
        for path in inspected
        if "mnist" in path.name.lower()
        or "ml_training_template" in path.read_text(encoding="utf-8").lower()
    ]
    assert stale == []


def test_initializer_refuses_unknown_committed_example_named_test(tmp_path: Path) -> None:
    copied = tmp_path / "customized"
    shutil.copytree(
        ROOT,
        copied,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", ".pytest_cache", "datasets", "runs"
        ),
    )
    custom = copied / "tests/test_mnist_custom_research.py"
    custom.write_text(
        '"""User-owned test that the template initializer must not delete."""\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=copied, check=True)
    subprocess.run(["git", "add", "."], cwd=copied, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Template Test",
            "-c",
            "user.email=template-test@example.invalid",
            "commit",
            "-qm",
            "customized template fixture",
        ],
        cwd=copied,
        check=True,
    )

    result = _run(
        "initialize_project.py",
        "--project-name",
        "Fresh Research Project",
        "--package-name",
        "fresh_project",
        "--root",
        str(copied),
        "--apply",
        check=False,
    )

    assert result.returncode != 0
    assert custom.is_file()
    assert "User-owned test" in custom.read_text(encoding="utf-8")
    assert (copied / "src/ml_training_template").is_dir()
