from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate_repo = _load_script("validate_repo.py")
check_readme_sync = _load_script("check_readme_sync.py")


def _write(path: Path, text: str = "placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _valid_repository(root: Path) -> Path:
    for relative in validate_repo.REQUIRED_PATHS:
        _write(root / relative)
    for name in ("training-manager", "add-experiment-logging", "runpod-training"):
        _write(
            root / ".agents/skills" / name / "SKILL.md",
            f"---\nname: {name}\ndescription: Test skill.\n---\n\n# Test\n",
        )
        _write(
            root / ".agents/skills" / name / "agents/openai.yaml",
            "interface:\n"
            f'  display_name: "{name}"\n'
            '  short_description: "A sufficiently long test skill description"\n'
            f'  default_prompt: "Use ${name} to test this repository."\n'
            "policy:\n  allow_implicit_invocation: true\n",
        )
    _write(
        root / "pyproject.toml",
        "[project]\nname = 'test-project'\nversion = '0.1.0'\n\n"
        "[tool.mltrain]\nadapter = 'test_project.project:adapter'\n",
    )
    _write(root / "src/test_project/__init__.py", "")
    _write(
        root / "src/test_project/project.py",
        "from mltrain.contracts import ExperimentConfig\n\n"
        "class Adapter:\n    config_model = ExperimentConfig\n\n"
        "adapter = Adapter()\n",
    )
    _write(
        root / "configs/research.md",
        "# Project Research\n\n## Goal\n\nTest.\n\n## Research Lines\n\n"
        "| Research line | Goal | Status | Latest evidence |\n"
        "| --- | --- | --- | --- |\n"
        "| [baseline](baseline/research.md) | Test. | Planned | — |\n\n"
        "## Global Decisions\n",
    )
    _write(
        root / "configs/baseline/research.md",
        "# Baseline\n\n## Goal\n\nTest.\n\n## Results\n",
    )
    config = root / "configs/baseline/exp-001.yml"
    _write(
        config,
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": {
                    "id": "exp-001",
                    "research_line": "baseline",
                    "goal": "Test",
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
    _write(
        root / "configs/registry.yml",
        yaml.safe_dump(
            {
                "schema_version": 1,
                "research_lines": {
                    "baseline": {
                        "experiments": {
                            "exp-001": {
                                "config": "configs/baseline/exp-001.yml",
                                "status": "planned",
                                "config_sha256": None,
                                "completed_run": None,
                                "promoted_artifact": None,
                            }
                        }
                    }
                },
            },
            sort_keys=False,
        ),
    )
    return config


def _promoted_repository(root: Path) -> Path:
    config = _valid_repository(root)
    artifact = root / "artifacts/baseline/exp-001/test-run"
    artifact.mkdir(parents=True)
    commit = "1" * 40
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    result = {
        "primary_metric_name": "validation/loss",
        "primary_metric": 0.25,
        "metrics": {"validation/loss": 0.25},
    }
    manifest = {
        "run_id": "test-run",
        "source": {"commit": commit},
        "config": {"source_sha256": config_hash},
    }
    files = {
        "manifest.json": yaml.safe_dump(manifest),
        "resolved_config.yml": config.read_text(),
        "result.json": yaml.safe_dump(result),
        "validation.json": yaml.safe_dump({"classification": "completed"}),
    }
    for name, content in files.items():
        _write(artifact / name, content)
    result_hash = hashlib.sha256((artifact / "result.json").read_bytes()).hexdigest()
    _write(
        artifact / "summary.json",
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": "exp-001",
                "research_line": "baseline",
                "source_commit": commit,
                "config_sha256": config_hash,
                "result_sha256": result_hash,
                "primary_metric": {"name": "validation/loss", "value": 0.25},
                "decision": "Promote",
            },
            sort_keys=False,
        ),
    )
    members = (
        "manifest.json",
        "resolved_config.yml",
        "result.json",
        "summary.json",
        "validation.json",
    )
    _write(
        artifact / "checksums.sha256",
        "".join(
            f"{hashlib.sha256((artifact / name).read_bytes()).hexdigest()}  {name}\n"
            for name in members
        ),
    )
    registry_path = root / "configs/registry.yml"
    registry = yaml.safe_load(registry_path.read_text())
    entry = registry["research_lines"]["baseline"]["experiments"]["exp-001"]
    entry.update(
        {
            "status": "promoted",
            "config_sha256": config_hash,
            "completed_run": "runs/baseline/exp-001/test-run",
            "promoted_artifact": "artifacts/baseline/exp-001/test-run",
        }
    )
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    research = root / "configs/baseline/research.md"
    research.write_text(
        research.read_text()
        + f"| exp-001 | `{commit}` | validation/loss = 0.25 | Promote | promoted |\n",
        encoding="utf-8",
    )
    project = root / "configs/research.md"
    project.write_text(
        project.read_text().replace(
            "| Planned | — |", "| Promoted | `exp-001` @ `11111111` |"
        ),
        encoding="utf-8",
    )
    return artifact


def test_repository_validator_accepts_minimal_contract(tmp_path: Path) -> None:
    _valid_repository(tmp_path)
    assert validate_repo.validate(tmp_path) == []


def test_repository_validator_requires_registration_and_line_research(tmp_path: Path) -> None:
    _valid_repository(tmp_path)
    (tmp_path / "configs/baseline/research.md").unlink()
    _write(
        tmp_path / "configs/other/exp-002.yml",
        "experiment:\n  id: exp-002\n  research_line: other\n",
    )

    errors = validate_repo.validate(tmp_path)

    assert any("not registered" in error for error in errors)
    assert any("has no research.md" in error for error in errors)


def test_repository_validator_detects_locked_config_drift(tmp_path: Path) -> None:
    config = _valid_repository(tmp_path)
    registry_path = tmp_path / "configs/registry.yml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entry = registry["research_lines"]["baseline"]["experiments"]["exp-001"]
    entry["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    config.write_text(config.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")

    errors = validate_repo.validate(tmp_path)

    assert any("locked experiment config changed" in error for error in errors)


def test_repository_validator_rejects_secret_files_and_host_paths(tmp_path: Path) -> None:
    _valid_repository(tmp_path)
    _write(tmp_path / ".env", "API" + "_KEY=super-secret-token-value\n")
    config = tmp_path / "configs/baseline/exp-001.yml"
    config.write_text(config.read_text(encoding="utf-8") + "data_root: /Users/alice/data\n")

    errors = validate_repo.validate(tmp_path)

    assert any("environment secret file" in error for error in errors)
    assert any("host-specific absolute path" in error for error in errors)


def test_readme_sync_compares_only_marked_contracts(tmp_path: Path) -> None:
    shared = (
        "<!-- sync:start quickstart -->\n"
        "```bash\nuv run mltrain train --config config.yml\n```\n"
        "<!-- sync:end quickstart -->\n"
    )
    _write(tmp_path / "README.md", "# English\n" + shared)
    _write(tmp_path / "README.zh-TW.md", "# 中文\n" + shared)
    assert check_readme_sync.check(tmp_path) == []

    chinese = tmp_path / "README.zh-TW.md"
    chinese.write_text(chinese.read_text().replace("train", "evaluate"), encoding="utf-8")
    assert check_readme_sync.check(tmp_path) == ["README sync block 'quickstart' differs"]


def test_readme_sync_rejects_duplicated_unmarked_commands(tmp_path: Path) -> None:
    marked = (
        "<!-- sync:start quickstart -->\n```bash\nuv sync\n```\n"
        "<!-- sync:end quickstart -->\n"
    )
    unmarked = "```bash\nuv run mltrain validate --run RUN\n```\n"
    _write(tmp_path / "README.md", marked + unmarked)
    _write(tmp_path / "README.zh-TW.md", marked + unmarked)
    assert check_readme_sync.check(tmp_path) == [
        "duplicated bash block is not synchronized: uv run mltrain validate --run RUN"
    ]


def test_repository_validator_rejects_evidence_path_traversal(tmp_path: Path) -> None:
    config = _valid_repository(tmp_path)
    registry_path = tmp_path / "configs/registry.yml"
    registry = yaml.safe_load(registry_path.read_text())
    entry = registry["research_lines"]["baseline"]["experiments"]["exp-001"]
    entry.update(
        {
            "status": "promoted",
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "completed_run": "runs/baseline/exp-001/../../escape",
            "promoted_artifact": "artifacts/baseline/exp-001/../../../escape",
        }
    )
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    errors = validate_repo.validate(tmp_path)

    assert any("canonical runs directory" in error for error in errors)
    assert any("canonical artifact directory" in error for error in errors)


def test_repository_validator_requires_promoted_artifact_but_not_ignored_run(
    tmp_path: Path,
) -> None:
    config = _valid_repository(tmp_path)
    registry_path = tmp_path / "configs/registry.yml"
    registry = yaml.safe_load(registry_path.read_text())
    entry = registry["research_lines"]["baseline"]["experiments"]["exp-001"]
    entry.update(
        {
            "status": "promoted",
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "completed_run": "runs/baseline/exp-001/run-does-not-exist",
            "promoted_artifact": "artifacts/baseline/exp-001/run-does-not-exist",
        }
    )
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    research = tmp_path / "configs/baseline/research.md"
    research.write_text(
        research.read_text()
        + "| exp-001 | `1111111111111111111111111111111111111111` | "
        "validation/loss = 0.25 | Promote | promoted |\n",
        encoding="utf-8",
    )

    errors = validate_repo.validate(tmp_path)

    assert any("promoted artifact is missing" in error for error in errors)
    assert not any("run-does-not-exist" in error and "runs/" in error for error in errors)


def test_repository_validator_uses_project_adapter_schema(tmp_path: Path) -> None:
    config = _valid_repository(tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw["unexpected"] = True
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    errors = validate_repo.validate(tmp_path)

    assert any("project schema rejected" in error for error in errors)


def test_repository_validator_parses_skill_openai_manifest(tmp_path: Path) -> None:
    _valid_repository(tmp_path)
    manifest = tmp_path / ".agents/skills/training-manager/agents/openai.yaml"
    value = yaml.safe_load(manifest.read_text())
    value["interface"]["default_prompt"] = "This forgot the explicit skill invocation."
    manifest.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    errors = validate_repo.validate(tmp_path)

    assert any("default_prompt must mention $training-manager" in error for error in errors)


def test_repository_validator_checks_project_research_index_state(tmp_path: Path) -> None:
    _valid_repository(tmp_path)
    research = tmp_path / "configs/research.md"
    research.write_text(
        research.read_text().replace("| Planned | — |", "| Promoted | `exp-001` @ `11111111` |"),
        encoding="utf-8",
    )

    errors = validate_repo.validate(tmp_path)

    assert any("should show baseline as Planned with no evidence" in error for error in errors)


def test_repository_validator_accepts_complete_promoted_evidence(tmp_path: Path) -> None:
    _promoted_repository(tmp_path)
    assert validate_repo.validate(tmp_path) == []


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("extra", "file set is invalid"),
        ("checksum", "checksum mismatch"),
        ("summary", "summary experiment is inconsistent"),
    ],
)
def test_repository_validator_rejects_tampered_promoted_evidence(
    tmp_path: Path, tamper: str, message: str
) -> None:
    artifact = _promoted_repository(tmp_path)
    if tamper == "extra":
        _write(artifact / "unexpected.txt", "not curated\n")
    elif tamper == "checksum":
        with (artifact / "result.json").open("a", encoding="utf-8") as stream:
            stream.write("# tampered\n")
    else:
        summary = yaml.safe_load((artifact / "summary.json").read_text())
        summary["experiment"] = "exp-999"
        (artifact / "summary.json").write_text(yaml.safe_dump(summary), encoding="utf-8")
        checksums = (artifact / "checksums.sha256").read_text().splitlines()
        checksums = [
            (
                hashlib.sha256((artifact / "summary.json").read_bytes()).hexdigest()
                + "  summary.json"
                if line.endswith("  summary.json")
                else line
            )
            for line in checksums
        ]
        (artifact / "checksums.sha256").write_text("\n".join(checksums) + "\n")

    errors = validate_repo.validate(tmp_path)

    assert any(message in error for error in errors)
