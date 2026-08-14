from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mltrain import lifecycle
from mltrain.contracts import ExperimentConfig, RunContext, RunResult, ValidationResult
from mltrain.profiling import RunProfiler


class FakeAdapter:
    config_model = ExperimentConfig

    def train(self, config: ExperimentConfig, context: RunContext) -> RunResult:
        context.log_metrics(1, {"validation/loss": 0.25})
        return RunResult(
            primary_metric_name="validation/loss",
            primary_metric=0.25,
            metrics={"validation/loss": 0.25},
            data_sha256="a" * 64,
            model_sha256="b" * 64,
        )

    def evaluate(
        self, config: ExperimentConfig, checkpoint: Path, context: RunContext
    ) -> RunResult:
        return self.train(config, context)

    def validate(self, config: ExperimentConfig, run_dir: Path) -> ValidationResult:
        return ValidationResult(passed=True, checks={"metric_finite": True})


ADAPTER = FakeAdapter()


def _config(*, mode: str = "strict") -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": {
            "id": "exp-001",
            "research_line": "baseline",
            "goal": "Test",
            "primary_metric": {"name": "validation/loss", "direction": "minimize"},
        },
        "reproducibility": {"seed": 42, "mode": mode},
        "runtime": {"device": "cpu", "strategy": "single", "world_size": 1},
        "data": {},
        "model": {},
        "training": {},
        "validation": {},
        "tracking": {"backend": "none"},
        "output": {"root": "runs"},
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _repository(
    tmp_path: Path, *, mode: str = "strict", dirty: bool = False
) -> tuple[Path, Path]:
    root = tmp_path
    (root / "pyproject.toml").write_text("[tool.mltrain]\nadapter='fake:adapter'\n")
    config_path = root / "configs/baseline/exp-001.yml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(_config(mode=mode), sort_keys=False), encoding="utf-8")
    (root / "configs/baseline/research.md").write_text(
        "# baseline\n\n## Goal\n\nTest.\n\n## Results\n\n"
        "| Experiment | Commit | Primary result | Decision | Status |\n"
        "| --- | --- | --- | --- | --- |\n",
        encoding="utf-8",
    )
    (root / "configs/research.md").write_text(
        "# Project Research\n\n## Goal\n\nTest.\n\n## Research Lines\n\n"
        "| Research line | Goal | Status | Latest evidence |\n"
        "| --- | --- | --- | --- |\n"
        "| [baseline](baseline/research.md) | Test. | Planned | — |\n\n"
        "## Global Decisions\n\n"
        "| Commit | Decision | Reason |\n"
        "| --- | --- | --- |\n",
        encoding="utf-8",
    )
    registry = {
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
    }
    (root / "configs/registry.yml").write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    (root / "artifacts").mkdir()
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("runs/\nartifacts/*\n", encoding="utf-8")
    (root / "artifacts/README.md").write_text("# Artifacts\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Lifecycle Test",
            "-c",
            "user.email=lifecycle-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run = root / "runs/baseline/exp-001/test-run"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        '{"step": 1, "validation/loss": 0.25}\n', encoding="utf-8"
    )
    (run / "checkpoint.pt").write_bytes(b"test checkpoint")
    resolved = yaml.safe_dump(_config(mode=mode), sort_keys=True)
    (run / "resolved_config.yml").write_text(resolved, encoding="utf-8")
    source_hash = lifecycle.sha256_file(config_path)
    lock_hash = lifecycle.sha256_file(root / "uv.lock")
    environment = {
        "python": "3.12.0",
        "platform": "test",
        "machine": "test",
        "packages": {},
        "lockfile_sha256": lock_hash,
        "container": "native",
        "image_digest": None,
        "oci_digest": None,
        "sif_sha256": None,
    }
    environment_hash = hashlib.sha256(
        json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "run_id": "test-run",
        "kind": "train",
        "status": "succeeded",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "source": {
            "commit": source_commit,
            "dirty": dirty,
            "dirty_fingerprint": "f" * 64 if dirty else None,
        },
        "config": {
            "path": "configs/baseline/exp-001.yml",
            "source_sha256": source_hash,
            "resolved_sha256": lifecycle.sha256_file(run / "resolved_config.yml"),
        },
        "experiment": _config(mode=mode)["experiment"],
        "command": [
            "python",
            "-m",
            "mltrain",
            "train",
            "--config",
            str(config_path),
        ],
        "reproducibility": {
            "seed": 42,
            "mode": mode,
            "deterministic_algorithms": mode == "strict",
            "cudnn_benchmark": mode != "strict",
        },
        "runtime": {"device": "cpu", "strategy": "single", "world_size": 1},
        "provenance": {
            "data_sha256": "a" * 64,
            "model_sha256": lifecycle.sha256_file(run / "checkpoint.pt"),
            **environment,
            "environment_sha256": environment_hash,
        },
    }
    _write_json(
        run / "result.json",
        {
            "primary_metric_name": "validation/loss",
            "primary_metric": 0.25,
            "metrics": {"validation/loss": 0.25},
            "checkpoint": "checkpoint.pt",
            "data_sha256": "a" * 64,
            "model_sha256": lifecycle.sha256_file(run / "checkpoint.pt"),
        },
    )
    manifest["result_sha256"] = lifecycle.sha256_file(run / "result.json")
    _write_json(run / "manifest.json", manifest)
    return root, run


def _patch_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle, "load_adapter", lambda: ADAPTER)


def _validate(root: Path, run: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.chdir(root)
    _patch_adapter(monkeypatch)
    return lifecycle.validate_run(run)


def _enable_profile(root: Path, run: Path) -> None:
    config_path = root / "configs/baseline/exp-001.yml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["profiling"] = {"enabled": True, "sample_interval_seconds": 60.0}
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    (run / "resolved_config.yml").write_text(
        yaml.safe_dump(raw, sort_keys=True), encoding="utf-8"
    )
    subprocess.run(["git", "add", str(config_path)], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Lifecycle Test",
            "-c",
            "user.email=lifecycle-test@example.invalid",
            "commit",
            "-qm",
            "enable profile",
        ],
        cwd=root,
        check=True,
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    profiler = RunProfiler(
        run,
        rank=0,
        is_primary=True,
        device="cpu",
        interval_seconds=60.0,
    )
    profiler.start()
    with profiler.stage("lifecycle/setup"):
        pass
    profiler.stop()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["source"]["commit"] = source_commit
    manifest["config"]["source_sha256"] = lifecycle.sha256_file(config_path)
    manifest["config"]["resolved_sha256"] = lifecycle.sha256_file(
        run / "resolved_config.yml"
    )
    manifest["profiling"] = profiler.evidence()
    _write_json(run / "manifest.json", manifest)


def test_record_result_locks_config_and_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    validation = _validate(root, run, monkeypatch)
    assert validation["classification"] == "completed"

    lifecycle.record_result(run, "Keep baseline")
    registry = yaml.safe_load((root / "configs/registry.yml").read_text())
    entry = registry["research_lines"]["baseline"]["experiments"]["exp-001"]
    assert entry["status"] == "completed"
    assert entry["config_sha256"] == lifecycle.sha256_file(
        root / "configs/baseline/exp-001.yml"
    )
    assert entry["completed_run"] == "runs/baseline/exp-001/test-run"
    research = (root / "configs/baseline/research.md").read_text()
    assert "validation/loss = 0.25" in research
    project_research = (root / "configs/research.md").read_text()
    commit = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert f"| Completed | `exp-001` @ `{commit}` |" in project_research

    with pytest.raises(ValueError, match="already recorded"):
        lifecycle.record_result(run, "Rewrite history")
    assert (root / "configs/baseline/research.md").read_text() == research


def test_promotion_gate_rejects_exploratory_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path, mode="performance")
    validation = _validate(root, run, monkeypatch)
    assert validation["classification"] == "exploratory"
    assert validation["checks"]["strict_reproducibility"] is False

    with pytest.raises(ValueError, match="only a completed run"):
        lifecycle.promote_run(run, "Do not promote")
    assert {path.name for path in (root / "artifacts").iterdir()} == {"README.md"}


def test_completed_run_promotes_small_reviewable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    validation = _validate(root, run, monkeypatch)
    assert validation["classification"] == "completed"
    lifecycle.record_result(run, "Promote baseline")

    destination = lifecycle.promote_run(run, "Promote baseline")

    assert destination.is_dir()
    assert {path.name for path in destination.iterdir()} == {
        "manifest.json",
        "result.json",
        "validation.json",
        "resolved_config.yml",
        "summary.json",
        "checksums.sha256",
    }
    registry = yaml.safe_load((root / "configs/registry.yml").read_text())
    entry = registry["research_lines"]["baseline"]["experiments"]["exp-001"]
    assert entry["status"] == "promoted"
    assert entry["promoted_artifact"] == "artifacts/baseline/exp-001/test-run"
    assert "| promoted |" in (root / "configs/baseline/research.md").read_text()
    commit = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert f"| Promoted | `exp-001` @ `{commit}` |" in (root / "configs/research.md").read_text()


def test_profiled_run_promotes_only_the_small_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    _enable_profile(root, run)
    validation = _validate(root, run, monkeypatch)
    assert validation["classification"] == "completed"

    artifact = lifecycle.promote_run(run, "Promote profiled baseline")

    assert (artifact / "profile-summary.json").is_file()
    assert not (artifact / "stages.rank-000.jsonl").exists()
    assert not (artifact / "resources.rank-000.jsonl").exists()
    summary = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
    assert summary["profile_summary_sha256"] == lifecycle.sha256_file(
        artifact / "profile-summary.json"
    )


def test_planned_run_promotes_and_records_all_evidence_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    validation = _validate(root, run, monkeypatch)
    assert validation["classification"] == "completed"

    destination = lifecycle.promote_run(run, "Directly promote baseline")

    registry = yaml.safe_load((root / "configs/registry.yml").read_text())
    entry = registry["research_lines"]["baseline"]["experiments"]["exp-001"]
    assert entry == {
        "config": "configs/baseline/exp-001.yml",
        "status": "promoted",
        "config_sha256": lifecycle.sha256_file(root / "configs/baseline/exp-001.yml"),
        "completed_run": "runs/baseline/exp-001/test-run",
        "promoted_artifact": "artifacts/baseline/exp-001/test-run",
    }
    line_research = (root / "configs/baseline/research.md").read_text()
    assert line_research.count("| exp-001 |") == 1
    assert "| Directly promote baseline | promoted |" in line_research
    assert "| Promoted | `exp-001` @ `" in (root / "configs/research.md").read_text()
    assert destination.is_dir()


@pytest.mark.parametrize("symlink_level", ["root", "line"])
def test_promotion_rejects_artifact_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symlink_level: str
) -> None:
    root, run = _repository(tmp_path)
    _validate(root, run, monkeypatch)
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    artifacts = root / "artifacts"
    if symlink_level == "root":
        for path in artifacts.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        artifacts.rmdir()
        artifacts.symlink_to(outside, target_is_directory=True)
    else:
        (artifacts / "baseline").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="artifact path must not contain symlinks"):
        lifecycle.promote_run(run, "Escape attempt")
    assert list(outside.iterdir()) == []


def test_completed_run_promotes_directly_from_planned_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    validation = _validate(root, run, monkeypatch)
    assert validation["classification"] == "completed"

    destination = lifecycle.promote_run(run, "Promote baseline")

    assert destination.is_dir()
    registry = yaml.safe_load((root / "configs/registry.yml").read_text())
    entry = registry["research_lines"]["baseline"]["experiments"]["exp-001"]
    assert entry["status"] == "promoted"
    assert entry["completed_run"] == "runs/baseline/exp-001/test-run"
    assert entry["config_sha256"] == lifecycle.sha256_file(
        root / "configs/baseline/exp-001.yml"
    )
    assert "| promoted |" in (root / "configs/baseline/research.md").read_text()


def test_promotion_rejects_symlinked_artifact_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    _validate(root, run, monkeypatch)
    outside = root / "outside-artifacts"
    outside.mkdir()
    (root / "artifacts/README.md").unlink()
    (root / "artifacts").rmdir()
    (root / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        lifecycle.promote_run(run, "Promote baseline")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("target", "failed_check"),
    [
        ("result", "result_hash"),
        ("resolved", "resolved_config_hash"),
        ("source", "source_config_unchanged"),
        ("manifest", "run_succeeded"),
    ],
)
def test_validate_run_detects_tampered_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    failed_check: str,
) -> None:
    root, run = _repository(tmp_path)
    if target == "result":
        _write_json(run / "result.json", {"primary_metric": 999.0})
    elif target == "resolved":
        with (run / "resolved_config.yml").open("a", encoding="utf-8") as stream:
            stream.write("# tampered\n")
    elif target == "source":
        with (root / "configs/baseline/exp-001.yml").open("a", encoding="utf-8") as stream:
            stream.write("# tampered\n")
    else:
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["status"] = "failed"
        _write_json(run / "manifest.json", manifest)

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "exploratory"
    assert validation["checks"][failed_check] is False


def test_dirty_source_is_exploratory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, run = _repository(tmp_path, dirty=True)
    validation = _validate(root, run, monkeypatch)
    assert validation["classification"] == "exploratory"
    assert validation["checks"]["clean_source_commit"] is False


def test_succeeded_run_requires_ordered_timezone_aware_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["finished_at"] = "2025-12-31T23:59:00+00:00"
    _write_json(run / "manifest.json", manifest)

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "exploratory"
    assert validation["checks"]["run_timestamps"] is False


def test_resolved_config_must_semantically_match_canonical_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    resolved_path = run / "resolved_config.yml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["reproducibility"]["seed"] = 43
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["config"]["resolved_sha256"] = lifecycle.sha256_file(resolved_path)
    manifest["reproducibility"]["seed"] = 43
    _write_json(run / "manifest.json", manifest)

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "exploratory"
    assert validation["checks"]["resolved_config_hash"] is True
    assert validation["checks"]["source_config_semantics"] is False


def test_source_commit_must_exist_in_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["source"]["commit"] = "f" * 40
    _write_json(run / "manifest.json", manifest)

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "exploratory"
    assert validation["checks"]["source_commit"] is True
    assert validation["checks"]["source_commit_exists"] is False


@pytest.mark.parametrize("metrics_value", [None, "", '{"validation/loss": "NaN"}\n'])
def test_canonical_metrics_must_be_nonempty_finite_and_match_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metrics_value: str | None,
) -> None:
    root, run = _repository(tmp_path)
    metrics = run / "metrics.jsonl"
    if metrics_value is None:
        metrics.unlink()
    else:
        metrics.write_text(metrics_value, encoding="utf-8")

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "exploratory"
    assert validation["checks"]["canonical_metrics_present"] is False
    assert validation["checks"]["canonical_metrics_match"] is False


def test_canonical_metrics_best_primary_value_must_match_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    (run / "metrics.jsonl").write_text(
        '{"step": 1, "validation/loss": 0.5}\n', encoding="utf-8"
    )

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "exploratory"
    assert validation["checks"]["canonical_metrics_present"] is True
    assert validation["checks"]["canonical_metrics_match"] is False


def test_nonmonotonic_training_metrics_accept_best_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    (run / "metrics.jsonl").write_text(
        '{"step": 1, "validation/loss": 0.25}\n'
        '{"step": 2, "validation/loss": 0.5}\n',
        encoding="utf-8",
    )

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "completed"
    assert validation["checks"]["canonical_metrics_match"] is True


def test_resolved_config_semantics_must_match_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    resolved = yaml.safe_load((run / "resolved_config.yml").read_text())
    resolved["experiment"]["goal"] = "Tampered semantic goal"
    (run / "resolved_config.yml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["config"]["resolved_sha256"] = lifecycle.sha256_file(
        run / "resolved_config.yml"
    )
    _write_json(run / "manifest.json", manifest)

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "exploratory"
    assert validation["checks"]["source_config_semantics"] is False


def test_completed_evaluate_run_uses_explicit_test_metric_and_archived_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run = _repository(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["kind"] = "evaluate"
    manifest["command"][3] = "evaluate"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["primary_metric_name"] = "test/loss"
    result["metrics"] = {"test/loss": result["primary_metric"]}
    (run / "metrics.jsonl").write_text(
        '{"step": 0, "test/loss": 0.25}\n', encoding="utf-8"
    )
    (run / "metrics.jsonl").write_text(
        '{"step": 0, "test/loss": 0.25}\n', encoding="utf-8"
    )
    _write_json(run / "result.json", result)
    manifest["result_sha256"] = lifecycle.sha256_file(run / "result.json")
    _write_json(run / "manifest.json", manifest)

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "completed"
    assert validation["primary_metric"] == {"name": "test/loss", "value": 0.25}


def test_train_run_cannot_replace_selection_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run = _repository(tmp_path)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["primary_metric_name"] = "test/loss"
    result["metrics"] = {"test/loss": result["primary_metric"]}
    _write_json(run / "result.json", result)
    (run / "metrics.jsonl").write_text(
        '{"step": 0, "test/loss": 0.25}\n', encoding="utf-8"
    )
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["result_sha256"] = lifecycle.sha256_file(run / "result.json")
    _write_json(run / "manifest.json", manifest)

    validation = _validate(root, run, monkeypatch)

    assert validation["classification"] == "exploratory"
    assert validation["checks"]["train_primary_metric_contract"] is False


def test_fabricated_completed_validation_cannot_bypass_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path, mode="performance")
    monkeypatch.chdir(root)
    _patch_adapter(monkeypatch)
    _write_json(run / "validation.json", {"classification": "completed"})

    with pytest.raises(ValueError, match=r"completed|validation"):
        lifecycle.promote_run(run, "Fabricated evidence")
    assert {path.name for path in (root / "artifacts").iterdir()} == {"README.md"}


def test_manifest_run_id_cannot_escape_canonical_artifact_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    _validate(root, run, monkeypatch)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["run_id"] = "../escaped"
    _write_json(run / "manifest.json", manifest)

    with pytest.raises(ValueError, match=r"run id|run_id|canonical"):
        lifecycle.promote_run(run, "Traversal attempt")
    assert not (root / "artifacts/baseline/escaped").exists()


def test_manifest_config_path_must_match_canonical_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    outside = root / "outside.yml"
    shutil.copy2(root / "configs/baseline/exp-001.yml", outside)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["config"]["path"] = "outside.yml"
    manifest["config"]["source_sha256"] = lifecycle.sha256_file(outside)
    _write_json(run / "manifest.json", manifest)

    monkeypatch.chdir(root)
    _patch_adapter(monkeypatch)
    with pytest.raises(ValueError, match="config path is not canonical"):
        lifecycle.validate_run(run)


def test_repeat_promotion_cannot_change_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    _validate(root, run, monkeypatch)
    lifecycle.record_result(run, "Promote baseline")
    lifecycle.promote_run(run, "Promote baseline")

    with pytest.raises(ValueError, match="promoted research row does not match run evidence"):
        lifecycle.promote_run(run, "Different conclusion")


def test_promotion_rolls_back_artifact_if_registry_transaction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = _repository(tmp_path)
    _validate(root, run, monkeypatch)
    lifecycle.record_result(run, "Promote baseline")
    registry_before = (root / "configs/registry.yml").read_bytes()
    line_before = (root / "configs/baseline/research.md").read_bytes()
    project_before = (root / "configs/research.md").read_bytes()

    monkeypatch.setattr(
        lifecycle,
        "_atomic_batch",
        lambda *_args: (_ for _ in ()).throw(OSError("simulated transaction failure")),
    )
    with pytest.raises(OSError, match="transaction failure"):
        lifecycle.promote_run(run, "Promote baseline")

    assert not (root / "artifacts/baseline/exp-001/test-run").exists()
    assert (root / "configs/registry.yml").read_bytes() == registry_before
    assert (root / "configs/baseline/research.md").read_bytes() == line_before
    assert (root / "configs/research.md").read_bytes() == project_before


def test_execute_records_tracker_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig.model_validate(_config())
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = RunContext(run_id="run", run_dir=run_dir, command=["mltrain", "train"])
    (run_dir / "metrics.jsonl").touch()
    manifest: dict[str, object] = {
        "status": "running",
        "reproducibility": {},
        "provenance": {},
        "tracking": {"backend": "broken", "degraded": False},
    }
    degraded_adapter = SimpleNamespace(
        config_model=ExperimentConfig,
        train=lambda _config, _context: RunResult(
            primary_metric_name="validation/loss",
            primary_metric=1.0,
            tracking_degraded=True,
        ),
    )
    monkeypatch.setattr(lifecycle, "load_adapter", lambda: degraded_adapter)
    monkeypatch.setattr(lifecycle, "load_config", lambda _path, _adapter: config)
    monkeypatch.setattr(lifecycle, "ensure_device", lambda _config: None)
    monkeypatch.setattr(
        lifecycle, "configure_reproducibility", lambda _config: {"deterministic_algorithms": True}
    )
    monkeypatch.setattr(lifecycle, "create_run", lambda *_args: (context, manifest))
    written: list[dict[str, object]] = []
    monkeypatch.setattr(
        lifecycle, "write_manifest", lambda _path, value: written.append(value.copy())
    )

    assert lifecycle.execute(config_path, "train") == run_dir
    assert (run_dir / "result.json").is_file()
    assert manifest["status"] == "succeeded"
    assert manifest["tracking"] == {"backend": "broken", "degraded": True}
    assert "progress" not in manifest


def test_execute_attaches_profiler_and_records_manifest_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _config()
    raw["profiling"] = {"enabled": True, "sample_interval_seconds": 0.2}
    config = ExperimentConfig.model_validate(raw)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").touch()
    context = RunContext(run_id="run", run_dir=run_dir, command=["mltrain", "train"])
    manifest: dict[str, object] = {
        "status": "running",
        "reproducibility": {},
        "provenance": {},
        "tracking": {"backend": "none", "degraded": False},
        "profiling": {
            "enabled": True,
            "schema_version": 1,
            "sample_interval_seconds": 0.2,
            "status": "pending",
            "degraded": False,
            "files": {},
        },
    }

    def train(_config: ExperimentConfig, run: RunContext) -> RunResult:
        with run.profile_stage("epoch/train", epoch=1):
            pass
        return RunResult(
            primary_metric_name="validation/loss",
            primary_metric=1.0,
            metrics={"validation/loss": 1.0},
        )

    adapter = SimpleNamespace(config_model=ExperimentConfig, train=train)
    monkeypatch.setattr(lifecycle, "load_adapter", lambda: adapter)
    monkeypatch.setattr(lifecycle, "load_config", lambda _path, _adapter: config)
    monkeypatch.setattr(lifecycle, "ensure_device", lambda _config: None)
    monkeypatch.setattr(lifecycle, "configure_reproducibility", lambda _config: {})
    monkeypatch.setattr(lifecycle, "create_run", lambda *_args: (context, manifest))
    monkeypatch.setattr(lifecycle, "write_manifest", lambda *_args: None)

    assert lifecycle.execute(config_path, "train") == run_dir

    profile = manifest["profiling"]
    assert isinstance(profile, dict)
    assert profile["status"] == "completed"
    assert profile["degraded"] is False
    assert set(profile["files"]) == {
        "stages.rank-000.jsonl",
        "resources.rank-000.jsonl",
        "summary.rank-000.json",
    }
    stages = (run_dir / "profile/stages.rank-000.jsonl").read_text(encoding="utf-8")
    assert '"stage":"lifecycle/setup"' in stages
    assert '"stage":"epoch/train"' in stages
    assert '"stage":"lifecycle/finalize"' in stages
