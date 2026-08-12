from __future__ import annotations

from pathlib import Path

import pytest

from mltrain import provenance, runtime
from mltrain.contracts import ExperimentConfig
from mltrain.runtime import distributed_state


def _ddp_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "id": "exp-001",
                "research_line": "baseline",
                "goal": "Test DDP bounds",
                "primary_metric": {"name": "validation/loss", "direction": "minimize"},
            },
            "reproducibility": {"seed": 42, "mode": "strict"},
            "runtime": {"device": "cpu", "strategy": "ddp", "world_size": 2},
            "data": {},
            "model": {},
            "training": {},
            "validation": {},
            "tracking": {"backend": "none"},
            "output": {"root": "runs"},
        }
    )


def test_git_status_failure_is_dirty_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_status(root: Path, *arguments: str) -> str | None:
        del root
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("cat-file", "-e", f"{'a' * 40}^{{commit}}"):
            return ""
        return None

    monkeypatch.setattr(provenance, "_git", failing_status)
    monkeypatch.setattr(provenance, "_git_bytes", lambda *_args: None)

    state = provenance.source_state(tmp_path)

    assert state["commit"] is None
    assert state["dirty"] is True
    assert isinstance(state["dirty_fingerprint"], str)
    assert len(state["dirty_fingerprint"]) == 64


def test_all_git_command_failures_are_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "_git", lambda *_args: None)
    monkeypatch.setattr(provenance, "_git_bytes", lambda *_args: None)

    state = provenance.source_state(tmp_path)

    assert state["commit"] is None
    assert state["dirty"] is True
    assert isinstance(state["dirty_fingerprint"], str)


@pytest.mark.parametrize(
    ("rank", "local_rank"),
    [(-1, 0), (2, 0), (0, -1), (0, 2)],
)
def test_ddp_rejects_rank_and_local_rank_outside_world_size(
    monkeypatch: pytest.MonkeyPatch, rank: int, local_rank: int
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("MLTRAIN_RUN_ID", "shared-run")

    with pytest.raises(RuntimeError, match=r"RANK|LOCAL_RANK"):
        distributed_state(_ddp_config())


def test_ddp_accepts_rank_and_local_rank_inside_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("MLTRAIN_RUN_ID", "shared-run")
    assert distributed_state(_ddp_config()) == (1, 2)


def test_cli_command_keeps_repository_python_path_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / ".venv/bin/python"
    monkeypatch.setattr(runtime, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(runtime.sys, "executable", str(interpreter))
    monkeypatch.setattr(runtime.sys, "argv", ["mltrain", "train", "--config", "exp.yml"])

    assert runtime.cli_command() == [
        ".venv/bin/python",
        "-m",
        "mltrain",
        "train",
        "--config",
        "exp.yml",
    ]
