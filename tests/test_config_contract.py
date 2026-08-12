from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import Field

from ml_training_template.project import adapter as mnist_adapter
from mltrain.config import load_config
from mltrain.contracts import ExperimentConfig, StrictModel


class DataConfig(StrictModel):
    root: Path
    batch_size: int = Field(gt=0)


class ModelConfig(StrictModel):
    name: str


class TrainingConfig(StrictModel):
    epochs: int = Field(gt=0)


class ValidationConfig(StrictModel):
    batch_size: int = Field(gt=0)


class FakeConfig(ExperimentConfig):
    data: DataConfig  # type: ignore[assignment]
    model: ModelConfig  # type: ignore[assignment]
    training: TrainingConfig  # type: ignore[assignment]
    validation: ValidationConfig  # type: ignore[assignment]


ADAPTER = SimpleNamespace(config_model=FakeConfig)


def valid_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": {
            "id": "exp-001",
            "research_line": "baseline",
            "goal": "Test the baseline",
            "primary_metric": {"name": "validation/loss", "direction": "minimize"},
        },
        "reproducibility": {"seed": 42, "mode": "strict"},
        "runtime": {"device": "cpu", "strategy": "single", "world_size": 1},
        "data": {"root": "datasets", "batch_size": 16},
        "model": {"name": "tiny"},
        "training": {"epochs": 1},
        "validation": {"batch_size": 32},
        "tracking": {"backend": "none"},
        "output": {"root": "runs"},
    }


def _config_file(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_project_model_strictly_loads_complete_config(tmp_path: Path) -> None:
    config = load_config(_config_file(tmp_path, valid_config()), ADAPTER)
    assert isinstance(config, FakeConfig)
    assert config.experiment.id == "exp-001"
    assert config.data.batch_size == 16


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"surprise": True}), "Extra inputs are not permitted"),
        (
            lambda value: value["data"].update({"surprise": True}),  # type: ignore[union-attr]
            "Extra inputs are not permitted",
        ),
        (
            lambda value: value["runtime"].update({"world_size": 2}),  # type: ignore[union-attr]
            "single strategy requires world_size=1",
        ),
        (
            lambda value: value["output"].update({"root": "/tmp/runs"}),  # type: ignore[union-attr]
            "exactly 'runs'",
        ),
    ],
)
def test_config_rejects_unknown_or_unsafe_values(tmp_path: Path, mutation, message: str) -> None:
    raw = valid_config()
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        load_config(_config_file(tmp_path, raw), ADAPTER)


def test_config_rejects_silent_ddp_mismatch(tmp_path: Path) -> None:
    raw = valid_config()
    raw["runtime"] = {"device": "cuda", "strategy": "ddp", "world_size": 1}
    with pytest.raises(ValueError, match="ddp strategy requires world_size>=2"):
        load_config(_config_file(tmp_path, raw), ADAPTER)


def test_bundled_experiment_matrix_uses_project_strict_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "configs/mnist-baseline").glob("exp-*.yml"))
    loaded = [load_config(path, mnist_adapter) for path in configs]
    assert [config.experiment.id for config in loaded] == ["exp-001", "exp-002", "exp-003"]
    assert [(config.runtime.device, config.runtime.strategy) for config in loaded] == [
        ("cpu", "single"),
        ("cuda", "single"),
        ("cuda", "ddp"),
    ]
