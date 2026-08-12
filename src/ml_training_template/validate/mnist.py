"""Semantic validation for MNIST result files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_training_template.config import MNISTConfig
    from mltrain.contracts import ValidationResult


def validate_mnist_run(config: MNISTConfig, run_dir: Path) -> ValidationResult:
    """Validate project outputs; generic provenance is validated by mltrain core."""

    from ml_training_template.data.mnist import resolve_data_root
    from ml_training_template.evidence import sha256_tree
    from mltrain.contracts import ValidationResult

    checks: dict[str, bool] = {}
    result_path = run_dir / "result.json"
    checks["result_exists"] = result_path.is_file()
    if not result_path.is_file():
        return ValidationResult(passed=False, checks=checks, notes=["result.json is missing"])

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return ValidationResult(
            passed=False,
            checks=checks,
            notes=[f"invalid result.json: {error}"],
        )
    if not isinstance(result, dict):
        return ValidationResult(
            passed=False,
            checks=checks,
            notes=["result.json must contain an object"],
        )

    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return ValidationResult(
            passed=False,
            checks={**checks, "manifest_readable": False},
            notes=[f"invalid manifest.json: {error}"],
        )
    if not isinstance(manifest, dict):
        return ValidationResult(
            passed=False,
            checks={**checks, "manifest_readable": False},
            notes=["manifest.json must contain an object"],
        )

    value = result.get("primary_metric")
    metric_name = result.get("primary_metric_name")
    metrics = result.get("metrics")
    kind = manifest.get("kind")
    checkpoint_value = result.get("checkpoint")
    checkpoint_path = Path(checkpoint_value) if isinstance(checkpoint_value, str) else Path("/")
    checks["primary_metric_finite"] = isinstance(value, (int, float)) and math.isfinite(value)
    checks["checkpoint_portable"] = (
        isinstance(checkpoint_value, str)
        and bool(checkpoint_value)
        and not checkpoint_path.is_absolute()
        and ".." not in checkpoint_path.parts
    )
    checks["recognized_run_kind"] = kind in {"train", "evaluate"}
    result_data_sha256 = result.get("data_sha256")
    try:
        current_data_sha256 = sha256_tree(resolve_data_root(config.data.root) / "MNIST")
    except (OSError, ValueError) as error:
        checks["dataset_authenticity"] = False
        dataset_note = f"could not verify MNIST dataset: {error}"
    else:
        checks["dataset_authenticity"] = result_data_sha256 == current_data_sha256
        dataset_note = "MNIST dataset hash differs from the run result"
    if kind == "train":
        checks["metric_semantics"] = metric_name == config.experiment.primary_metric.name
        checks["metrics_match_primary"] = (
            isinstance(metrics, dict)
            and metrics.get(metric_name) == value
        )
        checks["checkpoint_exists"] = checks["checkpoint_portable"] and (
            run_dir / checkpoint_path
        ).is_file()
    elif kind == "evaluate":
        checks["metric_semantics"] = metric_name == "test/loss"
        checks["metrics_match_primary"] = isinstance(metrics, dict) and (
            metrics.get(metric_name) == value
        )
        checks["checkpoint_exists"] = checks["checkpoint_portable"] and (
            run_dir / checkpoint_path
        ).is_file()
    errors = [name for name, passed in checks.items() if not passed]
    notes = list(errors)
    if not checks["dataset_authenticity"]:
        notes.append(dataset_note)
    return ValidationResult(passed=not errors, checks=checks, notes=notes)
