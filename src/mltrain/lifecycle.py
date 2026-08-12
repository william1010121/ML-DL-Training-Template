from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from mltrain.adapter import load_adapter
from mltrain.config import (
    canonical_config_path,
    canonical_yaml,
    load_config,
    load_registry,
    registry_entry,
    repository_root,
    sha256_file,
    validate_config_registration,
)
from mltrain.contracts import ExperimentConfig, ProjectAdapter, RunResult, ValidationResult
from mltrain.provenance import commit_exists, validate_run_id
from mltrain.runtime import (
    cli_command,
    configure_reproducibility,
    create_run,
    ensure_device,
    load_resolved,
    write_manifest,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_REF = re.compile(r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")
_SECRET = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password|private[_-]?key)"
    r"\s*[:=]\s*['\"]?[^\s'\"]+"
)
_MAX_DECISION_LENGTH = 280


def _markdown_cell(value: str) -> str:
    return " ".join(value.replace("|", "/").replace("`", "'").split())


def _safe_decision(value: str) -> str:
    decision = _markdown_cell(value)
    if not decision:
        raise ValueError("decision cannot be empty")
    if len(decision) > _MAX_DECISION_LENGTH:
        raise ValueError(f"decision must be at most {_MAX_DECISION_LENGTH} characters")
    if _SECRET.search(decision):
        raise ValueError("decision appears to contain a secret")
    return decision


def _atomic_bytes(path: Path, value: bytes) -> None:
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_batch(values: Mapping[Path, bytes]) -> None:
    """Replace tracked files as one recoverable batch on the same filesystem."""
    originals = {path: path.read_bytes() for path in values}
    staged: dict[Path, Path] = {}
    try:
        for path, value in values.items():
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", dir=path.parent
            )
            temporary = Path(raw_temporary)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            staged[path] = temporary
        for path, temporary in staged.items():
            temporary.replace(path)
    except BaseException:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for path, original in originals.items():
            _atomic_bytes(path, original)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metrics_primary(
    path: Path,
    name: str,
    *,
    direction: str | None = None,
) -> tuple[bool, float | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, None
    found: float | None = None
    try:
        with path.open(encoding="utf-8") as stream:
            for raw_line in stream:
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    return False, None
                value = record.get(name)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    numeric = float(value)
                    if found is None:
                        found = numeric
                    elif direction == "minimize":
                        found = min(found, numeric)
                    elif direction == "maximize":
                        found = max(found, numeric)
                    else:
                        found = numeric
    except (OSError, json.JSONDecodeError):
        return False, None
    return found is not None, found


def _run_times(manifest: Mapping[str, Any]) -> bool:
    try:
        started = datetime.fromisoformat(str(manifest["started_at"]))
        finished = datetime.fromisoformat(str(manifest["finished_at"]))
    except (KeyError, TypeError, ValueError):
        return False
    return started.tzinfo is not None and finished.tzinfo is not None and finished >= started


def _result(value: RunResult | Mapping[str, Any]) -> RunResult:
    return value if isinstance(value, RunResult) else RunResult.model_validate(value)


def _validation(value: ValidationResult | Mapping[str, Any]) -> ValidationResult:
    return value if isinstance(value, ValidationResult) else ValidationResult.model_validate(value)


def execute(config_path: Path, kind: str, checkpoint: Path | None = None) -> Path:
    adapter = load_adapter()
    config = load_config(config_path, adapter)
    ensure_device(config)
    flags = configure_reproducibility(config)
    context, manifest = create_run(config, config_path, cli_command(), kind)
    if context.is_primary:
        reproducibility = cast(dict[str, Any], manifest["reproducibility"])
        reproducibility.update(flags)
        write_manifest(context.run_dir, manifest)
    try:
        raw = (
            adapter.train(config, context)
            if kind == "train"
            else adapter.evaluate(config, checkpoint or Path(), context)
        )
        result = _result(raw)
        if context.is_primary:
            result_path = context.run_dir / "result.json"
            result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
            provenance = cast(dict[str, Any], manifest["provenance"])
            if result.data_sha256:
                provenance["data_sha256"] = result.data_sha256
            if result.model_sha256:
                provenance["model_sha256"] = result.model_sha256
            tracking = cast(dict[str, Any], manifest["tracking"])
            tracking["degraded"] = result.tracking_degraded
            manifest["status"] = "succeeded"
            manifest["finished_at"] = datetime.now(UTC).isoformat()
            manifest["result_sha256"] = sha256_file(result_path)
            write_manifest(context.run_dir, manifest)
    except BaseException as error:
        if context.is_primary:
            manifest["status"] = "failed" if isinstance(error, Exception) else "interrupted"
            manifest["finished_at"] = datetime.now(UTC).isoformat()
            write_manifest(context.run_dir, manifest)
        raise
    return context.run_dir


def _load_run(
    run_dir: Path, adapter: ProjectAdapter
) -> tuple[ExperimentConfig, dict[str, Any], dict[str, Any]]:
    root = repository_root()
    runs_root = root / "runs"
    if runs_root.exists() and runs_root.resolve() != runs_root:
        raise ValueError("runs/ must not be a symlink")
    resolved_run = run_dir.resolve()
    if not resolved_run.is_relative_to(runs_root):
        raise ValueError("run directory must be inside repository runs/")
    manifest = _json(resolved_run / "manifest.json")
    result = _json(resolved_run / "result.json")
    resolved = load_resolved(resolved_run / "resolved_config.yml")
    config = adapter.config_model.model_validate(resolved)
    if config.output.root != Path("runs"):
        raise ValueError("resolved output.root must be exactly 'runs'")
    run_id = validate_run_id(manifest.get("run_id"))
    expected = runs_root / config.experiment.research_line / config.experiment.id / run_id
    if resolved_run != expected:
        raise ValueError("run path does not match manifest experiment and run_id")
    source_path_value = cast(dict[str, Any], manifest.get("config", {})).get("path")
    expected_config = canonical_config_path(root, config)
    if source_path_value != expected_config.relative_to(root).as_posix():
        raise ValueError("manifest config path is not canonical")
    validate_config_registration(root, expected_config, config)
    return config, manifest, result


def validate_run(run_dir: Path) -> dict[str, Any]:
    root = repository_root()
    adapter = load_adapter()
    config, manifest, result = _load_run(run_dir, adapter)
    resolved_run = run_dir.resolve()
    resolved_path = resolved_run / "resolved_config.yml"
    result_path = resolved_run / "result.json"
    manifest_path = resolved_run / "manifest.json"
    metrics_path = resolved_run / "metrics.jsonl"
    lockfile = root / "uv.lock"
    source_path = canonical_config_path(root, config)
    checks: dict[str, bool] = {}
    notes: list[str] = []

    checks["manifest_schema"] = manifest.get("schema_version") == 1
    checks["run_succeeded"] = manifest.get("status") == "succeeded"
    checks["run_timestamps"] = _run_times(manifest)
    source = cast(dict[str, Any], manifest.get("source", {}))
    checks["source_commit"] = bool(_GIT_SHA.fullmatch(str(source.get("commit", ""))))
    checks["source_commit_exists"] = commit_exists(root, source.get("commit"))
    checks["clean_source_commit"] = (
        checks["source_commit"]
        and checks["source_commit_exists"]
        and not source.get("dirty", True)
    )
    if source.get("dirty"):
        checks["dirty_fingerprint"] = bool(
            _SHA256.fullmatch(str(source.get("dirty_fingerprint", "")))
        )
    manifest_config = cast(dict[str, Any], manifest.get("config", {}))
    checks["resolved_config_hash"] = (
        manifest_config.get("resolved_sha256") == sha256_file(resolved_path)
    )
    checks["source_config_unchanged"] = source_path.is_file() and (
        manifest_config.get("source_sha256") == sha256_file(source_path)
    )
    source_config = load_config(source_path, adapter)
    checks["source_config_semantics"] = (
        source_config.model_dump(mode="json") == config.model_dump(mode="json")
    )
    checks["result_hash"] = manifest.get("result_sha256") == sha256_file(result_path)

    expected_experiment = config.model_dump(mode="json")["experiment"]
    checks["manifest_experiment"] = manifest.get("experiment") == expected_experiment
    reproducibility = cast(dict[str, Any], manifest.get("reproducibility", {}))
    checks["manifest_seed"] = reproducibility.get("seed") == config.reproducibility.seed
    checks["manifest_reproducibility_mode"] = (
        reproducibility.get("mode") == config.reproducibility.mode
    )
    checks["strict_reproducibility"] = (
        config.reproducibility.mode == "strict"
        and reproducibility.get("deterministic_algorithms") is True
        and reproducibility.get("cudnn_benchmark") is False
    )
    expected_runtime = {
        "device": config.runtime.device,
        "strategy": config.runtime.strategy,
        "world_size": config.runtime.world_size,
    }
    checks["manifest_runtime"] = manifest.get("runtime") == expected_runtime

    command = manifest.get("command")
    command_valid = (
        isinstance(command, list)
        and bool(command)
        and all(isinstance(part, str) and part for part in command)
        and "mltrain" in command
        and manifest.get("kind") in {"train", "evaluate"}
        and manifest.get("kind") in command
    )
    if command_valid:
        assert isinstance(command, list)
        try:
            config_argument = Path(command[command.index("--config") + 1])
            if not config_argument.is_absolute():
                config_argument = root / config_argument
            command_valid = config_argument.resolve() == source_path
        except (ValueError, IndexError):
            command_valid = False
    checks["manifest_command"] = command_valid

    provenance = cast(dict[str, Any], manifest.get("provenance", {}))
    for field in ("data_sha256", "model_sha256", "environment_sha256", "lockfile_sha256"):
        checks[f"provenance_{field}"] = bool(_SHA256.fullmatch(str(provenance.get(field, ""))))
    checks["current_lockfile"] = lockfile.is_file() and (
        provenance.get("lockfile_sha256") == sha256_file(lockfile)
    )
    result_data = result.get("data_sha256")
    result_model = result.get("model_sha256")
    checks["result_data_hash"] = bool(_SHA256.fullmatch(str(result_data or ""))) and (
        result_data == provenance.get("data_sha256")
    )
    checks["result_model_hash"] = bool(_SHA256.fullmatch(str(result_model or ""))) and (
        result_model == provenance.get("model_sha256")
    )
    environment_keys = (
        "python",
        "platform",
        "machine",
        "packages",
        "lockfile_sha256",
        "container",
        "image_digest",
        "oci_digest",
        "sif_sha256",
    )
    environment_input = {key: provenance.get(key) for key in environment_keys}
    environment_encoded = json.dumps(
        environment_input, sort_keys=True, separators=(",", ":")
    ).encode()
    checks["environment_hash"] = provenance.get("environment_sha256") == hashlib.sha256(
        environment_encoded
    ).hexdigest()
    container = provenance.get("container", "native")
    checks["container_kind"] = container in {"native", "docker", "apptainer"}
    if container == "docker":
        checks["container_image_digest"] = bool(
            _DIGEST_REF.fullmatch(str(provenance.get("image_digest", "")))
        )
    elif container == "apptainer":
        checks["container_oci_digest"] = bool(
            _DIGEST_REF.fullmatch(str(provenance.get("oci_digest", "")))
        )
        checks["container_sif_sha256"] = bool(
            _SHA256.fullmatch(str(provenance.get("sif_sha256", "")))
        )
    run_kind = manifest.get("kind")
    expected_metric = config.experiment.primary_metric.name
    result_metric = result.get("primary_metric_name")
    checks["primary_metric_name"] = isinstance(result_metric, str) and bool(result_metric)
    checks["train_primary_metric_contract"] = (
        run_kind != "train" or result_metric == expected_metric
    )
    primary_value = result.get("primary_metric")
    checks["primary_metric_finite"] = (
        isinstance(primary_value, (int, float))
        and not isinstance(primary_value, bool)
        and math.isfinite(float(primary_value))
    )
    metrics = result.get("metrics")
    checks["primary_metric_matches"] = (
        isinstance(metrics, dict) and metrics.get(result_metric) == primary_value
    )
    metrics_present, metrics_primary = _metrics_primary(
        metrics_path,
        str(result_metric),
        direction=config.experiment.primary_metric.direction if run_kind == "train" else None,
    )
    checks["canonical_metrics_present"] = metrics_present
    checks["canonical_metrics_match"] = metrics_present and metrics_primary == primary_value
    checkpoint_value = result.get("checkpoint")
    if isinstance(checkpoint_value, str) and checkpoint_value:
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_absolute():
            checkpoint = resolved_run / checkpoint
        checkpoint = checkpoint.resolve()
        checks["checkpoint_inside_run"] = checkpoint.is_relative_to(resolved_run)
        checks["checkpoint_hash"] = (
            checks["checkpoint_inside_run"]
            and checkpoint.is_file()
            and sha256_file(checkpoint) == provenance.get("model_sha256")
        )
    else:
        checks["checkpoint_inside_run"] = False
        checks["checkpoint_hash"] = False
    try:
        project = _validation(adapter.validate(config, resolved_run))
        checks.update({f"project_{key}": value for key, value in project.checks.items()})
        checks["project_validation"] = project.passed
        notes.extend(project.notes)
    except Exception as error:
        checks["project_validation"] = False
        notes.append(f"project validation failed: {type(error).__name__}: {error}")
    completed = all(checks.values())
    classification = "completed" if completed else "exploratory"
    if not completed:
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        notes.append("failed checks: " + failed)
    validation = {
        "schema_version": 1,
        "validated_at": datetime.now(UTC).isoformat(),
        "classification": classification,
        "primary_metric": {"name": result_metric, "value": result.get("primary_metric")},
        "evidence_inputs": {
            "manifest_sha256": sha256_file(manifest_path),
            "resolved_config_sha256": sha256_file(resolved_path),
            "result_sha256": sha256_file(result_path),
            "metrics_sha256": sha256_file(metrics_path) if metrics_path.is_file() else None,
            "source_config_sha256": sha256_file(source_path),
            "lockfile_sha256": sha256_file(lockfile) if lockfile.is_file() else None,
        },
        "checks": checks,
        "notes": notes,
    }
    _atomic_json(resolved_run / "validation.json", validation)
    return validation


def _research_row(
    research: Path,
    experiment: str,
    commit: str,
    metric: str,
    decision: str,
    status: str,
    allow_promote: bool = False,
    write: bool = True,
) -> bytes:
    safe_decision = _safe_decision(decision)
    metric = _markdown_cell(metric)
    text = research.read_text(encoding="utf-8")
    prefix = f"| {experiment} |"
    existing = next((line for line in text.splitlines() if line.startswith(prefix)), None)
    row = f"| {experiment} | `{commit}` | {metric} | {safe_decision} | {status} |"
    if existing:
        if allow_promote and existing.endswith("| completed |"):
            expected = row[: -len("promoted |")] + "completed |"
            if existing != expected:
                raise ValueError("promotion must preserve the recorded result and decision")
            text = text.replace(existing, existing[: -len("completed |")] + "promoted |", 1)
        elif allow_promote and existing.endswith("| promoted |"):
            if existing != row:
                raise ValueError("promoted research row does not match run evidence")
        else:
            raise ValueError(f"research result already recorded: {experiment}")
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += row + "\n"
    value = text.encode()
    if not write:
        return value
    _atomic_bytes(research, value)
    return value


def _project_research_line(
    path: Path,
    line: str,
    experiment: str,
    commit: str,
    status: str,
    write: bool = True,
) -> bytes | None:
    # Repository validation requires this file. Keeping the lifecycle helper
    # tolerant makes it usable in isolated adapters and unit-test repositories.
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    prefix = f"| [{line}]("
    rows = text.splitlines()
    index = next((number for number, row in enumerate(rows) if row.startswith(prefix)), None)
    if index is None:
        raise ValueError(f"research line is missing from configs/research.md: {line}")
    cells = rows[index].split("|")
    if len(cells) != 6:
        raise ValueError(f"malformed research-line index row: {rows[index]}")
    cells[3] = f" {status.title()} "
    cells[4] = f" `{experiment}` @ `{commit[:8]}` "
    rows[index] = "|".join(cells)
    value = ("\n".join(rows) + "\n").encode()
    if not write:
        return value
    _atomic_bytes(path, value)
    return value


def record_result(run_dir: Path, decision: str) -> None:
    decision = _safe_decision(decision)
    root = repository_root()
    adapter = load_adapter()
    config, manifest, result = _load_run(run_dir, adapter)
    validation = validate_run(run_dir)
    status = validation.get("classification")
    if status not in {"exploratory", "completed"}:
        raise ValueError(f"invalid validation classification: {status}")
    line = config.experiment.research_line
    experiment = config.experiment.id
    registry_path = root / "configs" / "registry.yml"
    registry = load_registry(root)
    entry = registry_entry(registry, line, experiment)
    if entry.get("status") != "planned":
        raise ValueError(f"experiment already recorded with status {entry.get('status')}")
    expected_config = canonical_config_path(root, config).relative_to(root).as_posix()
    if entry.get("config") != expected_config:
        raise ValueError("registry config path does not match run config")
    config_evidence = cast(dict[str, Any], manifest["config"])
    entry["status"] = status
    entry["config_sha256"] = config_evidence["source_sha256"]
    entry["completed_run"] = run_dir.resolve().relative_to(root).as_posix()
    line_entry = cast(dict[str, Any], registry["research_lines"][line])
    line_entry["status"] = status
    source = cast(dict[str, Any], manifest["source"])
    commit = str(source.get("commit") or "uncommitted")
    if source.get("dirty"):
        commit += "+dirty." + str(source.get("dirty_fingerprint"))[:8]
    metric = f"{result['primary_metric_name']} = {float(result['primary_metric']):.6g}"
    research = root / "configs" / line / "research.md"
    research_value = _research_row(
        research, experiment, commit, metric, decision, str(status), write=False
    )
    project_research = root / "configs" / "research.md"
    project_value = _project_research_line(
        project_research, line, experiment, commit, str(status), write=False
    )
    updates: dict[Path, bytes] = {
        registry_path: canonical_yaml(registry).encode(),
        research: research_value,
    }
    if project_value is not None:
        updates[project_research] = project_value
    _atomic_batch(updates)


def _scan_promotable(path: Path) -> None:
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError(f"promotion input exceeds 10 MiB: {path.name}")
    text = path.read_text(encoding="utf-8")
    if _SECRET.search(text):
        raise ValueError(f"possible secret in promotion input: {path.name}")


def _verify_promoted_artifact(destination: Path, decision: str) -> None:
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("registered promoted artifact is missing or unsafe")
    checksums = destination / "checksums.sha256"
    summary_path = destination / "summary.json"
    if not checksums.is_file() or not summary_path.is_file():
        raise ValueError("registered promoted artifact is incomplete")
    required = {
        "manifest.json",
        "result.json",
        "validation.json",
        "resolved_config.yml",
        "summary.json",
    }
    expected: dict[str, str] = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or not _SHA256.fullmatch(parts[0]):
            raise ValueError("promoted artifact checksum file is invalid")
        name = parts[1]
        if Path(name).name != name or name in expected:
            raise ValueError("promoted artifact checksum path is unsafe")
        expected[name] = parts[0]
    if set(expected) != required:
        raise ValueError("promoted artifact checksum set is incomplete or unexpected")
    if {path.name for path in destination.iterdir()} != required | {"checksums.sha256"}:
        raise ValueError("promoted artifact contains unexpected files")
    for name, digest in expected.items():
        path = destination / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"promoted artifact checksum mismatch: {name}")
    summary = _json(summary_path)
    if summary.get("decision") != decision:
        raise ValueError("experiment was already promoted with a different decision")
    if summary.get("result_sha256") != sha256_file(destination / "result.json"):
        raise ValueError("promoted artifact summary result hash is invalid")
    manifest = _json(destination / "manifest.json")
    if summary.get("config_sha256") != cast(dict[str, Any], manifest.get("config", {})).get(
        "source_sha256"
    ):
        raise ValueError("promoted artifact summary config hash is invalid")


def _safe_artifact_destination(
    root: Path, line: str, experiment: str, run_id: str
) -> Path:
    artifacts = root / "artifacts"
    destination = artifacts / line / experiment / run_id
    if not destination.is_relative_to(artifacts):
        raise ValueError("artifact destination escaped artifacts/")
    for path in (artifacts, artifacts / line, artifacts / line / experiment, destination):
        if path.is_symlink():
            raise ValueError(f"artifact path must not contain symlinks: {path.relative_to(root)}")
        if path.exists() and not path.is_dir():
            raise ValueError(f"artifact parent is not a directory: {path.relative_to(root)}")
    if destination.resolve(strict=False) != destination:
        raise ValueError("artifact destination is not canonical")
    return destination


def promote_run(run_dir: Path, decision: str) -> Path:
    decision = _safe_decision(decision)
    root = repository_root()
    adapter = load_adapter()
    config, manifest, result = _load_run(run_dir, adapter)
    validation = validate_run(run_dir)
    if validation.get("classification") != "completed":
        raise ValueError("only a completed run can be promoted")
    line = config.experiment.research_line
    experiment = config.experiment.id
    registry_path = root / "configs" / "registry.yml"
    registry = load_registry(root)
    entry = registry_entry(registry, line, experiment)
    run_relative = run_dir.resolve().relative_to(root).as_posix()
    entry_status = entry.get("status")
    if entry_status == "planned":
        if any(
            entry.get(field) is not None
            for field in ("config_sha256", "completed_run", "promoted_artifact")
        ):
            raise ValueError("planned registry entry already contains result evidence")
    elif entry.get("completed_run") != run_relative:
        raise ValueError("promotion run does not match registry completed_run")

    destination = _safe_artifact_destination(
        root, line, experiment, str(manifest["run_id"])
    )
    destination_relative = destination.relative_to(root).as_posix()
    source = cast(dict[str, Any], manifest["source"])
    commit = str(source["commit"])
    metric = f"{result['primary_metric_name']} = {float(result['primary_metric']):.6g}"
    line_research = root / "configs" / line / "research.md"
    project_research = root / "configs" / "research.md"
    if entry_status == "promoted":
        if entry.get("promoted_artifact") != destination_relative:
            raise ValueError("registry points to a different promoted artifact")
        _research_row(
            line_research,
            experiment,
            commit,
            metric,
            decision,
            "promoted",
            allow_promote=True,
            write=False,
        )
        _verify_promoted_artifact(destination, decision)
        return destination
    if entry_status not in {"planned", "completed"}:
        raise ValueError(f"cannot promote experiment with status {entry_status}")
    if entry_status == "completed" and entry.get("promoted_artifact") is not None:
        raise ValueError("completed experiment already has a promoted artifact")

    line_research_value = _research_row(
        line_research,
        experiment,
        commit,
        metric,
        decision,
        "promoted",
        allow_promote=True,
        write=False,
    )
    project_research_value = _project_research_line(
        project_research, line, experiment, commit, "promoted", write=False
    )

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"promotion destination already exists: {destination}")
    names = ("manifest.json", "result.json", "validation.json", "resolved_config.yml")
    for name in names:
        _scan_promotable(run_dir.resolve() / name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _safe_artifact_destination(
        root, line, experiment, str(manifest["run_id"])
    )
    stage = Path(tempfile.mkdtemp(prefix=".promote-", dir=destination.parent))
    try:
        for name in names:
            shutil.copy2(run_dir.resolve() / name, stage / name)
        source = cast(dict[str, Any], manifest["source"])
        summary = {
            "schema_version": 1,
            "experiment": experiment,
            "research_line": line,
            "source_commit": source["commit"],
            "config_sha256": cast(dict[str, Any], manifest["config"])["source_sha256"],
            "result_sha256": sha256_file(stage / "result.json"),
            "primary_metric": {
                "name": result["primary_metric_name"],
                "value": result["primary_metric"],
            },
            "decision": decision,
        }
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checksum_lines = [
            f"{sha256_file(stage / name)}  {name}" for name in (*names, "summary.json")
        ]
        (stage / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        stage.replace(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    entry["status"] = "promoted"
    entry["config_sha256"] = cast(dict[str, Any], manifest["config"])["source_sha256"]
    entry["completed_run"] = run_relative
    entry["promoted_artifact"] = destination_relative
    line_entry = cast(dict[str, Any], registry["research_lines"][line])
    line_entry["status"] = "promoted"
    updates: dict[Path, bytes] = {
        registry_path: canonical_yaml(registry).encode(),
        line_research: line_research_value,
    }
    if project_research_value is not None:
        updates[project_research] = project_research_value
    try:
        _atomic_batch(updates)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination
