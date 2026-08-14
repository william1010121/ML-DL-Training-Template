from __future__ import annotations

import importlib
import os
import random
import secrets
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from mltrain.config import (
    canonical_yaml,
    repository_root,
    sha256_bytes,
    sha256_file,
    validate_config_registration,
)
from mltrain.contracts import ExperimentConfig, RunContext
from mltrain.provenance import (
    environment_state,
    model_config_hash,
    safe_run_id,
    source_state,
)


def distributed_state(config: ExperimentConfig) -> tuple[int, int]:
    try:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    except ValueError as error:
        raise RuntimeError("distributed rank variables must be integers") from error
    if world_size < 1 or not 0 <= rank < world_size:
        raise RuntimeError("RANK must satisfy 0 <= RANK < WORLD_SIZE")
    if local_world_size < 1 or not 0 <= local_rank < local_world_size:
        raise RuntimeError("LOCAL_RANK must satisfy 0 <= LOCAL_RANK < LOCAL_WORLD_SIZE")
    expected = config.runtime.world_size
    if config.runtime.strategy == "single" and (rank != 0 or world_size != 1):
        raise RuntimeError("single strategy cannot run under a distributed launcher")
    if config.runtime.strategy == "single" and (local_rank != 0 or local_world_size != 1):
        raise RuntimeError("single strategy requires LOCAL_RANK=0 and LOCAL_WORLD_SIZE=1")
    if config.runtime.strategy == "ddp":
        if world_size != expected:
            raise RuntimeError(f"config world_size={expected}, launcher WORLD_SIZE={world_size}")
        if "TORCHELASTIC_RUN_ID" not in os.environ and "MLTRAIN_RUN_ID" not in os.environ:
            raise RuntimeError("DDP requires a shared TORCHELASTIC_RUN_ID or MLTRAIN_RUN_ID")
    return rank, world_size


def configure_reproducibility(config: ExperimentConfig) -> dict[str, bool | str]:
    seed = config.reproducibility.seed
    random.seed(seed)
    try:
        numpy_random = importlib.import_module("numpy.random")
        numpy_random.seed(seed)
    except ImportError:
        pass
    strict = config.reproducibility.mode == "strict"
    if strict and config.runtime.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(strict)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not strict
        torch.backends.cudnn.deterministic = strict
    return {
        "deterministic_algorithms": strict,
        "cudnn_benchmark": not strict,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
    }


def ensure_device(config: ExperimentConfig) -> None:
    import torch

    device = config.runtime.device
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    if not available[device]:
        raise RuntimeError(f"requested device {device!r} is unavailable; refusing CPU fallback")


def _run_id(config: ExperimentConfig, source: dict[str, object]) -> str:
    if config.runtime.strategy == "ddp":
        shared = os.environ.get("MLTRAIN_RUN_ID") or os.environ["TORCHELASTIC_RUN_ID"]
        return safe_run_id(shared)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    commit = str(source.get("commit") or "uncommitted")[:8]
    return f"{timestamp}-{commit}-{secrets.token_hex(3)}"


def create_run(
    config: ExperimentConfig, config_path: Path, command: list[str], kind: str
) -> tuple[RunContext, dict[str, object]]:
    root = repository_root()
    if config.output.root != Path("runs"):
        raise ValueError("output.root must be exactly 'runs'")
    validate_config_registration(root, config_path, config)
    rank, world_size = distributed_state(config)
    source = source_state(root)
    run_id = _run_id(config, source)
    run_dir = (
        root
        / config.output.root
        / config.experiment.research_line
        / config.experiment.id
        / run_id
    )
    runs_root = root / "runs"
    if runs_root.exists() and runs_root.resolve() != runs_root:
        raise ValueError("runs/ must not be a symlink")
    if not run_dir.is_relative_to(runs_root):
        raise ValueError("run directory escaped runs/")
    context = RunContext(
        run_id=run_id,
        run_dir=run_dir,
        command=command,
        rank=rank,
        world_size=world_size,
        is_primary=rank == 0,
    )
    resolved = config.model_dump(mode="json")
    resolved_text = canonical_yaml(resolved)
    config_absolute = config_path.resolve()
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "resolved_config.yml").write_text(resolved_text, encoding="utf-8")
        (run_dir / "metrics.jsonl").touch()
    else:
        deadline = time.monotonic() + 30
        while not run_dir.is_dir() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not run_dir.is_dir():
            raise RuntimeError(f"rank zero did not create run directory: {run_dir}")

    environment = environment_state(root)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": kind,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "experiment": resolved["experiment"],
        "command": command,
        "source": source,
        "config": {
            "path": config_absolute.relative_to(root).as_posix(),
            "source_sha256": sha256_file(config_absolute),
            "resolved_sha256": sha256_bytes(resolved_text.encode()),
        },
        "reproducibility": {
            "seed": config.reproducibility.seed,
            "mode": config.reproducibility.mode,
        },
        "runtime": {
            "device": config.runtime.device,
            "strategy": config.runtime.strategy,
            "world_size": world_size,
        },
        "provenance": {
            "data_sha256": None,
            "model_sha256": model_config_hash(resolved["model"]),
            **environment,
        },
        "tracking": {"backend": config.tracking.backend, "degraded": False},
        "profiling": {
            "enabled": config.profiling.enabled,
            "schema_version": 1,
            "sample_interval_seconds": config.profiling.sample_interval_seconds,
            "status": "pending" if config.profiling.enabled else "disabled",
            "degraded": False,
            "files": {},
        },
    }
    if rank == 0:
        write_manifest(run_dir, manifest)
    return context, manifest


def write_manifest(run_dir: Path, manifest: dict[str, object]) -> None:
    import json

    temporary = run_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "manifest.json")


def load_resolved(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"resolved config is not a mapping: {path}")
    return value


def cli_command() -> list[str]:
    root = repository_root()
    executable = Path(sys.executable).resolve()
    try:
        interpreter = executable.relative_to(root).as_posix()
    except ValueError:
        # Keep promoted evidence portable when Python comes from a host-level
        # installation or cache outside the repository.
        interpreter = executable.name
    return [interpreter, "-m", "mltrain", *sys.argv[1:]]
