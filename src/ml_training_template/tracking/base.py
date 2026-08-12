"""Optional experiment-tracker interface used by project training code."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    """Minimum interface implemented by optional external tracking adapters."""

    degraded: bool

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None: ...

    def log_artifact(self, path: Path, *, name: str | None = None) -> None: ...

    def finish(self, *, status: str) -> None: ...


class NoOpTracker:
    """Tracker that intentionally sends no data outside the local run."""

    degraded = False

    def log_params(self, params: Mapping[str, Any]) -> None:
        del params

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        del metrics, step

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        del path, name

    def finish(self, *, status: str) -> None:
        del status


class ResilientTracker:
    """Warn and continue when an optional external tracker becomes unavailable."""

    def __init__(self, tracker: Tracker) -> None:
        self._tracker = tracker
        self.degraded = tracker.degraded

    def _call(self, method: str, *args: Any, **kwargs: Any) -> None:
        try:
            getattr(self._tracker, method)(*args, **kwargs)
        except Exception:  # external SDK failures must not lose a local run
            self.degraded = True
            warnings.warn(
                f"experiment tracker degraded during {method}; provider details suppressed",
                RuntimeWarning,
                stacklevel=2,
            )

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._call("log_params", params)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        self._call("log_metrics", metrics, step=step)

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        self._call("log_artifact", path, name=name)

    def finish(self, *, status: str) -> None:
        self._call("finish", status=status)


def create_tracker(backend: str) -> Tracker:
    """Create a tracker explicitly; unknown backends never silently degrade."""

    if backend == "noop":
        return NoOpTracker()
    raise ValueError(
        f"Unsupported tracking backend {backend!r}. Install and configure an explicit "
        "adapter before selecting it."
    )
