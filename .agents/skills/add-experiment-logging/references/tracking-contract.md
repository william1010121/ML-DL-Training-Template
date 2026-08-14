# Local-first tracking contract

## Stable interface

Implement a project-level `Tracker` protocol with operations equivalent to:

```python
log_params(params: Mapping[str, object]) -> None
log_metrics(metrics: Mapping[str, float], step: int) -> None
log_artifact(path: Path, *, name: str) -> None
finish(status: str) -> None
```

Keep a `NoOpTracker`. Construct adapters in the project tracking package and pass the protocol into training code. Never branch on provider names inside the training loop.

The canonical local writer is separate from optional tracking. It must write resolved config, manifest, JSONL metrics, and final result even when a provider is disabled or unavailable.

## Run-local profiling

Profiling is canonical local evidence, not an external tracker. Put raw stage and resource JSONL
under `runs/<line>/<experiment>/<run-id>/profile/`. Use `RunContext.profile_stage()` for stable
project stage names and one background sampler per rank. Rank zero may add system-wide and
all-device NVIDIA data; never let multiple ranks append to the same file.

Use monotonic time for duration and UTC timestamps only for cross-log correlation. A profiler
measures work; it does not enforce a deadline, stop training, or delete cloud resources. Keep the
sample interval bounded and record unavailable GPU telemetry explicitly instead of fabricating
zero utilization.

## Ephemeral progress

Use the stable `RunContext` progress interface for long project operations. Progress answers where
execution is now; profiling records how resources and time were used. Keep progress on stderr and
out of configs, manifests, metrics, results, and promoted artifacts. In an interactive terminal,
show bounded epoch/batch work with nested bars. In a non-TTY log, append at most once per 5%
advance or 30 seconds. Unknown totals receive stage start/completion/failure messages instead of a
fabricated percentage. Render shared DDP progress on rank zero only and preserve the original
exception if progress output fails.

## Failure behavior

Catch initialization, logging, and flush failures at the adapter boundary. Warn without exposing secrets, set a degraded-tracking field in the run manifest, and continue training. Do not report a successful provider sync if any queued data failed.

## Weights & Biases

Read credentials from `WANDB_API_KEY`. Initialize with project, run name, config, and resume policy from non-secret config fields. Log metrics with an explicit step. Store only artifact references when checkpoints are too large for the selected retention policy.

## MLflow

Read the tracking URI from `MLFLOW_TRACKING_URI` and credentials from environment variables supported by the deployment. Set the experiment and run explicitly. Flatten parameter keys deterministically and avoid logging secrets or entire environment dumps.

## TensorBoard

Write event files beneath the current ignored run directory. Emit the same metric names and steps as `metrics.jsonl`. Flush periodically and close the writer in `finish`.

## Tests

Test `NoOpTracker`, successful adapter calls with a fake client, initialization failure, mid-run failure, and finish failure. Assert that canonical local output survives every provider failure. For profiling, test disabled no-op behavior, monotonic stage timing, sampler degradation, rank isolation, summary aggregation, and evidence tampering. For progress, test TTY/plain/off selection, throttling, stderr-only output, rank isolation, and exception/interruption cleanup.
