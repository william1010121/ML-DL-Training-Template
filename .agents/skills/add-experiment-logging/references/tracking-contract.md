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

## Failure behavior

Catch initialization, logging, and flush failures at the adapter boundary. Warn without exposing secrets, set a degraded-tracking field in the run manifest, and continue training. Do not report a successful provider sync if any queued data failed.

## Weights & Biases

Read credentials from `WANDB_API_KEY`. Initialize with project, run name, config, and resume policy from non-secret config fields. Log metrics with an explicit step. Store only artifact references when checkpoints are too large for the selected retention policy.

## MLflow

Read the tracking URI from `MLFLOW_TRACKING_URI` and credentials from environment variables supported by the deployment. Set the experiment and run explicitly. Flatten parameter keys deterministically and avoid logging secrets or entire environment dumps.

## TensorBoard

Write event files beneath the current ignored run directory. Emit the same metric names and steps as `metrics.jsonl`. Flush periodically and close the writer in `finish`.

## Tests

Test `NoOpTracker`, successful adapter calls with a fake client, initialization failure, mid-run failure, and finish failure. Assert that canonical local output survives every provider failure.
