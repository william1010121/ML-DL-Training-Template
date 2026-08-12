---
name: add-experiment-logging
description: Add or modify optional experiment tracking in repositories created from this ML/DL training template. Use when integrating Weights & Biases, MLflow, TensorBoard, or another tracker; changing src/*/tracking/; or recording tracker state in run manifests. Preserve canonical local metrics and keep external services optional and failure-tolerant.
---

# Add Experiment Logging

Keep local run evidence authoritative. Treat external trackers as replaceable projections of that evidence.

## Workflow

1. Read `references/tracking-contract.md` before changing tracking code.
2. Read only the provider section needed for the requested integration.
3. Implement the repository `Tracker` protocol behind a provider-specific adapter. Do not import provider SDKs from the training loop.
4. Load credentials from environment variables at runtime. Never put tokens in YAML, manifests, examples, or tests.
5. Catch provider failures at the adapter boundary, emit a warning, mark tracking degraded in the manifest, and continue canonical local logging.
6. Test with the provider disabled and with a simulated provider failure. Run a real provider smoke test only when credentials and network access are explicitly available.

## Required behavior

- Always write `resolved_config.yml`, `manifest.json`, `metrics.jsonl`, and `result.json` locally.
- Log scalar metrics with explicit steps and stable names.
- Store large artifacts outside Git and record references plus checksums.
- Make `NoOpTracker` the safe default.
- Finish or flush the adapter without masking training errors.
