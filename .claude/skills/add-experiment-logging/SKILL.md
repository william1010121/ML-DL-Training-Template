---
name: add-experiment-logging
description: Add or modify optional experiment tracking, profiling, or progress reporting in repositories created from this ML/DL training template. Use when integrating Weights & Biases, MLflow, TensorBoard, another tracker, progress bars, detached-log progress, stage timers, CPU/GPU resource sampling, src/*/tracking/, or tracker/profile state in run manifests. Preserve canonical local evidence and keep external services optional and failure-tolerant.
---

# Add Experiment Logging (pointer)

The authoritative workflow is `.agents/skills/add-experiment-logging/SKILL.md`. Read it, and the
`.agents/skills/add-experiment-logging/references/tracking-contract.md` it cites, before changing
tracking, profiling, or progress code.

Rules worth knowing before you open those files:

- Canonical local evidence always wins. `resolved_config.yml`, `manifest.json`, `metrics.jsonl`,
  and `result.json` must be written even when a provider is disabled or fails.
- Implement the `Tracker` protocol behind a provider adapter; the training loop must not import a
  provider SDK. Read credentials from environment variables only.
- Progress is stderr-only presentation: dynamic bars on a TTY, throttled plain text in detached and
  CI logs, rank zero only. It is never experiment intent or evidence.
