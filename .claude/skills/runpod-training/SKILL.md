---
name: runpod-training
description: Run reproducible training on Runpod Pods. Use when adding or changing a Runpod controller, Pod creation and teardown, SSH endpoint readiness, source or artifact transfer, detached remote training, cost watchdogs, or Runpod support claims in repositories created from this template.
---

# Runpod Training (pointer)

The authoritative workflow is `.agents/skills/runpod-training/SKILL.md`. Read it, and the
`.agents/skills/runpod-training/references/transport-contract.md` it cites, before changing Pod
orchestration, SSH readiness, transfer, watchdog, or teardown behavior.

Rules worth knowing before you open those files:

- Treat either `ssh.direct` or `ssh.proxy` as SSH-ready. Never wait only for `publicIp`.
- Reuse `mltrain.runpod_transport`; keep the controller outside the project adapter and the normal
  config, manifest, validation, and promotion lifecycle authoritative.
- Retrieve and verify the complete run archive before deleting a Pod. A failed transfer or timeout
  is not research evidence, and only a real Pod run with validated evidence may be called verified.
