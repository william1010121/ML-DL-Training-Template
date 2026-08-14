---
name: runpod-training
description: Run reproducible training on Runpod Pods. Use when adding or changing a Runpod controller, Pod creation and teardown, SSH endpoint readiness, source or artifact transfer, detached remote training, cost watchdogs, or Runpod support claims in repositories created from this template.
---

# Runpod Training

Keep Runpod orchestration outside the project adapter. Reuse
`mltrain.runpod_transport` for connectivity and keep the normal config, run manifest, validation,
and promotion lifecycle authoritative.

## Workflow

1. Read `references/transport-contract.md` before changing a Runpod controller or transfer path.
2. Require a clean committed source revision and create a commit-addressed source archive.
3. Query the Runpod v2 Pod response for both `ssh.direct` and `ssh.proxy`. Treat either endpoint
   as SSH-ready; never wait only for `publicIp` or a direct TCP mapping.
4. Try direct SSH first when present. Fall back after a bounded failure to Basic SSH through
   `ssh.runpod.io`.
5. Use direct rsync for resumable transfers. Basic SSH has no SCP, SFTP, rsync, or port
   forwarding; use the template's PTY-safe base64 stream and verify SHA-256 before an atomic move.
6. Start training as a detached Pod-side process with `MLTRAIN_RUN_ID` fixed by the controller.
   Keep termination monitoring independent of the SSH session.
7. Retrieve and verify the complete run archive before deleting only the Pod recorded in the
   controller state. A failed transfer, eviction, or timeout is not research evidence.

## Stable helper

Inspect endpoint readiness without requiring a public IP:

```bash
uv run mltrain-runpod endpoint <pod-id>
```

The same helper provides `exec`, `upload`, and `download`. It reads `RUNPOD_API_KEY` or
`~/.runpod/config.toml`, never prints the key, prefers direct TCP, and falls back to proxy.

## Guardrails

- Use the opaque proxy username returned by `ssh.proxy`; never construct it from Pod or machine
  identifiers.
- Bound endpoint polling, remote execution, transfer size, total training time, and finalization
  reserve. Do not turn a missing public IP into an unbounded wait.
- Store Pod ID, endpoint mode, source commit, command, deadlines, run path, and artifact hashes in
  controller state. Do not store API keys, private-key contents, or secret environment values.
- Keep device selection fail-closed and preserve canonical local metrics even if controller or
  tracker reporting degrades.
- Call a path verified only after a real external Pod run produces matching validated evidence.
