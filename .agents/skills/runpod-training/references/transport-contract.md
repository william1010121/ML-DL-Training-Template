# Runpod transport contract

## Endpoint readiness

Use `GET https://api.runpod.io/v2/pods/<pod-id>` with a bearer API key. The Pod `ssh` object has
two nullable endpoints:

```json
{
  "ssh": {
    "direct": {
      "host": "195.26.233.3",
      "port": 34446,
      "username": "root"
    },
    "proxy": {
      "host": "ssh.runpod.io",
      "port": 22,
      "username": "opaque-routing-token"
    }
  }
}
```

The API key must be allowed to read Pods through the v2 API. Treat HTTP 401/403 as an API-key
scope or account-policy problem; do not fall back to guessing the proxy username from the Pod ID.

The proxy works without a public IP. It is an interactive shell only. Direct SSH requires a
published TCP port and supports SCP, SFTP, rsync, and port forwarding. See the official
[Runpod SSH guide](https://docs.runpod.io/pods/configuration/use-ssh) and
[v2 API schema](https://github.com/runpod/docs/blob/main/api-reference-v2/openapi.json).

Do not derive the proxy username. Runpod documents it as an opaque routing token; consume the
exact API field. Poll until either endpoint exists, then probe candidates in `direct, proxy`
order with a bounded connection timeout.

## Transfer behavior

`mltrain.runpod_transport` implements the shared behavior:

- Direct upload/download uses rsync with partial append verification, followed by a remote and
  local SHA-256 check.
- Proxy upload waits until terminal echo is disabled, sends base64 through the PTY, checks
  SHA-256 on the Pod, and atomically moves the completed file.
- Proxy download frames metadata and base64 with random markers, enforces the size bound, writes
  a local temporary file, and links it into place only after SHA-256 verification.
- Existing files are accepted only when their content hash matches. Different content is never
  overwritten.
- Remote paths must be simple absolute POSIX paths. Sources and downloaded artifacts must be
  regular non-symlink files.

The default Basic SSH transfer limit is 512 MiB. Prefer object storage, a network volume, or
Runpod's transfer tooling for larger assets; the proxy fallback is for source archives and
bounded run evidence, not datasets or checkpoints.

The Pod image must provide `bash`, GNU `base64`, `sha256sum`, and GNU `stat`. Direct transfers
also require `rsync` locally and on the Pod. Probe these prerequisites before starting a costly
training job.

## Controller state

Persist enough local state to resume safely:

```text
controller identity and schema version
owned Pod ID
source commit and source archive SHA-256
selected experiment config and MLTRAIN_RUN_ID
created, training, finalization, and deletion deadlines
last successful SSH mode: direct | proxy
remote process/control paths
retrieved artifact path and SHA-256
append-only lifecycle events
```

Never delete a Pod that is not owned by the saved controller state. A host-side watchdog is only
best effort if it cannot survive host loss; say so in user documentation. Prefer a provider-side
termination field when Runpod exposes one for the selected Pod API.

## Evidence boundary

Runpod transport success is not experiment success. The remote command must still produce the
normal `resolved_config.yml`, manifest, metrics, result, checkpoint, and validation inputs. After
retrieval, run the repository's normal `mltrain validate`; only a clean strict completed run may
be recorded or promoted.
