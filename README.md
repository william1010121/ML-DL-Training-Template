# ML/DL Training Template

**A reproducible, evidence-first PyTorch training template designed for native, Docker, and Apptainer-based HPC execution without coupling training code to a platform or tracker.**

[繁體中文](README.zh-TW.md)

Most training repositories make it easy to start a run and hard to explain it six weeks later. This template treats the experiment config as intent, the run manifest as evidence, and the research notes as the decision record. Data, weights, logs, and checkpoints stay outside Git; the small facts needed to reproduce a result stay reviewable.

It ships with a real MNIST reference project, strict Pydantic configuration, local-first metrics, a stable `mltrain` CLI, environment-only containers, and repository-local Codex skills for evolving the project safely.

## Why this template

- **One experiment, one complete config.** No hidden inheritance or machine-specific paths.
- **Claims require evidence.** A result is not promoted until its provenance and validation pass.
- **Clean by construction.** Datasets, checkpoints, complete runs, SIF files, and secrets are ignored.
- **One environment definition.** Docker is the source of truth; Apptainer consumes the same immutable OCI image.
- **Tracking is optional.** Canonical JSONL and manifests are always written locally; external trackers are adapters.
- **AI-agent friendly.** Repo-local skills preserve experiment numbering, research records, and dependency boundaries.

## Five-minute CPU start

Requirements: Git, Python 3.11–3.13, and [uv](https://docs.astral.sh/uv/). MNIST is downloaded explicitly; training never hides a network request or substitutes fake data.

<!-- sync:start quickstart -->
```bash
uv sync --extra cpu
uv run python scripts/download_mnist.py --root datasets
uv run mltrain train --config configs/mnist-baseline/exp-001.yml
```
<!-- sync:end quickstart -->

The run is written to `runs/mnist-baseline/exp-001/<run-id>/`. Inspect its `result.json`, then validate it explicitly:

<!-- sync:start validate-command -->
```bash
uv run mltrain validate --run runs/mnist-baseline/exp-001/<run-id>
```
<!-- sync:end validate-command -->

Validation classifies a run as `exploratory` or `completed`; it does not quietly edit the research record. Once you have made a decision, record it:

<!-- sync:start record-command -->
```bash
uv run mltrain record-result \
  --run runs/mnist-baseline/exp-001/<run-id> \
  --decision "Keep as the CPU baseline"
```
<!-- sync:end record-command -->

Use `promote` instead when a strict, clean, completed run should become tracked evidence:

<!-- sync:start promote-command -->
```bash
uv run mltrain promote \
  --run runs/mnist-baseline/exp-001/<run-id> \
  --decision "Promote as the CPU baseline"
```
<!-- sync:end promote-command -->

Promotion stores only small, reviewable metadata in `artifacts/`; it does not commit the dataset or checkpoint.

## Repository map

```text
.
├── .agents/skills/               # Repo-local project governance
├── src/
│   ├── mltrain/                   # Stable CLI, contracts, provenance gates
│   └── ml_training_template/      # Replaceable project adapter
│       ├── data/
│       ├── model/
│       ├── training/
│       ├── validate/
│       └── tracking/
├── configs/
│   ├── research.md                # Project goal, research-line index, global decisions
│   ├── registry.yml               # Experiment lifecycle and evidence pointers
│   └── mnist-baseline/
│       ├── research.md            # Goal and concise result decisions for this line
│       └── exp-001.yml … exp-003.yml
├── datasets/                      # Local data; contents ignored
├── checkpoints/                   # Local/pretrained weights; contents ignored
├── runs/                          # Complete local run records; ignored
├── artifacts/                     # Promoted small evidence; tracked
├── containers/                    # CPU/CUDA environment definitions
├── scripts/                       # Download and runtime helpers
└── tests/                          # Contracts, smoke tests, and repo hygiene
```

The split is deliberate: `src/.../data` and `src/.../model` contain code; `datasets/` and `checkpoints/` contain local bytes. `configs/research.md` is the project-wide map, while each `configs/<research-line>/research.md` owns that line's results.

The research-line slug is the join key across the folder name, every config's `experiment.research_line`, `configs/registry.yml`, and the top-level research index. Registry entries keep the machine-readable `config`, `status`, locked config hash, completed run, and promoted artifact pointers; the line-level `research.md` remains the human-readable result ledger. Keep those views consistent instead of duplicating new metadata in a second place.

## Experiment lifecycle

```text
planned config
    │
    ├─ train ───────> ignored run + manifest + metrics
    │                        │
    │                     validate
    │                        │
    └──────────────> exploratory or completed
                             │
                    record-result or promote
                             │
                 registry + line research + evidence
```

An experiment config becomes immutable after a result is recorded. Change a seed, dataset, model, or training choice by creating the next `exp-###.yml`, not by rewriting history. Failed and interrupted runs remain useful local diagnostics but do not become research claims.

Every run contains the resolved config, source revision and dirty state, command, environment identity, seed and determinism flags, dataset/model identities, canonical metrics, result, and validation state. Secret values are never part of this record.

## Start your own project

Create a repository from this GitHub Template and commit it first. Initialization removes and rewrites template files, so run it only from a clean worktree: `git status --short` must print nothing. Then preview the one-time operation:

<!-- sync:start initialize-command -->
```bash
git status --short
python .agents/skills/training-manager/scripts/initialize_project.py \
  --project-name "My Project" \
  --package-name my_project

python .agents/skills/training-manager/scripts/initialize_project.py \
  --project-name "My Project" \
  --package-name my_project \
  --apply
```
<!-- sync:end initialize-command -->

The first invocation is a dry-run. Review its diff before the `--apply` invocation. The initializer renames the project package, removes the MNIST implementation and configs, clears the template adapter setting, and resets the research registry. It intentionally leaves a non-runnable task shell rather than guessing your domain interface.

After initialization, finish the project in this order:

1. Implement the project config, data, model, training, validation, and `ProjectAdapter`; restore `[tool.mltrain].adapter` in `pyproject.toml`.
2. Create a meaningful research line and a complete first experiment config with the skill scripts below.
3. Regenerate and install the environment from the new project definition.
4. Replace the MNIST-oriented README quickstart and support claims with commands and evidence for the new task; initialization does not author domain documentation.

<!-- sync:start post-initialize-command -->
```bash
uv lock
uv sync --extra cpu
uv run python scripts/validate_repo.py
```
<!-- sync:end post-initialize-command -->

Only after these steps should the new project be expected to run. The template initializer is a safe de-specialization tool, not an application generator.

Create research lines and experiments through the same deterministic workflow:

<!-- sync:start scaffold-commands -->
```bash
python .agents/skills/training-manager/scripts/new_research_line.py baseline \
  --goal "Establish a reproducible baseline" --apply

python .agents/skills/training-manager/scripts/new_experiment.py baseline \
  --config path/to/full.yml --apply

python .agents/skills/training-manager/scripts/new_experiment.py baseline \
  --from configs/baseline/exp-001.yml --apply
```
<!-- sync:end scaffold-commands -->

These commands never overwrite files. Codex discovers the skills in `.agents/skills/`; `AGENTS.md` tells agents when the workflows are mandatory.

## Native, Docker, Apptainer, and Runpod

### Docker

The image contains the locked environment, not your application source. The helper bind-mounts the repository read-only at `/workspace` and mounts `datasets/`, `checkpoints/`, `runs/`, and `artifacts/` separately as writable directories. Editing Python does not require rebuilding the image.

<!-- sync:start docker-commands -->
```bash
docker build -f containers/Dockerfile --target cpu -t ml-training-template:cpu .
./scripts/docker-run cpu ml-training-template:cpu \
  train --config configs/mnist-baseline/exp-001.yml

docker build -f containers/Dockerfile --target cuda -t ml-training-template:cuda .
./scripts/docker-run cuda ml-training-template:cuda \
  train --config configs/mnist-baseline/exp-002.yml
```
<!-- sync:end docker-commands -->

The CUDA helper adds `--gpus all`. CUDA is fail-closed: requesting it without a compatible visible GPU is an error, never a silent CPU fallback.

Before launch, the Docker helper resolves the supplied local tag to Docker's content-addressed image/config ID (`sha256:...`), runs that ID, and records it in the manifest. This local image ID is not a registry OCI manifest digest. A release published to a registry has its own immutable OCI digest, which is the value used for an Apptainer pull.

### Apptainer / SingularityCE

Publish an immutable OCI image first, then set `OCI_IMAGE` to its complete `docker://...@sha256:...` URI. Do not use `latest` or another mutable tag.

<!-- sync:start apptainer-commands -->
```bash
./scripts/apptainer-run pull "$OCI_IMAGE" training.sif

./scripts/apptainer-run exec cuda training.sif \
  train --config configs/mnist-baseline/exp-002.yml
```
<!-- sync:end apptainer-commands -->

The pull helper records both the source OCI digest (`.oci-digest`) and the generated SIF checksum (`.sha256`); execution verifies both sidecars. There is intentionally no second `Singularity.def`: Docker/OCI remains the single environment definition. The host must provide a compatible NVIDIA driver, and site policy always wins.

For two GPUs on one node, both helpers can invoke `torchrun`. Give each launch a unique filesystem-safe run ID; the helper passes it to every rank as `MLTRAIN_RUN_ID`, configures a single-node rendezvous, and refuses a reused run directory. The following Docker and Apptainer paths are supported contracts but remain manual/unverified.

<!-- sync:start ddp-command -->
```bash
DOCKER_RUN_ID="exp-003-docker-$(uuidgen | tr '[:upper:]' '[:lower:]')"
./scripts/docker-run cuda ml-training-template:cuda \
  --launcher torchrun --nproc-per-node 2 --run-id "$DOCKER_RUN_ID" -- \
  train --config configs/mnist-baseline/exp-003.yml

APPTAINER_RUN_ID="exp-003-apptainer-$(uuidgen | tr '[:upper:]' '[:lower:]')"
./scripts/apptainer-run exec cuda training.sif \
  --launcher torchrun --nproc-per-node 2 --run-id "$APPTAINER_RUN_ID" -- \
  train --config configs/mnist-baseline/exp-003.yml
```
<!-- sync:end ddp-command -->

The repository validates `WORLD_SIZE=2` against the config and refuses DDP without the shared run identity. These helpers support single-node CUDA DDP only; multi-node Slurm orchestration is outside the v1 contract. If `uuidgen` is unavailable, provide another unique value matching `[A-Za-z0-9_.-]+` and at most 80 characters.

### Runpod Pods

`mltrain-runpod` is an opt-in SSH transport, not a hidden training lifecycle or Pod creator. It
queries Runpod's v2 Pod API, treats either `ssh.direct` or `ssh.proxy` as ready, tries direct TCP
first, and falls back to Basic SSH without waiting for a public IP. Direct transfers use rsync;
proxy transfers use a PTY-safe base64 stream and verify SHA-256 before an atomic move.

<!-- sync:start runpod-transport-commands -->
```bash
uv run mltrain-runpod endpoint "$POD_ID"
uv run mltrain-runpod upload "$POD_ID" source.tar /workspace/source.tar
uv run mltrain-runpod exec "$POD_ID" -- bash /workspace/start-training.sh
uv run mltrain-runpod download "$POD_ID" \
  /workspace/run.tar.gz runs/_runpod-downloads/run.tar.gz
```
<!-- sync:end runpod-transport-commands -->

The helper reads `RUNPOD_API_KEY` or `~/.runpod/config.toml` and never records the key; the key
must have v2 Pod read access. Runpod's proxy username is an opaque API value; the helper never
constructs it. Basic SSH does not support
SCP, SFTP, rsync, or port forwarding, and its default transfer limit here is 512 MiB. Keep datasets
and large checkpoints in images, object storage, or a network volume. Pod creation, detached
training, deadlines, and deletion remain the responsibility of a project controller following the
`runpod-training` skill.

## Support matrix

“Configured” means the contract and automation exist. “Verified” is reserved for an execution backed by evidence in this repository.

<!-- sync:start support-matrix -->
| Runtime | Architecture | Status | Evidence |
| --- | --- | --- | --- |
| Native CPU / MNIST | macOS arm64 | Locally verified | [Promoted exp-001 evidence](artifacts/mnist-baseline/exp-001/20260812T092650681503Z-439c052c-b51254/summary.json) |
| Native MPS | macOS arm64 | Manual / unverified | No promoted run yet |
| CPU Docker | Linux amd64 | CI-configured / locally unverified | Docker daemon unavailable during bootstrap |
| Single NVIDIA GPU | Linux amd64, CUDA 12.6 | Supported contract / manual-unverified | No GPU run in this repository |
| Single-node DDP | Linux amd64, 2× NVIDIA GPU | Supported contract / manual-unverified | No DDP run in this repository |
| Apptainer / SIF | Linux amd64 | Supported contract / manual-unverified | Apptainer unavailable during bootstrap |
| Runpod Pod SSH transport | Direct TCP or Basic SSH proxy | Contract-tested / live proxy unverified | Fallback and PTY transfer unit tests; no promoted GPU run |
| Multi-node Slurm | — | Out of scope for v1 | No launcher contract |
<!-- sync:end support-matrix -->

This table changes only when matching evidence exists. Linux CPU remains covered by CI configuration and contract tests, not by the local macOS training result.

## Verified reference

The bundled baseline was trained from a clean commit. Its official test set remained untouched; the reported result is from the deterministic 5,000-example validation split.

<!-- sync:start verified-evidence -->
| Field | Verified value |
| --- | --- |
| Experiment | `mnist-baseline/exp-001` |
| Source commit | `439c052c57242d7a5806e9070203c5ae2061cc77` |
| Date / platform | 2026-08-12 / macOS 26.3 arm64 |
| Runtime | Python 3.12.9 / PyTorch 2.13.0 / native CPU |
| Seed / split | 42 / deterministic 55,000 train + 5,000 validation |
| Primary result | `validation/loss = 0.0463606422` at epoch 5 |
| Secondary result | `validation/accuracy = 0.9848` |
| Evidence | [summary](artifacts/mnist-baseline/exp-001/20260812T092650681503Z-439c052c-b51254/summary.json) · [validation](artifacts/mnist-baseline/exp-001/20260812T092650681503Z-439c052c-b51254/validation.json) · [checksums](artifacts/mnist-baseline/exp-001/20260812T092650681503Z-439c052c-b51254/checksums.sha256) |
<!-- sync:end verified-evidence -->

## Reproducibility contract

- Configs are complete YAML documents validated with strict Pydantic models (`extra="forbid"`).
- `strict` mode enables deterministic behavior and fails when an operation cannot honor it. `performance` mode is explicitly exploratory and cannot be promoted.
- Device requests are fail-closed; the runner never changes CUDA or MPS to CPU on your behalf.
- `train` writes local evidence, `validate` judges it, and only an explicit decision changes tracked research records.
- The source commit in a result is the commit that produced the run, not the later documentation commit.
- `runs/` is the complete local record; `artifacts/` contains only promoted summaries, manifests, checksums, and small figures.
- Data, weights, SIF files, secrets, mutable image tags, and host-specific absolute paths do not belong in Git.
- External trackers may degrade with a warning; canonical local logging remains authoritative.

## Project boundary

`mltrain` is the stable, project-neutral core. The replaceable package implements a single `ProjectAdapter` selected by `[tool.mltrain]` in `pyproject.toml`. Project training may depend on its data, model, and tracking modules; project validation must not depend on training orchestration. This keeps a new research task replaceable without forking the governance layer.

The template is intentionally PyTorch-first. It does not bind training to Hydra, Lightning, W&B, MLflow, DVC, Git LFS, or a cloud platform. The Runpod transport is an optional standard-library integration and is never imported by the core lifecycle or project adapter. Add external tracking only when a project needs it; the `add-experiment-logging` skill documents the local-first adapter contract.

## Contributing

Keep changes small and evidence-backed. Run the repository checks before proposing a change, never edit a recorded experiment in place, and do not upgrade a support claim without a corresponding validated artifact. See `AGENTS.md` for the rules followed by both people and coding agents.

## License

[MIT](LICENSE) © 2026 ML/DL Training Template contributors.
