# Goal

Build reproducible AI training projects whose configs express intent, run manifests preserve what actually happened, and research notes contain only evidence-backed decisions. Keep the repository portable across native, Docker, and Apptainer runtimes without coupling project code to a platform, tracker, dataset, or machine.

## Required skills

Before changing source code, training or validation behavior, configs, experiments, registry state, research results, or promoted artifacts, read and follow `.agents/skills/training-manager/SKILL.md`. Use its deterministic scripts to initialize the template and create research lines or experiments; preview the dry-run first and apply explicitly. Never bypass its no-overwrite and immutable-experiment rules with manual edits.

Before adding or changing metrics, tracker behavior, W&B, MLflow, TensorBoard, or any logging adapter, also read and follow `.agents/skills/add-experiment-logging/SKILL.md`. Canonical local logs must remain available when an external tracker is absent or fails.

## Architecture invariants

- `src/mltrain/` is the stable governance and CLI layer. It must not import the project package directly; load the configured `ProjectAdapter` dynamically.
- The project package owns `data`, `model`, `training`, `validate`, and `tracking`. Training may depend on the other project modules. Validation must not depend on training orchestration.
- Keep experiment YAML complete and strict. Do not introduce hidden inheritance, host-specific absolute paths, mutable image tags, or secret interpolation.
- `configs/research.md` is the project-level goal, research-line index, and global decision log. Results belong in `configs/<research-line>/research.md`.
- After a result is recorded, its experiment config is immutable. Create the next `exp-###.yml` for any meaningful change.

## Evidence workflow

`train` writes only to the ignored run directory. `validate` classifies evidence. `record-result` or `promote` performs the explicit tracked update. Never hand-edit registry or research state to claim a result that the CLI did not validate.

A completed result requires a clean source commit, config/environment/data/model identities, the exact command, seed and determinism state, required metrics, and successful validation. Performance-mode or dirty runs are exploratory. Promotion is allowed only for completed strict runs.

The commit recorded in research is the source commit that produced the run, not the later evidence-documentation commit. Do not manufacture metrics, hashes, badges, support claims, or sample evidence.

## Repository hygiene

- Never commit dataset bytes, model weights, checkpoints, complete runs, SIF images, cache files, `.env`, credentials, or generated tracker directories.
- Track only source, complete configs, documentation, small tests/fixtures, and promoted metadata or small figures in `artifacts/`.
- Read credentials only from environment variables. Record names when needed, never values.
- Treat tracked files larger than 10 MiB or common secret patterns as validation failures unless a narrow reviewed allowlist says otherwise.
- Preserve user changes. Do not overwrite, reset, delete, or reformat unrelated work.

## Runtime and support claims

Docker images are environment-only: do not copy application source into them. Bind the repository read-only and give write access only to designated data, checkpoint, run, and artifact mounts. Apptainer SIF files must derive from immutable OCI digests; do not create a competing environment definition.

Device selection is fail-closed. Never silently fall back from CUDA or MPS to CPU. Mark GPU, DDP, MPS, Docker, Apptainer, or Slurm behavior as verified only when the repository contains matching evidence. Otherwise say configured, supported contract, manual/unverified, or out of scope.

## Checks

Before handing off a change, run the smallest relevant unit and contract tests plus repository validation. For broad changes, run Ruff, mypy, pytest, README synchronization, skill validation, and the available CPU smoke checks. Do not run formatters that rewrite unrelated files. If hardware or a runtime is unavailable, report the unrun check plainly instead of weakening the gate.
