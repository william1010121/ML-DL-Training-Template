---
name: training-manager
description: Govern reproducible experiments in repositories created from this ML/DL training template. Use when changing project source, experiment YAML, research records, run evidence, or repository structure; when initializing a new project; or when adding a research line or experiment under configs/. Keep configs, registry, research summaries, and promoted evidence consistent.
---

# Training Manager

Preserve the repository's separation between experiment intent, runtime evidence, and research decisions.

## Workflow

1. Read `references/repository-contract.md` before changing source, configs, registry, research records, or artifacts.
2. Inspect `configs/registry.yml` and the relevant `configs/<research-line>/research.md` before editing an experiment.
3. Use the bundled scripts for initialization and creation. Run without `--apply` first, inspect the diff, then rerun with `--apply`.
4. Treat an experiment config as mutable only while its registry status is `planned`. Create the next experiment after a result has been recorded.
5. When adding, removing, renaming, or widening a YAML option, update the project-specific Pydantic schema in `src/<project_package>/config.py`, the consuming runtime code, and focused tests in the same change. Never let the schema advertise behavior the implementation does not support.
6. Run the repository validators and focused tests after changes. Do not claim support without evidence.
7. Read and follow `../runpod-training/SKILL.md` before changing Runpod Pod orchestration,
   SSH readiness, remote transfer, watchdog, or teardown behavior.

## Deterministic operations

Initialize a repository once:

```bash
python .agents/skills/training-manager/scripts/initialize_project.py \
  --project-name "My Project" --package-name my_project
```

Create a research line:

```bash
python .agents/skills/training-manager/scripts/new_research_line.py \
  my-baseline --goal "Establish a reproducible baseline."
```

Create an experiment from a complete candidate config or a prior experiment:

```bash
python .agents/skills/training-manager/scripts/new_experiment.py \
  my-baseline --config /tmp/candidate.yml

python .agents/skills/training-manager/scripts/new_experiment.py \
  my-baseline --from configs/my-baseline/exp-001.yml \
  --goal "Test the next controlled change."
```

Add `--apply` only after reviewing dry-run output. Never bypass collision, registry, or schema failures by editing around the scripts.

Run initialization only from the clean, committed, untouched template. It rejects symlinks and modified example files, removes the stale `uv.lock`, and leaves the project adapter intentionally unset. Implement the new adapter, restore `[tool.mltrain].adapter`, then run `uv lock` before creating an experiment. Every candidate experiment is validated with that configured adapter before any file changes.

## Evidence rules

- Keep full runs in ignored `runs/`; keep datasets and checkpoints out of Git.
- Use `mltrain validate` before `mltrain record-result` or `mltrain promote`.
- Record the source commit that produced a result, not the later documentation commit.
- Keep dirty or incomplete runs exploratory. Promote only clean, strict, completed runs.
- Append decisions; never rewrite an existing result row to make a new conclusion fit.
