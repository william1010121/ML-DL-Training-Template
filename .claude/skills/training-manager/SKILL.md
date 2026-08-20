---
name: training-manager
description: Govern reproducible experiments in repositories created from this ML/DL training template. Use when changing project source, experiment YAML, research records, run evidence, or repository structure; when initializing a new project; or when adding a research line or experiment under configs/. Keep configs, registry, research summaries, and promoted evidence consistent.
---

# Training Manager (pointer)

The authoritative workflow is `.agents/skills/training-manager/SKILL.md`. Read it, and the
`.agents/skills/training-manager/references/repository-contract.md` it cites, before changing
source, configs, registry state, research records, or artifacts.

Rules worth knowing before you open those files:

- Use the bundled scripts under `.agents/skills/training-manager/scripts/`. Run each without
  `--apply` first, review the diff, then rerun with `--apply`.
- An experiment config is mutable only while its registry status is `planned`. After a result is
  recorded, create the next `exp-###.yml` instead of editing it.
- Never hand-edit `configs/registry.yml` or a `research.md` result row to claim something the CLI
  did not validate.
