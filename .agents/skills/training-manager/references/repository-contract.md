# Repository contract

## Boundaries

- Keep stable governance code in `src/mltrain/`.
- Keep project code in `src/<project_package>/{data,model,training,validate,tracking}/`.
- Keep experiment intent in full standalone `configs/<research-line>/exp-###.yml` files.
- Keep full runtime evidence in ignored `runs/` and curated small evidence in tracked `artifacts/`.
- Keep actual data in ignored `datasets/` and weights in ignored `checkpoints/`.

Do not introduce imports from `mltrain` into a specific project implementation beyond public contracts. Keep validation independent of training orchestration.

Provider integrations must remain opt-in. `src/mltrain/runpod_transport.py` may be imported by a
project controller, but the core lifecycle and project adapter must not import it. A project that
does not use Runpod must not need Runpod credentials, CLI tools, or network access.

## Configuration contract

Keep the project-specific Pydantic schema in `src/<project_package>/config.py`. Keep only project-neutral fields and contracts in `src/mltrain/`; do not add task-specific model, dataset, optimizer, metric, or tracker choices there.

Treat config changes according to their scope:

- For a new experiment using values already allowed by the schema, create the next `exp-###.yml`; do not change Python code.
- For a new YAML field or allowed value, update the project schema, the data/model/training/validation/tracking code that consumes it, and focused tests together.
- Remove an option from the schema when its runtime implementation is removed. Do not accept ignored or partially implemented fields.
- Keep `extra="forbid"` at every config level so misspellings and unsupported options fail before training.

Before applying a new experiment config, validate the complete YAML through the configured `ProjectAdapter.config_model`. A syntactically valid YAML mapping is not sufficient.

## Research records

Treat `configs/research.md` as the project map. It contains the project goal, a research-line index, and global decisions. Treat each `configs/<research-line>/research.md` as the concise record for that line, with one Goal and one Results table.

Use this result shape:

```text
| Experiment | Commit | Primary result | Decision | Status |
```

The directory is a research line, not an experiment. Number experiments independently within each line as `exp-001`, `exp-002`, and so on.

## Lifecycle

Register new configs as `planned`. A validated run may become `exploratory` or `completed`. Promotion is explicit and is allowed only for a strict completed run from a clean commit. Once a result is recorded, lock the config hash and create a new experiment for every later change.

Never silently fall back to another device, download data during training, commit secrets, or claim that a manual path is verified.
