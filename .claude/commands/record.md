---
description: Validate a run and record or promote it through the mltrain CLI
argument-hint: [run-path]
arguments: "run-path"
---

Turn the run at `$run-path` into a research result. The rules governing this workflow live in
`.agents/skills/training-manager/SKILL.md` and `AGENTS.md`; read them rather than relying on this
file to restate them.

1. Validate the evidence first and show me the JSON:

   ```bash
   uv run mltrain validate --run $run-path
   ```

2. Read the classification and tell me which outcome it allows.
3. Propose the decision text, then run exactly one of:

   ```bash
   uv run mltrain record-result --run $run-path --decision "<decision>"
   uv run mltrain promote --run $run-path --decision "<decision>"
   ```

Let the CLI be the only writer of registry and research state.
