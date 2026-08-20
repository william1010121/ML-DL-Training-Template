---
description: Validate a run and record or promote it through the mltrain CLI
argument-hint: [run-path]
arguments: "run-path"
---

Turn the run at `$run-path` into a research result.

1. Validate the evidence first and show me the JSON:

   ```bash
   uv run mltrain validate --run $run-path
   ```

2. Read the classification. A performance-mode or dirty run is exploratory; only a clean, strict,
   completed run may be promoted.
3. Propose the decision text, then run exactly one of:

   ```bash
   uv run mltrain record-result --run $run-path --decision "<decision>"
   uv run mltrain promote --run $run-path --decision "<decision>"
   ```

Never hand-edit `configs/registry.yml` or a `research.md` result row to claim a result the CLI did
not validate. The commit recorded in research is the source commit that produced the run, not the
later documentation commit.
