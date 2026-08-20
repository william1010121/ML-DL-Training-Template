---
description: Create a governed experiment in a research line via the deterministic script
argument-hint: [research-line]
arguments: "research-line"
---

Create the next experiment in the `$research-line` research line.

1. Read `.agents/skills/training-manager/SKILL.md` first.
2. Inspect `configs/registry.yml` and `configs/$research-line/research.md` to see the current
   experiment numbering and the last recorded result.
3. Run the script as a dry-run — no `--apply`:

   ```bash
   python .agents/skills/training-manager/scripts/new_experiment.py $research-line \
     --from configs/$research-line/exp-<latest>.yml \
     --goal "<the controlled change being tested>"
   ```

   Use `--config <path>` instead of `--from` when starting from a complete candidate config.
4. Show me the dry-run output and wait for my explicit confirmation.
5. Only after I confirm, rerun the same command with `--apply`.

Never edit around a collision, registry, or schema failure, and never hand-write an experiment file
that the script would have produced.
