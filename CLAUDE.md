# Claude Code instructions

@AGENTS.md

The imported contract above is the single source of truth for this repository. It is shared with
every coding agent; do not restate or fork its rules here.

## Repository skills

The authoritative skill content lives in `.agents/skills/<name>/SKILL.md` together with its
`references/*.md`. The skills under `.claude/skills/` are thin pointers so that Claude Code can
discover the same workflows; always read the `.agents/` source before acting on one.

| Work you are asked to do | Read first |
| --- | --- |
| Source, configs, experiments, registry, research records, promoted artifacts | `.agents/skills/training-manager/SKILL.md` |
| Metrics, trackers, profiling, progress reporting | `.agents/skills/add-experiment-logging/SKILL.md` |
| Runpod Pods, SSH readiness, remote transfer, teardown | `.agents/skills/runpod-training/SKILL.md` |

## Checks

Run the smallest relevant tests plus repository validation before handing off a change. For broad
changes run the full gate, which mirrors the CI quality job:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv run python scripts/validate_repo.py
uv run python scripts/validate_skills.py --official skip
uv run python scripts/check_readme_sync.py
```

Report any check you could not run instead of weakening the gate.

## Environment

Use `uv` for every Python operation; never call `pip`. The CLI entry points are `mltrain` and
`mltrain-runpod`, both run through `uv run`.
