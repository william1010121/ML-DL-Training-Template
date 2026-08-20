---
description: Run the repository check gate that mirrors the CI quality job
---

Run the repository checks in this order, stopping to report the first failure with its output:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv run python scripts/validate_repo.py
uv run python scripts/validate_skills.py --official skip
uv run python scripts/check_readme_sync.py
```

Then summarize which checks passed and which failed. If a check could not run at all, say so
plainly rather than reporting it as passed or weakening the gate. Do not run formatters that
rewrite unrelated files.
