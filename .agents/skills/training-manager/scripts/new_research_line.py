#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    SLUG_RE,
    apply_transaction,
    dump_yaml,
    insert_research_row,
    load_yaml,
    registry_lines,
    reject_symlinks,
    repo_root,
    show_diff,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Create and register one research line.")
    value.add_argument("slug", help="lowercase hyphenated research-line identifier")
    value.add_argument("--goal", required=True, help="one-sentence research-line goal")
    value.add_argument("--root", default=".", help="repository root (default: current directory)")
    value.add_argument("--apply", action="store_true", help="write changes; default is dry-run")
    return value


def main() -> int:
    args = parser().parse_args()
    if not SLUG_RE.fullmatch(args.slug):
        raise SystemExit("slug must contain lowercase letters, digits, and single hyphens")
    goal = " ".join(args.goal.split())
    if not goal:
        raise SystemExit("goal must not be empty")

    root = repo_root(args.root)
    reject_symlinks(root)
    line_dir = root / "configs" / args.slug
    line_research = line_dir / "research.md"
    top_research = root / "configs" / "research.md"
    registry_path = root / "configs" / "registry.yml"
    if line_dir.exists():
        raise SystemExit(f"refusing to overwrite existing research line: {line_dir}")

    registry = load_yaml(registry_path)
    lines = registry_lines(registry)
    if args.slug in lines:
        raise SystemExit(f"research line is already registered: {args.slug}")

    research_before = top_research.read_text(encoding="utf-8")
    research_after = insert_research_row(
        research_before,
        f"| [{args.slug}]({args.slug}/research.md) | {goal} | Planned | — |",
    )
    lines[args.slug] = {
        "goal": goal,
        "status": "planned",
        "experiments": {},
    }
    registry_before = registry_path.read_text(encoding="utf-8")
    registry_after = dump_yaml(registry)
    line_content = (
        f"# {args.slug}\n\n## Goal\n\n{goal}\n\n"
        "## Results\n\n"
        "| Experiment | Commit | Primary result | Decision | Status |\n"
        "| --- | --- | --- | --- | --- |\n"
    )

    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: create research line {args.slug}")
    show_diff(line_research, "", line_content)
    show_diff(top_research, research_before, research_after)
    show_diff(registry_path, registry_before, registry_after)
    if not args.apply:
        print("No files changed. Rerun with --apply after reviewing the diff.")
        return 0

    apply_transaction(
        root,
        new_files={line_research: line_content},
        replacements={top_research: research_after, registry_path: registry_after},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
