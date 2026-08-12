#!/usr/bin/env python3
"""Check that intentionally shared README fragments have not drifted.

Both README files mark verbatim, language-neutral fragments with:

    <!-- sync:start quickstart -->
    ...
    <!-- sync:end quickstart -->

The prose around a fragment may be translated; commands and support claims may not.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

START = re.compile(r"^<!-- sync:start ([a-z0-9][a-z0-9-]*) -->$")
END = re.compile(r"^<!-- sync:end ([a-z0-9][a-z0-9-]*) -->$")
BASH_BLOCK = re.compile(r"```bash\n(?P<body>.*?)\n```", re.DOTALL)


def shared_fragments(path: Path) -> dict[str, str]:
    fragments: dict[str, str] = {}
    name: str | None = None
    lines: list[str] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if match := START.fullmatch(line):
            if name is not None:
                raise ValueError(f"{path}:{line_number}: nested sync marker")
            name = match.group(1)
            if name in fragments:
                raise ValueError(f"{path}:{line_number}: duplicate sync block {name!r}")
            lines = []
            continue
        if match := END.fullmatch(line):
            if name != match.group(1):
                raise ValueError(f"{path}:{line_number}: unmatched sync end marker")
            fragments[name] = "\n".join(lines).strip()
            name = None
            lines = []
            continue
        if name is not None:
            lines.append(line.rstrip())

    if name is not None:
        raise ValueError(f"{path}: sync block {name!r} is not closed")
    return fragments


def check(root: Path) -> list[str]:
    english = root / "README.md"
    chinese = root / "README.zh-TW.md"
    errors: list[str] = []
    for path in (english, chinese):
        if not path.is_file():
            errors.append(f"missing {path.relative_to(root)}")
    if errors:
        return errors

    try:
        left = shared_fragments(english)
        right = shared_fragments(chinese)
    except ValueError as exc:
        return [str(exc)]

    if not left:
        errors.append("README.md has no synchronized blocks")
    if set(left) != set(right):
        errors.append(
            "README sync block names differ: "
            f"README.md={sorted(left)}, README.zh-TW.md={sorted(right)}"
        )
    for name in sorted(set(left) & set(right)):
        if left[name] != right[name]:
            errors.append(f"README sync block {name!r} differs")
    english_text = english.read_text(encoding="utf-8")
    chinese_text = chinese.read_text(encoding="utf-8")
    duplicated_bash = {
        match.group("body").strip() for match in BASH_BLOCK.finditer(english_text)
    } & {match.group("body").strip() for match in BASH_BLOCK.finditer(chinese_text)}
    synchronized_bash = {
        match.group("body").strip()
        for fragment in left.values()
        for match in BASH_BLOCK.finditer(fragment)
    }
    for block in sorted(duplicated_bash - synchronized_bash):
        first_line = block.splitlines()[0] if block else "<empty>"
        errors.append(f"duplicated bash block is not synchronized: {first_line}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    errors = check(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("README synchronized blocks match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
