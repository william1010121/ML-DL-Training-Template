#!/usr/bin/env python3
"""Validate repo-owned skill contracts and use Codex's official validator if present."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

FRONTMATTER = re.compile(r"\A---\n(?P<body>.*?)\n---(?:\n|\Z)", re.DOTALL)
SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
ALLOWED_FRONTMATTER = {"name", "description"}
OPENAI_TOP_LEVEL = {"interface", "dependencies", "policy"}
INTERFACE_FIELDS = {
    "display_name",
    "short_description",
    "icon_small",
    "icon_large",
    "brand_color",
    "default_prompt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official",
        choices=("auto", "require", "skip"),
        default="auto",
        help="run skill-creator quick_validate when available (default: auto)",
    )
    return parser.parse_args()


def _official_validator() -> Path | None:
    explicit = os.environ.get("SKILL_CREATOR_QUICK_VALIDATE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    candidate = codex_home / "skills/.system/skill-creator/scripts/quick_validate.py"
    if candidate.is_file():
        return candidate
    legacy_candidate = (
        Path.home() / ".agents/skills/.system/skill-creator/scripts/quick_validate.py"
    )
    return legacy_candidate if legacy_candidate.is_file() else None


def _validate_frontmatter(skill_file: Path, errors: list[str]) -> str | None:
    text = skill_file.read_text(encoding="utf-8")
    match = FRONTMATTER.match(text)
    if not match:
        errors.append(f"{skill_file}: missing or invalid YAML frontmatter")
        return None
    try:
        fields = yaml.safe_load(match.group("body"))
    except yaml.YAMLError as error:
        errors.append(f"{skill_file}: invalid YAML frontmatter: {error}")
        return None
    if not isinstance(fields, dict):
        errors.append(f"{skill_file}: frontmatter must be a mapping")
        return None
    unexpected = set(fields) - ALLOWED_FRONTMATTER
    if unexpected:
        errors.append(f"{skill_file}: unexpected frontmatter keys: {sorted(unexpected)}")
    name = fields.get("name")
    description = fields.get("description")
    if not isinstance(name, str) or not SKILL_NAME.fullmatch(name):
        errors.append(f"{skill_file}: name must be lowercase hyphen-case")
        return None
    if len(name) > MAX_SKILL_NAME_LENGTH:
        errors.append(f"{skill_file}: name exceeds {MAX_SKILL_NAME_LENGTH} characters")
    if not isinstance(description, str) or len(description.strip()) < 40:
        errors.append(f"{skill_file}: description is incomplete")
    elif "TODO" in description or len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"{skill_file}: description is invalid")
    return name


def _validate_manifest(skill_dir: Path, name: str, errors: list[str]) -> None:
    path = skill_dir / "agents/openai.yaml"
    if not path.is_file():
        errors.append(f"{path}: required manifest is missing")
        return
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        errors.append(f"{path}: invalid YAML: {error}")
        return
    if not isinstance(value, dict):
        errors.append(f"{path}: manifest must be a mapping")
        return
    unexpected = set(value) - OPENAI_TOP_LEVEL
    if unexpected:
        errors.append(f"{path}: unexpected top-level keys: {sorted(unexpected)}")
    interface = value.get("interface")
    if not isinstance(interface, dict):
        errors.append(f"{path}: interface must be a mapping")
        return
    unexpected_interface = set(interface) - INTERFACE_FIELDS
    if unexpected_interface:
        errors.append(f"{path}: unexpected interface keys: {sorted(unexpected_interface)}")
    for field in ("display_name", "short_description", "default_prompt"):
        if not isinstance(interface.get(field), str) or not interface[field].strip():
            errors.append(f"{path}: interface.{field} must be a non-empty string")
    short = interface.get("short_description", "")
    if isinstance(short, str) and not 25 <= len(short) <= 64:
        errors.append(f"{path}: short_description must be 25-64 characters")
    prompt = interface.get("default_prompt", "")
    if isinstance(prompt, str) and f"${name}" not in prompt:
        errors.append(f"{path}: default_prompt must mention ${name}")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    skills_root = root / ".agents" / "skills"
    errors: list[str] = []
    skill_files = sorted(skills_root.glob("*/SKILL.md"))
    if not skill_files:
        errors.append("no repository skills found")

    for skill_file in skill_files:
        name = _validate_frontmatter(skill_file, errors)
        if name is None:
            continue
        if name != skill_file.parent.name:
            errors.append(f"{skill_file}: name must be {skill_file.parent.name!r}")
        _validate_manifest(skill_file.parent, name, errors)

    official = None if args.official == "skip" else _official_validator()
    if official is None and args.official == "require":
        errors.append(
            "official skill-creator quick_validate.py is unavailable; set "
            "SKILL_CREATOR_QUICK_VALIDATE or install the system skill"
        )
    if official is not None:
        if not official.is_file():
            errors.append(f"official validator is not a file: {official}")
        else:
            for skill_file in skill_files:
                process = subprocess.run(
                    [sys.executable, str(official), str(skill_file.parent)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if process.returncode != 0:
                    message = (process.stdout + process.stderr).strip()
                    errors.append(f"official validation failed for {skill_file.parent}: {message}")

    if errors:
        print("Skill validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    official_status = " + official quick_validate" if official is not None else " (repo contract)"
    print(f"Validated {len(skill_files)} repository skills{official_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
