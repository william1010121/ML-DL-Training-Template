from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

from mltrain.lifecycle import execute, promote_run, record_result, validate_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mltrain", description="Reproducible training lifecycle")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="run a training experiment")
    train.add_argument("--config", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)

    validate = commands.add_parser("validate", help="validate run evidence")
    validate.add_argument("--run", type=Path, required=True)

    record = commands.add_parser("record-result", help="record a validated research result")
    record.add_argument("--run", type=Path, required=True)
    record.add_argument("--decision", required=True)

    promote = commands.add_parser("promote", help="promote completed evidence")
    promote.add_argument("--run", type=Path, required=True)
    promote.add_argument("--decision", required=True)
    return parser


def _die(message: str) -> NoReturn:
    raise SystemExit(f"mltrain: error: {message}")


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "train":
            print(execute(args.config, "train"))
        elif args.command == "evaluate":
            print(execute(args.config, "evaluate", args.checkpoint))
        elif args.command == "validate":
            print(json.dumps(validate_run(args.run), indent=2, sort_keys=True))
        elif args.command == "record-result":
            record_result(args.run, args.decision)
        elif args.command == "promote":
            print(promote_run(args.run, args.decision))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        _die(str(error))
