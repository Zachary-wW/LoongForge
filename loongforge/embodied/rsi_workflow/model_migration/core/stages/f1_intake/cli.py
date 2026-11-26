"""Thin JSON CLI for Stage 1 (`intake`): `init` / `analyze` / `freeze`.

Process exit code is intentionally separated from domain receipt/verdict status:
- Exit 0 whenever the command parses and produces a structured JSON receipt
  (including `waiting` or `blocked` receipts).
- Exit non-zero only on CLI usage errors or unexpected internal crashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .freeze import analyze_intake, freeze_intake, init_preparation


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m core.stages.f1_intake.cli",
        description="Thin JSON CLI for Stage 1 (intake) offline preparation and freeze.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_cmd = subparsers.add_parser("init", help="Initialize an intake preparation directory.")
    init_cmd.add_argument("--request", required=True, help="Path to task.yaml.")
    init_cmd.add_argument("--prep-dir", required=True, help="Preparation directory path.")
    init_cmd.add_argument("--preparation-id", default=None, help="Optional preparation ID.")
    init_cmd.add_argument("--request-id", default="intake-request-01", help="Optional request ID.")

    analyze_cmd = subparsers.add_parser(
        "analyze",
        help="Run source identity lock, AST discovery, asset triage, and optional probe.",
    )
    analyze_cmd.add_argument("--prep-dir", required=True, help="Preparation directory path.")
    analyze_cmd.add_argument(
        "--probe",
        choices=("none", "startup"),
        default="none",
        help="Whether to execute an isolated startup probe.",
    )
    analyze_cmd.add_argument(
        "--repo-root",
        default=None,
        help="Optional override path for local upstream Git checkout.",
    )
    analyze_cmd.add_argument("--handoff-dir", default=None, help="v2 report/evidence import directory; handoff_only.")
    analyze_cmd.add_argument(
        "--timeout-sec",
        type=float,
        default=60.0,
        help="Timeout in seconds for startup probe.",
    )

    freeze_cmd = subparsers.add_parser(
        "freeze",
        help="Validate and atomically publish an immutable intake package.",
    )
    freeze_cmd.add_argument("--prep-dir", required=True, help="Preparation directory path.")
    freeze_cmd.add_argument("--release-name", default=None, help="Optional release directory name.")
    freeze_cmd.add_argument("--attempt-id", default="intake-attempt-01", help="Attempt ID.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        payload = init_preparation(
            Path(args.request),
            Path(args.prep_dir),
            preparation_id=args.preparation_id,
            request_id=args.request_id,
        )
        _emit(payload)
        return 0

    if args.command == "analyze":
        payload = analyze_intake(
            Path(args.prep_dir),
            probe_mode=args.probe,
            repo_root=Path(args.repo_root) if args.repo_root else None,
            timeout_sec=args.timeout_sec,
            handoff_dir=args.handoff_dir,
        )
        _emit(payload)
        return 0

    if args.command == "freeze":
        result = freeze_intake(
            Path(args.prep_dir),
            release_name=args.release_name,
            attempt_id=args.attempt_id,
        )
        _emit(result.to_dict())
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
