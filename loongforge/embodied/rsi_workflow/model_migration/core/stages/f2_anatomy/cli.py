"""Thin JSON CLI for Stage 2 (`anatomy`): `capture` / `freeze` / `replay`.

Process exit code is separated from domain receipt/verdict status:
valid structured receipts (`captured`, `failed`, `blocked`, `pass`, `fail`) exit 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters.fake import DEFAULT_SAMPLE_IDS, FAKE_NEGATIVE_INJECTIONS
from .capture import capture_two_attempts, replay_sealed_anatomy
from .freeze import freeze_anatomy
from .spec import load_anatomy_spec


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m core.stages.f2_anatomy.cli",
        description="Thin JSON CLI for Stage 2 (anatomy) capture, freeze, and replay.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cap_cmd = subparsers.add_parser(
        "capture",
        help="Verify sealed intake package and run two independent single-step subprocesses.",
    )
    cap_cmd.add_argument("--intake-dir", required=True)
    cap_cmd.add_argument("--intake-manifest-sha256", required=True)
    cap_cmd.add_argument("--prep-dir", required=True)
    cap_cmd.add_argument("--case-id", default="pi05-cpu-mock-step-1")
    cap_cmd.add_argument("--spec", default=None, help="Strict bare or schema-2 wrapped AnatomySpec YAML (default: pi05 preset).")
    cap_cmd.add_argument("--seed", type=int, default=42)
    cap_cmd.add_argument("--world-size", type=int, default=1)
    cap_cmd.add_argument("--sample-ids", default=",".join(DEFAULT_SAMPLE_IDS))
    cap_cmd.add_argument("--inject-attempt1", default=None, choices=FAKE_NEGATIVE_INJECTIONS)
    cap_cmd.add_argument("--inject-attempt2", default=None, choices=FAKE_NEGATIVE_INJECTIONS)

    freeze_cmd = subparsers.add_parser(
        "freeze",
        help="Validate and atomically publish a sealed anatomy package.",
    )
    freeze_cmd.add_argument("--prep-dir", required=True)
    freeze_cmd.add_argument("--release-name", default=None)
    freeze_cmd.add_argument("--attempt-id", default="anatomy-attempt-01")

    replay_cmd = subparsers.add_parser(
        "replay",
        help="Verify a sealed anatomy package and replay a fresh subprocess against attempt-1.",
    )
    replay_cmd.add_argument("--package-dir", required=True)
    replay_cmd.add_argument("--manifest-sha256", required=True)
    replay_cmd.add_argument("--spec-manifest-sha256", required=True, help="Caller-held pre-capture spec digest.")
    replay_cmd.add_argument("--inject", default=None, choices=FAKE_NEGATIVE_INJECTIONS)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "capture":
        samples = tuple(s.strip() for s in args.sample_ids.split(",") if s.strip())
        res = capture_two_attempts(
            intake_package_dir=Path(args.intake_dir),
            intake_manifest_sha256=args.intake_manifest_sha256,
            prep_dir=Path(args.prep_dir),
            case_id=args.case_id,
            seed=args.seed,
            world_size=args.world_size,
            sample_ids=samples,
            inject_attempt1=args.inject_attempt1,
            inject_attempt2=args.inject_attempt2,
            spec=load_anatomy_spec(args.spec) if args.spec else None,
        )
        _emit(res.to_dict())
        return 0

    if args.command == "freeze":
        res = freeze_anatomy(
            Path(args.prep_dir),
            release_name=args.release_name,
            attempt_id=args.attempt_id,
        )
        _emit(res.to_dict())
        return 0

    if args.command == "replay":
        report = replay_sealed_anatomy(
            package_dir=Path(args.package_dir),
            manifest_sha256=args.manifest_sha256,
            spec_manifest_sha256=args.spec_manifest_sha256,
            inject=args.inject,
        )
        _emit(report.to_dict())
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
