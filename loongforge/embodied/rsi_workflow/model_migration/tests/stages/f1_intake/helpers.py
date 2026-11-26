from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "GIT_AUTHOR_NAME": "Test Bot",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test Bot",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def create_sample_repo(
    root: Path,
    *,
    with_submodule: bool = False,
    extra_commit: bool = False,
    ambiguous_entrypoints: bool = False,
) -> dict[str, Any]:
    """Build a deterministic local Git repository for offline CPU intake tests."""
    repo = root / "upstream_repo"
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-b", "main")

    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir(parents=True)
    (repo / "README.md").write_text(
        "# Fake Embodied Model\nRun `scripts/train.py --config configs/pi05.yaml`.\n",
        encoding="utf-8",
    )
    (repo / "configs" / "pi05.yaml").write_text(
        "model_variant: pi05\nbatch_size: 2\n",
        encoding="utf-8",
    )
    (repo / "scripts" / "train.py").write_text(
        '''"""Fake reference training script for CPU intake tests."""
import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fake embodied trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="startup")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--mutate-source", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mutate_source:
        Path("MUTATED_BY_PROBE.txt").write_text("bad\\n", encoding="utf-8")
    if args.mode == "fail":
        print("simulated startup failure", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir or os.environ.get("INTAKE_PROBE_OUTPUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": "startup_probe_ok",
        "config": args.config,
        "mode": args.mode,
        "source_commit": os.environ.get("INTAKE_PROBE_COMMIT"),
        "entrypoint_path": os.environ.get("INTAKE_PROBE_ENTRYPOINT"),
        "argv_sha256": os.environ.get("INTAKE_PROBE_ARGV_SHA256"),
        "probe_nonce": os.environ.get("INTAKE_PROBE_NONCE"),
    }
    (out_dir / "probe_marker.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )
    if ambiguous_entrypoints:
        (repo / "scripts" / "train_alt.py").write_text(
            'def main() -> int:\n    return 0\n\nif __name__ == "__main__":\n    main()\n',
            encoding="utf-8",
        )

    git(repo, "add", ".")
    git(repo, "commit", "-m", "initial commit")
    first_commit = git(repo, "rev-parse", "HEAD").stdout.strip()

    sub_commit = None
    if with_submodule:
        sub_origin = root / "sub_origin"
        sub_origin.mkdir(parents=True, exist_ok=True)
        git(sub_origin, "init", "-b", "main")
        (sub_origin / "lib.py").write_text("VALUE = 1\n", encoding="utf-8")
        git(sub_origin, "add", "lib.py")
        git(sub_origin, "commit", "-m", "init submodule")
        sub_commit = git(sub_origin, "rev-parse", "HEAD").stdout.strip()

        git(
            repo,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(sub_origin),
            "third_party/helper",
        )
        git(repo, "commit", "-m", "add submodule")
        first_commit = git(repo, "rev-parse", "HEAD").stdout.strip()

    second_commit = first_commit
    if extra_commit:
        (repo / "configs" / "pi05.yaml").write_text(
            "model_variant: pi05\nbatch_size: 4\n",
            encoding="utf-8",
        )
        git(repo, "add", "configs/pi05.yaml")
        git(repo, "commit", "-m", "update config")
        second_commit = git(repo, "rev-parse", "HEAD").stdout.strip()

    return {
        "repo": repo,
        "first_commit": first_commit,
        "head_commit": second_commit,
        "submodule_commit": sub_commit,
    }


def make_asset_file(root: Path, name: str, content: bytes) -> tuple[Path, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return path, digest


def write_task_yaml(
    path: Path,
    *,
    repo: Path,
    commit: str,
    entrypoint_path: str = "scripts/train.py",
    argv: list[str] | None = None,
    assets: list[dict[str, Any]] | None = None,
    startup_probe: bool = True,
    asset_mode: str = "owner_provided",
) -> Path:
    doc = {
        "schema_version": 1,
        "source": {
            "repo": str(repo),
            "commit": commit,
            "model_variant": "pi05",
            "behavior_reference": entrypoint_path,
        },
        "target": {
            "framework": "loongforge",
            "commit": "5f48c81d1209419fcd53efe8495141ec1be5437f",
        },
        "environment": {
            "name": "embodied-cpu-fixture",
            "gpu_count": 0,
            "image": None,
            "dependencies_lock": None,
            "python": "3.12",
        },
        "entrypoint": {
            "path": entrypoint_path,
            "argv": argv
            or ["python3", entrypoint_path, "--config", "configs/pi05.yaml"],
            "cwd": ".",
            "env": {},
            "topology": "single_process",
        },
        "assets": assets or [],
        "scope": {
            "static": True,
            "startup_probe": startup_probe,
            "train_step": False,
            "checkpoint_resume": False,
            "task_eval": False,
            "asset_mode": asset_mode,
        },
        "unknowns": [],
        "blocked_reasons": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path
