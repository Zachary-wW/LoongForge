from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from core.artifacts import PackageRef, publish
from core.stages.f1_intake.freeze import analyze_intake, freeze_intake, init_preparation

UPSTREAM_COMMIT = "7f3c2a9d4e5b6f70128394aa65b7c8d9e0f1a2b3"
FRAMEWORK_COMMIT = "5f48c81d1209419fcd53efe8495141ec1be5437f"


def _git(cwd: Path, *args: str) -> str:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "GIT_AUTHOR_NAME": "Test Bot",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test Bot",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def build_sealed_intake_package(
    root: Path,
    *,
    name: str = "intake-sealed-01",
    framework_commit: str = FRAMEWORK_COMMIT,
    asset_ready: bool = True,
    train_step: bool = True,
    interpreter: str = sys.executable,
    source_repo: str | None = None,
) -> dict[str, Any]:
    """Build a genuine sealed Stage 1 (`intake`) package produced by the real intake pipeline.

    When `asset_ready=True` and `train_step=True`, the returned package passes `IntakeChecker`
    with full offline git bundles and verified probe marker. When `asset_ready=False`, builds
    a sealed waiting-asset fixture explicitly to test Stage 2 fail-closed rejection.
    """
    repo = root / f"{name}_repo"
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "pi05.yaml").write_text(
        "model_variant: pi05\nbatch_size: 2\n", encoding="utf-8"
    )
    (repo / "scripts" / "train.py").write_text(
        '''import argparse, json, os
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    out = Path(os.environ.get("INTAKE_PROBE_OUTPUT_DIR", "."))
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_marker.json").write_text(json.dumps({
        "event": "startup_probe_ok",
        "config": args.config,
        "source_commit": os.environ.get("INTAKE_PROBE_COMMIT"),
        "entrypoint_path": os.environ.get("INTAKE_PROBE_ENTRYPOINT"),
        "argv_sha256": os.environ.get("INTAKE_PROBE_ARGV_SHA256"),
        "probe_nonce": os.environ.get("INTAKE_PROBE_NONCE"),
    }), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init upstream repo")
    upstream_commit = _git(repo, "rev-parse", "HEAD")

    assets_dir = root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = assets_dir / "pi05_base.params"
    ckpt_bytes = b"synthetic-checkpoint-bytes-v1"
    ckpt_file.write_bytes(ckpt_bytes)
    ckpt_sha = hashlib.sha256(ckpt_bytes).hexdigest()

    task_doc = {
        "schema_version": 1,
        "source": {
            "repo": source_repo or str(repo),
            "commit": upstream_commit,
            "model_variant": "pi05",
            "behavior_reference": "scripts/train.py",
        },
        "target": {
            "framework": "loongforge",
            "commit": framework_commit,
        },
        "environment": {
            "name": "embodied-cpu-mock",
            "gpu_count": 0,
            "image": None,
            "dependencies_lock": None,
            "python": "3.12",
        },
        "entrypoint": {
            "path": "scripts/train.py",
            "argv": [interpreter, "scripts/train.py", "--config", "configs/pi05.yaml"],
            "cwd": ".",
            "env": {},
            "topology": "single_process",
        },
        "assets": [
            {
                "id": "checkpoint",
                "kind": "checkpoint",
                "required_for": ["f1_startup_probe", "f2_train_step"],
                "source": "hf://example/pi05_base",
                "revision": "main",
                "expected_sha256": ckpt_sha,
                "access_policy": "owner_provided",
                "local_path": str(ckpt_file) if asset_ready else None,
            }
        ],
        "scope": {
            "static": True,
            "startup_probe": True,
            "train_step": train_step,
            "checkpoint_resume": False,
            "task_eval": False,
            "asset_mode": "owner_provided" if asset_ready else "manifest_only",
        },
        "unknowns": [],
        "blocked_reasons": [],
    }

    if asset_ready:
        task_path = root / f"{name}_task.yaml"
        task_path.write_text(yaml.safe_dump(task_doc, sort_keys=False), encoding="utf-8")
        prep = root / f"{name}_prep"
        init_preparation(task_path, prep)
        analyze_intake(prep, probe_mode="startup", repo_root=repo if source_repo else None)
        fr = freeze_intake(prep, release_name=name)
        assert fr.status == "completed" and fr.package_ref is not None, fr
        return {
            "package_ref": fr.package_ref,
            "upstream_commit": upstream_commit,
            "framework_commit": framework_commit,
            "checkpoint_path": ckpt_file,
            "checkpoint_sha256": ckpt_sha,
            "repo": repo,
        }

    # Negative fixture: artificially sealed waiting-asset package (must be rejected by IntakeChecker!)
    staging = root / f"{name}.staging"
    (staging / "inputs").mkdir(parents=True)
    (staging / "analysis").mkdir(parents=True)
    (staging / "inputs" / "task.yaml").write_text(
        yaml.safe_dump(task_doc, sort_keys=False), encoding="utf-8"
    )
    (staging / "analysis" / "resource_manifest.yaml").write_text(
        yaml.safe_dump({"overall_status": "waiting", "records": []}, sort_keys=False),
        encoding="utf-8",
    )
    ref: PackageRef = publish(staging)
    return {
        "package_ref": ref,
        "upstream_commit": upstream_commit,
        "framework_commit": framework_commit,
        "checkpoint_path": ckpt_file,
        "checkpoint_sha256": ckpt_sha,
        "repo": repo,
    }
