from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from core.artifacts import PackageRef, verify as verify_package
from core.contracts import load_gate_result, load_handoff_result
from core.stages.base import StageContext
from core.stages.f1_intake import IntakeStageRunner
from core.stages.f1_intake.cli import main as intake_cli
from core.stages.f1_intake.freeze import analyze_intake, freeze_intake, init_preparation
from validate.base import CheckContext
from validate.intake_check import IntakeChecker

from helpers import create_sample_repo, make_asset_file, write_task_yaml


@pytest.mark.parametrize("source_kind", ["remote", "other_local"])
@pytest.mark.parametrize("source_change", ["unchanged", "dirty", "missing"])
def test_freeze_rechecks_the_analyzed_repo_root_override(
    tmp_path: Path, source_kind: str, source_change: str,
) -> None:
    info = create_sample_repo(tmp_path / "actual")
    repo = info["repo"]
    task_path = write_task_yaml(
        tmp_path / "task.yaml", repo=repo, commit=info["head_commit"],
        startup_probe=False,
    )
    task = yaml.safe_load(task_path.read_text())
    task["source"]["repo"] = (
        "https://github.com/example/upstream.git" if source_kind == "remote"
        else str(create_sample_repo(tmp_path / "declared")["repo"])
    )
    task_path.write_text(yaml.safe_dump(task), encoding="utf-8")
    prep = tmp_path / "prep"
    init_preparation(task_path, prep)
    state = analyze_intake(prep, repo_root=repo)
    assert not state["blocked_reasons"], state
    assert state["resolved_repo"] == str(repo.resolve())
    assert json.loads((prep / "prep_state.json").read_text())["resolved_repo"] == str(repo.resolve())

    if source_change == "dirty":
        (repo / "configs/pi05.yaml").write_text("batch_size: 99\n")
    elif source_change == "missing":
        repo.rename(tmp_path / "moved-source")
    result = freeze_intake(prep)
    if source_change == "unchanged":
        assert result.status == "completed", result
        assert result.package_ref is not None
        assert "prep_state.json" not in verify_package(result.package_ref).manifest_paths
    else:
        assert result.status == "blocked", result
        assert result.package_ref is None
        assert ("SOURCE_WORKTREE_DIRTY" if source_change == "dirty" else "SOURCE_NOT_DIRECTORY") in result.reason_codes


def test_cli_freeze_uses_repo_root_saved_by_analyze(tmp_path: Path, capsys) -> None:
    info = create_sample_repo(tmp_path)
    repo = info["repo"]
    task_path = write_task_yaml(
        tmp_path / "task.yaml", repo=repo, commit=info["head_commit"],
        startup_probe=False,
    )
    task = yaml.safe_load(task_path.read_text())
    task["source"]["repo"] = "https://github.com/example/upstream.git"
    task_path.write_text(yaml.safe_dump(task))
    prep = tmp_path / "prep"
    assert intake_cli(["init", "--request", str(task_path), "--prep-dir", str(prep)]) == 0
    capsys.readouterr()
    assert intake_cli(["analyze", "--prep-dir", str(prep), "--repo-root", str(repo)]) == 0
    assert json.loads(capsys.readouterr().out)["resolved_repo"] == str(repo.resolve())
    (repo / "configs/pi05.yaml").write_text("batch_size: 99\n")
    assert intake_cli(["freeze", "--prep-dir", str(prep)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "blocked"
    assert receipt["reason_codes"] == ["SOURCE_WORKTREE_DIRTY"]
    assert receipt["package_dir"] is None


def test_end_to_end_init_analyze_freeze_and_independent_intake_checker(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path, with_submodule=True)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]
    ckpt_path, ckpt_sha = make_asset_file(tmp_path, "assets/pi05.ckpt", b"fake-weights")

    task_path = write_task_yaml(
        tmp_path / "task.yaml",
        repo=repo,
        commit=commit,
        argv=[sys.executable, "scripts/train.py", "--config", "configs/pi05.yaml"],
        assets=[
            {
                "id": "checkpoint",
                "kind": "checkpoint",
                "required_for": ["f1_startup_probe"],
                "source": "hf://example/pi05",
                "revision": "main",
                "expected_sha256": ckpt_sha,
                "access_policy": "owner_provided",
                "local_path": str(ckpt_path),
            }
        ],
        startup_probe=True,
        asset_mode="owner_provided",
    )

    prep_dir = tmp_path / "prep"
    init_res = init_preparation(task_path, prep_dir)
    assert init_res["status"] == "initialized"

    analyze_res = analyze_intake(prep_dir, probe_mode="startup")
    assert analyze_res["status"] == "analyzed"
    assert analyze_res["observed_scope"] == ["static", "startup_probe"]

    freeze_res = freeze_intake(prep_dir)
    assert freeze_res.status == "completed"
    assert freeze_res.package_ref is not None
    assert freeze_res.release_record_path is not None
    assert freeze_res.release_record_path.exists()

    # Verify sealed package integrity and required layout
    verified = verify_package(freeze_res.package_ref)
    expected_paths = {
        "inputs/task.yaml",
        "inputs/source_fingerprint.json",
        "analysis/reference_manifest.yaml",
        "analysis/reference_entrypoints.yaml",
        "analysis/launch_contract.yaml",
        "analysis/resource_manifest.yaml",
        "analysis/analysis_report.md",
        "handoff/result.json",
    }
    assert expected_paths <= verified.manifest_paths

    handoff = load_handoff_result(freeze_res.package_ref.package_dir / "handoff/result.json")
    assert handoff.execution_status == "completed"
    assert handoff.observed_scope == ["static", "startup_probe"]
    assert handoff.validator_verdict is not None

    # Independent checker verifies the sealed package
    checker = IntakeChecker()
    gate = checker.check(
        CheckContext(
            commit=commit,
            manifest_sha256=freeze_res.package_ref.manifest_sha256,
            package_dir=freeze_res.package_ref.package_dir,
            reference_dir=repo,
        )
    )
    assert gate.status == "pass"
    assert gate.binding.commit == commit
    assert gate.binding.manifest_sha256 == freeze_res.package_ref.manifest_sha256
    assert gate.binding.validator_sha256 == checker.validator_sha256()

    # Verdict JSON persisted on disk passes strict core loader
    verdict_disk = tmp_path / "verdict.json"
    verdict_disk.write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")
    assert load_gate_result(verdict_disk).status == "pass"


def test_unsealed_staging_dir_is_rejected_and_missing_asset_yields_waiting(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]
    expected_sha = "2468ace013579bdf2468ace013579bdf2468ace013579bdf2468ace013579bdf"

    task_path = write_task_yaml(
        tmp_path / "task_waiting.yaml",
        repo=repo,
        commit=commit,
        argv=[sys.executable, "scripts/train.py", "--config", "configs/pi05.yaml"],
        assets=[
            {
                "id": "checkpoint",
                "kind": "checkpoint",
                "required_for": ["f1_startup_probe"],
                "source": "hf://example/pi05",
                "revision": "main",
                "expected_sha256": expected_sha,
                "access_policy": "owner_provided",
                "local_path": None,
            }
        ],
        startup_probe=True,
        asset_mode="manifest_only",
    )
    prep_dir = tmp_path / "prep_wait"
    init_preparation(task_path, prep_dir)
    analyze_intake(prep_dir, probe_mode="none")
    freeze_res = freeze_intake(prep_dir)

    # Waiting assets cannot freeze as completed release!
    assert freeze_res.status == "waiting"
    assert freeze_res.package_ref is None
    assert "WAITING_ASSET" in freeze_res.reason_codes

    # Passing an unsealed directory to IntakeChecker must fail closed.
    checker = IntakeChecker()
    unsealed = tmp_path / "unsealed.staging"
    unsealed.mkdir()
    (unsealed / "inputs").mkdir()
    gate = checker.check(
        CheckContext(
            commit=commit,
            manifest_sha256=expected_sha,
            package_dir=unsealed,
            reference_dir=repo,
        )
    )
    assert gate.status in ("blocked", "error")
    assert "UNSEALED_PACKAGE_REJECTED" in gate.reason_codes
    assert gate.first_mismatch is not None


def test_intake_checker_negative_cases_tampered_manifest_wrong_commit_and_scope_mismatch(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path, extra_commit=True)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]
    ckpt_path, ckpt_sha = make_asset_file(tmp_path, "assets/pi05.ckpt", b"weights-ok")

    task_path = write_task_yaml(
        tmp_path / "task.yaml",
        repo=repo,
        commit=commit,
        argv=[sys.executable, "scripts/train.py", "--config", "configs/pi05.yaml"],
        assets=[
            {
                "id": "checkpoint",
                "kind": "checkpoint",
                "required_for": ["f1_startup_probe"],
                "source": "hf://example/pi05",
                "revision": "main",
                "expected_sha256": ckpt_sha,
                "access_policy": "owner_provided",
                "local_path": str(ckpt_path),
            }
        ],
        startup_probe=True,
        asset_mode="owner_provided",
    )

    prep_dir = tmp_path / "prep"
    init_preparation(task_path, prep_dir)
    analyze_intake(prep_dir, probe_mode="startup")
    freeze_res = freeze_intake(prep_dir)
    assert freeze_res.package_ref is not None
    ref: PackageRef = freeze_res.package_ref
    checker = IntakeChecker()

    # 1) Wrong expected commit
    wrong_commit_gate = checker.check(
        CheckContext(
            commit=info["first_commit"],
            manifest_sha256=ref.manifest_sha256,
            package_dir=ref.package_dir,
            reference_dir=repo,
        )
    )
    assert wrong_commit_gate.status == "blocked"
    assert "SOURCE_COMMIT_MISMATCH" in wrong_commit_gate.reason_codes
    assert wrong_commit_gate.metrics["expected"] == info["first_commit"]
    assert wrong_commit_gate.metrics["actual"] == commit
    assert wrong_commit_gate.metrics["evidence_path"] == "inputs/source_fingerprint.json"

    # 2) Tampered file inside sealed package (same byte length -> FILE_DIGEST_MISMATCH)
    report_file = ref.package_dir / "analysis" / "analysis_report.md"
    orig_bytes = report_file.read_bytes()
    tampered_bytes = b"X" + orig_bytes[1:]
    report_file.write_bytes(tampered_bytes)
    tampered_gate = checker.check(
        CheckContext(
            commit=commit,
            manifest_sha256=ref.manifest_sha256,
            package_dir=ref.package_dir,
            reference_dir=repo,
        )
    )
    assert tampered_gate.status == "error"
    assert "FILE_DIGEST_MISMATCH" in tampered_gate.reason_codes
    assert tampered_gate.metrics["evidence_path"] == "analysis/analysis_report.md"

    # 3) Scope mismatch: startup_probe requested in task, but probe was not executed (--probe none)
    prep_no_probe = tmp_path / "prep_no_probe"
    init_preparation(task_path, prep_no_probe)
    analyze_intake(prep_no_probe, probe_mode="none")
    freeze_no_probe = freeze_intake(prep_no_probe)
    assert freeze_no_probe.status == "blocked"
    assert "SCOPE_NOT_SATISFIED" in freeze_no_probe.reason_codes
    assert freeze_no_probe.package_ref is None


def test_intake_stage_runner_and_cli_roundtrip(tmp_path: Path) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]
    ckpt_path, ckpt_sha = make_asset_file(tmp_path, "assets/pi05.ckpt", b"weights-cli")

    task_path = write_task_yaml(
        tmp_path / "task_cli.yaml",
        repo=repo,
        commit=commit,
        argv=[sys.executable, "scripts/train.py", "--config", "configs/pi05.yaml"],
        assets=[
            {
                "id": "checkpoint",
                "kind": "checkpoint",
                "required_for": ["f1_startup_probe"],
                "source": "hf://example/pi05",
                "revision": "main",
                "expected_sha256": ckpt_sha,
                "access_policy": "owner_provided",
                "local_path": str(ckpt_path),
            }
        ],
        startup_probe=True,
        asset_mode="owner_provided",
    )

    # 1) StageRunner integration
    runner = IntakeStageRunner()
    stage_res = runner.run(
        StageContext(
            run_id="run-intake-1",
            stage="intake",
            workspace_dir=repo,
            output_dir=tmp_path / "runner_prep",
            parameters={"task_path": str(task_path), "probe_mode": "startup"},
        )
    )
    assert stage_res.status == "completed"
    assert stage_res.artifact_dir is not None
    assert stage_res.manifest_sha256 is not None

    # 2) Thin CLI: init -> analyze -> freeze -> independent intake_check CLI
    repo_root = Path(__file__).parents[3]
    cli_prep = tmp_path / "cli_prep"

    init_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.stages.f1_intake.cli",
            "init",
            "--request",
            str(task_path),
            "--prep-dir",
            str(cli_prep),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(init_proc.stdout)["status"] == "initialized"

    analyze_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.stages.f1_intake.cli",
            "analyze",
            "--prep-dir",
            str(cli_prep),
            "--probe",
            "startup",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(analyze_proc.stdout)["status"] == "analyzed"

    freeze_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.stages.f1_intake.cli",
            "freeze",
            "--prep-dir",
            str(cli_prep),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    freeze_payload = json.loads(freeze_proc.stdout)
    assert freeze_payload["status"] == "completed"
    package_dir = freeze_payload["package_dir"]
    manifest_sha = freeze_payload["manifest_sha256"]

    # Independent checker CLI verifies the sealed package
    check_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "validate.intake_check",
            "--package-dir",
            package_dir,
            "--manifest-sha256",
            manifest_sha,
            "--expected-commit",
            commit,
            "--reference-dir",
            str(repo),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    verdict_payload = json.loads(check_proc.stdout)
    assert verdict_payload["status"] == "pass"
    assert verdict_payload["binding"]["manifest_sha256"] == manifest_sha

    # Ensure CLI --help does not fake a completed receipt
    help_task = tmp_path / "task_help.yaml"
    raw = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    raw["entrypoint"]["argv"] = [sys.executable, "scripts/train.py", "--help"]
    help_task.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    help_prep = tmp_path / "help_prep"
    init_preparation(help_task, help_prep)
    analyze_intake(help_prep, probe_mode="startup")
    freeze_help = freeze_intake(help_prep)
    assert freeze_help.status == "blocked"
    assert "SCOPE_NOT_SATISFIED" in freeze_help.reason_codes


def test_static_only_intake_marks_unexecuted_scopes_not_run_and_cli_separates_exit_code(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]

    # 1) Static-only task (`startup_probe: false`) passes without running probe; unexecuted scopes stay `not_run`.
    static_task = write_task_yaml(
        tmp_path / "task_static.yaml",
        repo=repo,
        commit=commit,
        assets=[],
        startup_probe=False,
        asset_mode="manifest_only",
    )
    prep_static = tmp_path / "prep_static"
    init_preparation(static_task, prep_static)
    analyze_intake(prep_static, probe_mode="none")
    freeze_static = freeze_intake(prep_static)
    assert freeze_static.status == "completed"
    assert freeze_static.package_ref is not None
    assert freeze_static.verdict is not None
    scope_status = freeze_static.verdict["metrics"]["scope_status"]
    assert scope_status["static"] == "passed"
    assert scope_status["startup_probe"] == "not_run"
    assert scope_status["train_step"] == "not_run"

    # 2) CLI process exits 0 even when domain receipt status is `waiting` (exit code decoupled from verdict).
    waiting_task = write_task_yaml(
        tmp_path / "task_wait_cli.yaml",
        repo=repo,
        commit=commit,
        assets=[
            {
                "id": "checkpoint",
                "kind": "checkpoint",
                "required_for": ["f1_startup_probe"],
                "source": "hf://example/pi05",
                "revision": "main",
                "expected_sha256": "2468ace013579bdf2468ace013579bdf2468ace013579bdf2468ace013579bdf",
                "access_policy": "owner_provided",
                "local_path": None,
            }
        ],
        startup_probe=True,
        asset_mode="manifest_only",
    )
    repo_root = Path(__file__).parents[3]
    prep_wait = tmp_path / "prep_wait_cli"
    init_preparation(waiting_task, prep_wait)
    analyze_intake(prep_wait, probe_mode="none")
    wait_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.stages.f1_intake.cli",
            "freeze",
            "--prep-dir",
            str(prep_wait),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert wait_proc.returncode == 0
    wait_payload = json.loads(wait_proc.stdout)
    assert wait_payload["status"] == "waiting"
    assert "WAITING_ASSET" in wait_payload["reason_codes"]
    assert wait_payload["package_dir"] is None


def test_p0_p1_p2_regression_guards_fake_probe_forged_handoff_and_offline_restore(
    tmp_path: Path,
) -> None:
    import shutil
    from core.artifacts import publish as publish_pkg

    info = create_sample_repo(tmp_path / "src_host", with_submodule=True)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]
    ckpt_path, ckpt_sha = make_asset_file(tmp_path, "assets/pi05.ckpt", b"weights-reg")

    # 1) P0-1: `python -c pass` as entrypoint.argv must be rejected even when startup_probe=true
    fake_probe_task = write_task_yaml(
        tmp_path / "task_fake_probe.yaml",
        repo=repo,
        commit=commit,
        entrypoint_path="scripts/train.py",
        argv=[sys.executable, "-c", "pass"],
        assets=[
            {
                "id": "checkpoint",
                "kind": "checkpoint",
                "required_for": ["f1_startup_probe"],
                "source": "hf://example/pi05",
                "revision": "main",
                "expected_sha256": ckpt_sha,
                "access_policy": "owner_provided",
                "local_path": str(ckpt_path),
            }
        ],
        startup_probe=True,
        asset_mode="owner_provided",
    )
    prep_fake = tmp_path / "prep_fake_probe"
    init_preparation(fake_probe_task, prep_fake)
    an_fake = analyze_intake(prep_fake, probe_mode="startup")
    assert "PROBE_ARGV_ENTRYPOINT_MISMATCH" in an_fake["blocked_reasons"]
    assert "startup_probe" not in an_fake["observed_scope"]
    fr_fake = freeze_intake(prep_fake)
    assert fr_fake.status == "blocked"
    assert fr_fake.package_ref is None

    # Build a legitimate static-only sealed package (with submodule bundle included)
    good_task = write_task_yaml(
        tmp_path / "task_good_static.yaml",
        repo=repo,
        commit=commit,
        assets=[],
        startup_probe=False,
        asset_mode="manifest_only",
    )
    prep_good = tmp_path / "prep_good"
    init_preparation(good_task, prep_good)
    analyze_intake(prep_good, probe_mode="none")
    fr_good = freeze_intake(prep_good)
    assert fr_good.status == "completed"
    assert fr_good.package_ref is not None
    sealed_dir = fr_good.package_ref.package_dir
    sealed_sha = fr_good.package_ref.manifest_sha256

    # Ensure no dummy "f"*64 manifest SHA exists inside sealed package pre_publish_verdict
    pre_verdict_text = (
        sealed_dir / "evidence" / "validate" / "pre_publish_verdict.json"
    ).read_text(encoding="utf-8")
    assert "f" * 64 not in pre_verdict_text

    # 2) P1-3: Move/delete original host repo and verify sealed package STILL passes offline
    # by restoring root + recursive submodule from packaged bundles on a fresh path!
    moved_repo = tmp_path / "moved_away_repo"
    shutil.move(str(repo), str(moved_repo))
    checker = IntakeChecker()
    offline_gate = checker.check(
        CheckContext(
            commit=commit,
            manifest_sha256=sealed_sha,
            package_dir=sealed_dir,
            reference_dir=None,
        )
    )
    assert offline_gate.status == "pass"

    # 3) P0-2: Semantically forged handoff (blocked status, inflated scope, wrong verdict target)
    # re-sealed with valid manifest must STILL be rejected by IntakeChecker.
    forged_staging = tmp_path / "forged_handoff.staging"
    shutil.copytree(sealed_dir, forged_staging)
    (forged_staging / "artifact_manifest.json").unlink()
    handoff_file = forged_staging / "handoff" / "result.json"
    hdoc = json.loads(handoff_file.read_text(encoding="utf-8"))
    report_sha = __import__("hashlib").sha256(
        (forged_staging / "analysis" / "analysis_report.md").read_bytes()
    ).hexdigest()
    hdoc["execution_status"] = "blocked"
    hdoc["blocked_reasons"] = ["FORGED"]
    hdoc["observed_scope"] = ["static", "train_step_replay"]
    hdoc["validator_verdict"] = {
        "path": "analysis/analysis_report.md",
        "sha256": report_sha,
    }
    handoff_file.write_text(json.dumps(hdoc, indent=2), encoding="utf-8")
    forged_ref = publish_pkg(forged_staging)
    forged_gate = checker.check(
        CheckContext(
            commit=commit,
            manifest_sha256=forged_ref.manifest_sha256,
            package_dir=forged_ref.package_dir,
        )
    )
    assert forged_gate.status == "blocked"
    assert "HANDOFF_STATUS_NOT_COMPLETED" in forged_gate.reason_codes

    # 4) P0-2b: Completed handoff with inflated observed_scope is rejected
    forged_scope_staging = tmp_path / "forged_scope.staging"
    shutil.copytree(sealed_dir, forged_scope_staging)
    (forged_scope_staging / "artifact_manifest.json").unlink()
    hf2 = forged_scope_staging / "handoff" / "result.json"
    hdoc2 = json.loads(hf2.read_text(encoding="utf-8"))
    hdoc2["observed_scope"] = ["static", "train_step_replay"]
    hf2.write_text(json.dumps(hdoc2, indent=2), encoding="utf-8")
    forged_ref2 = publish_pkg(forged_scope_staging)
    gate_scope = checker.check(
        CheckContext(
            commit=commit,
            manifest_sha256=forged_ref2.manifest_sha256,
            package_dir=forged_ref2.package_dir,
        )
    )
    assert gate_scope.status == "blocked"
    assert "HANDOFF_SCOPE_EXCEEDED" in gate_scope.reason_codes

    # 5) P0-2c: Completed handoff whose validator_verdict points to analysis_report.md is rejected
    forged_vtarget = tmp_path / "forged_vtarget.staging"
    shutil.copytree(sealed_dir, forged_vtarget)
    (forged_vtarget / "artifact_manifest.json").unlink()
    hf3 = forged_vtarget / "handoff" / "result.json"
    hdoc3 = json.loads(hf3.read_text(encoding="utf-8"))
    hdoc3["validator_verdict"] = {
        "path": "analysis/analysis_report.md",
        "sha256": report_sha,
    }
    hf3.write_text(json.dumps(hdoc3, indent=2), encoding="utf-8")
    forged_ref3 = publish_pkg(forged_vtarget)
    gate_vtarget = checker.check(
        CheckContext(
            commit=commit,
            manifest_sha256=forged_ref3.manifest_sha256,
            package_dir=forged_ref3.package_dir,
        )
    )
    assert gate_vtarget.status == "blocked"
    assert "HANDOFF_VERDICT_INVALID" in gate_vtarget.reason_codes

    # 6) P2-1: Cross-document inconsistency (launch_contract disagrees with task.yaml) is rejected
    forged_launch = tmp_path / "forged_launch.staging"
    shutil.copytree(sealed_dir, forged_launch)
    (forged_launch / "artifact_manifest.json").unlink()
    lc_file = forged_launch / "analysis" / "launch_contract.yaml"
    lc_doc = yaml.safe_load(lc_file.read_text(encoding="utf-8"))
    lc_doc["entrypoint"]["topology"] = "forged_multi_node"
    lc_file.write_text(yaml.safe_dump(lc_doc, sort_keys=False), encoding="utf-8")
    forged_ref4 = publish_pkg(forged_launch)
    gate_launch = checker.check(
        CheckContext(
            commit=commit,
            manifest_sha256=forged_ref4.manifest_sha256,
            package_dir=forged_ref4.package_dir,
        )
    )
    assert gate_launch.status == "blocked"
    assert "LAUNCH_CONTRACT_TASK_MISMATCH" in gate_launch.reason_codes

    # 7) Batch 2 P1: forging a static pre_publish_verdict.json (missing real checks/checks_sha256)
    # and updating handoff SHA must STILL be rejected by IntakeChecker.
    forged_pre = tmp_path / "forged_pre_verdict.staging"
    shutil.copytree(sealed_dir, forged_pre)
    (forged_pre / "artifact_manifest.json").unlink()
    pv_file = forged_pre / "evidence" / "validate" / "pre_publish_verdict.json"
    pv_doc = json.loads(pv_file.read_text(encoding="utf-8"))
    pv_doc["checks"] = {"forged": True}
    pv_bytes = json.dumps(pv_doc, indent=2).encode("utf-8")
    pv_file.write_bytes(pv_bytes)
    hf4 = forged_pre / "handoff" / "result.json"
    hdoc4 = json.loads(hf4.read_text(encoding="utf-8"))
    hdoc4["validator_verdict"]["sha256"] = __import__("hashlib").sha256(pv_bytes).hexdigest()
    hf4.write_text(json.dumps(hdoc4, indent=2), encoding="utf-8")
    forged_ref5 = publish_pkg(forged_pre)
    gate_pre = checker.check(
        CheckContext(
            commit=commit,
            manifest_sha256=forged_ref5.manifest_sha256,
            package_dir=forged_ref5.package_dir,
        )
    )
    assert gate_pre.status == "blocked"
    assert "HANDOFF_VERDICT_BINDING_MISMATCH" in gate_pre.reason_codes
