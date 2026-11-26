from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from core.artifacts import PackageRef, verify as verify_package
from core.artifacts import publish as publish_pkg
from core.contracts import (
    load_anatomy_behavior_contract,
    load_gate_result,
    load_handoff_result,
)
from core.stages.f2_anatomy import AnatomyStageRunner
from core.stages.f2_anatomy.capture import capture_two_attempts
from core.stages.f2_anatomy.freeze import freeze_anatomy
from core.stages.base import StageContext
from validate.anatomy_check import AnatomyChecker
from validate.base import CheckContext

from anatomy_helpers import build_sealed_intake_package


def test_two_subprocess_capture_freeze_and_independent_anatomy_checker(
    tmp_path: Path,
) -> None:
    intake = build_sealed_intake_package(tmp_path)
    intake_ref: PackageRef = intake["package_ref"]
    prep_dir = tmp_path / "anatomy_prep"

    cap = capture_two_attempts(
        intake_package_dir=intake_ref.package_dir,
        intake_manifest_sha256=intake_ref.manifest_sha256,
        prep_dir=prep_dir,
        case_id="pi05-cpu-mock-step-1",
        seed=42,
        world_size=1,
    )
    assert cap.status == "captured"
    assert cap.attempt1_pid != cap.attempt2_pid
    assert cap.comparison.status == "pass"
    assert cap.execution_environment == "cpu_mock"
    assert cap.evidence_scope == "orchestration_only"

    freeze_res = freeze_anatomy(prep_dir)
    assert freeze_res.status == "completed"
    assert freeze_res.package_ref is not None
    assert freeze_res.release_record_path is not None
    assert freeze_res.release_record_path.exists()

    verified = verify_package(freeze_res.package_ref)
    required_files = {
        "inputs/anatomy_case.yaml",
        "analysis/step_anatomy.yaml",
        "analysis/behavior_contract.yaml",
        "analysis/observation_plan.yaml",
        "evidence/attempt-1/capture.json",
        "evidence/attempt-2/capture.json",
        "evidence/comparison.json",
        "handoff/result.json",
    }
    assert required_files <= verified.manifest_paths

    # Strict contract loaders validate the sealed files.
    contract = load_anatomy_behavior_contract(
        freeze_res.package_ref.package_dir / "analysis/behavior_contract.yaml"
    )
    assert contract.case.f1_manifest_sha256 == intake_ref.manifest_sha256
    assert contract.case.upstream_commit == intake["upstream_commit"]

    handoff = load_handoff_result(
        freeze_res.package_ref.package_dir / "handoff/result.json"
    )
    assert handoff.execution_status == "completed"
    assert handoff.stage == "f2"
    assert "train_step_replay" in handoff.observed_scope

    # Independent AnatomyChecker verifies sealed package + intake binding.
    checker = AnatomyChecker()
    gate = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=freeze_res.package_ref.manifest_sha256,
            package_dir=freeze_res.package_ref.package_dir,
            parameters={
                "intake_package_dir": str(intake_ref.package_dir),
                "intake_manifest_sha256": intake_ref.manifest_sha256,
                "spec_manifest_sha256": freeze_res.spec_manifest_sha256,
            },
        )
    )
    assert gate.status == "pass"
    assert gate.binding.validator_sha256 == checker.validator_sha256()
    assert gate.metrics["execution_environment"] == "cpu_mock"
    assert gate.metrics["evidence_scope"] == "orchestration_only"

    verdict_disk = tmp_path / "anatomy_verdict.json"
    verdict_disk.write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")
    assert load_gate_result(verdict_disk).status == "pass"


def test_anatomy_rejects_wrong_intake_manifest_sha_and_tampered_packages(
    tmp_path: Path,
) -> None:
    intake = build_sealed_intake_package(tmp_path)
    intake_ref: PackageRef = intake["package_ref"]

    # 1) Wrong intake manifest SHA rejected at capture time.
    bad_cap = capture_two_attempts(
        intake_package_dir=intake_ref.package_dir,
        intake_manifest_sha256="1" * 64,
        prep_dir=tmp_path / "bad_intake_prep",
    )
    assert bad_cap.status == "blocked"
    assert "MANIFEST_DIGEST_MISMATCH" in bad_cap.reason_codes

    # 1b) Batch 3 P0-1: waiting-asset intake fixture (fails IntakeChecker) is rejected by capture!
    waiting_intake = build_sealed_intake_package(
        tmp_path / "waiting_case",
        name="intake-waiting-01",
        asset_ready=False,
    )
    wait_cap = capture_two_attempts(
        intake_package_dir=waiting_intake["package_ref"].package_dir,
        intake_manifest_sha256=waiting_intake["package_ref"].manifest_sha256,
        prep_dir=tmp_path / "wait_anatomy_prep",
    )
    assert wait_cap.status == "blocked"

    # 2) Injected negative fault during attempt-2 blocks freeze (no sealed release).
    neg_prep = tmp_path / "neg_prep"
    neg_cap = capture_two_attempts(
        intake_package_dir=intake_ref.package_dir,
        intake_manifest_sha256=intake_ref.manifest_sha256,
        prep_dir=neg_prep,
        inject_attempt2="swap_cameras",
    )
    assert neg_cap.status == "failed"
    assert "CAMERA_ORDER_MISMATCH" in neg_cap.reason_codes
    neg_freeze = freeze_anatomy(neg_prep)
    assert neg_freeze.status == "blocked"
    assert neg_freeze.package_ref is None
    assert "CAMERA_ORDER_MISMATCH" in neg_freeze.reason_codes

    # 3) Sealed anatomy package tampered after publish -> AnatomyChecker fails closed.
    ok_prep = tmp_path / "ok_prep"
    capture_two_attempts(
        intake_package_dir=intake_ref.package_dir,
        intake_manifest_sha256=intake_ref.manifest_sha256,
        prep_dir=ok_prep,
    )
    ok_freeze = freeze_anatomy(ok_prep)
    assert ok_freeze.package_ref is not None
    ref = ok_freeze.package_ref

    checker = AnatomyChecker()
    # Wrong expected intake manifest SHA passed to checker
    wrong_intake_gate = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=ref.manifest_sha256,
            package_dir=ref.package_dir,
            parameters={
                "intake_package_dir": str(intake_ref.package_dir),
                "intake_manifest_sha256": "2" * 64,
                "spec_manifest_sha256": ok_freeze.spec_manifest_sha256,
            },
        )
    )
    assert wrong_intake_gate.status in ("blocked", "error")
    assert "INTAKE_MANIFEST_MISMATCH" in wrong_intake_gate.reason_codes

    # Tamper a byte inside sealed package (same length -> FILE_DIGEST_MISMATCH)
    target_file = ref.package_dir / "analysis" / "step_anatomy.yaml"
    raw_bytes = target_file.read_bytes()
    target_file.write_bytes(b"#" + raw_bytes[1:])
    tampered_gate = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=ref.manifest_sha256,
            package_dir=ref.package_dir,
            parameters={
                "intake_package_dir": str(intake_ref.package_dir),
                "intake_manifest_sha256": intake_ref.manifest_sha256,
                "spec_manifest_sha256": ok_freeze.spec_manifest_sha256,
            },
        )
    )
    assert tampered_gate.status == "error"
    assert "FILE_DIGEST_MISMATCH" in tampered_gate.reason_codes
    assert tampered_gate.metrics["evidence_path"] == "analysis/step_anatomy.yaml"


def test_anatomy_stage_runner_and_cli_capture_freeze_replay_and_checker_cli(
    tmp_path: Path,
) -> None:
    intake = build_sealed_intake_package(tmp_path)
    intake_ref: PackageRef = intake["package_ref"]

    # 1) AnatomyStageRunner end-to-end
    runner = AnatomyStageRunner()
    stage_res = runner.run(
        StageContext(
            run_id="run-anatomy-1",
            stage="anatomy",
            workspace_dir=tmp_path,
            output_dir=tmp_path / "runner_anatomy",
            upstream_artifacts={"intake": intake_ref.package_dir},
            parameters={
                "intake_manifest_sha256": intake_ref.manifest_sha256,
                "seed": 19,
            },
        )
    )
    assert stage_res.status == "completed"
    assert stage_res.artifact_dir is not None
    assert stage_res.manifest_sha256 is not None

    # 2) Thin CLI: capture -> freeze -> replay -> independent anatomy_check CLI
    repo_root = Path(__file__).parents[3]
    cli_prep = tmp_path / "cli_anatomy"

    cap_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.stages.f2_anatomy.cli",
            "capture",
            "--intake-dir",
            str(intake_ref.package_dir),
            "--intake-manifest-sha256",
            intake_ref.manifest_sha256,
            "--prep-dir",
            str(cli_prep),
            "--seed",
            "23",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    cap_payload = json.loads(cap_proc.stdout)
    assert cap_payload["status"] == "captured"
    assert cap_payload["attempt1_pid"] != cap_payload["attempt2_pid"]

    freeze_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.stages.f2_anatomy.cli",
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

    replay_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.stages.f2_anatomy.cli",
            "replay",
            "--package-dir",
            package_dir,
            "--manifest-sha256",
            manifest_sha,
            "--spec-manifest-sha256",
            cap_payload["spec_manifest_sha256"],
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    replay_payload = json.loads(replay_proc.stdout)
    assert replay_payload["status"] == "pass"

    # Negative replay injection via CLI still exits 0 with structured failure payload
    replay_neg_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.stages.f2_anatomy.cli",
            "replay",
            "--package-dir",
            package_dir,
            "--manifest-sha256",
            manifest_sha,
            "--spec-manifest-sha256",
            cap_payload["spec_manifest_sha256"],
            "--inject",
            "step_counter_only",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    replay_neg = json.loads(replay_neg_proc.stdout)
    assert replay_neg["status"] == "fail"
    assert "NO_REAL_PARAMETER_UPDATE" in replay_neg["reason_codes"]

    # Independent anatomy_check CLI
    check_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "validate.anatomy_check",
            "--package-dir",
            package_dir,
            "--manifest-sha256",
            manifest_sha,
            "--spec-manifest-sha256",
            cap_payload["spec_manifest_sha256"],
            "--expected-commit",
            intake["upstream_commit"],
            "--intake-dir",
            str(intake_ref.package_dir),
            "--intake-manifest-sha256",
            intake_ref.manifest_sha256,
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    verdict = json.loads(check_proc.stdout)
    assert verdict["status"] == "pass"
    assert verdict["binding"]["manifest_sha256"] == manifest_sha


def test_anatomy_checker_rejects_unsealed_staging_same_pid_and_fake_handoff(
    tmp_path: Path,
) -> None:
    intake = build_sealed_intake_package(tmp_path)
    intake_ref: PackageRef = intake["package_ref"]
    checker = AnatomyChecker()
    intake_params = {
        "intake_package_dir": str(intake_ref.package_dir),
        "intake_manifest_sha256": intake_ref.manifest_sha256,
    }

    # 1) Unsealed .staging directory rejected immediately.
    unsealed = tmp_path / "anatomy-fake.staging"
    unsealed.mkdir()
    gate_unsealed = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256="a" * 64,
            package_dir=unsealed,
            parameters=intake_params,
        )
    )
    assert gate_unsealed.status == "blocked"
    assert "UNSEALED_PACKAGE_REJECTED" in gate_unsealed.reason_codes

    prep_dir = tmp_path / "prep_for_forge"
    capture_two_attempts(
        intake_package_dir=intake_ref.package_dir,
        intake_manifest_sha256=intake_ref.manifest_sha256,
        prep_dir=prep_dir,
    )
    res = freeze_anatomy(prep_dir)
    assert res.package_ref is not None
    orig_dir = res.package_ref.package_dir
    # Retain the controller's trusted spec digest across all deliberately resealed forgeries.
    intake_params["spec_manifest_sha256"] = res.spec_manifest_sha256

    # 2) Batch 3 P0-1: omitting intake_package_dir / intake_manifest_sha256 MUST fail closed.
    gate_no_intake = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=res.package_ref.manifest_sha256,
            package_dir=orig_dir,
        )
    )
    assert gate_no_intake.status == "blocked"
    assert "INTAKE_REFERENCE_REQUIRED" in gate_no_intake.reason_codes

    # 3) Forge a sealed package where attempt-1 & attempt-2 share the same PID.
    forged_staging = tmp_path / "forged_same_pid.staging"
    shutil.copytree(orig_dir, forged_staging)
    (forged_staging / "artifact_manifest.json").unlink()
    a1 = json.loads((forged_staging / "evidence/attempt-1/capture.json").read_text(encoding="utf-8"))
    a2_path = forged_staging / "evidence/attempt-2/capture.json"
    a2 = json.loads(a2_path.read_text(encoding="utf-8"))
    a2["pid"] = a1["pid"]
    a2_path.write_text(json.dumps(a2, indent=2), encoding="utf-8")
    forged_ref = publish_pkg(forged_staging)

    gate_same_pid = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_ref.manifest_sha256,
            package_dir=forged_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_same_pid.status == "blocked"
    assert "SAME_PROCESS_ATTEMPTS" in gate_same_pid.reason_codes

    # 4) Batch 3 P0-2: forging attempt capture values (even with distinct fake PIDs)
    # is caught by independent two-subprocess replay!
    forged_replay_staging = tmp_path / "forged_replay.staging"
    shutil.copytree(orig_dir, forged_replay_staging)
    (forged_replay_staging / "artifact_manifest.json").unlink()
    for p in ("evidence/attempt-1/capture.json", "evidence/attempt-2/capture.json"):
        fp = forged_replay_staging / p
        d = json.loads(fp.read_text(encoding="utf-8"))
        d["loss"] = 999.0
        fp.write_text(json.dumps(d, indent=2), encoding="utf-8")
    forged_replay_ref = publish_pkg(forged_replay_staging)
    gate_replay = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_replay_ref.manifest_sha256,
            package_dir=forged_replay_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_replay.status == "blocked"
    assert "INDEPENDENT_REPLAY_MISMATCH" in gate_replay.reason_codes

    # 5) Batch 3 P0-3: replacing pre_publish_verdict.json with `{"foo":"bar"}` + updating handoff SHA fails!
    forged_pv_staging = tmp_path / "forged_pv.staging"
    shutil.copytree(orig_dir, forged_pv_staging)
    (forged_pv_staging / "artifact_manifest.json").unlink()
    pv_path = forged_pv_staging / "evidence/validate/pre_publish_verdict.json"
    fake_pv_bytes = b'{"foo":"bar"}\n'
    pv_path.write_bytes(fake_pv_bytes)
    h_path = forged_pv_staging / "handoff/result.json"
    h_doc = json.loads(h_path.read_text(encoding="utf-8"))
    h_doc["validator_verdict"]["sha256"] = hashlib.sha256(fake_pv_bytes).hexdigest()
    h_path.write_text(json.dumps(h_doc, indent=2), encoding="utf-8")
    forged_pv_ref = publish_pkg(forged_pv_staging)
    gate_pv = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_pv_ref.manifest_sha256,
            package_dir=forged_pv_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_pv.status == "blocked"
    assert "HANDOFF_VERDICT_BINDING_MISMATCH" in gate_pv.reason_codes

    # 6) Batch 3 P1-1: mutating attempt-2 labels (`real_gpu`/`model_behavior`) fails!
    forged_a2_staging = tmp_path / "forged_a2_tags.staging"
    shutil.copytree(orig_dir, forged_a2_staging)
    (forged_a2_staging / "artifact_manifest.json").unlink()
    a2_fp = forged_a2_staging / "evidence/attempt-2/capture.json"
    a2_d = json.loads(a2_fp.read_text(encoding="utf-8"))
    a2_d["execution_environment"] = "real_gpu"
    a2_d["evidence_scope"] = "model_behavior"
    a2_fp.write_text(json.dumps(a2_d, indent=2), encoding="utf-8")
    forged_a2_ref = publish_pkg(forged_a2_staging)
    gate_a2 = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_a2_ref.manifest_sha256,
            package_dir=forged_a2_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_a2.status == "blocked"
    assert "MOCK_SCOPE_TAG_MISSING" in gate_a2.reason_codes

    # 7) Batch 3 P1-2a: tampering framework_commit in inputs/anatomy_case.yaml fails!
    forged_fw_staging = tmp_path / "forged_fw.staging"
    shutil.copytree(orig_dir, forged_fw_staging)
    (forged_fw_staging / "artifact_manifest.json").unlink()
    case_fp = forged_fw_staging / "inputs/anatomy_case.yaml"
    case_d = yaml.safe_load(case_fp.read_text(encoding="utf-8"))
    case_d["framework_commit"] = "9" * 40
    case_fp.write_text(yaml.safe_dump(case_d, sort_keys=False), encoding="utf-8")
    forged_fw_ref = publish_pkg(forged_fw_staging)
    gate_fw = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_fw_ref.manifest_sha256,
            package_dir=forged_fw_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_fw.status == "blocked"
    assert "CROSS_DOCUMENT_MISMATCH" in gate_fw.reason_codes

    # 8) Batch 3 P1-2b: flipping all assertions required=False in behavior_contract.yaml fails!
    forged_assert_staging = tmp_path / "forged_assert.staging"
    shutil.copytree(orig_dir, forged_assert_staging)
    (forged_assert_staging / "artifact_manifest.json").unlink()
    bc_fp = forged_assert_staging / "analysis/behavior_contract.yaml"
    bc_d = yaml.safe_load(bc_fp.read_text(encoding="utf-8"))
    for item in bc_d["assertions"]:
        item["required"] = False
    bc_fp.write_text(yaml.safe_dump(bc_d, sort_keys=False), encoding="utf-8")
    forged_assert_ref = publish_pkg(forged_assert_staging)
    gate_assert = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_assert_ref.manifest_sha256,
            package_dir=forged_assert_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_assert.status == "blocked"
    assert "REQUIRED_ASSERTION_DISABLED" in gate_assert.reason_codes

    # 9) Batch 3 P2-1: tampering recorded comparison.json content (while keeping status='pass') fails!
    forged_cmp_staging = tmp_path / "forged_cmp.staging"
    shutil.copytree(orig_dir, forged_cmp_staging)
    (forged_cmp_staging / "artifact_manifest.json").unlink()
    cmp_fp = forged_cmp_staging / "evidence/comparison.json"
    cmp_d = json.loads(cmp_fp.read_text(encoding="utf-8"))
    cmp_d["summary"] = "tampered summary"
    cmp_fp.write_text(json.dumps(cmp_d, indent=2), encoding="utf-8")
    forged_cmp_ref = publish_pkg(forged_cmp_staging)
    gate_cmp = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_cmp_ref.manifest_sha256,
            package_dir=forged_cmp_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_cmp.status == "blocked"
    assert "RECORDED_COMPARISON_MISMATCH" in gate_cmp.reason_codes

    # 10) Forge a sealed package with a blocked handoff status.
    forged_handoff_staging = tmp_path / "forged_handoff.staging"
    shutil.copytree(orig_dir, forged_handoff_staging)
    (forged_handoff_staging / "artifact_manifest.json").unlink()
    handoff_path = forged_handoff_staging / "handoff/result.json"
    hdoc = json.loads(handoff_path.read_text(encoding="utf-8"))
    hdoc["execution_status"] = "blocked"
    hdoc["blocked_reasons"] = ["FORGED"]
    handoff_path.write_text(json.dumps(hdoc, indent=2), encoding="utf-8")
    forged_h_ref = publish_pkg(forged_handoff_staging)

    gate_fake_handoff = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_h_ref.manifest_sha256,
            package_dir=forged_h_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_fake_handoff.status == "blocked"
    assert "HANDOFF_STATUS_NOT_COMPLETED" in gate_fake_handoff.reason_codes

    # 11) Batch 3 P0: relabeling fake CPU package as `backend="jax"` MUST fail closed!
    forged_jax = tmp_path / "forged_jax.staging"
    shutil.copytree(orig_dir, forged_jax)
    (forged_jax / "artifact_manifest.json").unlink()
    bc_jax_fp = forged_jax / "analysis/behavior_contract.yaml"
    bc_jax = yaml.safe_load(bc_jax_fp.read_text(encoding="utf-8"))
    bc_jax["case"]["precision"]["backend"] = "jax"
    bc_jax["case"]["gpu_count"] = 1
    bc_jax_fp.write_text(yaml.safe_dump(bc_jax, sort_keys=False), encoding="utf-8")
    forged_jax_ref = publish_pkg(forged_jax)
    gate_jax = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_jax_ref.manifest_sha256,
            package_dir=forged_jax_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_jax.status == "blocked"
    assert "UNSUPPORTED_BACKEND_REPLAY_REQUIRED" in gate_jax.reason_codes

    # 12) Batch 3 P1-2: tampering `precision.dtype` in behavior_contract.yaml fails!
    forged_prec = tmp_path / "forged_prec.staging"
    shutil.copytree(orig_dir, forged_prec)
    (forged_prec / "artifact_manifest.json").unlink()
    bc_prec_fp = forged_prec / "analysis/behavior_contract.yaml"
    bc_prec = yaml.safe_load(bc_prec_fp.read_text(encoding="utf-8"))
    bc_prec["case"]["precision"]["dtype"] = "bfloat16"
    bc_prec_fp.write_text(yaml.safe_dump(bc_prec, sort_keys=False), encoding="utf-8")
    forged_prec_ref = publish_pkg(forged_prec)
    gate_prec = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_prec_ref.manifest_sha256,
            package_dir=forged_prec_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_prec.status == "blocked"
    assert "CROSS_DOCUMENT_MISMATCH" in gate_prec.reason_codes

    # 13) Batch 3 P1-3: forging `launch_receipt.command` fails!
    forged_cmd = tmp_path / "forged_cmd.staging"
    shutil.copytree(orig_dir, forged_cmd)
    (forged_cmd / "artifact_manifest.json").unlink()
    a1_cmd_fp = forged_cmd / "evidence/attempt-1/capture.json"
    a1_cmd_d = json.loads(a1_cmd_fp.read_text(encoding="utf-8"))
    a1_cmd_d["launch_receipt"]["command"] = ["/bin/sh", "-c", "true"]
    a1_cmd_fp.write_text(json.dumps(a1_cmd_d, indent=2), encoding="utf-8")
    forged_cmd_ref = publish_pkg(forged_cmd)
    gate_cmd = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_cmd_ref.manifest_sha256,
            package_dir=forged_cmd_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_cmd.status == "blocked"
    assert "LAUNCH_RECEIPT_INVALID" in gate_cmd.reason_codes

    # 14) Batch 3 P2-2: tampering observation_plan.boundaries fails!
    forged_obs = tmp_path / "forged_obs.staging"
    shutil.copytree(orig_dir, forged_obs)
    (forged_obs / "artifact_manifest.json").unlink()
    obs_fp = forged_obs / "analysis/observation_plan.yaml"
    obs_d = yaml.safe_load(obs_fp.read_text(encoding="utf-8"))
    obs_d["boundaries"] = []
    obs_fp.write_text(yaml.safe_dump(obs_d, sort_keys=False), encoding="utf-8")
    forged_obs_ref = publish_pkg(forged_obs)
    gate_obs = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_obs_ref.manifest_sha256,
            package_dir=forged_obs_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_obs.status == "blocked"
    assert "OBSERVATION_PLAN_MISMATCH" in gate_obs.reason_codes

    # 15) Batch 3 P2-2: inflating handoff.observed_scope with `task_eval` fails!
    forged_hscope = tmp_path / "forged_hscope.staging"
    shutil.copytree(orig_dir, forged_hscope)
    (forged_hscope / "artifact_manifest.json").unlink()
    hs_fp = forged_hscope / "handoff/result.json"
    hs_d = json.loads(hs_fp.read_text(encoding="utf-8"))
    hs_d["observed_scope"] = ["train_step_replay", "task_eval"]
    hs_fp.write_text(json.dumps(hs_d, indent=2), encoding="utf-8")
    forged_hs_ref = publish_pkg(forged_hscope)
    gate_hs = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=forged_hs_ref.manifest_sha256,
            package_dir=forged_hs_ref.package_dir,
            parameters=intake_params,
        )
    )
    assert gate_hs.status == "blocked"
    assert "HANDOFF_SCOPE_EXCEEDED" in gate_hs.reason_codes

    # 16) Verify passing GateResult exposes independent replay_pids & replay_receipts (and step_anatomy has no self-reported PIDs)
    step_anatomy_doc = yaml.safe_load((orig_dir / "analysis/step_anatomy.yaml").read_text(encoding="utf-8"))
    assert "attempt_pids" not in step_anatomy_doc
    assert step_anatomy_doc["precision"]["backend"] == "fake"
    ok_gate = checker.check(
        CheckContext(
            commit=intake["upstream_commit"],
            manifest_sha256=res.package_ref.manifest_sha256,
            package_dir=orig_dir,
            parameters=intake_params,
        )
    )
    assert ok_gate.status == "pass"
    assert len(ok_gate.metrics["replay_pids"]) == 2
    assert ok_gate.metrics["replay_pids"][0] != ok_gate.metrics["replay_pids"][1]
    assert len(ok_gate.metrics["replay_receipts"]) == 2
