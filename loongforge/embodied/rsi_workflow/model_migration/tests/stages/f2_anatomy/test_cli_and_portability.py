"""Review #2, #3 and #5 regressions at capture/CLI/validator boundaries."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest
import yaml

from core.contracts.common import load_yaml_document
from core.stages.f2_anatomy import capture as capture_module
from core.stages.f2_anatomy.capture import capture_two_attempts
from core.stages.f2_anatomy.freeze import freeze_anatomy
from validate.anatomy_check import AnatomyChecker
from validate.base import CheckContext

from anatomy_helpers import build_sealed_intake_package

ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def intake(tmp_path_factory):
    return build_sealed_intake_package(
        tmp_path_factory.mktemp("portable-intake"), interpreter="python3",
        source_repo="https://github.com/example/upstream.git",
    )


@pytest.fixture(scope="module")
def anatomy(tmp_path_factory, intake):
    prep = tmp_path_factory.mktemp("portable-anatomy")
    ref = intake["package_ref"]
    capture = capture_two_attempts(
        intake_package_dir=ref.package_dir, intake_manifest_sha256=ref.manifest_sha256,
        prep_dir=prep,
    )
    assert capture.status == "captured", capture
    frozen = freeze_anatomy(prep)
    assert frozen.status == "completed", frozen
    return frozen


def test_capture_executes_controlled_python_but_records_portable_command(tmp_path, intake, monkeypatch):
    commands = []
    original_run = subprocess.run

    def record_run(command, *args, **kwargs):
        if "-m" in command and "core.stages.f2_anatomy.adapters.fake" in command:
            commands.append(list(command))
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(capture_module.subprocess, "run", record_run)
    ref = intake["package_ref"]
    result = capture_two_attempts(
        intake_package_dir=ref.package_dir, intake_manifest_sha256=ref.manifest_sha256,
        prep_dir=tmp_path,
    )
    assert result.status == "captured", result
    assert len(commands) == 2
    for index, command in enumerate(commands, 1):
        assert command[0] == sys.executable
        assert Path(command[command.index("--output") + 1]).is_absolute()
        record = json.loads((tmp_path / f"work/evidence/attempt-{index}/capture.json").read_text())
        receipt = record["launch_receipt"]["command"]
        assert receipt[0] == "python3"
        assert receipt[receipt.index("--output") + 1] == f"evidence/attempt-{index}/capture.json"
        assert receipt[receipt.index("--spec-file") + 1] == "evidence/spec_freeze/anatomy_spec.yaml"


def test_moved_package_checks_and_replays_in_a_different_venv(tmp_path, intake, anatomy):
    moved = tmp_path / "moved-anatomy"
    ref = anatomy.package_ref
    shutil.copytree(ref.package_dir, moved)
    venv_dir = tmp_path / "validator-venv"
    # Copy the binary so even realpath differs from the capture interpreter.
    # Inherit already installed dependencies; no network or pip execution.
    venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=False).create(venv_dir)
    python = venv_dir / "bin/python3"
    assert python.resolve() != Path(sys.executable).resolve()
    intake_ref = intake["package_ref"]
    checked = subprocess.run([
        str(python), "-m", "validate.anatomy_check",
        "--package-dir", str(moved), "--manifest-sha256", ref.manifest_sha256,
        "--spec-manifest-sha256", anatomy.spec_manifest_sha256,
        "--expected-commit", intake["upstream_commit"],
        "--intake-dir", str(intake_ref.package_dir),
        "--intake-manifest-sha256", intake_ref.manifest_sha256,
    ], cwd=ROOT, capture_output=True, text=True, check=True)
    gate = json.loads(checked.stdout)
    assert gate["status"] == "pass", gate
    assert len(set(gate["metrics"]["replay_pids"])) == 2
    for receipt in gate["metrics"]["replay_receipts"]:
        command = receipt["command"]
        assert command[0] == "python3"
        assert command[command.index("--output") + 1] in ("replay-1.json", "replay-2.json")
    replayed = subprocess.run([
        str(python), "-m", "core.stages.f2_anatomy.cli", "replay",
        "--package-dir", str(moved), "--manifest-sha256", ref.manifest_sha256,
        "--spec-manifest-sha256", anatomy.spec_manifest_sha256,
    ], cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(replayed.stdout)["status"] == "pass"


@pytest.mark.parametrize("mutation", ["interpreter", "absolute_output", "traversal_output", "wrong_output"])
def test_forged_noncanonical_launch_receipts_fail_closed(tmp_path, intake, anatomy, mutation):
    from core.artifacts import MANIFEST_NAME, publish

    stage = tmp_path / "forged.staging"
    shutil.copytree(anatomy.package_ref.package_dir, stage)
    (stage / MANIFEST_NAME).unlink()
    path = stage / "evidence/attempt-1/capture.json"
    record = json.loads(path.read_text())
    command = record["launch_receipt"]["command"]
    if mutation == "interpreter":
        command[0] = "/untrusted/host/python3"
    else:
        command[command.index("--output") + 1] = {
            "absolute_output": "/untrusted/host/capture.json",
            "traversal_output": "../capture.json",
            "wrong_output": "evidence/attempt-2/capture.json",
        }[mutation]
    path.write_text(json.dumps(record))
    forged = publish(stage)
    intake_ref = intake["package_ref"]
    gate = AnatomyChecker().check(CheckContext(
        commit=intake["upstream_commit"], package_dir=forged.package_dir,
        manifest_sha256=forged.manifest_sha256,
        parameters={
            "intake_package_dir": str(intake_ref.package_dir),
            "intake_manifest_sha256": intake_ref.manifest_sha256,
            "spec_manifest_sha256": anatomy.spec_manifest_sha256,
        },
    ))
    assert gate.status == "blocked", gate
    assert gate.reason_codes == ("LAUNCH_RECEIPT_INVALID",)


@pytest.mark.parametrize("format", ["bare", "intake_wrapper", "anatomy_wrapper"])
def test_capture_cli_loads_bare_and_pipeline_wrapped_spec(tmp_path, intake, anatomy, format):
    document = load_yaml_document(anatomy.package_ref.package_dir / "analysis/anatomy_spec.yaml")
    if format == "bare":
        document = document["spec"]
    elif format == "intake_wrapper":
        document = {
            "schema_version": 2, "spec": document["spec"],
            "source_ref": None, "source_commit": intake["upstream_commit"],
            "request_id": "fixture-analysis",
        }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document))
    ref = intake["package_ref"]
    captured = subprocess.run([
        sys.executable, "-m", "core.stages.f2_anatomy.cli", "capture",
        "--intake-dir", str(ref.package_dir),
        "--intake-manifest-sha256", ref.manifest_sha256,
        "--prep-dir", str(tmp_path / "prep"), "--spec", str(spec_path),
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    assert json.loads(captured.stdout)["status"] == "captured"


@pytest.mark.parametrize("mutation", ["wrapper_version", "wrapper_unknown", "inner_unknown", "duplicate", "policy_digest"])
def test_capture_cli_rejects_malformed_wrapped_spec(tmp_path, intake, anatomy, mutation):
    document = load_yaml_document(anatomy.package_ref.package_dir / "analysis/anatomy_spec.yaml")
    if mutation == "wrapper_version":
        document["schema_version"] = 3
    elif mutation == "wrapper_unknown":
        document["disable_replay"] = True
    elif mutation == "inner_unknown":
        document["spec"]["disable_replay"] = True
    elif mutation == "policy_digest":
        document["anatomy_spec_sha256"] = "a" * 64
    spec_path = tmp_path / "spec.yaml"
    text = yaml.safe_dump(document)
    if mutation == "duplicate":
        text += "schema_version: 2\n"
    spec_path.write_text(text)
    ref = intake["package_ref"]
    captured = subprocess.run([
        sys.executable, "-m", "core.stages.f2_anatomy.cli", "capture",
        "--intake-dir", str(ref.package_dir),
        "--intake-manifest-sha256", ref.manifest_sha256,
        "--prep-dir", str(tmp_path / "prep"), "--spec", str(spec_path),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert captured.returncode != 0
    assert not (tmp_path / "prep/work/evidence/attempt-1/capture.json").exists()


@pytest.mark.parametrize("backend", ["pytorch", "jax"])
def test_nonfake_backend_dispatch_stays_fail_closed(tmp_path, intake, anatomy, monkeypatch, backend):
    from core.artifacts import MANIFEST_NAME, publish
    from core.stages.f1_intake.source import snapshot_source
    from core.stages.f2_anatomy.adapters.openpi_jax import OpenPIJaxAdapter, OpenPIPreflightResult

    observed = []

    def preflight(self, config):
        assert backend == "jax", "non-JAX backend must not invoke OpenPI JAX preflight"
        # Remote source.repo is metadata: preflight must get a verified, locally
        # restored pinned tree, not Path("https://...") or an untrusted host path.
        assert config.upstream_repo.is_dir()
        snapshot_source(config.upstream_repo, intake["upstream_commit"])
        observed.append(config.upstream_repo)
        return OpenPIPreflightResult(
            status="ready", execution_environment="openpi_jax", reason_codes=(),
            missing_assets=(), summary="fixture-only preflight", runtime_available=True,
        )

    monkeypatch.setattr(OpenPIJaxAdapter, "preflight", preflight)
    stage = tmp_path / f"{backend}.staging"
    shutil.copytree(anatomy.package_ref.package_dir, stage)
    (stage / MANIFEST_NAME).unlink()
    path = stage / "analysis/behavior_contract.yaml"
    contract = load_yaml_document(path)
    contract["case"]["precision"]["backend"] = backend
    path.write_text(yaml.safe_dump(contract))
    forged = publish(stage)
    ref = intake["package_ref"]
    gate = AnatomyChecker().check(CheckContext(
        commit=intake["upstream_commit"], manifest_sha256=forged.manifest_sha256,
        package_dir=forged.package_dir, parameters={
            "intake_package_dir": str(ref.package_dir),
            "intake_manifest_sha256": ref.manifest_sha256,
            "spec_manifest_sha256": anatomy.spec_manifest_sha256,
        },
    ))
    assert gate.status == "blocked", gate
    assert gate.reason_codes == ("UNSUPPORTED_BACKEND_REPLAY_REQUIRED",)
    assert gate.metrics["actual"]["backend"] == backend
    if backend == "pytorch":
        assert not observed
        assert "preflight" not in gate.summary
        assert "jax" not in gate.summary.lower()
        assert "preflight_status" not in gate.metrics["actual"]
    else:
        assert len(observed) == 1
        assert gate.metrics["actual"]["preflight_status"] == "ready"
        assert not observed[0].exists(), "temporary restored source must be cleaned up"
