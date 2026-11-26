import copy
import dataclasses
from pathlib import Path

import pytest
import yaml

from core.contracts import load_anatomy_behavior_contract
from core.stages.f2_anatomy.adapters.fake import FakeStepAdapter, FakeStepConfig
from core.stages.f2_anatomy.capture import capture_two_attempts, replay_sealed_anatomy
from core.stages.f2_anatomy.compare import compare_attempts
from core.stages.f2_anatomy.contract import ActionSpec, AnatomySpec
from core.stages.f2_anatomy.freeze import freeze_anatomy
from validate.anatomy_check import AnatomyChecker
from validate.base import CheckContext

from anatomy_helpers import build_sealed_intake_package

SPECS = [
    AnatomySpec(spec_id="mono-flow", required_camera_roles=("front",)),
    AnatomySpec(spec_id="stereo-flow", required_camera_roles=("front", "wrist")),
    AnatomySpec(spec_id="quad-flow", required_camera_roles=("front", "left", "right", "rear")),
    AnatomySpec(
        spec_id="mono-ar", required_camera_roles=("front",), required_rng_consumers=(),
        action=ActionSpec(
            representation="discrete_tokens", head="autoregressive", vocabulary_size=128,
            label_shift=1, tokenizer_id="cpu-fixture-v1", codebook_id="cpu-fixture-codebook",
            codebook_sha256="c" * 64, bos_token_id=0, eos_token_id=1, pad_token_id=2,
        ),
    ),
    AnatomySpec(spec_id="no-rng", required_camera_roles=("front",), required_rng_consumers=()),
    AnatomySpec(spec_id="optional-wrist", required_camera_roles=("front",), optional_camera_roles=("wrist",)),
    AnatomySpec(spec_id="no-vision", required_camera_roles=(), required_rng_consumers=()),
]


@pytest.fixture(scope="module")
def intake(tmp_path_factory):
    return build_sealed_intake_package(tmp_path_factory.mktemp("spec-intake"))


@pytest.mark.parametrize("spec", SPECS, ids=lambda x: x.spec_id)
def test_custom_spec_capture_freeze_and_independent_dual_replay(tmp_path: Path, intake, spec) -> None:
    ref = intake["package_ref"]
    cap = capture_two_attempts(
        intake_package_dir=ref.package_dir, intake_manifest_sha256=ref.manifest_sha256,
        prep_dir=tmp_path, spec=spec,
    )
    assert cap.status == "captured", cap
    frozen = freeze_anatomy(tmp_path)
    assert frozen.status == "completed", frozen
    package = frozen.package_ref
    contract = load_anatomy_behavior_contract(package.package_dir / "analysis/behavior_contract.yaml")
    assert contract.case.anatomy_spec_sha256 == spec.sha256()
    assert contract.validator.sha256 == AnatomyChecker().validator_sha256()
    gate = AnatomyChecker().check(CheckContext(
        commit=intake["upstream_commit"], manifest_sha256=package.manifest_sha256,
        package_dir=package.package_dir, parameters={
            "intake_package_dir": str(ref.package_dir), "intake_manifest_sha256": ref.manifest_sha256,
            "spec_manifest_sha256": cap.spec_manifest_sha256,
        },
    ))
    assert gate.status == "pass", gate
    assert len(set(gate.metrics["replay_pids"])) == 2
    assert replay_sealed_anatomy(package_dir=package.package_dir, manifest_sha256=package.manifest_sha256, spec_manifest_sha256=cap.spec_manifest_sha256).status == "pass"


@pytest.mark.parametrize("field", ["tokens", "labels", "loss_mask", "attention_mask", "padding_mask", "shape", "semantics"])
def test_discrete_action_evidence_tampering_rejected_even_on_both_attempts(field: str) -> None:
    spec = SPECS[3]
    base = FakeStepAdapter().execute_step(FakeStepConfig(spec=spec))
    altered = copy.deepcopy(base)
    evidence = altered["action_evidence"]
    if field in ("tokens", "labels"):
        evidence[field][0] = spec.action.vocabulary_size
    elif field in ("loss_mask", "attention_mask", "padding_mask"):
        evidence[field] = []
    elif field == "shape":
        evidence[field][1] += 1
    else:
        evidence[field]["units"] = "meters"
    assert compare_attempts(base, altered, spec=spec).status == "fail"
    assert compare_attempts(altered, altered, spec=spec).status == "fail"


@pytest.mark.parametrize("key,value", [
    ("coordinate_mode", "absolute"), ("normalization", "zscore"),
    ("units", "meters"), ("horizon", 99), ("dimension", 99),
])
def test_continuous_semantic_boundary_tampering_fails(key, value) -> None:
    base = FakeStepAdapter().execute_step(FakeStepConfig(spec=SPECS[0]))
    altered = copy.deepcopy(base)
    altered["action_evidence"]["semantics"][key] = value
    assert compare_attempts(base, altered, spec=SPECS[0]).status == "fail"


@pytest.mark.parametrize("injection", ["swap_action_tokens", "swap_action_masks", "swap_cameras"])
def test_new_negative_injections_are_compared(injection) -> None:
    spec = SPECS[3]
    base = FakeStepAdapter().execute_step(FakeStepConfig(spec=spec))
    altered = FakeStepAdapter().execute_step(FakeStepConfig(spec=spec, inject=injection))
    assert compare_attempts(base, altered, spec=spec).status == "fail"


def test_spec_identity_cannot_drift_or_be_inferred_from_attempts() -> None:
    rec = FakeStepAdapter().execute_step(FakeStepConfig(spec=SPECS[0]))
    assert compare_attempts(rec, rec, spec=SPECS[2]).status == "fail"
    assert compare_attempts(rec, rec).status == "fail"
    with pytest.raises(dataclasses.FrozenInstanceError):
        SPECS[0].spec_id = "mutated"
    with pytest.raises(ValueError):
        AnatomySpec.from_dict({**SPECS[0].to_dict(), "disable_forward_gate": True})


@pytest.mark.parametrize("mutation", ["missing", "tampered", "wrong_commit", "wrong_intake"])
def test_spec_missing_or_changed_after_capture_blocks_freeze(tmp_path, intake, mutation) -> None:
    ref = intake["package_ref"]
    cap = capture_two_attempts(
        intake_package_dir=ref.package_dir, intake_manifest_sha256=ref.manifest_sha256,
        prep_dir=tmp_path, spec=SPECS[0],
    )
    assert cap.status == "captured"
    path = tmp_path / "work/analysis/anatomy_spec.yaml"
    if mutation == "missing":
        path.unlink()
    else:
        doc = yaml.safe_load(path.read_text())
        if mutation == "tampered":
            doc["spec"]["required_camera_roles"] = []
        else:
            doc["upstream_commit" if mutation == "wrong_commit" else "intake_manifest_sha256"] = "a" * (
                40 if mutation == "wrong_commit" else 64
            )
        path.write_text(yaml.safe_dump(doc))
    assert freeze_anatomy(tmp_path).status != "completed"


def test_empty_camera_rng_policy_is_explicit_and_cannot_hide_missing_evidence() -> None:
    rec = FakeStepAdapter().execute_step(FakeStepConfig())
    rec["cameras"] = {}
    rec["rng_events"] = []
    assert compare_attempts(rec, rec).status == "fail"
    spec = SPECS[-1]
    explicit = FakeStepAdapter().execute_step(FakeStepConfig(spec=spec))
    assert compare_attempts(explicit, explicit, spec=spec).status == "pass"
    explicit["rng_events"] = [{"consumer": "undeclared"}]
    assert compare_attempts(explicit, explicit, spec=spec).status == "fail"


def test_repeated_capture_cannot_change_the_pre_capture_spec(tmp_path, intake) -> None:
    ref = intake["package_ref"]
    cap = capture_two_attempts(
        intake_package_dir=ref.package_dir, intake_manifest_sha256=ref.manifest_sha256,
        prep_dir=tmp_path, spec=SPECS[0],
    )
    assert cap.status == "captured"
    changed = capture_two_attempts(
        intake_package_dir=ref.package_dir, intake_manifest_sha256=ref.manifest_sha256,
        prep_dir=tmp_path, spec=SPECS[2],
    )
    assert changed.status == "blocked"
    assert "ANATOMY_SPEC_INVALID" in changed.reason_codes


@pytest.mark.parametrize("key", ["loss", "grad_global_norm"])
def test_numerical_endpoints_cannot_be_omitted_or_nonfinite(key) -> None:
    rec = FakeStepAdapter().execute_step(FakeStepConfig())
    del rec[key]
    assert compare_attempts(rec, rec).status == "fail"
    rec[key] = float("nan")
    assert compare_attempts(rec, rec).status == "fail"
