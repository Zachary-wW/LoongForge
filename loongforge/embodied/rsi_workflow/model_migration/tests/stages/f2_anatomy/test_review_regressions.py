"""Regression seams for the independent Batch B/C review."""
import copy
import json
import shutil
from dataclasses import replace

import pytest
import yaml

from core.stages.f2_anatomy.adapters.fake import FakeStepAdapter, FakeStepConfig
from core.stages.f2_anatomy.capture import capture_two_attempts
from core.stages.f2_anatomy.compare import compare_attempts
from core.stages.f2_anatomy.contract import ActionSpec, AnatomySpec
from core.contracts.anatomy_spec import ViewSpec
from core.artifacts import MANIFEST_NAME, publish
from core.contracts.common import load_yaml_document
from core.stages.f2_anatomy.capture import replay_sealed_anatomy
from core.stages.f2_anatomy.freeze import freeze_anatomy
from validate.anatomy_check import AnatomyChecker
from validate.base import CheckContext
from validate.intake_check import IntakeChecker
from anatomy_helpers import build_sealed_intake_package


def discrete_spec():
    return AnatomySpec(
        spec_id="review-ar", required_camera_roles=("front",), required_rng_consumers=(),
        action=ActionSpec(
            representation="discrete_tokens", head="autoregressive", vocabulary_size=64,
            label_shift=1, tokenizer_id="cpu-fixture-v1", codebook_id="cpu-fixture-codebook",
            codebook_sha256="c" * 64, bos_token_id=0, eos_token_id=1, pad_token_id=2,
        ),
    )


def test_both_attempts_with_unshifted_labels_must_fail():
    spec = discrete_spec()
    record = FakeStepAdapter().execute_step(FakeStepConfig(spec=spec))
    record["action_evidence"]["labels"] = list(record["action_evidence"]["tokens"])
    assert compare_attempts(record, copy.deepcopy(record), spec=spec).status == "fail"


@pytest.mark.parametrize("field,value", [
    ("shape", [1, 3, 1, 1]), ("dtype", "float16"), ("mask", False),
])
def test_both_attempts_with_invalid_view_spec_must_fail(field, value):
    spec = AnatomySpec(spec_id="review-mono", required_camera_roles=("front",))
    record = FakeStepAdapter().execute_step(FakeStepConfig(spec=spec))
    record["cameras"]["front"][field] = value
    assert compare_attempts(record, copy.deepcopy(record), spec=spec).status == "fail"


def test_validator_versions_are_explicitly_upgraded():
    assert IntakeChecker.validator_version == AnatomyChecker.validator_version == "2"


def test_anatomy_release_survives_relocation_and_deleted_preparation(tmp_path):
    intake = build_sealed_intake_package(tmp_path / "intake")
    ref = intake["package_ref"]
    prep = tmp_path / "prep"
    capture = capture_two_attempts(
        intake_package_dir=ref.package_dir, intake_manifest_sha256=ref.manifest_sha256, prep_dir=prep,
    )
    assert capture.status == "captured"
    # This digest is retained by the caller before moving the release, not read from the moved bundle.
    trusted_spec_sha = capture.spec_manifest_sha256
    frozen = freeze_anatomy(prep)
    assert frozen.status == "completed", frozen
    moved = tmp_path / "moved"
    shutil.copytree(frozen.package_ref.package_dir, moved)
    shutil.rmtree(prep)
    gate = AnatomyChecker().check(CheckContext(
        commit=intake["upstream_commit"], manifest_sha256=frozen.package_ref.manifest_sha256, package_dir=moved,
        parameters={"intake_package_dir": str(ref.package_dir), "intake_manifest_sha256": ref.manifest_sha256,
                    "spec_manifest_sha256": trusted_spec_sha},
    ))
    assert gate.status == "pass", gate
    assert replay_sealed_anatomy(
        package_dir=moved, manifest_sha256=frozen.package_ref.manifest_sha256,
        spec_manifest_sha256=trusted_spec_sha,
    ).status == "pass"


@pytest.mark.parametrize("mutation", [
    "shared_permutation", "shared_permutation_relabelled", "wrong_layout_coordinates",
    "duplicate_eos", "missing_bos", "pad_in_payload", "cross_sample_shift",
])
def test_each_discrete_attempt_rejects_joint_layout_and_special_token_errors(mutation):
    spec = discrete_spec()
    action = spec.action
    record = FakeStepAdapter().execute_step(FakeStepConfig(spec=spec))
    evidence = record["action_evidence"]
    if mutation.startswith("shared_permutation"):
        evidence["tokens"][1], evidence["tokens"][2] = evidence["tokens"][2], evidence["tokens"][1]
        if mutation == "shared_permutation_relabelled":
            evidence["labels"] = action.shifted_labels(evidence["tokens"])
        else:
            evidence["labels"][1], evidence["labels"][2] = evidence["labels"][2], evidence["labels"][1]
    elif mutation == "wrong_layout_coordinates":
        evidence["token_coordinates"][1], evidence["token_coordinates"][2] = (
            evidence["token_coordinates"][2], evidence["token_coordinates"][1],
        )
    elif mutation == "cross_sample_shift":
        evidence["labels"] = evidence["tokens"][1:] + [action.pad_token_id]
    else:
        position = 0 if mutation == "missing_bos" else 1
        value = 3 if mutation == "missing_bos" else action.eos_token_id if mutation == "duplicate_eos" else action.pad_token_id
        evidence["tokens"][position] = evidence["canonical_tokens"][position] = value
        evidence["labels"] = action.shifted_labels(evidence["tokens"])
    result = compare_attempts(record, copy.deepcopy(record), spec=spec)
    assert result.status == "fail", result
    assert result.reason_codes == ("ACTION_EVIDENCE_INVALID",)


@pytest.mark.parametrize("layout", [
    ("batch", "horizon", "dimension"), ("batch", "dimension", "horizon"),
])
@pytest.mark.parametrize("shift", [1, 2])
def test_declared_token_layout_shift_and_padding_masks_are_validated(layout, shift):
    spec = replace(discrete_spec(), action=replace(discrete_spec().action, token_layout=layout, label_shift=shift))
    record = FakeStepAdapter().execute_step(FakeStepConfig(spec=spec))
    assert compare_attempts(record, copy.deepcopy(record), spec=spec).status == "pass"
    assert record["action_evidence"]["labels"][spec.action.horizon * spec.action.dimension - 1] == spec.action.pad_token_id


VIEW_SPECS = [
    AnatomySpec(spec_id="mono-uint8", required_camera_roles=("front",), views=(
        ViewSpec("front", shape=(3, 16, 20), dtype="uint8"),
    )),
    AnatomySpec(spec_id="quad-mixed", required_camera_roles=("front", "left", "right", "rear"), views=(
        ViewSpec("front", shape=(3, 16, 20), dtype="float16"),
        ViewSpec("left", shape=(3, 12, 16), dtype="float32"),
        ViewSpec("right", shape=(1, 8, 12), dtype="uint8"),
        ViewSpec("rear", shape=(3, 20, 12), dtype="bfloat16"),
    )),
    *[AnatomySpec(
        spec_id=f"optional-wrist-{enabled}", required_camera_roles=("front",),
        optional_camera_roles=("wrist",), required_rng_consumers=(), views=(
            ViewSpec("front", shape=(3, 16, 20), dtype="uint8"),
            ViewSpec("wrist", required=False, enabled=enabled, shape=(3, 8, 12), dtype="float16"),
        ),
    ) for enabled in (False, True)],
    AnatomySpec(
        spec_id="disabled-optional-only", required_camera_roles=(), optional_camera_roles=("wrist",),
        required_rng_consumers=(), views=(ViewSpec("wrist", required=False, enabled=False),),
    ),
]


@pytest.fixture(scope="module")
def review_intake(tmp_path_factory):
    return build_sealed_intake_package(tmp_path_factory.mktemp("review-intake"))


@pytest.mark.parametrize("spec", VIEW_SPECS, ids=lambda spec: spec.spec_id)
def test_mixed_view_shapes_optional_switches_and_spec_derived_boundaries(tmp_path, review_intake, spec):
    ref = review_intake["package_ref"]
    capture = capture_two_attempts(
        intake_package_dir=ref.package_dir, intake_manifest_sha256=ref.manifest_sha256,
        prep_dir=tmp_path, spec=spec,
    )
    assert capture.status == "captured", capture
    frozen = freeze_anatomy(tmp_path)
    assert frozen.status == "completed", frozen
    record = json.loads((frozen.package_ref.package_dir / "evidence/attempt-1/capture.json").read_text())
    assert set(record["cameras"]) == {view.role for view in spec.views if view.enabled}
    for view in spec.views:
        if view.enabled:
            assert record["cameras"][view.role]["shape"] == [len(record["sample_ids"]), *view.shape]
            assert record["cameras"][view.role]["dtype"] == view.dtype
    plan = load_yaml_document(frozen.package_ref.package_dir / "analysis/observation_plan.yaml")
    boundaries = {item["name"] for item in plan["boundaries"]}
    assert ("camera_role_content_map" in boundaries) == bool(record["cameras"])
    assert ("rng_consumer_events" in boundaries) == bool(spec.required_rng_consumers)
    assert "model-forward-count" in {item["id"] for item in load_yaml_document(
        frozen.package_ref.package_dir / "analysis/behavior_contract.yaml",
    )["assertions"]}


@pytest.mark.parametrize("mutation", ["disabled_present", "disabled_null", "enabled_missing", "bad_placeholder", "bad_digest"])
def test_optional_and_placeholder_rules_cannot_be_bypassed_by_matching_attempts(mutation):
    spec = VIEW_SPECS[3] if mutation == "enabled_missing" else VIEW_SPECS[2]
    record = FakeStepAdapter().execute_step(FakeStepConfig(spec=spec))
    if mutation == "disabled_present":
        record["cameras"]["wrist"] = copy.deepcopy(record["cameras"]["front"])
    elif mutation == "disabled_null":
        record["cameras"]["wrist"] = None
    elif mutation == "enabled_missing":
        del record["cameras"]["wrist"]
    elif mutation == "bad_placeholder":
        record["cameras"]["front"]["placeholder"] = "zeros"
    else:
        record["cameras"]["front"]["sha256"] = "a" * 64
    assert compare_attempts(record, copy.deepcopy(record), spec=spec).status == "fail"


@pytest.fixture(scope="module")
def review_anatomy(tmp_path_factory, review_intake):
    prep = tmp_path_factory.mktemp("review-anatomy")
    ref = review_intake["package_ref"]
    capture = capture_two_attempts(
        intake_package_dir=ref.package_dir, intake_manifest_sha256=ref.manifest_sha256, prep_dir=prep,
    )
    assert capture.status == "captured"
    frozen = freeze_anatomy(prep)
    assert frozen.status == "completed", frozen
    return frozen


@pytest.mark.parametrize("mutation", ["missing_trust", "wrong_trust", "tampered_inner", "resealed_inner", "absolute_lock"])
def test_spec_release_requires_external_trust_even_if_outer_bundle_is_resealed(tmp_path, review_intake, review_anatomy, mutation):
    ref = review_intake["package_ref"]
    stage = tmp_path / "forged.staging"
    shutil.copytree(review_anatomy.package_ref.package_dir, stage)
    trusted_sha = review_anatomy.spec_manifest_sha256
    if mutation in {"tampered_inner", "resealed_inner"}:
        inner = stage / "evidence/spec_freeze"
        doc = load_yaml_document(inner / "anatomy_spec.yaml")
        doc["spec"]["spec_id"] = "forged-policy"
        (inner / "anatomy_spec.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
        if mutation == "resealed_inner":
            (inner / MANIFEST_NAME).unlink()
            inner.rename(inner.with_name("spec_freeze.staging"))
            forged = publish(inner.with_name("spec_freeze.staging"))
            lock = json.loads((stage / "analysis/spec_lock.json").read_text())
            lock["manifest_sha256"] = forged.manifest_sha256
            (stage / "analysis/spec_lock.json").write_text(json.dumps(lock))
    elif mutation == "absolute_lock":
        lock = json.loads((stage / "analysis/spec_lock.json").read_text())
        lock["package_dir"] = str(review_anatomy.package_ref.package_dir / "evidence/spec_freeze")
        (stage / "analysis/spec_lock.json").write_text(json.dumps(lock))
    elif mutation == "wrong_trust":
        trusted_sha = "a" * 64
    (stage / MANIFEST_NAME).unlink()
    forged_outer = publish(stage)
    params = {"intake_package_dir": str(ref.package_dir), "intake_manifest_sha256": ref.manifest_sha256}
    if mutation != "missing_trust":
        params["spec_manifest_sha256"] = trusted_sha
    gate = AnatomyChecker().check(CheckContext(
        commit=review_intake["upstream_commit"], manifest_sha256=forged_outer.manifest_sha256,
        package_dir=forged_outer.package_dir, parameters=params,
    ))
    assert gate.status == "blocked", gate
    assert gate.reason_codes == (("SPEC_TRUST_REQUIRED",) if mutation == "missing_trust" else ("ANATOMY_SPEC_INVALID",))


@pytest.mark.parametrize("mutation", ["old_version", "legacy_identity"])
def test_anatomy_v2_validator_binding_rejects_old_version_and_identity(tmp_path, review_intake, review_anatomy, mutation):
    stage = tmp_path / "old-receipt.staging"
    shutil.copytree(review_anatomy.package_ref.package_dir, stage)
    path = stage / "analysis/behavior_contract.yaml"
    contract = load_yaml_document(path)
    contract["validator"]["version"] = "1"
    if mutation == "legacy_identity":
        contract["validator"]["id"] = "stages.verify.anatomy_check"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    (stage / MANIFEST_NAME).unlink()
    forged = publish(stage)
    ref = review_intake["package_ref"]
    gate = AnatomyChecker().check(CheckContext(
        commit=review_intake["upstream_commit"], manifest_sha256=forged.manifest_sha256,
        package_dir=forged.package_dir, parameters={
            "intake_package_dir": str(ref.package_dir), "intake_manifest_sha256": ref.manifest_sha256,
            "spec_manifest_sha256": review_anatomy.spec_manifest_sha256,
        },
    ))
    assert gate.status == "blocked", gate
    assert gate.reason_codes == ((
        "LEGACY_VALIDATOR_RUNTIME_REQUIRED" if mutation == "legacy_identity" else "VALIDATOR_SHA_MISMATCH"
    ),)
    # Mutated identity is not a real historical v1 sealed-package compatibility fixture.


@pytest.mark.parametrize("attempt", [1, 2])
def test_external_spec_digest_is_required_on_each_attempt(tmp_path, review_intake, review_anatomy, attempt):
    stage = tmp_path / "wrong-attempt.staging"
    shutil.copytree(review_anatomy.package_ref.package_dir, stage)
    path = stage / f"evidence/attempt-{attempt}/capture.json"
    record = json.loads(path.read_text())
    record["spec_manifest_sha256"] = "a" * 64
    path.write_text(json.dumps(record))
    (stage / MANIFEST_NAME).unlink()
    forged = publish(stage)
    ref = review_intake["package_ref"]
    gate = AnatomyChecker().check(CheckContext(
        commit=review_intake["upstream_commit"], manifest_sha256=forged.manifest_sha256,
        package_dir=forged.package_dir, parameters={
            "intake_package_dir": str(ref.package_dir), "intake_manifest_sha256": ref.manifest_sha256,
            "spec_manifest_sha256": review_anatomy.spec_manifest_sha256,
        },
    ))
    assert gate.status == "blocked", gate
    assert gate.reason_codes == ("ATTEMPT_CASE_MISMATCH",)
