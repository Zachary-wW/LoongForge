import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from core.artifacts import MANIFEST_NAME, publish, verify
from core.contracts import ContractValidationError, IntakeTaskPackV2, ModelAnalysisReport, load_intake_task_pack
from core.stages.f1_intake.freeze import analyze_intake, freeze_intake, init_preparation
from core.stages.f1_intake.handoff import canonical_sha256
from core.stages.f1_intake.source import create_source_bundle
from core.stages.f2_anatomy.capture import capture_two_attempts
from core.stages.f2_anatomy.cli import main as anatomy_cli
from core.stages.f2_anatomy.contract import ActionSpec, AnatomySpec
from core.stages.f2_anatomy.freeze import freeze_anatomy
from core.stages.f3_scaffold import ScaffoldPlan
from validate.base import CheckContext
from validate.intake_check import IntakeChecker

from helpers import create_sample_repo, git, write_task_yaml

ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def v2_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2-inputs")
    info = create_sample_repo(root)
    repo = info["repo"]
    params = repo / "scripts/hyperparameters.py"
    params.write_text("BATCH_SIZE = 2\nEFFECTIVE_BATCH_SIZE = 4\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "training defaults")
    commit = git(repo, "rev-parse", "HEAD").stdout.strip()
    path = write_task_yaml(root / "task.yaml", repo=repo, commit=commit, startup_probe=False)
    task = yaml.safe_load(path.read_text())
    task["schema_version"] = 2
    task["scope"]["train_step"] = True

    skill_repo = root / "skill-repo"
    skill_repo.mkdir()
    git(skill_repo, "init", "-b", "main")
    playbook = b"# Model-adaptation fixture\nOffline read-only analysis playbook.\n"
    (skill_repo / "SKILL.md").write_bytes(playbook)
    git(skill_repo, "add", ".")
    git(skill_repo, "commit", "-m", "playbook")
    skill_commit = git(skill_repo, "rev-parse", "HEAD").stdout.strip()
    handoff = root / "handoff"
    bundle = create_source_bundle(skill_repo, skill_commit, handoff / "evidence/skill.bundle")

    def code_location(path, symbol, start, end):
        return {
            "kind": "code", "commit": commit, "path": path,
            "blob_sha256": hashlib.sha256((repo / path).read_bytes()).hexdigest(),
            "symbol": symbol, "line_start": start, "line_end": end,
        }

    tree = ast.parse((repo / "scripts/train.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    main_source = code_location("scripts/train.py", "main", main.lineno, main.end_lineno)
    code_default = code_location("scripts/hyperparameters.py", "BATCH_SIZE", 1, 1)
    effective_source = code_location("scripts/hyperparameters.py", "EFFECTIVE_BATCH_SIZE", 2, 2)
    paper = handoff / "evidence/papers/paper.txt"
    paper.parent.mkdir(parents=True)
    paper.write_text("Paper-1\nversion-v1\nsection: 3.2\npage: 7\nbatch_size default = 2.\n")
    paper_source = {
        "kind": "paper", "paper_id": "Paper-1", "version": "version-v1", "section": "section: 3.2",
        "page": 7, "path": "evidence/papers/paper.txt", "sha256": hashlib.sha256(paper.read_bytes()).hexdigest(),
        "excerpt": "batch_size default = 2.", "value_literal": "2",
    }

    profile = yaml.safe_load((ROOT / "core/profiles/loongforge_embodied.yaml").read_text())
    sides = {
        "model_registry": "model", "data_and_transforms": "data", "config_map": "train",
        "trainer_builder": "train", "eval_adapter": "train", "regression_runner": "train",
    }
    extensions = []
    for name, description in profile["extension_points"].items():
        registry = description.get("registry_path") or description.get("factory_registry_path") or description["entry_script"]
        extensions.append({
            "extension_point": name, "side": sides[name], "registry_path": registry,
            "target_files": [f"loongforge/embodied/model/fixture/{name}.py"], "reuse_components": [],
        })
    catalog = {
        "schema_version": 1, "compatibility_profile": "internal", "version": "fixture-v1",
        "target_commit": task["target"]["commit"], "extensions": extensions,
    }
    spec = AnatomySpec(
        spec_id="skill-mono-ar", required_camera_roles=("front",), required_rng_consumers=(),
        action=ActionSpec(
            representation="discrete_tokens", head="autoregressive", vocabulary_size=64,
            label_shift=1, tokenizer_id="cpu-fixture-v1", codebook_id="cpu-fixture-codebook",
            codebook_sha256="c" * 64, bos_token_id=0, eos_token_id=1, pad_token_id=2,
        ),
    )
    report = {
        "schema_version": 2, "request_id": "handoff-1", "status": "completed",
        "source_commit": commit, "target_commit": task["target"]["commit"],
        "slots": [{
            "slot_id": f"custom:{x['extension_point']}", "extension_point": x["extension_point"],
            "side": x["side"], "status": "applicable", "sources": [main_source],
            "target_files": x["target_files"], "reason": None,
        } for x in extensions],
        "training_parameters": [{
            "name": "batch_size", "default_value": 2, "effective_value": 4,
            "sources": [code_default, paper_source], "selected_source": 0,
            "resolution": None, "effective_source": effective_source,
        }],
        "anatomy_spec": spec.to_dict(),
        "anatomy_spec_source": {"source_ref": main_source, "analysis_basis": "Pinned training entrypoint; explicitly approved CPU format fixture, not real VLA behavior."},
    }
    report_path = handoff / "model_analysis.json"
    report_path.write_text(json.dumps(report, sort_keys=True))
    task["slot_catalog"] = catalog
    task["analysis_handoff"] = {
        "schema_version": 2, "request_id": "handoff-1", "boundary": "handoff_only",
        "source_commit": commit, "target_commit": task["target"]["commit"],
        "skill": {
            "name": "model-adaptation", "repo": str(skill_repo), "commit": skill_commit,
            "playbook_path": "SKILL.md", "playbook_sha256": hashlib.sha256(playbook).hexdigest(),
            "bundle_sha256": bundle.sha256,
        },
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "catalog_sha256": canonical_sha256(catalog), "read_paths": ["scripts"], "write_paths": ["analysis"],
    }
    return {"task": task, "report": report, "handoff": handoff, "spec": spec}


def prepare(root: Path, inputs, mutate=None):
    handoff = root / "handoff"
    shutil.copytree(inputs["handoff"], handoff)
    task = copy.deepcopy(inputs["task"])
    report = copy.deepcopy(inputs["report"])
    if mutate is not None:
        mutate(task, report, handoff)
    report_path = handoff / "model_analysis.json"
    report_path.write_text(json.dumps(report, sort_keys=True))
    task["analysis_handoff"]["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    task_path = root / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    prep = root / "prep"
    init_preparation(task_path, prep)
    return prep, analyze_intake(prep, handoff_dir=handoff), task_path


@pytest.mark.parametrize("valid_handoff", [True, False])
def test_reanalyze_same_preparation_drops_previous_handoff_evidence(tmp_path, v2_inputs, valid_handoff):
    prep, state, _ = prepare(tmp_path / "first", v2_inputs)
    assert not state["blocked_reasons"], state
    old_evidence = prep / "work/evidence/papers/paper.txt"
    assert old_evidence.is_file()

    handoff = tmp_path / "replacement-handoff"
    shutil.copytree(v2_inputs["handoff"], handoff)
    report = copy.deepcopy(v2_inputs["report"])
    paper_source = report["training_parameters"][0]["sources"][1]
    paper_source["path"] = "evidence/papers/replacement.txt"
    paper = handoff / paper_source["path"]
    (handoff / "evidence/papers/paper.txt").rename(paper)
    if not valid_handoff:
        paper.write_text("invalid replacement evidence")
    raw_report = json.dumps(report, sort_keys=True).encode()
    (handoff / "model_analysis.json").write_bytes(raw_report)
    task = copy.deepcopy(v2_inputs["task"])
    task["analysis_handoff"]["report_sha256"] = hashlib.sha256(raw_report).hexdigest()
    task_path = tmp_path / "replacement-task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    init_preparation(task_path, prep)
    state = analyze_intake(prep, handoff_dir=handoff)
    assert not old_evidence.exists()
    result = freeze_intake(prep)
    if valid_handoff:
        assert not state["blocked_reasons"], state
        assert result.status == "completed", result
        paths = verify(result.package_ref).manifest_paths
        assert "evidence/papers/replacement.txt" in paths
        assert "evidence/papers/paper.txt" not in paths
    else:
        assert "V2_HANDOFF_INVALID" in state["blocked_reasons"]
        assert result.status == "blocked"
        assert not list((prep / "work/evidence").rglob("*"))


@pytest.fixture(scope="module")
def sealed_v2(tmp_path_factory, v2_inputs):
    prep, state, _ = prepare(tmp_path_factory.mktemp("v2-sealed"), v2_inputs)
    assert not state["blocked_reasons"], state
    result = freeze_intake(prep)
    assert result.status == "completed", result
    return result.package_ref


def test_v2_import_receipt_scaffold_mapping_and_frozen_spec_chain(tmp_path, v2_inputs, sealed_v2) -> None:
    task = load_intake_task_pack(sealed_v2.package_dir / "inputs/task.yaml")
    assert isinstance(task, IntakeTaskPackV2)
    gate = IntakeChecker().check(CheckContext(
        commit=task.source.commit, manifest_sha256=sealed_v2.manifest_sha256, package_dir=sealed_v2.package_dir,
    ))
    assert gate.status == "pass", gate
    receipt = json.loads((sealed_v2.package_dir / "analysis/skill_handoff_receipt.json").read_text())
    assert receipt["boundary"] == "handoff_only"
    assert receipt["harness_executed"] is False
    report = ModelAnalysisReport.model_validate(v2_inputs["report"])
    assert isinstance(report.anatomy_spec, AnatomySpec)
    assert receipt["anatomy_spec_sha256"] == report.anatomy_spec.sha256()
    assert receipt["anatomy_spec_source_sha256"] == canonical_sha256(
        report.anatomy_spec_source.model_dump(mode="json"),
    )
    policy = yaml.safe_load((sealed_v2.package_dir / "analysis/anatomy_spec.yaml").read_text())
    assert policy["source_ref"] == report.anatomy_spec_source.model_dump(mode="json")
    verified = verify(sealed_v2)
    assert "analysis/training_parameters.yaml" in verified.manifest_paths

    cap = capture_two_attempts(
        intake_package_dir=sealed_v2.package_dir, intake_manifest_sha256=sealed_v2.manifest_sha256,
        prep_dir=tmp_path,
    )
    assert cap.status == "captured", cap
    record = json.loads((tmp_path / "work/evidence/attempt-1/capture.json").read_text())
    assert record["anatomy_spec_sha256"] == v2_inputs["spec"].sha256()
    anatomy = freeze_anatomy(tmp_path)
    assert anatomy.status == "completed", anatomy
    plan = ScaffoldPlan.from_intake(
        intake_package=sealed_v2.package_dir, intake_manifest_sha256=sealed_v2.manifest_sha256,
        anatomy_manifest_sha256=anatomy.package_ref.manifest_sha256, source_commit=task.source.commit,
        anatomy_package_dir=anatomy.package_ref.package_dir, spec_manifest_sha256=cap.spec_manifest_sha256,
    )
    assert plan.anatomy_reference_status == "verified"
    assert {x.side for x in plan.bindings} == {"data", "model", "train"}
    assert all(x.slot_ids[0].startswith("custom:") for x in plan.bindings)
    assert plan.metadata["compatibility_profile"] == "internal"
    mismatch = capture_two_attempts(
        intake_package_dir=sealed_v2.package_dir, intake_manifest_sha256=sealed_v2.manifest_sha256,
        prep_dir=tmp_path / "mismatched", spec=AnatomySpec(),
    )
    assert mismatch.status == "blocked"


def test_capture_cli_consumes_the_actual_v2_intake_spec_wrapper(tmp_path, sealed_v2, capsys):
    assert anatomy_cli([
        "capture", "--intake-dir", str(sealed_v2.package_dir),
        "--intake-manifest-sha256", sealed_v2.manifest_sha256,
        "--prep-dir", str(tmp_path),
        "--spec", str(sealed_v2.package_dir / "analysis/anatomy_spec.yaml"),
    ]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "captured", receipt
    assert receipt["spec_manifest_sha256"]


@pytest.mark.parametrize("mutation", [
    "partial", "wrong_commit", "forged_symbol", "forged_lines", "blob_sha", "outside_permission",
    "wrong_slot_side", "wrong_slot_target", "parameter_default", "parameter_effective",
    "paper_section", "paper_digest", "paper_conflict", "skill_sha", "catalog_drift",
])
def test_v2_independent_negative_evidence_cases(tmp_path, v2_inputs, mutation) -> None:
    def mutate(task, report, handoff):
        if mutation == "partial":
            report["status"] = "partial"
        elif mutation == "wrong_commit":
            report["source_commit"] = "a" * 40
        elif mutation in {"forged_symbol", "forged_lines", "blob_sha"}:
            source = report["slots"][0]["sources"][0]
            source[{"forged_symbol": "symbol", "forged_lines": "line_start", "blob_sha": "blob_sha256"}[mutation]] = (
                "fake_symbol" if mutation == "forged_symbol" else 1 if mutation == "forged_lines" else "a" * 64
            )
        elif mutation == "outside_permission":
            task["analysis_handoff"]["read_paths"] = ["configs"]
        elif mutation == "wrong_slot_side":
            report["slots"][0]["side"] = "data"
        elif mutation == "wrong_slot_target":
            report["slots"][0]["target_files"] = ["wrong/target.py"]
        elif mutation == "parameter_default":
            report["training_parameters"][0]["default_value"] = 99
        elif mutation == "parameter_effective":
            report["training_parameters"][0]["effective_value"] = 99
        elif mutation in {"paper_section", "paper_digest"}:
            paper = report["training_parameters"][0]["sources"][1]
            paper["section" if mutation == "paper_section" else "sha256"] = "missing-section" if mutation == "paper_section" else "a" * 64
        elif mutation == "paper_conflict":
            path = handoff / "evidence/papers/paper.txt"
            text = path.read_text().replace("default = 2.", "default = 3.")
            path.write_text(text)
            paper = report["training_parameters"][0]["sources"][1]
            paper.update(excerpt="batch_size default = 3.", value_literal="3", sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        elif mutation == "skill_sha":
            task["analysis_handoff"]["skill"]["playbook_sha256"] = "a" * 64
        else:
            task["slot_catalog"]["version"] = "unapproved-v2"
    prep, state, _ = prepare(tmp_path, v2_inputs, mutate)
    assert "V2_HANDOFF_INVALID" in state["blocked_reasons"]
    assert freeze_intake(prep).status == "blocked"


@pytest.mark.parametrize("filename", [
    "slot_hits.json", "training_parameters.yaml", "slot_catalog.yaml",
    "skill_handoff_receipt.json", "anatomy_spec.yaml",
])
def test_checker_rejects_resealed_modified_v2_artifacts(tmp_path, sealed_v2, filename) -> None:
    staging = tmp_path / "forged.staging"
    shutil.copytree(sealed_v2.package_dir, staging)
    (staging / MANIFEST_NAME).unlink()
    path = staging / "analysis" / filename
    path.write_text("{}\n")
    forged = publish(staging)
    task = load_intake_task_pack(forged.package_dir / "inputs/task.yaml")
    gate = IntakeChecker().check(CheckContext(
        commit=task.source.commit, manifest_sha256=forged.manifest_sha256, package_dir=forged.package_dir,
    ))
    assert gate.status != "pass"
    assert "V2_HANDOFF_INVALID" in gate.reason_codes


def test_v2_missing_handoff_is_fail_closed_and_v1_loader_is_unchanged(tmp_path, v2_inputs) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(v2_inputs["task"]))
    prep = tmp_path / "prep"
    init_preparation(path, prep)
    state = analyze_intake(prep)
    assert "V2_HANDOFF_INVALID" in state["blocked_reasons"]
    bad = copy.deepcopy(v2_inputs["task"])
    bad["schema_version"] = 1
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ContractValidationError):
        load_intake_task_pack(path)
    del bad["analysis_handoff"], bad["slot_catalog"]
    path.write_text(yaml.safe_dump(bad))
    assert load_intake_task_pack(path).schema_version == 1


@pytest.mark.parametrize("path_field", ["read_paths", "write_paths"])
def test_v2_permissions_cannot_escape_or_claim_harness_execution(tmp_path, v2_inputs, path_field) -> None:
    task = copy.deepcopy(v2_inputs["task"])
    task["analysis_handoff"][path_field] = ["../escape"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(task))
    with pytest.raises(ContractValidationError):
        load_intake_task_pack(path)


def test_v2_rejects_python_skill_api_boundary_and_missing_not_applicable_reason(tmp_path, v2_inputs) -> None:
    task = copy.deepcopy(v2_inputs["task"])
    task["analysis_handoff"]["boundary"] = "python_api"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(task))
    with pytest.raises(ContractValidationError):
        load_intake_task_pack(path)

    def mutate(task, report, handoff):
        report["slots"][-1].update(status="not_applicable", sources=[], target_files=[], reason=None)
    prep, state, _ = prepare(tmp_path, v2_inputs, mutate)
    assert "V2_HANDOFF_INVALID" in state["blocked_reasons"]
    assert freeze_intake(prep).status == "blocked"


def test_missing_v2_artifact_is_not_ignored_by_checker(tmp_path, sealed_v2) -> None:
    staging = tmp_path / "missing.staging"
    shutil.copytree(sealed_v2.package_dir, staging)
    (staging / MANIFEST_NAME).unlink()
    (staging / "analysis/training_parameters.yaml").unlink()
    forged = publish(staging)
    task = load_intake_task_pack(forged.package_dir / "inputs/task.yaml")
    gate = IntakeChecker().check(CheckContext(
        commit=task.source.commit, manifest_sha256=forged.manifest_sha256, package_dir=forged.package_dir,
    ))
    assert gate.status == "blocked"
    assert "V2_HANDOFF_MISSING" in gate.reason_codes


def test_not_applicable_slot_requires_explicit_reason(tmp_path, v2_inputs) -> None:
    def mutate(task, report, handoff):
        slot = report["slots"][-1]
        slot.update(status="not_applicable", sources=[], target_files=[], reason="No task eval in this fixture")
    prep, state, _ = prepare(tmp_path, v2_inputs, mutate)
    assert not state["blocked_reasons"], state
    assert freeze_intake(prep).status == "completed"


@pytest.mark.parametrize("mutation", [
    "missing_source", "missing_basis", "empty_basis", "forged_symbol", "forged_lines",
    "forged_blob", "forged_commit", "arbitrary_spec", "missing_view_policy",
])
def test_typed_anatomy_policy_requires_reproducible_source_even_with_updated_report_hash(tmp_path, v2_inputs, mutation):
    def mutate(task, report, handoff):
        provenance = report["anatomy_spec_source"]
        if mutation == "missing_source":
            del report["anatomy_spec_source"]
        elif mutation == "missing_basis":
            del provenance["analysis_basis"]
        elif mutation == "empty_basis":
            provenance["analysis_basis"] = " "
        elif mutation.startswith("forged_"):
            key, value = {
                "forged_symbol": ("symbol", "not_a_symbol"),
                "forged_lines": ("line_end", 9999),
                "forged_blob": ("blob_sha256", "a" * 64),
                "forged_commit": ("commit", "a" * 40),
            }[mutation]
            provenance["source_ref"][key] = value
        elif mutation == "arbitrary_spec":
            report["anatomy_spec"] = {"trust_me": True}
        else:
            report["anatomy_spec"]["views"] = []
    prep, state, _ = prepare(tmp_path, v2_inputs, mutate)
    assert "V2_HANDOFF_INVALID" in state["blocked_reasons"], state
    assert freeze_intake(prep).status == "blocked"


@pytest.mark.parametrize("target", ["source_ref", "source_digest", "policy_digest"])
def test_resealed_policy_source_and_receipt_forgery_are_independently_reconstructed(tmp_path, sealed_v2, target):
    staging = tmp_path / "forged.staging"
    shutil.copytree(sealed_v2.package_dir, staging)
    if target == "source_ref":
        path = staging / "analysis/anatomy_spec.yaml"
        doc = yaml.safe_load(path.read_text())
        doc["source_ref"]["analysis_basis"] = "new unapproved interpretation"
        path.write_text(yaml.safe_dump(doc))
    else:
        path = staging / "analysis/skill_handoff_receipt.json"
        doc = json.loads(path.read_text())
        doc["anatomy_spec_source_sha256" if target == "source_digest" else "anatomy_spec_sha256"] = "a" * 64
        path.write_text(json.dumps(doc))
    (staging / MANIFEST_NAME).unlink()
    forged = publish(staging)
    task = load_intake_task_pack(forged.package_dir / "inputs/task.yaml")
    gate = IntakeChecker().check(CheckContext(
        commit=task.source.commit, manifest_sha256=forged.manifest_sha256, package_dir=forged.package_dir,
    ))
    assert gate.status == "blocked", gate
    assert "V2_HANDOFF_INVALID" in gate.reason_codes


@pytest.fixture(scope="module")
def sealed_v2_anatomy(tmp_path_factory, sealed_v2):
    prep = tmp_path_factory.mktemp("v2-anatomy")
    cap = capture_two_attempts(
        intake_package_dir=sealed_v2.package_dir, intake_manifest_sha256=sealed_v2.manifest_sha256,
        prep_dir=prep,
    )
    assert cap.status == "captured", cap
    result = freeze_anatomy(prep)
    assert result.status == "completed", result
    return result


def test_scaffold_bare_anatomy_sha_is_explicitly_unverified(sealed_v2, v2_inputs):
    plan = ScaffoldPlan.from_intake(
        intake_package=sealed_v2.package_dir, intake_manifest_sha256=sealed_v2.manifest_sha256,
        anatomy_manifest_sha256="a" * 64, source_commit=v2_inputs["task"]["source"]["commit"],
    )
    assert plan.anatomy_manifest_sha256 == "a" * 64
    assert plan.anatomy_reference_status == "unverified_external"


@pytest.mark.parametrize("mutation", ["missing_spec_trust", "wrong_spec_trust", "wrong_outer_sha", "missing_package", "malformed_bare_sha"])
def test_scaffold_cannot_promote_rejected_anatomy_references_to_verified(tmp_path, sealed_v2, sealed_v2_anatomy, v2_inputs, mutation):
    result = sealed_v2_anatomy
    kwargs = {
        "intake_package": sealed_v2.package_dir, "intake_manifest_sha256": sealed_v2.manifest_sha256,
        "anatomy_manifest_sha256": result.package_ref.manifest_sha256,
        "source_commit": v2_inputs["task"]["source"]["commit"],
        "anatomy_package_dir": result.package_ref.package_dir, "spec_manifest_sha256": result.spec_manifest_sha256,
    }
    if mutation == "missing_spec_trust":
        del kwargs["spec_manifest_sha256"]
    elif mutation == "wrong_spec_trust":
        kwargs["spec_manifest_sha256"] = "a" * 64
    elif mutation == "wrong_outer_sha":
        kwargs["anatomy_manifest_sha256"] = "a" * 64
    elif mutation == "missing_package":
        kwargs["anatomy_package_dir"] = tmp_path / "nonexistent"
    else:
        kwargs.pop("anatomy_package_dir")
        kwargs["anatomy_manifest_sha256"] = "not-a-digest"
    with pytest.raises(ValueError):
        ScaffoldPlan.from_intake(**kwargs)


@pytest.mark.parametrize("mutation", ["old_version", "legacy_identity"])
def test_intake_v2_validator_receipts_do_not_silently_accept_v1_semantics(tmp_path, sealed_v2, v2_inputs, mutation):
    staging = tmp_path / "forged.staging"
    shutil.copytree(sealed_v2.package_dir, staging)
    handoff_path = staging / "handoff/result.json"
    handoff = json.loads(handoff_path.read_text())
    verdict_path = staging / handoff["validator_verdict"]["path"]
    verdict = json.loads(verdict_path.read_text())
    verdict["validator_version"] = "1"
    if mutation == "legacy_identity":
        verdict["validator_id"] = "stages.verify.intake_check"
    verdict_path.write_text(json.dumps(verdict))
    handoff["validator_verdict"]["sha256"] = hashlib.sha256(verdict_path.read_bytes()).hexdigest()
    handoff_path.write_text(json.dumps(handoff))
    (staging / MANIFEST_NAME).unlink()
    forged = publish(staging)
    gate = IntakeChecker().check(CheckContext(
        commit=v2_inputs["task"]["source"]["commit"], manifest_sha256=forged.manifest_sha256,
        package_dir=forged.package_dir,
    ))
    assert gate.status == "blocked", gate
    assert gate.reason_codes == ((
        "LEGACY_VALIDATOR_RUNTIME_REQUIRED" if mutation == "legacy_identity" else "HANDOFF_VERDICT_BINDING_MISMATCH"
    ),)
    # This is an identity rejection test, NOT a replay of a real historical sealed v1 fixture.
