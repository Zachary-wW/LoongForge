from core.stages.f5_verify import VerifyStageRunner
from pathlib import Path

import pytest
import yaml

from core.contracts import (
    AnatomyBehaviorContract,
    ContractParseError,
    ContractValidationError,
    F1TaskPack,
    F2BehaviorContract,
    GateBinding,
    GateDecision,
    GateResult,
    IntakeTaskPack,
    StageSpec,
    load_anatomy_behavior_contract,
    load_intake_task_pack,
    load_pipeline_profile,
)
from core.orchestrator.judger import GateBinding as EngineGateBinding
from core.orchestrator.judger import GateResult as EngineGateResult
from core.rsi import PlaceholderRSISupervisor, RSISupervisor
from core.stages import StageContext, StageRunner
from core.stages.f4_adapt import AdaptStageRunner
from core.stages.f2_anatomy import AnatomyStageRunner
from core.stages.f6_deliver import DeliverStageRunner
from core.stages.f1_intake import IntakeStageRunner
from core.stages.f3_scaffold import LOONGFORGE_EXTENSION_POINTS, ScaffoldStageRunner
from validate import (
    AnatomyChecker,
    CheckContext,
    ConstructChecker,
    EvalChecker,
    GateChecker,
    IntakeChecker,
    ParityChecker,
    TrainResumeChecker,
    validator_source_sha256,
)


REPO_ROOT = Path(__file__).parents[2]
COMMIT_A = "7f3c2a9d4e5b6f70128394aa65b7c8d9e0f1a2b3"
MANIFEST_A = "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace"
VAL_SHA_1 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"


def test_unified_contract_aliases_and_new_models_validate() -> None:
    assert IntakeTaskPack is F1TaskPack
    assert AnatomyBehaviorContract is F2BehaviorContract
    assert callable(load_intake_task_pack)
    assert callable(load_anatomy_behavior_contract)

    spec = StageSpec(
        id="scaffold",
        stage="scaffold",
        depends_on=["anatomy"],
        isolation="git_worktree",
        skills=["model-adaptation", "vla-model-eval-adapter"],
        extension_points=list(LOONGFORGE_EXTENSION_POINTS),
    )
    assert spec.stage == "scaffold"

    gate = GateResult(
        gate_id="intake",
        check="intake",
        status="pass",
        binding=GateBinding(
            commit=COMMIT_A,
            manifest_sha256=MANIFEST_A,
            validator_id="validate.intake_check",
            validator_sha256=VAL_SHA_1,
        ),
        summary="ok",
    )
    decision = GateDecision(
        status="pass",
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
        results=[gate],
        summary="all pass",
    )
    assert decision.status == "pass"


def test_stage_and_checker_placeholders_conform_to_protocols(tmp_path: Path) -> None:
    # Stage 1 (`intake`) and Stage 2 (`anatomy`) are implemented; remaining stages stay placeholders.
    assert isinstance(IntakeStageRunner(), StageRunner)
    assert isinstance(AnatomyStageRunner(), StageRunner)

    placeholder_runners = [
        ScaffoldStageRunner(),
        AdaptStageRunner(),
        VerifyStageRunner(allow_untrusted=True),
        DeliverStageRunner(),
    ]
    ctx = StageContext(
        run_id="run-1",
        stage="intake",
        workspace_dir=tmp_path,
        output_dir=tmp_path / "out",
    )
    for runner in placeholder_runners:
        assert isinstance(runner, StageRunner)
        with pytest.raises(NotImplementedError):
            runner.run(ctx)

    # IntakeChecker and AnatomyChecker are implemented; remaining checkers stay placeholders.
    intake_checker = IntakeChecker()
    assert isinstance(intake_checker, GateChecker)
    assert len(intake_checker.validator_sha256()) == 64

    anatomy_checker = AnatomyChecker()
    assert isinstance(anatomy_checker, GateChecker)
    assert len(anatomy_checker.validator_sha256()) == 64

    placeholder_checkers = [
        ConstructChecker(),
        ParityChecker(),
        TrainResumeChecker(),
        EvalChecker(),
    ]
    check_ctx = CheckContext(
        commit=COMMIT_A,
        manifest_sha256=MANIFEST_A,
        package_dir=tmp_path,
    )
    for checker in placeholder_checkers:
        assert isinstance(checker, GateChecker)
        assert len(checker.validator_sha256()) == 64
        with pytest.raises(NotImplementedError):
            checker.check(check_ctx)

    supervisor = PlaceholderRSISupervisor()
    assert isinstance(supervisor, RSISupervisor)
    with pytest.raises(NotImplementedError):
        supervisor.propose(())


def test_profile_declares_all_extension_points_skills_and_stateless_tools() -> None:
    profile_path = REPO_ROOT / "core" / "profiles" / "loongforge_embodied.yaml"
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    # Every stage entry validates directly against strict StageSpec without translation.
    validated_stages = [StageSpec.model_validate(entry) for entry in raw["stages"]]
    assert [s.stage for s in validated_stages] == [
        "intake",
        "anatomy",
        "scaffold",
        "adapt",
        "verify",
        "deliver",
    ]

    # Full profile validates schema_version, 40-char commit SHA, path_base, and real paths.
    profile = load_pipeline_profile(profile_path, verify_paths=True)
    assert profile.schema_version == 1
    assert len(profile.target_framework.reference_commit) == 40
    assert profile.target_framework.reference_commit == (
        "5f48c81d1209419fcd53efe8495141ec1be5437f"
    )
    assert set(profile.extension_points.keys()) == set(LOONGFORGE_EXTENSION_POINTS)
    assert {item.name for item in profile.skills} == {
        "model-adaptation",
        "precision-alignment",
        "vla-model-eval-adapter",
    }
    skills_by_name = {item.name: item for item in profile.skills}
    assert skills_by_name["model-adaptation"].optional is False
    assert skills_by_name["precision-alignment"].optional is False
    assert skills_by_name["vla-model-eval-adapter"].optional is True
    assert skills_by_name["vla-model-eval-adapter"].path is None
    assert {item.filename for item in profile.stateless_tools} == {
        "codenorm.py",
        "diff_live_code.sh",
        "diff_weight_probe.py",
        "read_tb.py",
        "reap_gpus.sh",
    }


def test_profile_loader_rejects_short_commit_duplicate_keys_and_missing_paths(
    tmp_path: Path,
) -> None:
    profile_path = REPO_ROOT / "core" / "profiles" / "loongforge_embodied.yaml"
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    # Short 7-char commit must be rejected by strict CommitHash.
    short_commit_doc = dict(raw)
    short_commit_doc["target_framework"] = dict(raw["target_framework"])
    short_commit_doc["target_framework"]["reference_commit"] = "5f48c81"
    short_path = tmp_path / "short.yaml"
    short_path.write_text(yaml.safe_dump(short_commit_doc), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        load_pipeline_profile(short_path)

    # Missing tool path rejected when verify_paths=True.
    missing_tool_doc = dict(raw)
    missing_tool_doc["stateless_tools"] = [dict(raw["stateless_tools"][0])]
    missing_tool_doc["stateless_tools"][0]["path"] = "does/not/exist.py"
    missing_path = tmp_path / "missing_tool.yaml"
    missing_path.write_text(yaml.safe_dump(missing_tool_doc), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        load_pipeline_profile(
            missing_path,
            verify_paths=True,
            repo_root=REPO_ROOT,
            workspace_root=REPO_ROOT.parent,
        )

    # Duplicate YAML keys rejected.
    dup_path = tmp_path / "dup.yaml"
    dup_path.write_text(
        profile_path.read_text(encoding="utf-8").replace(
            "schema_version: 1", "schema_version: 1\nschema_version: 1", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractParseError) as parse_err:
        load_pipeline_profile(dup_path)
    assert parse_err.value.reason_code == "DUPLICATE_YAML_KEY"

    # Non-optional skill with path=None must be rejected by strict schema.
    bad_req_skill_doc = dict(raw)
    bad_req_skill_doc["skills"] = [dict(item) for item in raw["skills"]]
    bad_req_skill_doc["skills"][0]["path"] = None
    bad_req_skill_doc["skills"][0]["optional"] = False
    bad_req_path = tmp_path / "req_skill_no_path.yaml"
    bad_req_path.write_text(yaml.safe_dump(bad_req_skill_doc), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        load_pipeline_profile(bad_req_path)


def test_verify_stage_runner_fail_closed_defaults_and_checker_crash_isolation(
    tmp_path: Path,
) -> None:
    # Fail-closed: trusted_validators cannot be omitted unless allow_untrusted=True.
    with pytest.raises(ValueError, match="trusted_validators"):
        VerifyStageRunner([IntakeChecker()])

    # Fail-closed: string required_gates rejected.
    with pytest.raises(TypeError):
        VerifyStageRunner(
            [IntakeChecker()],
            allow_untrusted=True,
            required_gates="intake",  # type: ignore[arg-type]
        )

    class _PassChecker:
        gate_id = "intake"
        check_name = "intake"
        validator_id = "test.pass_checker"

        def validator_sha256(self) -> str:
            return validator_source_sha256(type(self))

        def check(self, context: CheckContext) -> EngineGateResult:
            return EngineGateResult(
                gate_id=self.gate_id,
                check="intake",
                status="pass",
                binding=EngineGateBinding(
                    commit=context.commit,
                    manifest_sha256=context.manifest_sha256,
                    validator_id=self.validator_id,
                    validator_sha256=self.validator_sha256(),
                ),
                summary="pass ok",
            )

    class _CrashChecker:
        gate_id = "anatomy"
        check_name = "anatomy"
        validator_id = "test.crash_checker"

        def validator_sha256(self) -> str:
            return validator_source_sha256(type(self))

        def check(self, context: CheckContext) -> EngineGateResult:
            raise RuntimeError("boom inside checker")

    ctx = CheckContext(
        commit=COMMIT_A,
        manifest_sha256=MANIFEST_A,
        package_dir=tmp_path,
    )
    pass_checker = _PassChecker()
    crash_checker = _CrashChecker()

    # 1) All passing with trusted validator digest -> pass.
    runner_ok = VerifyStageRunner(
        [pass_checker],
        trusted_validators={pass_checker.validator_id: pass_checker.validator_sha256()},
    )
    assert runner_ok.required_gates == ("intake",)
    decision_ok = runner_ok.evaluate(ctx)
    assert decision_ok.status == "pass"

    # 2) Partial checker subset against declared required_gates -> error (MISSING_REQUIRED_GATE).
    runner_partial = VerifyStageRunner(
        [pass_checker],
        trusted_validators={pass_checker.validator_id: pass_checker.validator_sha256()},
        required_gates=("intake", "anatomy"),
    )
    decision_partial = runner_partial.evaluate(ctx)
    assert decision_partial.status == "error"
    assert "MISSING_REQUIRED_GATE" in decision_partial.reason_codes

    # 3) Checker exception is isolated into structured CHECKER_CRASH error verdict.
    runner_crash = VerifyStageRunner(
        [pass_checker, crash_checker],
        trusted_validators={
            pass_checker.validator_id: pass_checker.validator_sha256(),
            crash_checker.validator_id: crash_checker.validator_sha256(),
        },
    )
    decision_crash = runner_crash.evaluate(ctx)
    assert decision_crash.status == "error"
    assert "CHECKER_CRASH" in decision_crash.reason_codes
    assert "anatomy" in decision_crash.failing_gates
    assert decision_crash.first_mismatch == "anatomy:RuntimeError"


def test_validator_source_sha256_changes_when_shared_helper_changes(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "shared_helper.py"
    helper.write_text("THRESHOLD = 1e-3\n", encoding="utf-8")

    sha_before = validator_source_sha256(IntakeChecker, extra_sources=[helper])
    helper.write_text("THRESHOLD = 1e-2\n", encoding="utf-8")
    sha_after = validator_source_sha256(IntakeChecker, extra_sources=[helper])

    assert sha_before != sha_after
    assert validator_source_sha256(IntakeChecker) != validator_source_sha256(
        AnatomyChecker
    )
