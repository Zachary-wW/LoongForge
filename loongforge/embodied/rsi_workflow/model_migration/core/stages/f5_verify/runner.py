"""Stage 5 orchestration; concrete checkers live in validate/."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from core.orchestrator.judger import (VALID_CHECKS, GateBinding, GateDecision, GateEvidence, GateResult, decide)
from core.stages.base import StageContext, StageResult
from validate.base import CheckContext, GateChecker

class VerifyStageRunner:
    """Executes registered independent checkers and aggregates via `core.orchestrator.judger.decide`."""

    stage_name = "verify"

    def __init__(
        self,
        checkers: Iterable[GateChecker] = (),
        *,
        trusted_validators: Mapping[str, str] | None = None,
        required_gates: Iterable[str] | None = None,
        allow_untrusted: bool = False,
    ) -> None:
        if isinstance(required_gates, str):
            raise TypeError("required_gates must be an iterable of gate IDs, not str")
        self.checkers: tuple[GateChecker, ...] = tuple(checkers)
        if trusted_validators is None and not allow_untrusted:
            raise ValueError(
                "trusted_validators is required for fail-closed verification "
                "(pass allow_untrusted=True only for explicit test overrides)"
            )
        self.trusted_validators = (
            dict(trusted_validators) if trusted_validators is not None else None
        )
        self.allow_untrusted = allow_untrusted
        if required_gates is not None:
            self.required_gates: tuple[str, ...] = tuple(required_gates)
        else:
            self.required_gates = tuple(checker.gate_id for checker in self.checkers)

    def evaluate(self, context: CheckContext) -> GateDecision:
        results: list[GateResult] = []
        for checker in self.checkers:
            try:
                result = checker.check(context)
            except Exception as error:
                try:
                    val_sha = checker.validator_sha256()
                except Exception:
                    val_sha = "f" * 64
                check_name = (
                    checker.check_name
                    if checker.check_name in VALID_CHECKS
                    else "intake"
                )
                result = GateResult(
                    gate_id=checker.gate_id,
                    check=check_name,
                    status="error",
                    binding=GateBinding(
                        commit=context.commit,
                        manifest_sha256=context.manifest_sha256,
                        validator_id=checker.validator_id,
                        validator_sha256=val_sha,
                    ),
                    reason_codes=("CHECKER_CRASH",),
                    first_mismatch=f"{checker.gate_id}:{type(error).__name__}",
                    summary=f"checker {checker.gate_id!r} raised {type(error).__name__}: {error}",
                    evidence=(
                        GateEvidence(
                            key="exception",
                            detail=f"{type(error).__name__}: {error}",
                        ),
                    ),
                )
            results.append(result)

        return decide(
            results,
            expected_commit=context.commit,
            expected_manifest_sha256=context.manifest_sha256,
            trusted_validators=self.trusted_validators,
            required_gates=self.required_gates,
        )

    def run(self, context: StageContext) -> StageResult:
        raise NotImplementedError(
            "VerifyStageRunner.run orchestration binding lands once Stage 1/2 checkers are wired"
        )
