"""Independent checker placeholder for LoongForge training & checkpoint resume checks."""

from __future__ import annotations

from core.orchestrator.judger import GateResult

from .base import CheckContext, validator_source_sha256


class TrainResumeChecker:
    """Checks LoongForge training regression and checkpoint save/resume consistency."""

    gate_id = "train_resume"
    check_name = "train_resume"
    validator_id = "validate.train_resume_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(type(self))

    def check(self, context: CheckContext) -> GateResult:
        raise NotImplementedError(
            "TrainResumeChecker is not implemented in this milestone"
        )
