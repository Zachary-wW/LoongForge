"""Independent checker placeholder for closed-loop embodied simulation evaluation."""

from __future__ import annotations

from core.orchestrator.judger import GateResult

from .base import CheckContext, validator_source_sha256


class EvalChecker:
    """Checks LoongForge eval Factory/PayloadBuilder/ActionDecoder and closed-loop metrics."""

    gate_id = "task_eval"
    check_name = "task_eval"
    validator_id = "validate.eval_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(type(self))

    def check(self, context: CheckContext) -> GateResult:
        raise NotImplementedError(
            "EvalChecker is not implemented in this milestone"
        )
