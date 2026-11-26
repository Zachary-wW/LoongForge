"""Independent checker placeholder for numerical/input/forward/grad parity alignment."""

from __future__ import annotations

from core.orchestrator.judger import GateResult

from .base import CheckContext, validator_source_sha256


class ParityChecker:
    """Checks data/forward/loss/grad parity against frozen Stage 2 anatomy contract."""

    gate_id = "parity"
    check_name = "parity"
    validator_id = "validate.parity_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(type(self))

    def check(self, context: CheckContext) -> GateResult:
        raise NotImplementedError(
            "ParityChecker is not implemented in this milestone"
        )
