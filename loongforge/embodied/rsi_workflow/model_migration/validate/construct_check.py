"""Independent checker placeholder for LoongForge model/config/registry construction."""

from __future__ import annotations

from core.orchestrator.judger import GateResult

from .base import CheckContext, validator_source_sha256


class ConstructChecker:
    """Checks LoongForge 6 extension-point registration and smoke construction."""

    gate_id = "construct"
    check_name = "construct"
    validator_id = "validate.construct_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(type(self))

    def check(self, context: CheckContext) -> GateResult:
        raise NotImplementedError(
            "ConstructChecker is not implemented in this milestone"
        )
