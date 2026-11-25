"""Deterministic gate verdict and decision contracts."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import Field, model_validator

from .common import CommitHash, Sha256, StrictModel, load_yaml_document
from .errors import ContractValidationError

GateOutcome: TypeAlias = Literal["pass", "repairable", "blocked", "error"]
CheckName: TypeAlias = Literal[
    "intake",
    "anatomy",
    "construct",
    "parity",
    "train_resume",
    "task_eval",
]


class GateBinding(StrictModel):
    """Cryptographic and commit provenance bound to a gate result."""

    commit: CommitHash
    manifest_sha256: Sha256
    validator_id: str = Field(min_length=1)
    validator_sha256: Sha256


class GateEvidence(StrictModel):
    """Compact structured evidence item produced by an independent checker."""

    key: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    path: str | None = None
    digest: Sha256 | None = None


class GateResult(StrictModel):
    """Structured outcome from a single independent validator/checker."""

    schema_version: Literal[1] = 1
    gate_id: str = Field(min_length=1)
    check: CheckName
    status: GateOutcome
    binding: GateBinding
    reason_codes: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    first_mismatch: str | None = None
    evidence: list[GateEvidence] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def non_pass_requires_reason_codes(self) -> GateResult:
        if self.status != "pass" and not self.reason_codes:
            raise ValueError("non-pass gate result must include at least one reason_code")
        if any(not code for code in self.reason_codes):
            raise ValueError("gate reason_codes must be non-empty strings")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("gate reason_codes must be unique")
        return self


class GateDecision(StrictModel):
    """Aggregated deterministic decision across one or more GateResults."""

    schema_version: Literal[1] = 1
    status: GateOutcome
    expected_commit: CommitHash
    expected_manifest_sha256: Sha256
    results: list[GateResult] = Field(default_factory=list)
    failing_gates: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    first_mismatch: str | None = None
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> GateDecision:
        if self.status == "pass" and self.failing_gates:
            raise ValueError("pass decision cannot report failing_gates")
        if self.status != "pass" and not self.reason_codes:
            raise ValueError("non-pass decision must include at least one reason_code")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("decision reason_codes must be unique")
        if len(self.failing_gates) != len(set(self.failing_gates)):
            raise ValueError("decision failing_gates must be unique")
        return self


def load_gate_result(path: Any) -> GateResult:
    """Load and strictly validate a persisted GateResult (JSON or YAML)."""
    try:
        return GateResult.model_validate(load_yaml_document(path))
    except ContractValidationError:
        raise
    except Exception as error:
        validation_error = getattr(error, "errors", None)
        if callable(validation_error):
            raise ContractValidationError(validation_error()) from error
        raise


def load_gate_decision(path: Any) -> GateDecision:
    """Load and strictly validate a persisted GateDecision (JSON or YAML)."""
    try:
        return GateDecision.model_validate(load_yaml_document(path))
    except ContractValidationError:
        raise
    except Exception as error:
        validation_error = getattr(error, "errors", None)
        if callable(validation_error):
            raise ContractValidationError(validation_error()) from error
        raise


__all__ = [
    "CheckName",
    "GateBinding",
    "GateDecision",
    "GateEvidence",
    "GateOutcome",
    "GateResult",
    "load_gate_decision",
    "load_gate_result",
]
