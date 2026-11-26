"""Deterministic gate verdict aggregation with provenance binding enforcement.

Pure Python stdlib (no Pydantic dependency) so `core/orchestrator/` stays stdlib + PyYAML
only, while remaining structurally identical to `core.contracts.verdict`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

GateOutcome: TypeAlias = Literal["pass", "repairable", "blocked", "error"]
CheckName: TypeAlias = Literal[
    "intake",
    "anatomy",
    "construct",
    "parity",
    "train_resume",
    "task_eval",
]

_VALID_OUTCOMES: frozenset[str] = frozenset({"pass", "repairable", "blocked", "error"})
VALID_CHECKS: frozenset[str] = frozenset(
    {
        "intake",
        "anatomy",
        "construct",
        "parity",
        "train_resume",
        "task_eval",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

_PRECEDENCE: dict[GateOutcome, int] = {
    "pass": 0,
    "repairable": 1,
    "blocked": 2,
    "error": 3,
}


def _is_valid_sha256(value: str) -> bool:
    return bool(_SHA256_RE.match(value)) and value != "0" * len(value)


def _is_valid_commit(value: str) -> bool:
    return bool(_COMMIT_RE.match(value)) and value != "0" * len(value)


@dataclass(frozen=True, slots=True)
class GateBinding:
    """Commit, manifest SHA-256, and validator SHA-256 bound to a gate result."""

    commit: str
    manifest_sha256: str
    validator_id: str
    validator_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "manifest_sha256": self.manifest_sha256,
            "validator_id": self.validator_id,
            "validator_sha256": self.validator_sha256,
        }


@dataclass(frozen=True, slots=True)
class GateEvidence:
    """Compact structured evidence reference from an independent checker."""

    key: str
    detail: str
    path: str | None = None
    digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("evidence key must be a non-empty string")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("evidence detail must be a non-empty string")
        if self.digest is not None and not _is_valid_sha256(self.digest):
            raise ValueError("evidence digest must be a valid non-zero sha256")
        object.__setattr__(self, "key", self.key.strip())
        object.__setattr__(self, "detail", self.detail.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "detail": self.detail,
            "path": self.path,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateEvidence:
        return cls(
            key=data["key"],
            detail=data["detail"],
            path=data.get("path"),
            digest=data.get("digest"),
        )


@dataclass(frozen=True, slots=True)
class GateResult:
    """Structured result emitted by one independent checker."""

    gate_id: str
    check: CheckName
    status: GateOutcome
    binding: GateBinding
    summary: str
    reason_codes: tuple[str, ...] = ()
    first_mismatch: str | None = None
    evidence: tuple[GateEvidence, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gate_id or not self.gate_id.strip():
            raise ValueError("gate_id must be a non-empty string")
        if self.check not in VALID_CHECKS:
            raise ValueError(
                f"invalid check name {self.check!r}; expected one of {sorted(VALID_CHECKS)}"
            )
        if self.status not in _VALID_OUTCOMES:
            raise ValueError(f"invalid gate outcome: {self.status!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be a non-empty string")
        codes = tuple(self.reason_codes)
        if any(not isinstance(code, str) or not code.strip() for code in codes):
            raise ValueError("reason_codes must contain non-empty strings")
        if len(codes) != len(set(codes)):
            raise ValueError("gate reason_codes must be unique")
        if self.status != "pass" and not codes:
            raise ValueError("non-pass GateResult must include at least one reason_code")
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "metrics", dict(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "gate_id": self.gate_id,
            "check": self.check,
            "status": self.status,
            "binding": self.binding.to_dict(),
            "reason_codes": list(self.reason_codes),
            "summary": self.summary,
            "first_mismatch": self.first_mismatch,
            "evidence": [item.to_dict() for item in self.evidence],
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateResult:
        if data.get("schema_version", 1) != 1:
            raise ValueError(f"unsupported schema_version: {data.get('schema_version')!r}")
        binding_data = data["binding"]
        binding = GateBinding(
            commit=binding_data["commit"],
            manifest_sha256=binding_data["manifest_sha256"],
            validator_id=binding_data["validator_id"],
            validator_sha256=binding_data["validator_sha256"],
        )
        evidence = tuple(
            GateEvidence.from_dict(item) for item in data.get("evidence", ())
        )
        return cls(
            gate_id=data["gate_id"],
            check=data["check"],
            status=data["status"],
            binding=binding,
            summary=data["summary"],
            reason_codes=tuple(data.get("reason_codes", ())),
            first_mismatch=data.get("first_mismatch"),
            evidence=evidence,
            metrics=dict(data.get("metrics", {})),
        )


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Aggregated deterministic decision returned by `decide(...)`."""

    status: GateOutcome
    expected_commit: str
    expected_manifest_sha256: str
    results: tuple[GateResult, ...]
    failing_gates: tuple[str, ...]
    reason_codes: tuple[str, ...]
    first_mismatch: str | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "expected_commit": self.expected_commit,
            "expected_manifest_sha256": self.expected_manifest_sha256,
            "results": [item.to_dict() for item in self.results],
            "failing_gates": list(self.failing_gates),
            "reason_codes": list(self.reason_codes),
            "first_mismatch": self.first_mismatch,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateDecision:
        if data.get("schema_version", 1) != 1:
            raise ValueError(f"unsupported schema_version: {data.get('schema_version')!r}")
        return cls(
            status=data["status"],
            expected_commit=data["expected_commit"],
            expected_manifest_sha256=data["expected_manifest_sha256"],
            results=tuple(
                GateResult.from_dict(item) for item in data.get("results", ())
            ),
            failing_gates=tuple(data.get("failing_gates", ())),
            reason_codes=tuple(data.get("reason_codes", ())),
            first_mismatch=data.get("first_mismatch"),
            summary=data["summary"],
        )


def decide(
    results: Iterable[GateResult],
    *,
    expected_commit: str,
    expected_manifest_sha256: str,
    trusted_validators: Mapping[str, str] | None = None,
    required_gates: Iterable[str] | None = None,
) -> GateDecision:
    """Deterministically aggregate independent GateResults into a GateDecision.

    Fail-closed rules:
    - Invalid expected commit/manifest SHA -> `error`
    - Empty results -> `error` (`EMPTY_GATE_RESULTS`)
    - Duplicate gate_id -> `error` (`DUPLICATE_GATE_ID`)
    - Missing required gate -> `error` (`MISSING_REQUIRED_GATE`)
    - Binding mismatch on commit, manifest SHA-256, or validator SHA-256 -> `error`
    - Otherwise, highest-severity outcome wins:
      `error` > `blocked` > `repairable` > `pass`
    """
    result_list = tuple(results)
    reason_codes: list[str] = []
    failing_gates: list[str] = []
    first_mismatch: str | None = None
    worst: GateOutcome = "pass"

    def _add_reason(code: str) -> None:
        if code not in reason_codes:
            reason_codes.append(code)

    def _add_failing(gate_id: str) -> None:
        if gate_id not in failing_gates:
            failing_gates.append(gate_id)

    def _escalate(outcome: GateOutcome) -> None:
        nonlocal worst
        if _PRECEDENCE[outcome] > _PRECEDENCE[worst]:
            worst = outcome

    if not _is_valid_commit(expected_commit):
        _escalate("error")
        _add_reason("INVALID_EXPECTED_COMMIT")
    if not _is_valid_sha256(expected_manifest_sha256):
        _escalate("error")
        _add_reason("INVALID_EXPECTED_MANIFEST_SHA")
    if isinstance(required_gates, str):
        raise TypeError("required_gates must be an iterable of gate IDs, not str")

    if not result_list:
        _escalate("error")
        _add_reason("EMPTY_GATE_RESULTS")
        return GateDecision(
            status="error",
            expected_commit=expected_commit,
            expected_manifest_sha256=expected_manifest_sha256,
            results=result_list,
            failing_gates=tuple(failing_gates),
            reason_codes=tuple(reason_codes),
            first_mismatch=None,
            summary="no gate results supplied to judger",
        )

    seen_gate_ids: set[str] = set()
    for result in result_list:
        if result.gate_id in seen_gate_ids:
            _escalate("error")
            _add_failing(result.gate_id)
            _add_reason("DUPLICATE_GATE_ID")
        seen_gate_ids.add(result.gate_id)

        binding = result.binding
        binding_failed = False
        if not _is_valid_commit(binding.commit) or binding.commit != expected_commit:
            binding_failed = True
            _add_reason("COMMIT_BINDING_MISMATCH")
        if (
            not _is_valid_sha256(binding.manifest_sha256)
            or binding.manifest_sha256 != expected_manifest_sha256
        ):
            binding_failed = True
            _add_reason("MANIFEST_BINDING_MISMATCH")
        if not binding.validator_id.strip() or not _is_valid_sha256(
            binding.validator_sha256
        ):
            binding_failed = True
            _add_reason("INVALID_VALIDATOR_BINDING")
        elif trusted_validators is not None:
            expected_val_sha = trusted_validators.get(binding.validator_id)
            if expected_val_sha is None:
                binding_failed = True
                _add_reason("UNTRUSTED_VALIDATOR")
            elif binding.validator_sha256 != expected_val_sha:
                binding_failed = True
                _add_reason("VALIDATOR_SHA_MISMATCH")

        if binding_failed:
            _escalate("error")
            _add_failing(result.gate_id)
            if first_mismatch is None:
                first_mismatch = f"{result.gate_id}:binding_mismatch"
            continue

        if result.status != "pass":
            _escalate(result.status)
            _add_failing(result.gate_id)
            for code in result.reason_codes:
                _add_reason(code)
            if first_mismatch is None and result.first_mismatch is not None:
                first_mismatch = result.first_mismatch

    if required_gates is not None:
        for gate_id in required_gates:
            if gate_id not in seen_gate_ids:
                _escalate("error")
                _add_failing(gate_id)
                _add_reason("MISSING_REQUIRED_GATE")

    if worst == "pass":
        summary = f"all {len(result_list)} gate(s) passed with verified provenance"
    else:
        summary = (
            f"decision={worst}; failing_gates={','.join(failing_gates) or 'none'}; "
            f"reasons={','.join(reason_codes)}"
        )

    return GateDecision(
        status=worst,
        expected_commit=expected_commit,
        expected_manifest_sha256=expected_manifest_sha256,
        results=result_list,
        failing_gates=tuple(failing_gates),
        reason_codes=tuple(reason_codes),
        first_mismatch=first_mismatch,
        summary=summary,
    )
