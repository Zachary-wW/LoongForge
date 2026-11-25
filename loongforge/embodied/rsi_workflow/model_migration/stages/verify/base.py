"""Unified checker protocol and helper utilities for Stage 5 (`verify`)."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from engine import judger as _judger_module
from engine.judger import (
    VALID_CHECKS,
    GateBinding,
    GateDecision,
    GateEvidence,
    GateResult,
    decide,
)
from stages.base import StageContext, StageResult


def _shared_validator_sources() -> tuple[Path, ...]:
    """Return canonical shared verification modules bound into every validator digest."""
    paths: list[Path] = [Path(__file__).resolve()]
    judger_file = inspect.getsourcefile(_judger_module)
    if judger_file is not None:
        paths.append(Path(judger_file).resolve())
    return tuple(paths)


def validator_source_sha256(
    obj: Any,
    *,
    extra_sources: Iterable[str | Path] = (),
) -> str:
    """Compute a composite SHA-256 over the checker module + shared verify/judger modules.

    Any change to the checker itself, `stages/verify/base.py`, `engine/judger.py`,
    or explicitly listed helper modules changes the resulting validator SHA-256.
    """
    source_file = inspect.getsourcefile(obj)
    if source_file is None:
        raise ValueError(f"cannot locate source file for validator {obj!r}")

    resolved_files: dict[str, Path] = {}
    for candidate in (
        Path(source_file).resolve(),
        *_shared_validator_sources(),
        *(Path(p).resolve() for p in extra_sources),
    ):
        resolved_files[candidate.as_posix()] = candidate

    digest = hashlib.sha256()
    for key in sorted(resolved_files):
        file_path = resolved_files[key]
        content = file_path.read_bytes()
        file_sha = hashlib.sha256(content).hexdigest()
        # Bind canonical module basename/relative marker + file hash deterministically.
        digest.update(f"{file_path.name}:{file_sha}\n".encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Immutable inputs supplied to an independent checker."""

    commit: str
    manifest_sha256: str
    package_dir: Path
    workspace_dir: Path | None = None
    reference_dir: Path | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GateChecker(Protocol):
    """Protocol implemented by every independent checker under `stages/verify/`."""

    gate_id: str
    check_name: str
    validator_id: str

    def validator_sha256(self) -> str:
        """Return the SHA-256 hash binding this validator implementation."""
        ...

    def check(self, context: CheckContext) -> GateResult:
        """Independently inspect artifacts/code and emit a bound `GateResult`."""
        ...


class VerifyStageRunner:
    """Executes registered independent checkers and aggregates via `engine.judger.decide`."""

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
