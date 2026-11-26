"""Unified checker protocol and helper utilities for Stage 5 (`verify`)."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.orchestrator import judger as _judger_module
from core.orchestrator.judger import (
    GateResult,
)


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

    Any change to the checker itself, `validate/base.py`, `core/orchestrator/judger.py`,
    or explicitly listed helper modules changes the resulting validator SHA-256.
    """
    source_file = inspect.getsourcefile(obj)
    if source_file is None:
        raise ValueError(f"cannot locate source file for validator {obj!r}")

    resolved_files: dict[str, Path] = {}
    package_root = Path(__file__).resolve().parents[1]
    for candidate in (
        Path(source_file).resolve(),
        *_shared_validator_sources(),
        *(Path(p).resolve() for p in extra_sources),
    ):
        try:
            marker = candidate.relative_to(package_root).as_posix()
        except ValueError:
            marker = f"external/{candidate.name}"
        if marker in resolved_files and resolved_files[marker] != candidate:
            raise ValueError(f"ambiguous validator source marker: {marker}")
        resolved_files[marker] = candidate

    digest = hashlib.sha256()
    for key in sorted(resolved_files):
        file_path = resolved_files[key]
        content = file_path.read_bytes()
        file_sha = hashlib.sha256(content).hexdigest()
        # Bind canonical module basename/relative marker + file hash deterministically.
        digest.update(f"{key}:{file_sha}\n".encode("utf-8"))
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
    """Protocol implemented by every independent checker under `validate/`."""

    gate_id: str
    check_name: str
    validator_id: str

    def validator_sha256(self) -> str:
        """Return the SHA-256 hash binding this validator implementation."""
        ...

    def check(self, context: CheckContext) -> GateResult:
        """Independently inspect artifacts/code and emit a bound `GateResult`."""
        ...
