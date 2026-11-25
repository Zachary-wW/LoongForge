"""Pinned-commit AST entrypoint and configuration discovery (`git ls-tree` + blob + `ast`)."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeAlias

from .source import _run_git

DiscoveryStatus: TypeAlias = Literal["resolved", "blocked"]

_CONFIG_SUFFIXES: frozenset[str] = frozenset({".yaml", ".yml", ".json", ".toml"})
_ENTRY_KEYWORDS: tuple[str, ...] = ("train", "finetune", "sft", "run", "main", "eval")


class DiscoveryError(ValueError):
    """Structured failure raised when pinned-tree discovery cannot verify an entrypoint."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        evidence_path: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        self.reason_code = reason_code
        self.message = message
        self.evidence_path = evidence_path
        self.expected = expected
        self.actual = actual
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "message": self.message,
            "evidence_path": self.evidence_path,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True, slots=True)
class EntrypointCandidate:
    """Executable Python entrypoint discovered strictly from a pinned Git tree."""

    path: str
    blob_oid: str
    blob_sha256: str
    symbol: str
    line_range: tuple[int, int]
    has_main_guard: bool
    cli_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "blob_oid": self.blob_oid,
            "blob_sha256": self.blob_sha256,
            "symbol": self.symbol,
            "line_range": [self.line_range[0], self.line_range[1]],
            "has_main_guard": self.has_main_guard,
            "cli_flags": list(self.cli_flags),
        }


@dataclass(frozen=True, slots=True)
class ConfigCandidate:
    """Configuration file discovered from the pinned Git tree."""

    path: str
    blob_oid: str
    blob_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "blob_oid": self.blob_oid,
            "blob_sha256": self.blob_sha256,
        }


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Structured discovery output written to `analysis/reference_entrypoints.yaml`."""

    commit: str
    tree_oid: str
    status: DiscoveryStatus
    selected: EntrypointCandidate | None
    candidates: tuple[EntrypointCandidate, ...]
    configs: tuple[ConfigCandidate, ...]
    blocked_reasons: tuple[str, ...] = ()
    readme_clues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "tree_oid": self.tree_oid,
            "status": self.status,
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "candidates": [item.to_dict() for item in self.candidates],
            "configs": [item.to_dict() for item in self.configs],
            "blocked_reasons": list(self.blocked_reasons),
            "readme_clues": list(self.readme_clues),
        }


def _normalize_rel_path(path: str) -> str:
    posix = PurePosixPath(path)
    if (
        "\\" in path
        or posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise DiscoveryError(
            "INVALID_ENTRYPOINT_PATH",
            f"entrypoint path must be relative and normalized: {path!r}",
            evidence_path=path,
        )
    return posix.as_posix()


def _list_pinned_blobs(repo_root: Path, commit: str) -> dict[str, str]:
    proc = _run_git(repo_root, "ls-tree", "-r", "-z", commit, check=False)
    if proc.returncode != 0:
        raise DiscoveryError(
            "PINNED_TREE_READ_FAILED",
            f"failed to list tree for commit {commit}: {proc.stderr.strip()}",
            evidence_path=str(repo_root),
            expected=commit,
        )
    blobs: dict[str, str] = {}
    for entry in proc.stdout.split("\0"):
        if not entry:
            continue
        meta, _, rel_path = entry.partition("\t")
        parts = meta.split()
        if len(parts) < 3:
            continue
        _, obj_type, oid = parts[0], parts[1], parts[2]
        if obj_type == "blob":
            blobs[rel_path] = oid
    return blobs


def _read_blob_bytes(repo_root: Path, blob_oid: str) -> bytes:
    proc = _run_git(repo_root, "cat-file", "blob", blob_oid, binary=True)
    return bytes(proc.stdout)


def _verify_worktree_matches_blob(
    repo_root: Path,
    rel_path: str,
    blob_bytes: bytes,
) -> None:
    worktree_file = repo_root / rel_path
    if not worktree_file.is_file():
        raise DiscoveryError(
            "ENTRYPOINT_MISSING_IN_WORKTREE",
            f"pinned entrypoint {rel_path!r} is missing or deleted in worktree",
            evidence_path=rel_path,
            expected=hashlib.sha256(blob_bytes).hexdigest(),
            actual=None,
        )
    actual_bytes = worktree_file.read_bytes()
    if actual_bytes != blob_bytes:
        raise DiscoveryError(
            "ENTRYPOINT_WORKTREE_MISMATCH",
            f"worktree bytes for {rel_path!r} differ from pinned commit blob",
            evidence_path=rel_path,
            expected=hashlib.sha256(blob_bytes).hexdigest(),
            actual=hashlib.sha256(actual_bytes).hexdigest(),
        )


def _has_main_guard(tree: ast.Module) -> tuple[bool, tuple[int, int] | None]:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            return True, (start, end)
    return False, None


def _extract_cli_flags(tree: ast.Module) -> tuple[str, ...]:
    flags: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                for arg in node.args:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.startswith("-")
                        and arg.value not in flags
                    ):
                        flags.append(arg.value)
    return tuple(flags)


def _analyze_python_blob(
    rel_path: str,
    blob_oid: str,
    blob_bytes: bytes,
) -> EntrypointCandidate | None:
    try:
        source_text = blob_bytes.decode("utf-8")
        tree = ast.parse(source_text, filename=rel_path)
    except (UnicodeDecodeError, SyntaxError):
        return None

    blob_sha256 = hashlib.sha256(blob_bytes).hexdigest()
    has_guard, guard_range = _has_main_guard(tree)
    cli_flags = _extract_cli_flags(tree)

    functions: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            functions[node.name] = (start, end)

    symbol: str | None = None
    line_range: tuple[int, int] | None = None

    for preferred in ("main", "train", "run"):
        if preferred in functions:
            symbol = preferred
            line_range = functions[preferred]
            break

    if symbol is None and has_guard and guard_range is not None:
        symbol = "__main__"
        line_range = guard_range

    if symbol is None:
        return None

    stem = PurePosixPath(rel_path).stem.lower()
    is_likely_entry = has_guard or any(k in stem for k in _ENTRY_KEYWORDS)
    if not is_likely_entry:
        return None

    assert line_range is not None
    return EntrypointCandidate(
        path=rel_path,
        blob_oid=blob_oid,
        blob_sha256=blob_sha256,
        symbol=symbol,
        line_range=line_range,
        has_main_guard=has_guard,
        cli_flags=cli_flags,
    )


def _extract_readme_clues(
    repo_root: Path,
    blobs: dict[str, str],
) -> tuple[str, ...]:
    clues: list[str] = []
    for path, oid in sorted(blobs.items()):
        if PurePosixPath(path).name.lower().startswith("readme"):
            try:
                text = _read_blob_bytes(repo_root, oid).decode("utf-8", errors="ignore")
            except Exception:
                continue
            for candidate_path in sorted(blobs):
                if candidate_path.endswith(".py") and candidate_path in text:
                    if candidate_path not in clues:
                        clues.append(candidate_path)
    return tuple(clues)


def discover_entrypoints(
    repo_root: str | Path,
    commit: str,
    *,
    entry_hint: str | None = None,
) -> DiscoveryResult:
    """Discover executable entrypoints and configs strictly from `commit`'s Git tree."""
    repo = Path(repo_root).resolve()
    tree_oid = _run_git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
    blobs = _list_pinned_blobs(repo, commit)

    configs: list[ConfigCandidate] = []
    candidates: list[EntrypointCandidate] = []

    for rel_path, blob_oid in sorted(blobs.items()):
        suffix = PurePosixPath(rel_path).suffix.lower()
        if suffix in _CONFIG_SUFFIXES:
            blob_bytes = _read_blob_bytes(repo, blob_oid)
            configs.append(
                ConfigCandidate(
                    path=rel_path,
                    blob_oid=blob_oid,
                    blob_sha256=hashlib.sha256(blob_bytes).hexdigest(),
                )
            )
        elif suffix == ".py":
            blob_bytes = _read_blob_bytes(repo, blob_oid)
            candidate = _analyze_python_blob(rel_path, blob_oid, blob_bytes)
            if candidate is not None:
                candidates.append(candidate)

    readme_clues = _extract_readme_clues(repo, blobs)

    if entry_hint is not None:
        normalized_hint = _normalize_rel_path(entry_hint)
        if normalized_hint not in blobs:
            raise DiscoveryError(
                "ENTRYPOINT_NOT_IN_PINNED_TREE",
                f"entrypoint {normalized_hint!r} is not present in pinned commit {commit}",
                evidence_path=normalized_hint,
                expected=f"in tree {tree_oid}",
                actual="absent",
            )
        hint_oid = blobs[normalized_hint]
        hint_bytes = _read_blob_bytes(repo, hint_oid)
        _verify_worktree_matches_blob(repo, normalized_hint, hint_bytes)

        matched = next((c for c in candidates if c.path == normalized_hint), None)
        if matched is None:
            # Even if the hinted script has no `if __name__ == "__main__"`, parse its AST
            # explicitly to record a top-level function or module symbol.
            try:
                tree = ast.parse(hint_bytes.decode("utf-8"), filename=normalized_hint)
            except (UnicodeDecodeError, SyntaxError) as error:
                raise DiscoveryError(
                    "ENTRYPOINT_SYNTAX_ERROR",
                    f"entrypoint {normalized_hint!r} failed AST parse: {error}",
                    evidence_path=normalized_hint,
                ) from error
            line_count = max(1, len(hint_bytes.splitlines()))
            first_fn = next(
                (
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                None,
            )
            symbol = first_fn.name if first_fn is not None else "<module>"
            line_range = (
                (first_fn.lineno, getattr(first_fn, "end_lineno", first_fn.lineno) or first_fn.lineno)
                if first_fn is not None
                else (1, line_count)
            )
            matched = EntrypointCandidate(
                path=normalized_hint,
                blob_oid=hint_oid,
                blob_sha256=hashlib.sha256(hint_bytes).hexdigest(),
                symbol=symbol,
                line_range=line_range,
                has_main_guard=False,
                cli_flags=_extract_cli_flags(tree),
            )
            candidates.append(matched)
            candidates.sort(key=lambda item: item.path)

        return DiscoveryResult(
            commit=commit,
            tree_oid=tree_oid,
            status="resolved",
            selected=matched,
            candidates=tuple(candidates),
            configs=tuple(configs),
            blocked_reasons=(),
            readme_clues=readme_clues,
        )

    # No explicit hint: resolve uniquely or fail closed with structured ambiguity evidence.
    if len(candidates) == 1:
        sole = candidates[0]
        sole_bytes = _read_blob_bytes(repo, sole.blob_oid)
        _verify_worktree_matches_blob(repo, sole.path, sole_bytes)
        return DiscoveryResult(
            commit=commit,
            tree_oid=tree_oid,
            status="resolved",
            selected=sole,
            candidates=tuple(candidates),
            configs=tuple(configs),
            blocked_reasons=(),
            readme_clues=readme_clues,
        )

    reason = "ENTRYPOINT_NOT_FOUND" if not candidates else "ENTRYPOINT_AMBIGUOUS"
    return DiscoveryResult(
        commit=commit,
        tree_oid=tree_oid,
        status="blocked",
        selected=None,
        candidates=tuple(candidates),
        configs=tuple(configs),
        blocked_reasons=(reason,),
        readme_clues=readme_clues,
    )
