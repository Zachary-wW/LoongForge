"""Isolated read-only startup probe with pre/post source immutability verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeAlias

from .source import (
    SourceError,
    make_tree_writable,
    materialize_readonly_source_copy,
    snapshot_source,
    verify_source_unchanged,
)

ProbeStatus: TypeAlias = Literal[
    "not_run",
    "cli_parsed",
    "passed",
    "failed",
    "blocked",
]

_HELP_FLAGS: frozenset[str] = frozenset({"-h", "--help", "-?", "/?"})
_RESERVED_ISOLATION_ENV_KEYS: frozenset[str] = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "PYTHONPYCACHEPREFIX",
        "PYTHONDONTWRITEBYTECODE",
        "INTAKE_PROBE_OUTPUT_DIR",
        "INTAKE_PROBE_NONCE",
        "INTAKE_PROBE_COMMIT",
        "INTAKE_PROBE_ENTRYPOINT",
        "INTAKE_PROBE_ARGV_SHA256",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    }
)
_PYTHON_EXEC_FLAGS: frozenset[str] = frozenset({"-c", "-m"})
_ALLOWED_INTERPRETER_RE = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_PROBE_MARKER_FILENAME = "probe_marker.json"
_PROBE_MARKER_EVENT = "startup_probe_ok"
_SENSITIVE_ENV_TOKENS: tuple[str, ...] = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Recorded evidence from an isolated startup probe (or `not_run`)."""

    status: ProbeStatus
    argv: tuple[str, ...]
    cwd: str
    sanitized_env: dict[str, str]
    exit_code: int | None
    observed_scope: tuple[str, ...]
    source_unchanged: bool
    before_identity_sha256: str | None
    after_identity_sha256: str | None
    stdout_path: str | None = None
    stderr_path: str | None = None
    stdout_sha256: str | None = None
    stderr_sha256: str | None = None
    entrypoint_path: str | None = None
    marker_path: str | None = None
    marker_sha256: str | None = None
    marker_payload: dict[str, Any] | None = None
    argv_sha256: str | None = None
    probe_nonce: str | None = None
    reason_codes: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "sanitized_env": dict(self.sanitized_env),
            "exit_code": self.exit_code,
            "observed_scope": list(self.observed_scope),
            "source_unchanged": self.source_unchanged,
            "before_identity_sha256": self.before_identity_sha256,
            "after_identity_sha256": self.after_identity_sha256,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "entrypoint_path": self.entrypoint_path,
            "marker_path": self.marker_path,
            "marker_sha256": self.marker_sha256,
            "marker_payload": self.marker_payload,
            "argv_sha256": self.argv_sha256,
            "probe_nonce": self.probe_nonce,
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
        }


def sanitize_env(env: Mapping[str, str]) -> dict[str, str]:
    """Remove potentially sensitive environment variables before persisting receipts."""
    sanitized: dict[str, str] = {}
    for key, value in sorted(env.items()):
        upper = key.upper()
        if any(token in upper for token in _SENSITIVE_ENV_TOKENS):
            continue
        sanitized[key] = str(value)
    return sanitized


def _is_help_invocation(argv: Sequence[str]) -> bool:
    return any(arg in _HELP_FLAGS for arg in argv[1:])


def compute_argv_sha256(argv: Sequence[str]) -> str:
    encoded = json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_trusted_interpreter(executable: str) -> str | None:
    """Resolve `executable` to the controlled Python binary path or return None.

    - Bare `python` / `python3` / `python3.X` is NEVER resolved via task PATH; it is
      strictly mapped to the current controlled `os.path.realpath(sys.executable)`.
    - Explicit paths must match `python[3[.X]]` and resolve via `os.path.realpath`
      strictly to `os.path.realpath(sys.executable)`. All other explicit paths
      (including writable system prefixes or compiled/script wrappers) are rejected.
    """
    if not executable or not executable.strip():
        return None
    exec_name = Path(executable).name
    if not _ALLOWED_INTERPRETER_RE.match(exec_name):
        return None

    current_python = os.path.realpath(sys.executable)
    if "/" in executable or "\\" in executable:
        exec_path = Path(executable)
        try:
            resolved_str = os.path.realpath(exec_path)
            resolved = Path(resolved_str)
            if not resolved.is_file():
                return None
        except OSError:
            return None
        if resolved_str == current_python:
            return current_python
        return None

    # Bare interpreter name (`python`, `python3`, etc.): bind strictly to controlled `sys.executable`.
    return current_python


def _is_trusted_interpreter(executable: str) -> bool:
    return resolve_trusted_interpreter(executable) is not None


def validate_probe_argv(
    argv: Sequence[str],
    *,
    entrypoint_path: str | None,
    cwd: str = ".",
) -> tuple[bool, str | None, str]:
    """Verify `argv[0]` is a trusted Python interpreter and `argv` invokes `entrypoint_path`."""
    if not argv:
        return False, "EMPTY_PROBE_ARGV", "probe argv must be non-empty"

    if not _is_trusted_interpreter(argv[0]):
        return (
            False,
            "UNTRUSTED_PROBE_LAUNCHER",
            (
                f"probe argv[0]={argv[0]!r} is not an allowed Python interpreter "
                "(arbitrary wrapper scripts are rejected)"
            ),
        )

    if entrypoint_path is None:
        return True, None, "no entrypoint_path constraint supplied"

    expected_posix = PurePosixPath(entrypoint_path).as_posix()
    cwd_posix = PurePosixPath(cwd)
    if cwd_posix.as_posix() in {"", "."}:
        expected_from_cwd = expected_posix
    else:
        try:
            expected_from_cwd = PurePosixPath(expected_posix).relative_to(cwd_posix).as_posix()
        except ValueError:
            expected_from_cwd = expected_posix

    for token in argv[1:]:
        if token == "--":
            break
        if token in _PYTHON_EXEC_FLAGS:
            return (
                False,
                "PROBE_ARGV_ENTRYPOINT_MISMATCH",
                f"probe argv uses disallowed inline/module flag {token!r} instead of {expected_posix!r}",
            )

    script_arg: str | None = None
    for token in argv[1:]:
        if token.startswith("-"):
            continue
        script_arg = token
        break

    if script_arg is None:
        return (
            False,
            "PROBE_ARGV_ENTRYPOINT_MISMATCH",
            f"probe argv does not reference discovered entrypoint {expected_posix!r}",
        )

    normalized_arg = PurePosixPath(script_arg.replace("\\", "/")).as_posix()
    if normalized_arg not in {expected_posix, expected_from_cwd}:
        return (
            False,
            "PROBE_ARGV_ENTRYPOINT_MISMATCH",
            (
                f"probe argv script {script_arg!r} does not match discovered entrypoint "
                f"{expected_posix!r}"
            ),
        )
    return True, None, "argv matches discovered entrypoint"


def _parse_probe_marker(
    out_dir: Path,
    *,
    expected_commit: str,
    expected_entrypoint: str | None,
    expected_argv_sha256: str,
    expected_nonce: str,
) -> tuple[dict[str, Any] | None, Path | None, str | None, str | None]:
    """Require a structured JSON `probe_marker.json` bound to commit, entrypoint, argv, and nonce."""
    marker_file = out_dir / _PROBE_MARKER_FILENAME
    if not marker_file.is_file() or marker_file.is_symlink():
        return None, None, None, "PROBE_MARKER_MISSING"
    raw = marker_file.read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return None, marker_file, None, "PROBE_MARKER_INVALID"
    if not isinstance(parsed, dict) or parsed.get("event") != _PROBE_MARKER_EVENT:
        return None, marker_file, None, "PROBE_MARKER_INVALID"
    if parsed.get("source_commit") != expected_commit:
        return None, marker_file, None, "PROBE_MARKER_BINDING_MISMATCH"
    if expected_entrypoint is not None and parsed.get("entrypoint_path") != expected_entrypoint:
        return None, marker_file, None, "PROBE_MARKER_BINDING_MISMATCH"
    if parsed.get("argv_sha256") != expected_argv_sha256:
        return None, marker_file, None, "PROBE_MARKER_BINDING_MISMATCH"
    if parsed.get("probe_nonce") != expected_nonce:
        return None, marker_file, None, "PROBE_MARKER_BINDING_MISMATCH"
    return parsed, marker_file, hashlib.sha256(raw).hexdigest(), None


def run_startup_probe(
    *,
    repo_root: str | Path,
    expected_commit: str,
    argv: Sequence[str],
    cwd: str = ".",
    env: Mapping[str, str] | None = None,
    output_dir: str | Path,
    entrypoint_path: str | None = None,
    timeout_sec: float = 60.0,
    use_readonly_copy: bool = True,
) -> ProbeResult:
    """Execute a controlled startup probe in an isolated output/cache directory.

    Hard guarantees:
    - Validates `argv` genuinely targets `entrypoint_path` (rejecting `python -c ...`).
    - Rejects task env attempts to override reserved isolation variables.
    - Materializes a chmod-read-only offline checkout copy so transient or persistent
      writes to the source tree fail at OS permission level AND snapshots source before/after.
    - `--help` / `-h` only yields `status="cli_parsed"` and `observed_scope=("cli_parsed",)`.
    - Exit code 0 alone is NEVER sufficient: requires structured `probe_marker.json`
      with `event == "startup_probe_ok"` inside `output_dir`.
    """
    argv_tuple = tuple(argv)
    argv_sha = compute_argv_sha256(argv_tuple) if argv_tuple else None
    resolved_interpreter = (
        resolve_trusted_interpreter(argv_tuple[0]) if argv_tuple else None
    )
    argv_ok, argv_reason, argv_detail = validate_probe_argv(
        argv_tuple,
        entrypoint_path=entrypoint_path,
        cwd=cwd,
    )
    if not argv_ok:
        return ProbeResult(
            status="blocked",
            argv=argv_tuple,
            cwd=cwd,
            sanitized_env=sanitize_env(env or {}),
            exit_code=None,
            observed_scope=(),
            source_unchanged=True,
            before_identity_sha256=None,
            after_identity_sha256=None,
            entrypoint_path=entrypoint_path,
            argv_sha256=argv_sha,
            reason_codes=(argv_reason or "PROBE_ARGV_ENTRYPOINT_MISMATCH",),
            detail=argv_detail,
        )

    repo = Path(repo_root).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    tmp_dir = out_dir / "tmp"
    pycache_dir = out_dir / "pycache"
    for directory in (cache_dir, tmp_dir, pycache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    before = snapshot_source(repo, expected_commit, require_clean=True)

    declared_env = dict(env or {})
    forbidden_overrides = sorted(
        key
        for key in declared_env
        if key.upper() in _RESERVED_ISOLATION_ENV_KEYS
    )
    if forbidden_overrides:
        return ProbeResult(
            status="blocked",
            argv=argv_tuple,
            cwd=cwd,
            sanitized_env=sanitize_env(declared_env),
            exit_code=None,
            observed_scope=(),
            source_unchanged=True,
            before_identity_sha256=before.identity_sha256,
            after_identity_sha256=before.identity_sha256,
            entrypoint_path=entrypoint_path,
            reason_codes=("RESERVED_ENV_OVERRIDE",),
            detail=(
                "task env cannot override reserved isolation variables: "
                + ", ".join(forbidden_overrides)
            ),
        )

    resolved_cwd = (repo / cwd).resolve()
    try:
        rel_cwd = resolved_cwd.relative_to(repo)
    except ValueError:
        return ProbeResult(
            status="blocked",
            argv=argv_tuple,
            cwd=cwd,
            sanitized_env=sanitize_env(env or {}),
            exit_code=None,
            observed_scope=(),
            source_unchanged=True,
            before_identity_sha256=before.identity_sha256,
            after_identity_sha256=before.identity_sha256,
            entrypoint_path=entrypoint_path,
            reason_codes=("PROBE_CWD_ESCAPES_REPO",),
            detail=f"probe cwd {cwd!r} escapes source repo root",
        )

    probe_nonce = secrets.token_hex(16)
    assert argv_sha is not None
    run_env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    }
    run_env.update(declared_env)
    run_env.update(
        {
            "HOME": str(out_dir),
            "TMPDIR": str(tmp_dir),
            "XDG_CACHE_HOME": str(cache_dir),
            "PYTHONPYCACHEPREFIX": str(pycache_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "INTAKE_PROBE_OUTPUT_DIR": str(out_dir),
            "INTAKE_PROBE_NONCE": probe_nonce,
            "INTAKE_PROBE_COMMIT": expected_commit,
            "INTAKE_PROBE_ENTRYPOINT": entrypoint_path or "",
            "INTAKE_PROBE_ARGV_SHA256": argv_sha,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HTTP_PROXY": "http://127.0.0.1:0",
            "HTTPS_PROXY": "http://127.0.0.1:0",
            "ALL_PROXY": "http://127.0.0.1:0",
            "NO_PROXY": "",
        }
    )
    recorded_env = sanitize_env(declared_env)

    stdout_file = out_dir / "probe.stdout.log"
    stderr_file = out_dir / "probe.stderr.log"
    marker_existing = out_dir / _PROBE_MARKER_FILENAME
    if marker_existing.exists():
        marker_existing.unlink()

    timed_out = False
    exit_code: int | None
    ro_temp_parent: Path | None = None
    ro_copy_dir: Path | None = None
    exec_cwd = resolved_cwd
    assert resolved_interpreter is not None
    exec_argv = [resolved_interpreter, *argv_tuple[1:]]
    try:
        if use_readonly_copy:
            ro_temp_parent = Path(tempfile.mkdtemp(prefix="intake-ro-src-"))
            ro_copy_dir = materialize_readonly_source_copy(
                repo,
                before,
                ro_temp_parent / "src",
            )
            exec_cwd = (ro_copy_dir / rel_cwd).resolve()

        proc = subprocess.run(
            exec_argv,
            cwd=exec_cwd,
            env=run_env,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        stdout_bytes = proc.stdout
        stderr_bytes = proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_bytes = exc.stdout if isinstance(exc.stdout, bytes) else b""
        stderr_bytes = exc.stderr if isinstance(exc.stderr, bytes) else b""
        exit_code = None
    except OSError as exc:
        stdout_bytes = b""
        stderr_bytes = str(exc).encode("utf-8")
        exit_code = None
    finally:
        if ro_copy_dir is not None:
            make_tree_writable(ro_copy_dir)
        if ro_temp_parent is not None and ro_temp_parent.exists():
            shutil.rmtree(ro_temp_parent, ignore_errors=True)

    stdout_file.write_bytes(stdout_bytes)
    stderr_file.write_bytes(stderr_bytes)
    stdout_sha = hashlib.sha256(stdout_bytes).hexdigest()
    stderr_sha = hashlib.sha256(stderr_bytes).hexdigest()

    after = snapshot_source(repo, expected_commit, require_clean=False)
    source_unchanged = True
    try:
        verify_source_unchanged(before, after)
    except SourceError:
        source_unchanged = False

    if not source_unchanged:
        return ProbeResult(
            status="blocked",
            argv=argv_tuple,
            cwd=cwd,
            sanitized_env=recorded_env,
            exit_code=exit_code,
            observed_scope=(),
            source_unchanged=False,
            before_identity_sha256=before.identity_sha256,
            after_identity_sha256=after.identity_sha256,
            stdout_path=str(stdout_file),
            stderr_path=str(stderr_file),
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            entrypoint_path=entrypoint_path,
            reason_codes=("SOURCE_MUTATED_DURING_ANALYSIS",),
            detail="upstream source repository was modified during startup probe",
        )

    if timed_out:
        return ProbeResult(
            status="failed",
            argv=argv_tuple,
            cwd=cwd,
            sanitized_env=recorded_env,
            exit_code=None,
            observed_scope=(),
            source_unchanged=True,
            before_identity_sha256=before.identity_sha256,
            after_identity_sha256=after.identity_sha256,
            stdout_path=str(stdout_file),
            stderr_path=str(stderr_file),
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            reason_codes=("PROBE_TIMEOUT",),
            detail=f"startup probe timed out after {timeout_sec}s",
        )

    if exit_code is None:
        return ProbeResult(
            status="failed",
            argv=argv_tuple,
            cwd=cwd,
            sanitized_env=recorded_env,
            exit_code=None,
            observed_scope=(),
            source_unchanged=True,
            before_identity_sha256=before.identity_sha256,
            after_identity_sha256=after.identity_sha256,
            stdout_path=str(stdout_file),
            stderr_path=str(stderr_file),
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            reason_codes=("PROBE_LAUNCH_ERROR",),
            detail="failed to launch startup probe process",
        )

    if exit_code != 0:
        stderr_text = stderr_bytes.decode("utf-8", errors="ignore")
        if use_readonly_copy and (
            "PermissionError" in stderr_text or "Read-only file system" in stderr_text
        ):
            return ProbeResult(
                status="blocked",
                argv=argv_tuple,
                cwd=cwd,
                sanitized_env=recorded_env,
                exit_code=exit_code,
                observed_scope=(),
                source_unchanged=True,
                before_identity_sha256=before.identity_sha256,
                after_identity_sha256=after.identity_sha256,
                stdout_path=str(stdout_file),
                stderr_path=str(stderr_file),
                stdout_sha256=stdout_sha,
                stderr_sha256=stderr_sha,
                entrypoint_path=entrypoint_path,
                reason_codes=("SOURCE_WRITE_ATTEMPT_BLOCKED",),
                detail="probe attempted to write into read-only materialized source tree",
            )
        return ProbeResult(
            status="failed",
            argv=argv_tuple,
            cwd=cwd,
            sanitized_env=recorded_env,
            exit_code=exit_code,
            observed_scope=(),
            source_unchanged=True,
            before_identity_sha256=before.identity_sha256,
            after_identity_sha256=after.identity_sha256,
            stdout_path=str(stdout_file),
            stderr_path=str(stderr_file),
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            entrypoint_path=entrypoint_path,
            reason_codes=("PROBE_EXIT_NONZERO",),
            detail=f"startup probe exited with code {exit_code}",
        )

    if _is_help_invocation(argv_tuple):
        return ProbeResult(
            status="cli_parsed",
            argv=argv_tuple,
            cwd=cwd,
            sanitized_env=recorded_env,
            exit_code=0,
            observed_scope=("cli_parsed",),
            source_unchanged=True,
            before_identity_sha256=before.identity_sha256,
            after_identity_sha256=after.identity_sha256,
            stdout_path=str(stdout_file),
            stderr_path=str(stderr_file),
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            entrypoint_path=entrypoint_path,
            argv_sha256=argv_sha,
            probe_nonce=probe_nonce,
            reason_codes=(),
            detail="--help only confirms CLI argument parsing, never startup_probe",
        )

    marker_payload, marker_file, marker_sha, marker_err = _parse_probe_marker(
        out_dir,
        expected_commit=expected_commit,
        expected_entrypoint=entrypoint_path,
        expected_argv_sha256=argv_sha,
        expected_nonce=probe_nonce,
    )
    if marker_payload is None or marker_file is None or marker_sha is None:
        return ProbeResult(
            status="blocked",
            argv=argv_tuple,
            cwd=cwd,
            sanitized_env=recorded_env,
            exit_code=0,
            observed_scope=(),
            source_unchanged=True,
            before_identity_sha256=before.identity_sha256,
            after_identity_sha256=after.identity_sha256,
            stdout_path=str(stdout_file),
            stderr_path=str(stderr_file),
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            entrypoint_path=entrypoint_path,
            argv_sha256=argv_sha,
            probe_nonce=probe_nonce,
            reason_codes=(marker_err or "PROBE_MARKER_MISSING",),
            detail=(
                "exit code 0 alone is insufficient: startup probe must emit structured "
                f"{_PROBE_MARKER_FILENAME} bound to commit, entrypoint, argv_sha256, and probe_nonce"
            ),
        )

    return ProbeResult(
        status="passed",
        argv=argv_tuple,
        cwd=cwd,
        sanitized_env=recorded_env,
        exit_code=0,
        observed_scope=("startup_probe",),
        source_unchanged=True,
        before_identity_sha256=before.identity_sha256,
        after_identity_sha256=after.identity_sha256,
        stdout_path=str(stdout_file),
        stderr_path=str(stderr_file),
        stdout_sha256=stdout_sha,
        stderr_sha256=stderr_sha,
        entrypoint_path=entrypoint_path,
        marker_path=str(marker_file),
        marker_sha256=marker_sha,
        marker_payload=marker_payload,
        argv_sha256=argv_sha,
        probe_nonce=probe_nonce,
        reason_codes=(),
        detail="isolated read-only startup probe completed with verified marker",
    )
