"""Git source identity lock, pre/post immutability checks, and offline `git bundle` archive."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SourceError(ValueError):
    """Structured failure raised when upstream source identity cannot be trusted."""

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
class SubmoduleRecord:
    """Identity record for one recursive Git submodule."""

    path: str
    url: str | None
    gitlink_commit: str
    head_commit: str | None
    dirty: bool
    initialized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "url": self.url,
            "gitlink_commit": self.gitlink_commit,
            "head_commit": self.head_commit,
            "dirty": self.dirty,
            "initialized": self.initialized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubmoduleRecord:
        return cls(
            path=str(data["path"]),
            url=data.get("url"),
            gitlink_commit=str(data["gitlink_commit"]),
            head_commit=data.get("head_commit"),
            dirty=bool(data["dirty"]),
            initialized=bool(data["initialized"]),
        )


@dataclass(frozen=True, slots=True)
class WorktreeStatus:
    """Categorized Git porcelain worktree status."""

    staged: tuple[str, ...]
    unstaged: tuple[str, ...]
    untracked: tuple[str, ...]
    ignored: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not (self.staged or self.unstaged or self.untracked or self.ignored)

    def is_clean_with_allowlist(self, allowed_ignored: Iterable[str] = ()) -> bool:
        allow = set(allowed_ignored)
        disallowed_ignored = [item for item in self.ignored if item not in allow]
        return not (
            self.staged or self.unstaged or self.untracked or disallowed_ignored
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "staged": list(self.staged),
            "unstaged": list(self.unstaged),
            "untracked": list(self.untracked),
            "ignored": list(self.ignored),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorktreeStatus:
        return cls(
            staged=tuple(data.get("staged", ())),
            unstaged=tuple(data.get("unstaged", ())),
            untracked=tuple(data.get("untracked", ())),
            ignored=tuple(data.get("ignored", ())),
        )


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Canonical identity snapshot of an upstream Git repository."""

    repo_root: str
    head_commit: str
    tree_oid: str
    object_format: str
    submodules: tuple[SubmoduleRecord, ...]
    worktree_status: WorktreeStatus
    worktree_digest: str
    identity_sha256: str
    allowed_ignored: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        return self.worktree_status.is_clean_with_allowlist(
            self.allowed_ignored
        ) and all(
            sub.initialized
            and not sub.dirty
            and sub.head_commit == sub.gitlink_commit
            for sub in self.submodules
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "head_commit": self.head_commit,
            "tree_oid": self.tree_oid,
            "object_format": self.object_format,
            "submodules": [sub.to_dict() for sub in self.submodules],
            "worktree_status": self.worktree_status.to_dict(),
            "worktree_digest": self.worktree_digest,
            "identity_sha256": self.identity_sha256,
            "allowed_ignored": list(self.allowed_ignored),
            "is_clean": self.is_clean,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSnapshot:
        return cls(
            repo_root=str(data["repo_root"]),
            head_commit=str(data["head_commit"]),
            tree_oid=str(data["tree_oid"]),
            object_format=str(data["object_format"]),
            submodules=tuple(
                SubmoduleRecord.from_dict(item) for item in data.get("submodules", ())
            ),
            worktree_status=WorktreeStatus.from_dict(data["worktree_status"]),
            worktree_digest=str(data["worktree_digest"]),
            identity_sha256=str(data["identity_sha256"]),
            allowed_ignored=tuple(data.get("allowed_ignored", ())),
        )


@dataclass(frozen=True, slots=True)
class SourceBundleInfo:
    """Metadata for a verified offline `git bundle` artifact."""

    bundle_path: str
    commit: str
    tree_oid: str
    sha256: str
    size_bytes: int
    rel_path: str | None = None
    submodule_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": self.bundle_path,
            "rel_path": self.rel_path,
            "submodule_path": self.submodule_path,
            "commit": self.commit,
            "tree_oid": self.tree_oid,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _run_git(
    repo_root: Path,
    *args: str,
    check: bool = True,
    binary: bool = False,
) -> subprocess.CompletedProcess[Any]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        env=env,
        text=not binary,
        capture_output=True,
        check=check,
    )


def _parse_worktree_status(repo_root: Path) -> WorktreeStatus:
    proc = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
    )
    raw = proc.stdout
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    ignored: list[str] = []

    entries = raw.split("\0")
    idx = 0
    while idx < len(entries):
        entry = entries[idx]
        idx += 1
        if not entry:
            continue
        code = entry[:2]
        path = entry[3:]
        if code[0] in {"R", "C"} and idx < len(entries):
            # Skip original path token for renames/copies in -z format.
            idx += 1
        if code == "??":
            untracked.append(path)
        elif code == "!!":
            ignored.append(path)
        else:
            if code[0] != " ":
                staged.append(path)
            if code[1] != " ":
                unstaged.append(path)

    return WorktreeStatus(
        staged=tuple(sorted(staged)),
        unstaged=tuple(sorted(unstaged)),
        untracked=tuple(sorted(untracked)),
        ignored=tuple(sorted(ignored)),
    )


def _read_submodule_urls(repo_root: Path) -> dict[str, str]:
    gitmodules = repo_root / ".gitmodules"
    if not gitmodules.is_file():
        return {}
    proc = _run_git(
        repo_root,
        "config",
        "--file",
        str(gitmodules),
        "--get-regexp",
        r"^submodule\..*\.(path|url)$",
        check=False,
    )
    if proc.returncode != 0:
        return {}
    by_name: dict[str, dict[str, str]] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(" ")
        parts = key.split(".")
        if len(parts) < 3:
            continue
        field = parts[-1]
        name = ".".join(parts[1:-1])
        by_name.setdefault(name, {})[field] = value.strip()
    return {
        info["path"]: info["url"]
        for info in by_name.values()
        if "path" in info and "url" in info
    }


def _collect_submodules(
    repo_root: Path,
    commit: str,
    *,
    prefix: str = "",
) -> list[SubmoduleRecord]:
    ls_proc = _run_git(repo_root, "ls-tree", "-r", "-z", commit)
    urls = _read_submodule_urls(repo_root)
    records: list[SubmoduleRecord] = []

    for item in ls_proc.stdout.split("\0"):
        if not item:
            continue
        meta, _, rel_path = item.partition("\t")
        parts = meta.split()
        if len(parts) < 3:
            continue
        mode, obj_type, oid = parts[0], parts[1], parts[2]
        if mode != "160000" or obj_type != "commit":
            continue

        full_rel = f"{prefix}{rel_path}" if prefix else rel_path
        sub_dir = repo_root / rel_path
        initialized = False
        head_commit: str | None = None
        dirty = False

        if sub_dir.is_dir() and (sub_dir / ".git").exists():
            head_proc = _run_git(sub_dir, "rev-parse", "HEAD", check=False)
            if head_proc.returncode == 0:
                initialized = True
                head_commit = head_proc.stdout.strip()
                sub_status = _parse_worktree_status(sub_dir)
                dirty = not sub_status.is_clean

        records.append(
            SubmoduleRecord(
                path=full_rel,
                url=urls.get(rel_path),
                gitlink_commit=oid,
                head_commit=head_commit,
                dirty=dirty,
                initialized=initialized,
            )
        )
        if initialized and head_commit is not None:
            records.extend(
                _collect_submodules(
                    sub_dir,
                    head_commit,
                    prefix=f"{full_rel}/",
                )
            )

    records.sort(key=lambda rec: rec.path)
    return records


def _hash_worktree_contents(repo_root: Path) -> str:
    """Compute a deterministic digest of all non-.git filesystem entries under `repo_root`."""
    digest = hashlib.sha256()
    entries: list[str] = []
    for root, dir_names, file_names in os.walk(
        repo_root,
        topdown=True,
        followlinks=False,
    ):
        dir_names[:] = sorted(d for d in dir_names if d != ".git")
        root_path = Path(root)
        for fname in sorted(file_names):
            if fname == ".git":
                continue
            full_path = root_path / fname
            rel = full_path.relative_to(repo_root).as_posix()
            entries.append(rel)

    for rel in sorted(entries):
        full_path = repo_root / rel
        if full_path.is_symlink():
            target = os.readlink(full_path)
            digest.update(f"L:{rel}:{target}\n".encode("utf-8"))
        elif full_path.is_file():
            file_hash = _sha256_file(full_path)
            digest.update(f"F:{rel}:{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_source(
    repo_root: str | Path,
    expected_commit: str,
    *,
    require_clean: bool = True,
    allowed_ignored: Iterable[str] = (),
) -> SourceSnapshot:
    """Capture and verify canonical Git source identity without mutating the repository."""
    raw_root = Path(repo_root)
    if not raw_root.exists() or not raw_root.is_dir():
        raise SourceError(
            "SOURCE_NOT_DIRECTORY",
            f"source repo_root does not exist or is not a directory: {raw_root}",
            evidence_path=str(raw_root),
        )

    top_proc = _run_git(raw_root, "rev-parse", "--show-toplevel", check=False)
    if top_proc.returncode != 0:
        raise SourceError(
            "SOURCE_NOT_GIT_REPO",
            f"source directory is not a Git repository: {raw_root}",
            evidence_path=str(raw_root),
        )
    canonical_root = Path(top_proc.stdout.strip()).resolve()

    head_proc = _run_git(canonical_root, "rev-parse", "--verify", "HEAD^{commit}", check=False)
    if head_proc.returncode != 0:
        raise SourceError(
            "SOURCE_HEAD_MISSING",
            "source repository has no valid HEAD commit",
            evidence_path=str(canonical_root),
        )
    head_commit = head_proc.stdout.strip()
    if head_commit != expected_commit:
        raise SourceError(
            "SOURCE_COMMIT_MISMATCH",
            f"source HEAD {head_commit} does not match expected commit {expected_commit}",
            evidence_path="inputs/source_fingerprint.json",
            expected=expected_commit,
            actual=head_commit,
        )

    tree_oid = _run_git(canonical_root, "rev-parse", f"{head_commit}^{{tree}}").stdout.strip()
    fmt_proc = _run_git(
        canonical_root,
        "rev-parse",
        "--show-object-format",
        check=False,
    )
    object_format = fmt_proc.stdout.strip() if fmt_proc.returncode == 0 else "sha1"

    worktree_status = _parse_worktree_status(canonical_root)
    submodules = tuple(_collect_submodules(canonical_root, head_commit))
    worktree_digest = _hash_worktree_contents(canonical_root)
    allowed_ignored_tuple = tuple(sorted(set(allowed_ignored)))

    # Host-independent identity hash: intentionally excludes host absolute `repo_root`
    # and submodule URLs (which may be local paths) so offline bundle restore on a
    # different path or machine preserves the exact same `identity_sha256`.
    identity_payload = {
        "head_commit": head_commit,
        "tree_oid": tree_oid,
        "object_format": object_format,
        "submodules": [
            {
                "path": sub.path,
                "gitlink_commit": sub.gitlink_commit,
                "head_commit": sub.head_commit,
                "dirty": sub.dirty,
                "initialized": sub.initialized,
            }
            for sub in submodules
        ],
        "worktree_status": worktree_status.to_dict(),
        "worktree_digest": worktree_digest,
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    snapshot = SourceSnapshot(
        repo_root=str(canonical_root),
        head_commit=head_commit,
        tree_oid=tree_oid,
        object_format=object_format,
        submodules=submodules,
        worktree_status=worktree_status,
        worktree_digest=worktree_digest,
        identity_sha256=identity_sha256,
        allowed_ignored=allowed_ignored_tuple,
    )

    if require_clean:
        for sub in submodules:
            if not sub.initialized:
                raise SourceError(
                    "SOURCE_SUBMODULE_UNINITIALIZED",
                    f"submodule {sub.path!r} is not initialized",
                    evidence_path=sub.path,
                    expected=sub.gitlink_commit,
                    actual=None,
                )
            if sub.head_commit != sub.gitlink_commit:
                raise SourceError(
                    "SOURCE_SUBMODULE_MISMATCH",
                    (
                        f"submodule {sub.path!r} HEAD {sub.head_commit} != "
                        f"gitlink {sub.gitlink_commit}"
                    ),
                    evidence_path=sub.path,
                    expected=sub.gitlink_commit,
                    actual=sub.head_commit,
                )
            if sub.dirty:
                raise SourceError(
                    "SOURCE_SUBMODULE_DIRTY",
                    f"submodule {sub.path!r} has uncommitted changes",
                    evidence_path=sub.path,
                )
        if worktree_status.staged or worktree_status.unstaged:
            raise SourceError(
                "SOURCE_WORKTREE_DIRTY",
                "upstream source repository has staged or unstaged modifications",
                evidence_path="inputs/source_fingerprint.json",
                expected="clean",
                actual={
                    "staged": list(worktree_status.staged),
                    "unstaged": list(worktree_status.unstaged),
                },
            )
        if worktree_status.untracked:
            raise SourceError(
                "SOURCE_WORKTREE_UNTRACKED",
                "upstream source repository contains untracked files",
                evidence_path="inputs/source_fingerprint.json",
                expected=[],
                actual=list(worktree_status.untracked),
            )
        disallowed_ignored = [
            item
            for item in worktree_status.ignored
            if item not in allowed_ignored_tuple
        ]
        if disallowed_ignored:
            raise SourceError(
                "SOURCE_WORKTREE_IGNORED",
                (
                    "upstream source repository contains non-allowlisted ignored files: "
                    + ", ".join(disallowed_ignored)
                ),
                evidence_path="inputs/source_fingerprint.json",
                expected=list(allowed_ignored_tuple),
                actual=disallowed_ignored,
            )

    return snapshot


def verify_source_unchanged(
    before: SourceSnapshot,
    after: SourceSnapshot,
) -> None:
    """Assert that the upstream source repository was completely untouched during analysis."""
    if (
        before.head_commit != after.head_commit
        or before.tree_oid != after.tree_oid
        or before.worktree_digest != after.worktree_digest
        or before.identity_sha256 != after.identity_sha256
    ):
        raise SourceError(
            "SOURCE_MUTATED_DURING_ANALYSIS",
            "upstream source repository changed between pre- and post-analysis snapshots",
            evidence_path="inputs/source_fingerprint.json",
            expected=before.identity_sha256,
            actual=after.identity_sha256,
        )


def create_source_bundle(
    repo_root: str | Path,
    commit: str,
    bundle_path: str | Path,
    *,
    rel_path: str | None = None,
    submodule_path: str | None = None,
) -> SourceBundleInfo:
    """Create an offline `git bundle` containing `commit` without touching `repo_root`."""
    repo = Path(repo_root).resolve()
    verify_proc = _run_git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}", check=False)
    if verify_proc.returncode != 0:
        raise SourceError(
            "SOURCE_COMMIT_NOT_FOUND",
            f"commit {commit} does not exist in repository {repo}",
            evidence_path=str(repo),
            expected=commit,
            actual=None,
        )
    resolved_commit = verify_proc.stdout.strip()
    tree_oid = _run_git(repo, "rev-parse", f"{resolved_commit}^{{tree}}").stdout.strip()

    target = Path(bundle_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Create bundle via a temporary bare clone/fetch ref so the source repo stays untouched
    # and bundling works for both full and shallow (`--depth 1`) repositories even when
    # `commit` is not the current HEAD or a branch tip.
    with tempfile.TemporaryDirectory(prefix="intake-bundle-") as tmp_dir:
        stage_repo = Path(tmp_dir) / "stage.git"
        _run_git(Path(tmp_dir), "init", "--bare", str(stage_repo))
        fetch_proc = _run_git(
            stage_repo,
            "fetch",
            "--no-tags",
            "--update-shallow",
            str(repo),
            f"{resolved_commit}:refs/heads/pinned",
            check=False,
        )
        if fetch_proc.returncode != 0:
            raise SourceError(
                "BUNDLE_CREATE_FAILED",
                f"failed to fetch pinned commit {resolved_commit}: {fetch_proc.stderr.strip()}",
                evidence_path=str(repo),
                expected=resolved_commit,
                actual=fetch_proc.stderr.strip(),
            )
        _run_git(stage_repo, "bundle", "create", str(target), "refs/heads/pinned")

    bundle_sha = _sha256_file(target)
    return SourceBundleInfo(
        bundle_path=str(target),
        rel_path=rel_path,
        submodule_path=submodule_path,
        commit=resolved_commit,
        tree_oid=tree_oid,
        sha256=bundle_sha,
        size_bytes=target.stat().st_size,
    )


def create_recursive_source_bundles(
    repo_root: str | Path,
    snapshot: SourceSnapshot,
    output_dir: str | Path,
    *,
    bundle_rel_prefix: str = "evidence/source_bundles",
) -> dict[str, Any]:
    """Create offline `git bundle` archives for root and all recursive submodules."""
    repo = Path(repo_root).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root_filename = "root.bundle"
    root_rel = f"{bundle_rel_prefix}/{root_filename}"
    root_info = create_source_bundle(
        repo,
        snapshot.head_commit,
        out_dir / root_filename,
        rel_path=root_rel,
        submodule_path=None,
    )

    sub_infos: list[dict[str, Any]] = []
    for idx, sub in enumerate(snapshot.submodules):
        sub_repo = repo / sub.path
        safe_name = f"submodule_{idx}_{sub.path.replace('/', '__')}.bundle"
        sub_rel = f"{bundle_rel_prefix}/{safe_name}"
        sub_info = create_source_bundle(
            sub_repo,
            sub.gitlink_commit,
            out_dir / safe_name,
            rel_path=sub_rel,
            submodule_path=sub.path,
        )
        sub_infos.append(sub_info.to_dict())

    return {
        "root": root_info.to_dict(),
        "submodules": sub_infos,
    }


def restore_from_bundle_manifest(
    package_dir: str | Path,
    bundles_manifest: dict[str, Any],
    restore_dir: str | Path,
) -> SourceSnapshot:
    """Offline-restore root and recursive submodules from packaged bundles and return snapshot."""
    pkg = Path(package_dir).resolve()
    dest = Path(restore_dir).resolve()
    root_entry = bundles_manifest["root"]
    root_bundle = pkg / str(root_entry["rel_path"])
    verify_source_bundle(
        root_bundle,
        expected_commit=str(root_entry["commit"]),
        expected_sha256=str(root_entry["sha256"]),
        restore_dir=dest,
    )

    for sub_entry in bundles_manifest.get("submodules", []):
        sub_rel = str(sub_entry["rel_path"])
        sub_path = str(sub_entry["submodule_path"])
        sub_bundle = pkg / sub_rel
        sub_dest = dest / sub_path
        if sub_dest.exists() and not any(sub_dest.iterdir()):
            sub_dest.rmdir()
        verify_source_bundle(
            sub_bundle,
            expected_commit=str(sub_entry["commit"]),
            expected_sha256=str(sub_entry["sha256"]),
            restore_dir=sub_dest,
        )

    return snapshot_source(
        dest,
        str(root_entry["commit"]),
        require_clean=True,
    )


def materialize_readonly_source_copy(
    repo_root: str | Path,
    snapshot: SourceSnapshot,
    dest_dir: str | Path,
) -> Path:
    """Materialize a clean, read-only offline checkout copy of `repo_root` at `dest_dir`."""
    dest = Path(dest_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="intake-ro-bundle-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bundles = create_recursive_source_bundles(
            repo_root,
            snapshot,
            tmp_path / "bundles",
            bundle_rel_prefix="bundles",
        )
        restore_from_bundle_manifest(tmp_path, bundles, dest)

    # Strip write permissions recursively on non-.git files and directories (bottom-up).
    for root, dir_names, file_names in os.walk(dest, topdown=False, followlinks=False):
        root_path = Path(root)
        for fname in file_names:
            fpath = root_path / fname
            if fpath.is_symlink():
                continue
            mode = fpath.lstat().st_mode
            os.chmod(fpath, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        for dname in dir_names:
            dpath = root_path / dname
            if dpath.is_symlink():
                continue
            mode = dpath.lstat().st_mode
            os.chmod(dpath, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root_mode = dest.lstat().st_mode
    os.chmod(dest, root_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return dest


def make_tree_writable(path: str | Path) -> None:
    """Restore owner write bits so temporary read-only trees can be cleaned up cleanly."""
    root_target = Path(path)
    if not root_target.exists():
        return
    for root, dir_names, file_names in os.walk(root_target, topdown=True, followlinks=False):
        root_path = Path(root)
        try:
            os.chmod(root_path, root_path.lstat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
        except OSError:
            pass
        for dname in dir_names:
            dpath = root_path / dname
            if not dpath.is_symlink():
                try:
                    os.chmod(dpath, dpath.lstat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
                except OSError:
                    pass
        for fname in file_names:
            fpath = root_path / fname
            if not fpath.is_symlink():
                try:
                    os.chmod(fpath, fpath.lstat().st_mode | stat.S_IWUSR)
                except OSError:
                    pass


def verify_source_bundle(
    bundle_path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str | None = None,
    restore_dir: str | Path | None = None,
) -> SourceBundleInfo:
    """Verify an offline `git bundle` digest, integrity (`git bundle verify`), and commit restore."""
    target = Path(bundle_path).resolve()
    if not target.is_file():
        raise SourceError(
            "BUNDLE_MISSING",
            f"source bundle does not exist: {target}",
            evidence_path=str(target),
        )

    actual_sha = _sha256_file(target)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise SourceError(
            "BUNDLE_DIGEST_MISMATCH",
            "source bundle SHA-256 does not match expected digest",
            evidence_path=str(target),
            expected=expected_sha256,
            actual=actual_sha,
        )

    with tempfile.TemporaryDirectory(prefix="intake-bundle-verify-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        check_repo = tmp_path / "check"
        check_repo.mkdir()
        _run_git(check_repo, "init")
        verify_proc = _run_git(check_repo, "bundle", "verify", str(target), check=False)
        if verify_proc.returncode != 0:
            raise SourceError(
                "BUNDLE_VERIFY_FAILED",
                f"git bundle verify failed: {verify_proc.stderr.strip()}",
                evidence_path=str(target),
            )

        list_proc = _run_git(check_repo, "bundle", "list-heads", str(target), check=False)
        if list_proc.returncode != 0:
            raise SourceError(
                "BUNDLE_LIST_FAILED",
                f"git bundle list-heads failed: {list_proc.stderr.strip()}",
                evidence_path=str(target),
            )
        heads = [
            line.split()[0]
            for line in list_proc.stdout.splitlines()
            if line.strip()
        ]
        if expected_commit not in heads:
            raise SourceError(
                "BUNDLE_COMMIT_MISMATCH",
                f"expected commit {expected_commit} not advertised by bundle",
                evidence_path=str(target),
                expected=expected_commit,
                actual=heads,
            )

        clone_target = (
            Path(restore_dir).resolve()
            if restore_dir is not None
            else tmp_path / "restored"
        )
        if clone_target.exists():
            raise SourceError(
                "RESTORE_DIR_EXISTS",
                f"restore_dir already exists: {clone_target}",
                evidence_path=str(clone_target),
            )
        clone_target.mkdir(parents=True)
        _run_git(clone_target, "init")
        unbundle_proc = _run_git(
            clone_target,
            "bundle",
            "unbundle",
            str(target),
            check=False,
        )
        if unbundle_proc.returncode != 0:
            raise SourceError(
                "BUNDLE_RESTORE_FAILED",
                f"git bundle unbundle failed: {unbundle_proc.stderr.strip()}",
                evidence_path=str(target),
                expected=expected_commit,
                actual=unbundle_proc.stderr.strip(),
            )
        co_proc = _run_git(
            clone_target,
            "checkout",
            "--detach",
            expected_commit,
            check=False,
        )
        if co_proc.returncode != 0:
            raise SourceError(
                "BUNDLE_RESTORE_FAILED",
                f"git checkout after unbundle failed: {co_proc.stderr.strip()}",
                evidence_path=str(target),
                expected=expected_commit,
                actual=co_proc.stderr.strip(),
            )
        restored_head = _run_git(clone_target, "rev-parse", "HEAD").stdout.strip()
        if restored_head != expected_commit:
            raise SourceError(
                "BUNDLE_RESTORE_MISMATCH",
                f"restored commit {restored_head} != expected {expected_commit}",
                evidence_path=str(target),
                expected=expected_commit,
                actual=restored_head,
            )
        tree_oid = _run_git(clone_target, "rev-parse", f"{restored_head}^{{tree}}").stdout.strip()

    return SourceBundleInfo(
        bundle_path=str(target),
        commit=expected_commit,
        tree_oid=tree_oid,
        sha256=actual_sha,
        size_bytes=target.stat().st_size,
    )
