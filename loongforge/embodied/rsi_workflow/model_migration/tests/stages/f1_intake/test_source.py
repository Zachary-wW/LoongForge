from __future__ import annotations

from pathlib import Path

import pytest

from core.stages.f1_intake.source import (
    SourceError,
    create_source_bundle,
    snapshot_source,
    verify_source_bundle,
    verify_source_unchanged,
)

from helpers import create_sample_repo, git


def test_snapshot_source_records_full_identity_and_verifies_unchanged(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path, with_submodule=True)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]

    before = snapshot_source(repo, commit)
    assert before.repo_root == str(repo.resolve())
    assert before.head_commit == commit
    assert len(before.tree_oid) in (40, 64)
    assert before.object_format in ("sha1", "sha256")
    assert before.is_clean is True
    assert len(before.submodules) == 1
    assert before.submodules[0].path == "third_party/helper"
    assert before.submodules[0].gitlink_commit == info["submodule_commit"]
    assert before.submodules[0].head_commit == info["submodule_commit"]
    assert before.submodules[0].dirty is False

    after = snapshot_source(repo, commit)
    verify_source_unchanged(before, after)


def test_snapshot_source_rejects_commit_mismatch_dirty_untracked_and_submodule_drift(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path, with_submodule=True, extra_commit=True)
    repo: Path = info["repo"]

    # Wrong expected commit
    with pytest.raises(SourceError) as wrong_commit:
        snapshot_source(repo, info["first_commit"])
    assert wrong_commit.value.reason_code == "SOURCE_COMMIT_MISMATCH"

    # Unstaged dirty file
    cfg = repo / "configs" / "pi05.yaml"
    original = cfg.read_text(encoding="utf-8")
    cfg.write_text(original + "dirty: true\n", encoding="utf-8")
    with pytest.raises(SourceError) as dirty_err:
        snapshot_source(repo, info["head_commit"])
    assert dirty_err.value.reason_code == "SOURCE_WORKTREE_DIRTY"

    # Staged modification must also be rejected
    git(repo, "add", "configs/pi05.yaml")
    with pytest.raises(SourceError) as staged_err:
        snapshot_source(repo, info["head_commit"])
    assert staged_err.value.reason_code == "SOURCE_WORKTREE_DIRTY"
    git(repo, "reset", "HEAD", "configs/pi05.yaml")
    cfg.write_text(original, encoding="utf-8")

    # Untracked file
    untracked = repo / "stray.txt"
    untracked.write_text("stray\n", encoding="utf-8")
    with pytest.raises(SourceError) as untracked_err:
        snapshot_source(repo, info["head_commit"])
    assert untracked_err.value.reason_code == "SOURCE_WORKTREE_UNTRACKED"
    untracked.unlink()

    # Submodule commit mismatch (detached ahead of gitlink)
    sub_dir = repo / "third_party" / "helper"
    (sub_dir / "lib.py").write_text("VALUE = 2\n", encoding="utf-8")
    git(sub_dir, "add", "lib.py")
    git(sub_dir, "commit", "-m", "advance submodule")
    with pytest.raises(SourceError) as sub_err:
        snapshot_source(repo, info["head_commit"])
    assert sub_err.value.reason_code in {
        "SOURCE_SUBMODULE_MISMATCH",
        "SOURCE_WORKTREE_DIRTY",
    }


def test_verify_source_unchanged_detects_post_analysis_mutation(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]

    before = snapshot_source(repo, commit)
    (repo / "README.md").write_text("mutated!\n", encoding="utf-8")
    after = snapshot_source(repo, commit, require_clean=False)

    with pytest.raises(SourceError) as err:
        verify_source_unchanged(before, after)
    assert err.value.reason_code == "SOURCE_MUTATED_DURING_ANALYSIS"


def test_git_bundle_create_verify_and_offline_restore_across_branch_switch(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path, extra_commit=True)
    repo: Path = info["repo"]
    target_commit = info["first_commit"]

    # Switch repo to another branch/commit while bundling pinned commit
    git(repo, "checkout", "-b", "other-branch", info["head_commit"])
    bundle_path = tmp_path / "artifacts" / "source.bundle"
    bundle_info = create_source_bundle(repo, target_commit, bundle_path)
    assert bundle_path.exists()
    assert bundle_info.commit == target_commit
    assert len(bundle_info.sha256) == 64

    restore_dir = tmp_path / "restored"
    verify_result = verify_source_bundle(
        bundle_path,
        expected_commit=target_commit,
        expected_sha256=bundle_info.sha256,
        restore_dir=restore_dir,
    )
    assert verify_result.commit == target_commit
    assert (restore_dir / "configs" / "pi05.yaml").read_text(encoding="utf-8") == (
        "model_variant: pi05\nbatch_size: 2\n"
    )

    # Tampering bundle breaks verification
    bundle_path.write_bytes(bundle_path.read_bytes() + b"tamper")
    with pytest.raises(SourceError) as bad:
        verify_source_bundle(
            bundle_path,
            expected_commit=target_commit,
            expected_sha256=bundle_info.sha256,
        )
    assert bad.value.reason_code == "BUNDLE_DIGEST_MISMATCH"


def test_ignored_files_fail_closed_by_default_and_host_path_is_excluded_from_identity(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path / "host_a", with_submodule=True)
    repo: Path = info["repo"]

    # Add .gitignore and commit it so we can test an ignored local override file.
    (repo / ".gitignore").write_text("local_overrides.yaml\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-m", "add gitignore")
    commit_with_ignore = git(repo, "rev-parse", "HEAD").stdout.strip()

    # Clean snapshot before creating ignored file
    snap_clean = snapshot_source(repo, commit_with_ignore)
    assert snap_clean.is_clean is True

    # Creating an ignored local_overrides.yaml must fail closed by default (P1-2)
    override = repo / "local_overrides.yaml"
    override.write_text("lr: 999\n", encoding="utf-8")
    with pytest.raises(SourceError) as ign_err:
        snapshot_source(repo, commit_with_ignore)
    assert ign_err.value.reason_code == "SOURCE_WORKTREE_IGNORED"

    # Explicit allowlist permits only declared ignored paths
    snap_allowed = snapshot_source(
        repo,
        commit_with_ignore,
        allowed_ignored=("local_overrides.yaml",),
    )
    assert snap_allowed.is_clean is True
    override.unlink()

    # P1-3: Offline bundle restore to a DIFFERENT host directory path produces identical identity_sha256
    from core.stages.f1_intake.source import (
        create_recursive_source_bundles,
        restore_from_bundle_manifest,
    )

    bundles_root = tmp_path / "pkg_bundles"
    bundles_manifest = create_recursive_source_bundles(
        repo,
        snap_clean,
        bundles_root / "evidence" / "source_bundles",
        bundle_rel_prefix="evidence/source_bundles",
    )
    restored_other_host = tmp_path / "host_b_completely_different_path" / "checkout"
    restored_snap = restore_from_bundle_manifest(
        bundles_root,
        bundles_manifest,
        restored_other_host,
    )
    assert restored_snap.repo_root != snap_clean.repo_root
    assert restored_snap.identity_sha256 == snap_clean.identity_sha256
    verify_source_unchanged(snap_clean, restored_snap)


def test_shallow_clone_bundle_create_and_offline_unbundle_restore(
    tmp_path: Path,
) -> None:
    """Regression test from real `openpi` dry-run: `--depth 1` shallow clones must
    bundle cleanly via `fetch --update-shallow` and restore via `git bundle unbundle`.
    """
    origin_info = create_sample_repo(tmp_path / "origin_host", extra_commit=True)
    origin_repo: Path = origin_info["repo"]
    pinned_commit: str = origin_info["head_commit"]

    shallow_repo = tmp_path / "shallow_checkout"
    git(
        tmp_path,
        "clone",
        "--depth",
        "1",
        f"file://{origin_repo}",
        str(shallow_repo),
    )
    assert (
        git(shallow_repo, "rev-parse", "--is-shallow-repository").stdout.strip()
        == "true"
    )

    snap_before = snapshot_source(shallow_repo, pinned_commit)
    bundle_path = tmp_path / "bundles" / "shallow.bundle"
    bundle_info = create_source_bundle(shallow_repo, pinned_commit, bundle_path)
    assert bundle_info.commit == pinned_commit

    restore_dir = tmp_path / "restored_from_shallow"
    restored_info = verify_source_bundle(
        bundle_path,
        expected_commit=pinned_commit,
        expected_sha256=bundle_info.sha256,
        restore_dir=restore_dir,
    )
    assert restored_info.commit == pinned_commit
    assert restored_info.tree_oid == snap_before.tree_oid
    snap_restored = snapshot_source(restore_dir, pinned_commit)
    assert snap_restored.identity_sha256 == snap_before.identity_sha256
