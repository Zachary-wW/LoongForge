from __future__ import annotations

from pathlib import Path

import pytest

from core.stages.f1_intake.discovery import DiscoveryError, discover_entrypoints

from helpers import create_sample_repo


def test_discover_entrypoints_reads_pinned_tree_ast_and_configs(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]

    result = discover_entrypoints(
        repo,
        commit,
        entry_hint="scripts/train.py",
    )
    assert result.status == "resolved"
    assert result.selected is not None
    assert result.selected.path == "scripts/train.py"
    assert result.selected.symbol == "main"
    assert len(result.selected.blob_oid) in (40, 64)
    assert result.selected.line_range[0] >= 1
    assert result.selected.line_range[1] >= result.selected.line_range[0]
    assert any(cfg.path == "configs/pi05.yaml" for cfg in result.configs)


def test_discover_entrypoints_rejects_deleted_untracked_and_worktree_mismatch(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]

    # Untracked file not in pinned tree
    (repo / "scripts" / "untracked_train.py").write_text(
        "def main():\n    return 0\n", encoding="utf-8"
    )
    with pytest.raises(DiscoveryError) as untracked_err:
        discover_entrypoints(
            repo,
            commit,
            entry_hint="scripts/untracked_train.py",
        )
    assert untracked_err.value.reason_code == "ENTRYPOINT_NOT_IN_PINNED_TREE"

    # Worktree modified bytes differ from pinned blob
    train_py = repo / "scripts" / "train.py"
    original = train_py.read_text(encoding="utf-8")
    train_py.write_text(original + "\n# mutated\n", encoding="utf-8")
    with pytest.raises(DiscoveryError) as mismatch_err:
        discover_entrypoints(repo, commit, entry_hint="scripts/train.py")
    assert mismatch_err.value.reason_code == "ENTRYPOINT_WORKTREE_MISMATCH"

    # Entrypoint deleted from worktree
    train_py.unlink()
    with pytest.raises(DiscoveryError) as deleted_err:
        discover_entrypoints(repo, commit, entry_hint="scripts/train.py")
    assert deleted_err.value.reason_code == "ENTRYPOINT_MISSING_IN_WORKTREE"


def test_discover_entrypoints_blocks_on_ambiguous_candidates_without_hint(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path, ambiguous_entrypoints=True)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]

    result = discover_entrypoints(repo, commit, entry_hint=None)
    assert result.status == "blocked"
    assert "ENTRYPOINT_AMBIGUOUS" in result.blocked_reasons
    assert result.selected is None
    assert {c.path for c in result.candidates} >= {
        "scripts/train.py",
        "scripts/train_alt.py",
    }
