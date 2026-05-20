"""
Tests for the third_party/Loong-Megatron submodule update.

PR change: submodule pointer updated from d2141966341d1937d4aae634914f3b98492fa5b6
to b6a9b202807a0025279771b3022ee2cf5ec2fd3c.

These tests verify the correctness of the submodule configuration and pointer
after the update.
"""

import configparser
import os
import subprocess
from pathlib import Path

import pytest

# Absolute path to the repository root
REPO_ROOT = Path(__file__).parent.parent

# Expected values after the PR update
SUBMODULE_NAME = "third_party/Loong-Megatron"
EXPECTED_COMMIT = "b6a9b202807a0025279771b3022ee2cf5ec2fd3c"
PREVIOUS_COMMIT = "d2141966341d1937d4aae634914f3b98492fa5b6"
EXPECTED_URL = "https://github.com/baidu-baige/Loong-Megatron.git"
EXPECTED_BRANCH = "loong-main/core_v0.15.0"
SUBMODULE_PATH = REPO_ROOT / "third_party" / "Loong-Megatron"
GITMODULES_PATH = REPO_ROOT / ".gitmodules"


class TestGitmodulesConfig:
    """Tests for .gitmodules configuration correctness."""

    def test_gitmodules_file_exists(self):
        """The .gitmodules file must exist at the repository root."""
        assert GITMODULES_PATH.exists(), f".gitmodules not found at {GITMODULES_PATH}"

    def test_submodule_section_present(self):
        """The .gitmodules file must contain a section for Loong-Megatron."""
        config = configparser.ConfigParser()
        config.read(str(GITMODULES_PATH))
        section = f'submodule "{SUBMODULE_NAME}"'
        assert config.has_section(section), (
            f'Expected section [{section}] in .gitmodules. '
            f"Sections found: {config.sections()}"
        )

    def test_submodule_path_configured(self):
        """The submodule path in .gitmodules must match third_party/Loong-Megatron."""
        config = configparser.ConfigParser()
        config.read(str(GITMODULES_PATH))
        section = f'submodule "{SUBMODULE_NAME}"'
        assert config.has_option(section, "path"), "Submodule 'path' key missing in .gitmodules"
        assert config.get(section, "path") == SUBMODULE_NAME, (
            f"Expected path '{SUBMODULE_NAME}', "
            f"got '{config.get(section, 'path')}'"
        )

    def test_submodule_url_configured(self):
        """The submodule URL must point to the correct upstream repository."""
        config = configparser.ConfigParser()
        config.read(str(GITMODULES_PATH))
        section = f'submodule "{SUBMODULE_NAME}"'
        assert config.has_option(section, "url"), "Submodule 'url' key missing in .gitmodules"
        actual_url = config.get(section, "url")
        assert actual_url == EXPECTED_URL, (
            f"Expected URL '{EXPECTED_URL}', got '{actual_url}'"
        )

    def test_submodule_branch_configured(self):
        """The submodule branch must be loong-main/core_v0.15.0."""
        config = configparser.ConfigParser()
        config.read(str(GITMODULES_PATH))
        section = f'submodule "{SUBMODULE_NAME}"'
        assert config.has_option(section, "branch"), (
            "Submodule 'branch' key missing in .gitmodules"
        )
        actual_branch = config.get(section, "branch")
        assert actual_branch == EXPECTED_BRANCH, (
            f"Expected branch '{EXPECTED_BRANCH}', got '{actual_branch}'"
        )


class TestSubmoduleDirectoryStructure:
    """Tests for the submodule directory layout inside third_party/."""

    def test_third_party_directory_exists(self):
        """The third_party/ directory must exist at the repository root."""
        third_party = REPO_ROOT / "third_party"
        assert third_party.is_dir(), f"third_party/ directory not found at {REPO_ROOT}"

    def test_submodule_directory_exists(self):
        """The third_party/Loong-Megatron directory must exist."""
        assert SUBMODULE_PATH.exists(), (
            f"Submodule directory not found: {SUBMODULE_PATH}"
        )

    def test_submodule_is_directory(self):
        """third_party/Loong-Megatron must be a directory, not a regular file."""
        assert SUBMODULE_PATH.is_dir(), (
            f"Expected a directory at {SUBMODULE_PATH}, but it is not"
        )


class TestSubmoduleCommitPointer:
    """Tests for the git submodule commit pointer after the PR update."""

    def _get_submodule_commit(self):
        """Return the commit hash recorded in the parent repo for the submodule."""
        result = subprocess.run(
            ["git", "submodule", "status", "--", str(SUBMODULE_PATH)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"git submodule status failed: {result.stderr}"
        )
        # Output format: " <commit> <path> [<description>]"
        # Leading character is ' ', '+', or '-' indicating status
        output = result.stdout.strip()
        assert output, "git submodule status returned empty output"
        # Strip leading status character and extract first 40-char SHA
        commit = output.lstrip(" +-").split()[0]
        return commit

    def test_submodule_points_to_expected_commit(self):
        """The submodule must reference the new commit introduced in this PR."""
        commit = self._get_submodule_commit()
        assert commit == EXPECTED_COMMIT, (
            f"Submodule points to '{commit}', expected '{EXPECTED_COMMIT}'"
        )

    def test_submodule_does_not_point_to_previous_commit(self):
        """Regression: submodule must NOT reference the old pre-PR commit."""
        commit = self._get_submodule_commit()
        assert commit != PREVIOUS_COMMIT, (
            f"Submodule still points to the old commit '{PREVIOUS_COMMIT}'; "
            "the PR update was not applied correctly"
        )

    def test_submodule_commit_is_full_sha(self):
        """The recorded submodule commit must be a full 40-character SHA-1 hash."""
        commit = self._get_submodule_commit()
        assert len(commit) == 40, (
            f"Expected a 40-character SHA, got '{commit}' (length {len(commit)})"
        )
        assert all(c in "0123456789abcdef" for c in commit), (
            f"Commit '{commit}' contains non-hex characters"
        )

    def test_submodule_commit_prefix_matches(self):
        """The submodule commit must start with the known short-SHA prefix b6a9b20."""
        commit = self._get_submodule_commit()
        assert commit.startswith("b6a9b20"), (
            f"Expected commit starting with 'b6a9b20', got '{commit}'"
        )


class TestGitSubmoduleStatusOutput:
    """Tests that inspect raw git submodule status for Loong-Megatron."""

    def _run_git_ls_tree(self):
        """Return the ls-tree output for the submodule gitlink entry."""
        result = subprocess.run(
            ["git", "ls-tree", "HEAD", "--", SUBMODULE_NAME],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result

    def test_ls_tree_returns_gitlink_entry(self):
        """git ls-tree must list the submodule as a gitlink (mode 160000)."""
        result = self._run_git_ls_tree()
        assert result.returncode == 0, f"git ls-tree failed: {result.stderr}"
        output = result.stdout.strip()
        assert output, "git ls-tree returned empty output for the submodule path"
        # Gitlink entries have mode 160000 and type 'commit'
        assert "160000" in output, (
            f"Expected gitlink mode 160000 in ls-tree output, got: '{output}'"
        )
        assert "commit" in output, (
            f"Expected type 'commit' in ls-tree output, got: '{output}'"
        )

    def test_ls_tree_contains_new_commit_hash(self):
        """git ls-tree output must contain the new commit hash from this PR."""
        result = self._run_git_ls_tree()
        assert result.returncode == 0, f"git ls-tree failed: {result.stderr}"
        output = result.stdout.strip()
        assert EXPECTED_COMMIT in output, (
            f"Expected commit '{EXPECTED_COMMIT}' in ls-tree output, got: '{output}'"
        )

    def test_ls_tree_does_not_contain_old_commit_hash(self):
        """Regression: git ls-tree output must NOT contain the old commit hash."""
        result = self._run_git_ls_tree()
        assert result.returncode == 0, f"git ls-tree failed: {result.stderr}"
        output = result.stdout.strip()
        assert PREVIOUS_COMMIT not in output, (
            f"Old commit '{PREVIOUS_COMMIT}' still present in ls-tree output; "
            "submodule was not updated"
        )

    def test_ls_tree_submodule_name_in_output(self):
        """git ls-tree output must reference the correct submodule path."""
        result = self._run_git_ls_tree()
        assert result.returncode == 0, f"git ls-tree failed: {result.stderr}"
        assert SUBMODULE_NAME in result.stdout, (
            f"Expected '{SUBMODULE_NAME}' in ls-tree output, got: '{result.stdout}'"
        )
