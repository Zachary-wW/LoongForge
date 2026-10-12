# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
RESOLVE_SCRIPT = (
    REPOSITORY_ROOT
    / ".github"
    / "scripts"
    / "self_runner"
    / "resolve_execution_slot.sh"
)
REGRESSION_SLOT_SCRIPT = (
    REPOSITORY_ROOT
    / ".github"
    / "scripts"
    / "self_runner"
    / "run_release_regression_slot.sh"
)


def write_slot_config(tmp_path: Path) -> Path:
    config = tmp_path / "execution-slots.env"
    config.write_text(
        "\n".join(
            [
                "LOONGFORGE_EXECUTION_SLOT_PRIMARY_MODE=local",
                "LOONGFORGE_EXECUTION_SLOT_PRIMARY_DOCKER_HOST=",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_MODE=ssh",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_ALIAS=private-secondary",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_DOCKER_HOST=ssh://private-secondary/private.sock",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_ROOT=/private/root",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_SOURCE=/private/source",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_CONFIG=/private/config",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config.chmod(0o600)
    return config


def test_execution_slot_scripts_are_valid_shell():
    for script in (RESOLVE_SCRIPT, REGRESSION_SLOT_SCRIPT):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert result.stdout == ""
        assert result.stderr == ""


def test_secondary_slot_resolution_is_internal_and_complete(tmp_path):
    config = write_slot_config(tmp_path)
    github_env = tmp_path / "github.env"
    env = os.environ.copy()
    env.update(
        {
            "LOONGFORGE_EXECUTION_SLOT_CONFIG": str(config),
            "GITHUB_ENV": str(github_env),
        }
    )

    result = subprocess.run(
        ["bash", str(RESOLVE_SCRIPT), "secondary"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""

    values = dict(
        line.split("=", 1)
        for line in github_env.read_text(encoding="utf-8").splitlines()
    )
    assert values["LOONGFORGE_EXECUTION_MODE"] == "ssh"
    assert values["LOONGFORGE_EXECUTION_ALIAS"] == "private-secondary"
    assert values["DOCKER_HOST"] == "ssh://private-secondary/private.sock"
    assert values["LOONGFORGE_EXECUTION_SOURCE"] == "/private/source"


def test_remote_slot_script_archives_evidence():
    script = REGRESSION_SLOT_SCRIPT.read_text(encoding="utf-8")
    assert "LOONGFORGE_RELEASE_EVIDENCE_ROOT" in script
    assert "$evidence_root/$run_id/loongforge-release-artifacts" in script


def test_insecure_slot_configuration_fails_closed(tmp_path):
    config = write_slot_config(tmp_path)
    config.chmod(0o640)
    env = os.environ.copy()
    env["LOONGFORGE_EXECUTION_SLOT_CONFIG"] = str(config)

    result = subprocess.run(
        ["bash", str(RESOLVE_SCRIPT), "secondary"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "execution-slot: insecure configuration"
    assert str(tmp_path) not in result.stderr
