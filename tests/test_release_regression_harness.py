# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parent.parent
RELEASE_REGRESSION_SCRIPT = (
    REPOSITORY_ROOT
    / ".github"
    / "scripts"
    / "self_runner"
    / "run_release_regression.sh"
)


def test_release_regression_script_is_valid_shell():
    result = subprocess.run(
        ["bash", "-n", str(RELEASE_REGRESSION_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_release_regression_sets_single_node_distributed_environment():
    script = RELEASE_REGRESSION_SCRIPT.read_text(encoding="utf-8")

    for name, value in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("MASTER_ADDR", "127.0.0.1"),
    ):
        assert f'-e "{name}={value}"' in script

    assert '-e "MASTER_PORT=$master_port"' in script
    assert "master_port=$((20000 + ($$ % 30000)))" in script
