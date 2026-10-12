# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).parents[1]
    / ".github"
    / "scripts"
    / "self_runner"
    / "promote_to_dockerhub.sh"
)


def test_promotion_script_is_valid_shell():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_promotion_script_uses_regctl_copy():
    script = SCRIPT.read_text(encoding="utf-8")
    assert "image copy" in script
    assert "--host" in script
    assert "LOONGFORGE_REGCTL_BIN" in script
    assert "docker push" not in script
