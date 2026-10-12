# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / ".github" / "scripts" / "check_docker_storage.sh"


def test_storage_check_script_is_valid_shell():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_storage_check_uses_remote_df_for_ssh_docker_host(tmp_path):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"$1\" == info ]]; then echo /remote/docker-data; exit 0; fi",
                "exit 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"$1\" == -o ]]; then shift 2; fi",
                "case \"$2\" in \"df -Pk '/remote/docker-data'\"*) echo 262144000; exit 0;; esac",
                "exit 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "DOCKER_BIN": str(fake_docker),
            "PATH": f"{tmp_path}:{env['PATH']}",
            "LOONGFORGE_EXECUTION_ALIAS": "private-secondary",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "250"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == "docker-storage: ok\n"
