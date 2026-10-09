# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path


def run_loader(
    tmp_path: Path,
    *,
    credential_mode: int = 0o600,
    config_mode: int = 0o600,
    complete: bool = True,
):
    internal = tmp_path / "internal-credentials"
    public = tmp_path / "public-credentials"
    internal.write_text("IREGISTRY_USERNAME=user\nIREGISTRY_PASSWORD=secret\n", encoding="utf-8")
    public.write_text("DOCKERHUB_USERNAME=user\nDOCKERHUB_TOKEN=token\n", encoding="utf-8")
    internal.chmod(credential_mode)
    public.chmod(credential_mode)
    config = tmp_path / "release-config"
    values = [
        "DOCKER_HOST=ssh://publisher",
        "LOONGFORGE_RELEASE_PUBLISHER_ALIAS=publisher",
        f"LOONGFORGE_IREGISTRY_CREDENTIALS={internal}",
    ]
    if complete:
        values.append(f"LOONGFORGE_DOCKERHUB_CREDENTIALS={public}")
    config.write_text("\n".join(values) + "\n", encoding="utf-8")
    config.chmod(config_mode)
    env = os.environ.copy()
    env["LOONGFORGE_RELEASE_CONFIG"] = str(config)
    return subprocess.run(
        ["bash", "-c", "source .github/scripts/self_runner/load_release_config.sh"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_config_loader_accepts_private_complete_config(tmp_path):
    result = run_loader(tmp_path)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_release_config_loader_rejects_incomplete_config_without_paths(tmp_path):
    result = run_loader(tmp_path, complete=False)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "release-config: incomplete"
    assert str(tmp_path) not in result.stderr


def test_release_config_loader_rejects_group_readable_credentials(tmp_path):
    result = run_loader(tmp_path, credential_mode=0o640)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "release-config: insecure credentials"
    assert str(tmp_path) not in result.stderr


def test_release_config_loader_rejects_group_readable_config(tmp_path):
    result = run_loader(tmp_path, config_mode=0o640)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "release-config: insecure configuration"
    assert str(tmp_path) not in result.stderr
