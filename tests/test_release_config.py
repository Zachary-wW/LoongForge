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
    with_credentials: bool = True,
    config_complete: bool = True,
):
    internal = tmp_path / "internal-credentials"
    public = tmp_path / "public-credentials"
    internal.write_text("IREGISTRY_USERNAME=user\nIREGISTRY_PASSWORD=secret\n", encoding="utf-8")
    public.write_text("DOCKERHUB_USERNAME=user\nDOCKERHUB_TOKEN=token\n", encoding="utf-8")
    internal.chmod(credential_mode)
    public.chmod(credential_mode)
    config = tmp_path / "release-config"
    values = ["DOCKER_HOST=ssh://publisher"]
    if config_complete:
        values.append("LOONGFORGE_RELEASE_PUBLISHER_ALIAS=publisher")
    config.write_text("\n".join(values) + "\n", encoding="utf-8")
    config.chmod(config_mode)
    env = os.environ.copy()
    env["LOONGFORGE_RELEASE_CONFIG"] = str(config)
    if with_credentials:
        env["LOONGFORGE_IREGISTRY_CREDENTIALS"] = str(internal)
        env["LOONGFORGE_DOCKERHUB_CREDENTIALS"] = str(public)
    else:
        env.pop("LOONGFORGE_IREGISTRY_CREDENTIALS", None)
        env.pop("LOONGFORGE_DOCKERHUB_CREDENTIALS", None)
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
    result = run_loader(tmp_path, with_credentials=False)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "release-config: unavailable"
    assert str(tmp_path) not in result.stderr


def test_release_config_loader_rejects_config_missing_destination_fields(tmp_path):
    result = run_loader(tmp_path, config_complete=False)
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


def test_credentials_loader_masks_only_its_registry_under_set_e(tmp_path):
    hub_credentials = tmp_path / "hub-credentials"
    hub_credentials.write_text(
        "DOCKERHUB_USERNAME=hub-user\nDOCKERHUB_TOKEN=hub-token-value\n",
        encoding="utf-8",
    )
    hub_credentials.chmod(0o600)
    env = os.environ.copy()
    env["LOONGFORGE_DOCKERHUB_CREDENTIALS"] = str(hub_credentials)
    for name in ("DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN", "IREGISTRY_PASSWORD"):
        env.pop(name, None)
    result = subprocess.run(
        [
            "bash", "-c",
            "set -euo pipefail; "
            "source .github/scripts/self_runner/load_release_credentials.sh dockerhub; "
            'test "$DOCKERHUB_TOKEN" = hub-token-value && echo loaded',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "loaded" in result.stdout
    assert "::add-mask::hub-token-value" in result.stdout
    assert "::add-mask::hub-user" in result.stdout
    assert "iregistry" not in result.stdout
