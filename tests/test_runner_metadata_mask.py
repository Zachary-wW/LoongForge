# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess


def test_mask_script_masks_config_and_additional_values(tmp_path):
    image_config = tmp_path / "image-config"
    image_config.write_text("PRIVATE_PATH=/private/image/path\nEMPTY=\n", encoding="utf-8")
    release_config = tmp_path / "release-config"
    release_config.write_text("DOCKER_HOST=ssh://publisher\n", encoding="utf-8")
    extra_values = tmp_path / "private-values"
    extra_values.write_text("# comment\nprivate.example.invalid\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "RUNNER_NAME": "private-runner-name",
        "CI_CONFIG_PATH_IMAGE": str(image_config),
        "LOONGFORGE_RELEASE_CONFIG": str(release_config),
        "LOONGFORGE_RUNNER_MASK_VALUES": str(extra_values),
    })
    result = subprocess.run(
        ["bash", ".github/scripts/self_runner/mask_runner_metadata.sh"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for value in (
        "private-runner-name",
        str(image_config),
        str(release_config),
        "/private/image/path",
        "ssh://publisher",
        "private.example.invalid",
    ):
        assert f"::add-mask::{value}\n" in result.stdout
    assert "::add-mask::\n" not in result.stdout
