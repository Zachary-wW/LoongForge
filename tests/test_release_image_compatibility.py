# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import importlib.util


MODULE_PATH = Path(__file__).parents[1] / ".github" / "scripts" / "image_policy.py"
SPEC = importlib.util.spec_from_file_location("image_policy", MODULE_PATH)
image_policy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(image_policy)


def test_dockerfile_restores_diffusers_after_lerobot():
    dockerfile = (Path(__file__).parents[1] / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "diffusers==0.38.0" in dockerfile
    assert "from diffusers import ConfigMixin, ModelMixin" in dockerfile
    assert dockerfile.index("install_lerobot") < dockerfile.index('install "diffusers==0.38.0"')


def test_image_policy_ignores_daemon_graph_driver_metadata():
    metadata = image_policy.image_metadata(
        '{"GraphDriver":{"Data":{"UpperDir":"/ssd4/private/docker-data/diff"}},"Labels":{}}',
        "",
    )
    assert "/ssd4/private/docker-data" not in metadata


def test_image_policy_accepts_inspect_array_and_ignores_graph_driver():
    metadata = image_policy.image_metadata(
        '[{"GraphDriver":{"Data":{"UpperDir":"/ssd4/private/docker-data/diff"}},"Labels":{}}]',
        "",
    )
    assert "/ssd4/private/docker-data" not in metadata
