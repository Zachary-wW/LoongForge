# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_dockerfile_restores_diffusers_after_lerobot():
    dockerfile = (Path(__file__).parents[1] / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "diffusers==0.38.0" in dockerfile
    assert "from diffusers import ConfigMixin, ModelMixin" in dockerfile
    assert dockerfile.index("install_lerobot") < dockerfile.index('install "diffusers==0.38.0"')
