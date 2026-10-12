# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(
    ".github/scripts/self_runner/cleanup_debug_tags.py",
)


def load_cleanup_module():
    spec = importlib.util.spec_from_file_location("cleanup_debug_tags", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_image_reference_parses_registry_host_repository_and_tag():
    module = load_cleanup_module()
    host, repository, tag = module.split_image_reference(
        "registry.example.test/team/project:v0.2.3-ci_test",
    )
    assert host == "registry.example.test"
    assert repository == "team/project"
    assert tag == "v0.2.3-ci_test"


def test_split_image_reference_parses_dockerhub_reference():
    module = load_cleanup_module()
    host, repository, tag = module.split_image_reference(
        "docker.io/loongforge/loongforge:v0.2.3-ci_test",
    )
    assert host == "docker.io"
    assert module.normalize_dockerhub_repository(repository) == "loongforge/loongforge"
    assert tag == "v0.2.3-ci_test"


def test_split_image_reference_requires_a_tag():
    module = load_cleanup_module()
    with pytest.raises(ValueError):
        module.split_image_reference("registry.example.test/team/project")


def test_internal_registry_cleanup_treats_missing_tag_as_success(monkeypatch, capsys):
    module = load_cleanup_module()
    monkeypatch.setattr(
        module,
        "http_request",
        lambda *args, **kwargs: (404, {}, b""),
    )
    module.delete_internal_registry_tag(
        host="registry.example.test",
        image="registry.example.test/team/project:v0.2.3-ci_test",
        username="user",
        password="secret",
    )
    assert "internal registry: tag already absent" in capsys.readouterr().out
