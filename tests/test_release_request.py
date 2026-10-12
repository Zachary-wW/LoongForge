# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / ".github" / "scripts" / "release_request.py"
SPEC = importlib.util.spec_from_file_location("release_request", MODULE_PATH)
release_request = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules["release_request"] = release_request
SPEC.loader.exec_module(release_request)


def request(**overrides):
    values = {
        "build_image": "true",
        "run_regression": "true",
        "publish_internal": "true",
        "publish_dockerhub": "true",
    }
    values.update(overrides)
    return release_request.parse_release_request(**values)


def test_full_release_is_valid():
    result = request()
    assert result.build_image is True
    assert result.run_regression is True
    assert result.publish_internal is True
    assert result.publish_dockerhub is True


def test_build_only_is_valid():
    result = request(
        run_regression="false",
        publish_internal="false",
        publish_dockerhub="false",
    )
    assert result.build_image is True
    assert result.run_regression is False


def test_rehearsal_is_valid():
    result = request(publish_internal="false", publish_dockerhub="false")
    assert result.run_regression is True
    assert result.publish_internal is False


def test_regression_can_use_existing_image():
    result = request(
        build_image="false",
        publish_internal="false",
        publish_dockerhub="false",
        existing_image_ref="registry.example.test/team/project:v1",
    )
    assert result.build_image is False
    assert result.run_regression is True


def test_dockerhub_promotion_can_use_existing_internal_image():
    result = request(
        build_image="false",
        run_regression="false",
        publish_internal="false",
        existing_internal_image_ref="registry.example.test/team/project:v1",
        skip_regression_reason="promote previously validated image",
    )
    assert result.publish_dockerhub is True


def test_publishing_without_regression_requires_reason():
    with pytest.raises(release_request.ReleaseRequestError):
        request(run_regression="false")


def test_regression_requires_build_or_existing_image():
    with pytest.raises(release_request.ReleaseRequestError):
        request(
            build_image="false",
            publish_internal="false",
            publish_dockerhub="false",
        )


def test_at_least_one_stage_must_be_enabled():
    with pytest.raises(release_request.ReleaseRequestError):
        request(
            build_image="false",
            run_regression="false",
            publish_internal="false",
            publish_dockerhub="false",
        )


def test_invalid_debug_suffix_is_rejected():
    with pytest.raises(release_request.ReleaseRequestError):
        request(debug_tag_suffix="bad suffix")


def test_invalid_image_reference_is_rejected():
    with pytest.raises(release_request.ReleaseRequestError):
        request(existing_image_ref="not-an-image")
