# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / ".github"
    / "scripts"
    / "validate_release_models.py"
)
SOURCE = Path(__file__).parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("validate_release_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_current_release_models_are_valid():
    module = load_script()
    assert module.validate_llm_model(SOURCE, "deepseek_v2_lite") is None
    assert module.validate_vla_model(SOURCE, "pi05_ddp") is None


def test_unknown_models_are_rejected():
    module = load_script()
    assert module.validate_llm_model(SOURCE, "unknown_model") is not None
    assert module.validate_vla_model(SOURCE, "unknown_model") is not None


def test_cli_validates_models():
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--llm-model",
            "deepseek_v2_lite",
            "--vla-model",
            "pi05_ddp",
            "--source",
            str(SOURCE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
