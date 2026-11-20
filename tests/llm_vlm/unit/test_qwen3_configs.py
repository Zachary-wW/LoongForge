from pathlib import Path

import yaml


QWEN3_CONFIG_DIR = Path(__file__).parents[3] / "configs/models/qwen3"


def test_qwen3_configs_use_qwen3_registry_key():
    for path in sorted(QWEN3_CONFIG_DIR.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        assert config["model_type"] == "qwen3", path
