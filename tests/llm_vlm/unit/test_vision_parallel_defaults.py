from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ENCODER_DIR = ROOT / "configs" / "models" / "image_encoder"
QWEN25VL_TEST_CONFIG = ROOT / "tests" / "llm_vlm" / "configs" / "qwen2.5_vl_7b.yaml"


def test_standard_vision_tp_defaults_to_legacy_one():
    configs = sorted(IMAGE_ENCODER_DIR.glob("*.yaml"))
    assert configs

    for path in configs:
        config = yaml.safe_load(path.read_text())
        assert config.get("tensor_model_parallel_size") == 1, path.name


def test_qwen25vl_test_config_keeps_explicit_vision_tp2_override():
    text = QWEN25VL_TEST_CONFIG.read_text()
    assert text.count("++model.image_encoder.tensor_model_parallel_size=2") == 2
