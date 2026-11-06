# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Lazy foundation model registry."""

import importlib

from transformers import AutoModel


_FOUNDATION_MODELS = {
    "deepseek": (".deepseek", "DeepseekConfig", "DeepseekModelWithMTP"),
    "deepseek_v4": (".deepseek_v4", "DeepseekV4Config", "DeepseekV4Model"),
    "ernie4_5_vl": (".ernie4_5_vl", "ErnieMoeConfig", "ErnieMoeModel"),
    "glm": (".glm", "GlmConfig", "GlmModelWithMTP"),
    "glm5_next": (".glm5_next", "Glm5NextConfig", "Glm5NextModel"),
    "internlm": (".internlm", "InternLMConfig", "InternLMModel"),
    "internlm2.5": (".internlm", "InternLMConfig", "InternLMModel"),
    "kimi_k3": (".kimi_k3", "KimiK3Config", "KimiK3Model"),
    "kimi_k3_llm": (".kimi_k3", "KimiK3Config", "KimiK3Model"),
    "llama": (".llama", "LLaMAConfig", "LLaMAModel"),
    "llama2": (".llama", "LLaMAConfig", "LLaMAModel"),
    "llama3": (".llama", "LLaMAConfig", "LLaMAModel"),
    "llama3.1": (".llama", "LLaMAConfig", "LLaMAModel"),
    "minimax": (".minimax", "MinimaxConfig", "MinimaxModelWithMTP"),
    "mimo": (".mimo", "MimoConfig", "MimoModelWithMTP"),
    "qwen": (".qwen2", "Qwen2Config", "Qwen2Model"),
    "qwen1.5": (".qwen2", "Qwen2Config", "Qwen2Model"),
    "qwen2": (".qwen2", "Qwen2Config", "Qwen2Model"),
    "qwen2.5": (".qwen2", "Qwen2Config", "Qwen2Model"),
    "qwen3": (".qwen3", "Qwen3Config", "Qwen3Model"),
    "qwen3.5": (".qwen3_5", "Qwen35Config", "Qwen35Model"),
    "qwen3_next": (".qwen3_next", "Qwen3NextConfig", "Qwen3NextModel"),
}

_REGISTERED_MODEL_TYPES = set()

_SYMBOL_MODULES = {
    "DeepseekConfig": "loongforge.models.foundation.deepseek.deepseek_config",
}


def __getattr__(name: str):
    """Resolve public model classes without importing every foundation family."""
    if name in _SYMBOL_MODULES:
        module = importlib.import_module(_SYMBOL_MODULES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    for module_name, config_name, model_name in _FOUNDATION_MODELS.values():
        if name not in (config_name, model_name):
            continue
        module = importlib.import_module(module_name, package=__name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_model_family(model_type: str):
    """Import and register one foundation family without loading unrelated models."""
    normalized = str(model_type).lower()
    registration = _FOUNDATION_MODELS.get(normalized)
    if registration is None:
        raise ValueError(f"Unsupported foundation model type: {model_type}")
    if normalized in _REGISTERED_MODEL_TYPES:
        return

    module_name, config_name, model_name = registration
    module = importlib.import_module(module_name, package=__name__)
    config_class = getattr(module, config_name)
    model_class = getattr(module, model_name)
    AutoModel.register(config_class, model_class, exist_ok=True)
    _REGISTERED_MODEL_TYPES.add(normalized)
