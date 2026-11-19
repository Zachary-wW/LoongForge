"""Unit coverage for MiMo training lifecycle hooks."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.module import Float16Module

from loongforge.models.mimo_models.mimo_vlm_model import LoongForgeMimoVLMModel


class _FakeMimoModel:
    def __init__(self):
        self.calls = []
        self.handles = {object(): object()}

    def enable_forward_pre_hook(self):
        self.calls.append(("enable_forward_pre_hook",))

    def disable_forward_pre_hook(self, param_sync=True):
        self.calls.append(("disable_forward_pre_hook", param_sync))

    def start_param_sync(self, *args, **kwargs):
        self.calls.append(("start_param_sync", args, kwargs))

    def start_grad_sync(self, *args, **kwargs):
        self.calls.append(("start_grad_sync", args, kwargs))

    def free_overlap_buffers(self):
        self.calls.append(("free_overlap_buffers",))

    def zero_grad_buffer(self):
        self.calls.append(("zero_grad_buffer",))

    @property
    def remove_forward_pre_hook_handles(self):
        return self.handles


def test_loongforge_mimo_wrapper_delegates_training_hooks():
    fake_mimo = _FakeMimoModel()
    wrapper = torch.nn.Module.__new__(LoongForgeMimoVLMModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.mimo_model = fake_mimo

    wrapper.enable_forward_pre_hook()
    wrapper.disable_forward_pre_hook(param_sync=False)
    wrapper.start_param_sync(force_sync=True)
    wrapper.start_grad_sync()
    wrapper.free_overlap_buffers()
    wrapper.zero_grad_buffer()

    assert wrapper.remove_forward_pre_hook_handles is fake_mimo.handles
    assert fake_mimo.calls == [
        ("enable_forward_pre_hook",),
        ("disable_forward_pre_hook", False),
        ("start_param_sync", (), {"force_sync": True}),
        ("start_grad_sync", (), {}),
        ("free_overlap_buffers",),
        ("zero_grad_buffer",),
    ]


def test_mimo_loads_hf_converted_weights_after_module_wrapping():
    config = SimpleNamespace(fp16=False, bf16=True, virtual_pipeline_model_parallel_size=None)
    language_model = torch.nn.Linear(2, 2, bias=False)
    image_encoder = torch.nn.Linear(2, 2, bias=False)
    images = torch.nn.Module()
    images.encoders = torch.nn.ModuleDict({"image_encoder": image_encoder})
    mimo = torch.nn.Module()
    mimo.language_model = Float16Module(config, language_model)
    mimo.modality_submodules = torch.nn.ModuleDict({"images": Float16Module(config, images)})
    wrapper = torch.nn.Module.__new__(LoongForgeMimoVLMModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.mimo_model = mimo

    language_weights = torch.full((2, 2), 3.0)
    vision_weights = torch.full((2, 2), 5.0)
    state_dict = {
        "foundation_model.weight": language_weights,
        "encoder_model.image_encoder.weight": vision_weights,
    }

    wrapper.load_state_dict(state_dict, strict=True)

    torch.testing.assert_close(language_model.weight.float(), language_weights)
    torch.testing.assert_close(image_encoder.weight.float(), vision_weights)
    with pytest.raises(RuntimeError, match="Missing key"):
        wrapper.load_state_dict({"foundation_model.weight": language_weights}, strict=True)


def test_mimo_shared_embedding_weight_unwraps_lifecycle_wrappers():
    config = SimpleNamespace(
        fp16=False,
        bf16=True,
        virtual_pipeline_model_parallel_size=None,
    )
    expected = torch.nn.Parameter(torch.ones(2, 2))

    class _LanguageModel(torch.nn.Module):
        share_embeddings_and_output_weights = True

        def shared_embedding_or_output_weight(self):
            return expected

    mimo = SimpleNamespace(language_model=Float16Module(config, _LanguageModel()))
    wrapper = torch.nn.Module.__new__(LoongForgeMimoVLMModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.mimo_model = mimo

    assert wrapper.shared_embedding_or_output_weight() is expected
