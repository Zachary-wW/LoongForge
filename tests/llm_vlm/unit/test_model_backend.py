# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from loongforge.models.foundation.deepseek_v4.deepseek_v4_config import DeepseekV4Config
from loongforge.models.model_backend import (
    GPTBackend,
    HybridBackend,
    get_model_backend,
)


def test_registry_routes_gpt_and_hybrid_configs():
    assert isinstance(get_model_backend(SimpleNamespace(model_backend="gpt")), GPTBackend)
    assert isinstance(get_model_backend(SimpleNamespace(model_backend="hybrid")), HybridBackend)
    assert DeepseekV4Config.model_backend == "hybrid"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown model backend"):
        get_model_backend(SimpleNamespace(model_backend="missing"))


def test_pretrain_forward_contract_uses_keyword_arguments():
    backend = get_model_backend(SimpleNamespace(model_backend="gpt"))
    tokens = torch.zeros(2, 4, dtype=torch.long)
    labels = torch.zeros(2, 4, dtype=torch.long)
    loss_mask = torch.ones(2, 4)
    attention_mask = torch.ones(2, 1, 4, 4)
    position_ids = torch.arange(4).expand(2, -1)

    kwargs = backend.pretrain_forward_kwargs(
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        extra_block_kwargs=None,
    )

    assert kwargs == {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
        "extra_block_kwargs": None,
    }


def test_backend_extracts_structured_loss():
    backend = get_model_backend(SimpleNamespace(model_backend="hybrid"))
    loss = torch.tensor(1.25)

    assert backend.extract_loss(SimpleNamespace(loss=loss)) == loss
    assert backend.extract_loss(loss) == loss
    with pytest.raises(TypeError, match="neither a tensor"):
        backend.extract_loss(SimpleNamespace(loss=None))
