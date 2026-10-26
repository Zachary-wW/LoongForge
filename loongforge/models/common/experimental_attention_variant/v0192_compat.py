# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility surface for DSv4 features newer than the v0.19.2 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from megatron.core.transformer.experimental_attention_variant.dsa import (
    fused_qk_topk_naive,
)
from megatron.core.transformer.transformer_config import TransformerConfig

_DSA_INDEX_SHARE_HOLDER_ATTR = "_dsa_index_share_topk_holder"


@dataclass
class DSAIndexShareCarrier:
    """Per-forward holder for cross-layer DSA top-k sharing."""

    _dsa_index_share_topk_holder: Optional[dict] = None


def get_dsa_index_share_topk_holder(
    packed_seq_params,
    index_share_carrier,
    skip_topk: bool,
    layer_number: int,
    source_layer: int,
) -> Optional[dict]:
    """Return the holder used by GLM DSA IndexShare."""
    carrier = index_share_carrier if index_share_carrier is not None else packed_seq_params
    if carrier is None:
        if not skip_topk:
            return None
        raise RuntimeError(
            "DSA index sharing has no per-forward carrier at "
            f"layer_number={layer_number}, source_layer={source_layer}"
        )
    holder = getattr(carrier, _DSA_INDEX_SHARE_HOLDER_ATTR, None)
    if holder is None:
        holder = {}
        setattr(carrier, _DSA_INDEX_SHARE_HOLDER_ATTR, holder)
    return holder


def is_dsa_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """Return whether the layer reuses a previous layer's top-k indices."""
    skip_topk_offset = max(skip_topk_offset, 1)
    return max(layer_number - skip_topk_offset, 0) % topk_freq != 0


def source_dsa_compute_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> int:
    """Return the computing layer that owns this layer's top-k indices."""
    skip_topk_offset = max(skip_topk_offset, 1)
    if layer_number <= skip_topk_offset:
        return layer_number
    return layer_number - ((layer_number - skip_topk_offset) % topk_freq)


def validate_dsa_index_share_pipeline_split(
    config: TransformerConfig, local_layer_ids
) -> None:
    """Reject a PP split that would require cross-stage top-k sharing."""
    topk_freq = config.dsa_indexer_topk_freq
    if config.experimental_attention_variant != "dsa" or topk_freq <= 1 or not local_layer_ids:
        return
    layer_number = local_layer_ids[0] + 1
    if not is_dsa_skip_topk_layer(
        layer_number, config.dsa_indexer_skip_topk_offset, topk_freq
    ):
        return
    raise RuntimeError(
        "DSA index-share pipeline split is invalid: local layer "
        f"{layer_number} requires an indexer from an earlier pipeline stage."
    )


def batch_of_row(cu_seqlens_q: Tensor, total_q: Optional[int] = None) -> Tensor:
    """Map each packed query row to its sequence index."""
    if total_q is None:
        total_q = int(cu_seqlens_q[-1].item())
    num_sequences = cu_seqlens_q.shape[0] - 1
    row_idx = torch.arange(total_q, device=cu_seqlens_q.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens_q[1:], right=True).clamp(
        max=max(num_sequences - 1, 0)
    )


def local_to_global_flat(
    local_idxs: Tensor,
    batch_size: int,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
) -> Tensor:
    """Convert per-sequence local indices to flat SBHD/THD indices."""
    if cu_seqlens_q is not None and cu_seqlens_kv is not None:
        rows = local_idxs.shape[0]
        batch_ids = batch_of_row(cu_seqlens_q, total_q=rows)
        offsets = cu_seqlens_kv[batch_ids].to(local_idxs.dtype).unsqueeze(-1)
        return torch.where(local_idxs >= 0, local_idxs + offsets, -1).int()
    b, sq, topk = local_idxs.shape
    if batch_size not in (-1, b):
        raise ValueError(f"batch_size={batch_size} does not match index batch {b}")
    flat = local_idxs.permute(1, 0, 2).reshape(sq * b, topk)
    batch_ids = torch.arange(sq * b, device=local_idxs.device).remainder(b)
    offsets = batch_ids.to(local_idxs.dtype).unsqueeze(-1)
    return torch.where(flat >= 0, flat * b + offsets, -1).int()


def build_flat_topk_idxs(
    *idx_groups: Tensor,
    batch_size: int,
    compact: bool = False,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
):
    """Combine and globalize one or more top-k index groups."""
    combined = local_to_global_flat(
        torch.cat(idx_groups, dim=-1),
        batch_size,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
    )
    if not compact:
        return combined, None
    valid = combined >= 0
    order = valid.int().argsort(dim=-1, descending=True, stable=True)
    return combined.gather(-1, order), valid.sum(dim=-1).int()


def _causal_mask_for_compressed_kv(
    sq: int, sk: int, ratio: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    q_pos = torch.arange(sq, device=device).unsqueeze(1)
    k_pos = torch.arange(sk, device=device).unsqueeze(0)
    valid = k_pos < torch.div(q_pos + 1, max(ratio, 1), rounding_mode="floor").clamp(min=1)
    return torch.where(valid, 0.0, float("-inf")).to(dtype=dtype).unsqueeze(0)


def fused_qk_topk_naive_thd(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk: int,
    cu_seqlens_q: Tensor,
    cu_seqlens_kv: Tensor,
    ratio: int,
):
    """THD reference implementation of indexer scoring and top-k."""
    total_q = q.shape[0]
    topk = torch.full(
        (total_q, index_topk), -1, dtype=torch.int64, device=q.device
    )
    for batch_idx in range(cu_seqlens_q.numel() - 1):
        q_start = int(cu_seqlens_q[batch_idx])
        q_end = int(cu_seqlens_q[batch_idx + 1])
        k_start = int(cu_seqlens_kv[batch_idx])
        k_end = int(cu_seqlens_kv[batch_idx + 1])
        if q_start == q_end or k_start == k_end:
            continue
        q_b = q[q_start:q_end].unsqueeze(1)
        k_b = k[k_start:k_end].unsqueeze(1)
        w_b = weights[q_start:q_end].unsqueeze(1)
        mask = _causal_mask_for_compressed_kv(
            q_end - q_start, k_end - k_start, ratio, q.device, q.dtype
        )
        _, topk_b = fused_qk_topk_naive(q_b, k_b, w_b, index_topk, mask)
        topk[q_start:q_end, : topk_b.shape[-1]] = topk_b.squeeze(0)
    return None, topk


def dsa_sparse_attn(
    query: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
    is_thd: bool = False,
) -> Tensor:
    """Reference sparse attention used by the v0.19.2 compatibility path."""
    del indexer_topk
    if not is_thd:
        raise NotImplementedError("Use the backend fused SBHD path on v0.19.2.")
    if query.ndim != 3 or kv.ndim != 2:
        raise ValueError("THD query/kv must be 2-D or 3-D tensors")
    safe = topk_idxs.long().clamp(min=0)
    valid = topk_idxs >= 0
    gathered_kv = kv[safe]
    scores = torch.einsum("qhd,qkd->qhk", query.float(), gathered_kv.float())
    scores = scores * softmax_scale
    if attn_sink is not None:
        scores = scores + attn_sink.float().view(1, -1, 1)
    if topk_length is not None:
        positions = torch.arange(topk_idxs.shape[-1], device=query.device).unsqueeze(0)
        valid &= positions < topk_length.unsqueeze(1)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("qhk,qkd->qhd", probs, gathered_kv.float())
    return output.flatten(-2, -1).to(query.dtype)


def fused_indexer_sparse_attn(
    query,
    kv_full,
    attn_sink,
    window_idxs,
    q_indexer,
    k_indexer,
    weights,
    indexer_topk,
    ratio,
    softmax_scale,
    indexer_softmax_scale=1.0,
    loss_coeff=0.0,
    sparse_loss=False,
    kv_offset=0,
    calculate_per_token_loss=False,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    cu_seqlens_kv_full=None,
    cu_seqlens_compressed_idx=None,
    max_seqlen_q=None,
    max_seqlen_compressed_idx=None,
    compressed_kv=None,
    cu_seqlens_q_unpadded=None,
):
    """Reference THD fused indexer/attention entry for the compatibility path."""
    del (
        indexer_softmax_scale,
        sparse_loss,
        kv_offset,
        calculate_per_token_loss,
        max_seqlen_q,
        max_seqlen_compressed_idx,
        compressed_kv,
        cu_seqlens_q_unpadded,
    )
    if cu_seqlens_q is None:
        raise NotImplementedError("Use the v0.19.2 cuDNN fused SBHD path.")
    _, compressed_topk = fused_qk_topk_naive_thd(
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        cu_seqlens_q,
        cu_seqlens_compressed_idx,
        ratio,
    )
    groups = [window_idxs]
    if compressed_topk.shape[-1]:
        groups.append(compressed_topk)
    flat_kv_seqlens = (
        cu_seqlens_kv_full if cu_seqlens_kv_full is not None else cu_seqlens_kv
    )
    flat_idxs, _ = build_flat_topk_idxs(
        *groups,
        batch_size=-1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=flat_kv_seqlens,
    )
    output = dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        flat_idxs,
        softmax_scale,
        is_thd=True,
    )
    indexer_loss = query.new_zeros((), dtype=torch.float32) * loss_coeff
    return output, indexer_loss


__all__ = [
    "DSAIndexShareCarrier",
    "batch_of_row",
    "build_flat_topk_idxs",
    "dsa_sparse_attn",
    "fused_indexer_sparse_attn",
    "fused_qk_topk_naive_thd",
    "get_dsa_index_share_topk_holder",
    "validate_dsa_index_share_pipeline_split",
]
