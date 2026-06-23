# Phase 0 Metadata — DeepSeek-V4-Flash-Base

Run ID: ds_v4_loop_01
Phase: 0
Attempt: 1
Model: DeepSeek-V4-Flash-Base
Date: 2026-06-23

## Summary

Phase 0 analysis of DeepSeek-V4-Flash-Base for LoongForge adaptation.
Candidate family: deepseek_v3.

## Key Architectural Differences from V3

- Manifold-Constrained Hyper-Connections (mHC) with hc_mult=4 parallel residual streams
- Compressed Sparse Attention (CSA) / Heavily Compressed Attention (HCA) compressors
- Lightning Indexer for top-k compressed entry selection
- Hash-MoE bootstrap routing for first 3 layers (tid2eid lookup)
- Grouped output projection (o_groups=8, o_lora_rank=1024)
- Interleaved RoPE on trailing qk_rope_head_dim=64 of head_dim=512
- Per-head learnable attention sinks
- Inverse RoPE rotation on attention output
- FP8 block-wise quantized expert weights
- Shared-KV MQA (num_key_value_heads=1, K=V same tensor)

<!-- adapt-skill: run=ds_v4_loop_01 phase=0 attempt=1 kind=base -->
