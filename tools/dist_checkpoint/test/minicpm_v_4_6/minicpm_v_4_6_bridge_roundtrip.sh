#! /bin/bash
# MiniCPM-V-4.6 offline HF -> MCore -> HF roundtrip gate.

set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/data/weizhihao/agent/LoongForge"}
export MEGATRON_PATH=${MEGATRON_PATH:-"/data/weizhihao/agent/Loong-Megatron"}

HF_MODEL_PATH=${HF_MODEL_PATH:-"/data/weizhihao/MiniCPM-V-4.6"}
SAVE_ROOT=${SAVE_ROOT:-"/data/weizhihao/agent/minicpm_v_4_6_bridge_roundtrip"}
MCORE_PATH=${MCORE_PATH:-"$SAVE_ROOT/mcore"}
REBUILT_HF_PATH=${REBUILT_HF_PATH:-"$SAVE_ROOT/rebuilt_hf"}
REPORT_PATH=${REPORT_PATH:-"$SAVE_ROOT/roundtrip_comparison.json"}

rm -rf "$SAVE_ROOT"
mkdir -p "$SAVE_ROOT"

LOAD="$HF_MODEL_PATH" \
SAVE="$MCORE_PATH" \
LOONGFORGE_PATH="$LOONGFORGE_PATH" \
MEGATRON_PATH="$MEGATRON_PATH" \
PYTHONPATH="$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-}" \
bash "$LOONGFORGE_PATH/examples/minicpm_v_4_6/checkpoint_convert/convert_minicpm_v_4_6_hf_to_mcore.sh"

LOAD="$MCORE_PATH/release" \
SAVE="$REBUILT_HF_PATH" \
ORIGINAL_HF_PATH="$HF_MODEL_PATH" \
LOONGFORGE_PATH="$LOONGFORGE_PATH" \
MEGATRON_PATH="$MEGATRON_PATH" \
PYTHONPATH="$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-}" \
bash "$LOONGFORGE_PATH/examples/minicpm_v_4_6/checkpoint_convert/convert_minicpm_v_4_6_mcore_to_hf.sh"

python - "$HF_MODEL_PATH/model.safetensors" "$REBUILT_HF_PATH/model.safetensors" "$REPORT_PATH" <<'PY'
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import safe_open

original_path, rebuilt_path, report_path = sys.argv[1:4]

with safe_open(original_path, framework="pt", device="cpu") as original, safe_open(
    rebuilt_path, framework="pt", device="cpu"
) as rebuilt:
    original_keys = set(original.keys())
    rebuilt_keys = set(rebuilt.keys())
    missing = sorted(original_keys - rebuilt_keys)
    unexpected = sorted(rebuilt_keys - original_keys)
    shape_mismatch = []
    dtype_mismatch = []
    value_mismatch = []
    max_abs = 0.0
    max_abs_key = None
    mean_abs_sum = 0.0
    value_count = 0

    for key in sorted(original_keys & rebuilt_keys):
        original_tensor = original.get_tensor(key)
        rebuilt_tensor = rebuilt.get_tensor(key)
        if tuple(original_tensor.shape) != tuple(rebuilt_tensor.shape):
            shape_mismatch.append(
                {"key": key, "original": list(original_tensor.shape), "rebuilt": list(rebuilt_tensor.shape)}
            )
            continue
        if original_tensor.dtype != rebuilt_tensor.dtype:
            dtype_mismatch.append({"key": key, "original": str(original_tensor.dtype), "rebuilt": str(rebuilt_tensor.dtype)})

        diff = (original_tensor.float() - rebuilt_tensor.float()).abs()
        tensor_max = diff.max().item() if diff.numel() else 0.0
        mean_abs_sum += diff.sum().item()
        value_count += diff.numel()
        if tensor_max > max_abs:
            max_abs = tensor_max
            max_abs_key = key
        if not torch.equal(original_tensor, rebuilt_tensor):
            value_mismatch.append(
                {"key": key, "max_abs": tensor_max, "mean_abs": diff.mean().item() if diff.numel() else 0.0}
            )

report = {
    "status": "passed",
    "original_key_count": len(original_keys),
    "rebuilt_key_count": len(rebuilt_keys),
    "missing_count": len(missing),
    "unexpected_count": len(unexpected),
    "shape_mismatch_count": len(shape_mismatch),
    "dtype_mismatch_count": len(dtype_mismatch),
    "value_mismatch_count": len(value_mismatch),
    "max_abs": max_abs,
    "max_abs_key": max_abs_key,
    "mean_abs_global": mean_abs_sum / max(value_count, 1),
    "missing_sample": missing[:20],
    "unexpected_sample": unexpected[:20],
    "shape_mismatch_sample": shape_mismatch[:20],
    "dtype_mismatch_sample": dtype_mismatch[:20],
    "value_mismatch_sample": value_mismatch[:20],
}
if any(
    report[key] != 0
    for key in [
        "missing_count",
        "unexpected_count",
        "shape_mismatch_count",
        "dtype_mismatch_count",
        "value_mismatch_count",
    ]
):
    report["status"] = "failed"

Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if report["status"] != "passed":
    raise SystemExit(1)
PY
