#! /bin/bash

set -euo pipefail

SWIFT_VENV=${SWIFT_VENV:-"/data/weizhihao/agent/venvs/ms-swift-minicpmv46"}
HF_TRANSFORMERS_PATH=${HF_TRANSFORMERS_PATH:-"/data/weizhihao/agent/hf-transformers-minicpmv46/src"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"/data/weizhihao/MiniCPM-V-4.6"}
LOONG_BATCH_DIR=${LOONG_BATCH_DIR:-"/data/weizhihao/agent/train_runs/minicpm_v_4_6_pixmo_count/loongforge_exported_input_10step/exported_batches"}
RUN_DIR=${RUN_DIR:-"/data/weizhihao/agent/train_runs/minicpm_v_4_6_pixmo_count/swift_loong_batches_10step"}
MAX_STEPS=${MAX_STEPS:-10}

mkdir -p "$RUN_DIR"

source "$SWIFT_VENV/bin/activate"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=$HF_TRANSFORMERS_PATH:${PYTHONPATH:-}

python "$(dirname "$0")/train_loong_batches_swift.py" \
    --model "$HF_MODEL_PATH" \
    --batch-dir "$LOONG_BATCH_DIR" \
    --output-dir "$RUN_DIR" \
    --max-steps "$MAX_STEPS" \
    --gradient-accumulation-steps 16 \
    --learning-rate 5.0e-6 \
    --warmup-steps "${LR_WARMUP_ITERS:-5}" \
    --adam-eps 1.0e-5 \
    --seed 1234 \
    --attn-implementation flash_attention_2 \
    --downsample-mode "${DOWNSAMPLE_MODE:-4x}" 2>&1 | tee "$RUN_DIR/train.log"
