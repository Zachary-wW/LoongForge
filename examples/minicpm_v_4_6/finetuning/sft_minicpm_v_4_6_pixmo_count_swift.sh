#! /bin/bash

set -euo pipefail

SWIFT_VENV=${SWIFT_VENV:-"/data/weizhihao/agent/venvs/ms-swift-minicpmv46"}
HF_TRANSFORMERS_PATH=${HF_TRANSFORMERS_PATH:-"/data/weizhihao/agent/hf-transformers-minicpmv46/src"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"/data/weizhihao/MiniCPM-V-4.6"}
DATA_PATH=${DATA_PATH:-"/data/weizhihao/agent/datasets/pixmo-count/swift/pixmo_count_openai.jsonl"}
RUN_DIR=${RUN_DIR:-"/data/weizhihao/agent/train_runs/minicpm_v_4_6_pixmo_count/swift"}
MAX_STEPS=${MAX_STEPS:-100}

mkdir -p "$RUN_DIR"

source "$SWIFT_VENV/bin/activate"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export DOWNSAMPLE_MODE=${DOWNSAMPLE_MODE:-"4x"}
export PYTHONPATH=$HF_TRANSFORMERS_PATH:${PYTHONPATH:-}

swift sft \
    --model "$HF_MODEL_PATH" \
    --model_type minicpmv4_6 \
    --template minicpmv4_6 \
    --dataset "$DATA_PATH" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --bf16 true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-05 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --packing false \
    --max_steps "$MAX_STEPS" \
    --seed 1234 \
    --data_seed 1234 \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --freeze_vit false \
    --freeze_aligner false \
    --attn_impl flash_attn \
    --loss_scale ignore_empty_think \
    --enable_channel_loss true \
    --save_steps 10000000 \
    --logging_steps 1 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 1 \
    --load_from_cache_file false \
    --output_dir "$RUN_DIR" \
    --logging_dir "$RUN_DIR/tensorboard" \
    --report_to tensorboard \
    --add_non_thinking_prefix true 2>&1 | tee "$RUN_DIR/train.log"
