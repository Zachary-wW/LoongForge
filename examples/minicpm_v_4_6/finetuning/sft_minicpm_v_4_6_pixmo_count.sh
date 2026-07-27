#! /bin/bash

set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/data/weizhihao/agent/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/data/weizhihao/agent/LoongForge"}
HF_TRANSFORMERS_PATH=${HF_TRANSFORMERS_PATH:-"/data/weizhihao/agent/hf-transformers-minicpmv46/src"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"/data/weizhihao/MiniCPM-V-4.6"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/data/weizhihao/agent/minicpm_v_4_6_script_smoke"}
RAW_DATA_PATH=${RAW_DATA_PATH:-"/data/weizhihao/agent/datasets/pixmo-count/loongforge/pixmo_count_openai.jsonl"}
DATA_PATH=${DATA_PATH:-"/data/weizhihao/agent/datasets/pixmo-count/loongforge_tokenized"}
RUN_DIR=${RUN_DIR:-"/data/weizhihao/agent/train_runs/minicpm_v_4_6_pixmo_count/loongforge"}
export DOWNSAMPLE_MODE=${DOWNSAMPLE_MODE:-"4x"}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"

if [[ ! -d "$DATA_PATH/preprocess/0" ]]; then
    rm -rf "$DATA_PATH"
    PYTHONPATH=$HF_TRANSFORMERS_PATH:$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
        python ${LOONGFORGE_PATH}/tools/data_preprocess/llm/preprocess_sft_data.py \
            --input "$RAW_DATA_PATH" \
            --output-path "$DATA_PATH/preprocess/0" \
            --seq-length 4096 \
            --chat-template minicpm-v-4.6-hf \
            --chat-template-kwargs '{"add_generation_prompt":false}' \
            --downsample-mode "$DOWNSAMPLE_MODE" \
            --tokenizer-type HFTokenizer \
            --hf-tokenizer-path "$HF_MODEL_PATH" \
            --workers 1 \
            --split 100,0,0 \
            --sft-dataset-config ${LOONGFORGE_PATH}/configs/data/sft_dataset_config.yaml \
            --sft-dataset openai
fi

MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/minicpm_v_4_6.yaml

GPUS_PER_NODE=${GPUS_PER_NODE:-${GPU_PER_NODE:-1}}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash}
TRAIN_ITERS=${TRAIN_ITERS:-100}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-4096}
LR=${LR:-5.0e-6}
MIN_LR=${MIN_LR:-5.0e-6}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6017"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
)

if [[ -n "$LR_WARMUP_ITERS" ]]; then
    LR_WARMUP_ARGS=(--lr-warmup-iters "$LR_WARMUP_ITERS")
else
    LR_WARMUP_ARGS=(--lr-warmup-fraction 0.05)
fi

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $HF_MODEL_PATH
    --data-path "$DATA_PATH"
    --dataloader-type single
    --split 100,0,0
    --num-workers 0
    --is-tokenized-data
    --chat-template minicpm-v-4.6-hf
    --chat-template-kwargs '{"add_generation_prompt":false}'
    --sft-dataset-config ${LOONGFORGE_PATH}/configs/data/sft_dataset_config.yaml
    --sft-dataset openai
)

TRAINING_ARGS=(
    --training-phase sft
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --micro-batch-size 1
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr $LR
    --min-lr $MIN_LR
    --clip-grad 1.0
    --weight-decay 0.0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    --train-iters $TRAIN_ITERS
    --eval-iters 0
    --lr-decay-iters $TRAIN_ITERS
    --lr-decay-style constant
    ${LR_WARMUP_ARGS[@]}
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save "$RUN_DIR/checkpoints"
    --save-interval 10000000
    --ckpt-format torch
    --finetune
    --no-load-optim
    --no-load-rng
    --no-save-optim
    --no-save-rng
)

MODEL_PARALLEL_ARGS=(
    --attention-backend $ATTENTION_BACKEND
    --pipeline-model-parallel-size 1
    --tensor-model-parallel-size 1
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "$RUN_DIR/tensorboard"
    --tensorboard-log-interval 1
)

PYTHONPATH=$HF_TRANSFORMERS_PATH:$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} 2>&1 | tee "$RUN_DIR/logs/train.log"
