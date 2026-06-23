#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# DeepSeek-V4-Flash-Base pretrain launch script
# V4: Shared-KV MQA + MoE + mHC + Hash-MoE bootstrap

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-""}

TOKENIZER_PATH=${TOKENIZER_PATH:-""}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-""}

GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --model-name deepseek_v4
    --rotary-base 10000
    --norm-epsilon 1e-6
    --multi-latent-attention
    --enable-hyper-connections
    --num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --rotary-interleaved
    --rotary-percent 0.125
    --apply-rope-fusion
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --eod-mask-loss
    --data-path $DATA_PATH
    --split 98,2,0
)

TRAINING_ARGS=(
    --training-phase pretrain
    --seq-length 4096
    --max-position-embeddings 1048576
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 512
    --lr 1e-4
    --min-lr 1e-5
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --norm-epsilon 1e-6
    --train-iters 2000
    --lr-decay-iters 2000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 500
    --eval-interval 100
    --eval-iters 10
)

# ── MoE-specific parameters ────────────────────────────────────────────────
MOE_ARGS=(
    --moe-router-load-balancing-type none
    --moe-router-topk 6
    --moe-aux-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-router-score-function sqrtsoftplus
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --routed-scaling-factor 1.5
    --num-hash-layers 3
    --swiglu-limit 10.0
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 32
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
    --distributed-backend nccl
)

# ── MTP ─────────────────────────────────────────────────────────────────────
MTP_ARGS=(
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${WANDB_NAME}
    )
fi

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MTP_ARGS[@]} \
    ${LOGGING_ARGS[@]}
