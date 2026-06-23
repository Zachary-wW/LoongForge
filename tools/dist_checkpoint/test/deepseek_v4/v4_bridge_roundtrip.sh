#! /bin/bash
# HF Checkpoint Roundtrip Test — DeepSeek-V4-Flash-Base

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARNING

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/loongforge-ckpt/DeepSeek-V4-Flash-Base"}
SAVE_HF_PATH=${SAVE_HF_PATH:-"/workspace/loongforge-ckpt/deepseek-v4-roundtrip-output"}

GPUS_PER_NODE=8

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"12345"}
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
    --model-name deepseek-v4
    --multi-latent-attention
    --rotary-base 10000
    --original-max-position-embeddings 65536
    --mscale 1.0
    --mscale-all-dim 1.0
    --norm-epsilon 1e-6
    --rotary-scaling-factor 16
    --enable-fa-within-mla
    --rotary-interleaved
    --rotary-percent 0.125
)

TOKENIZER_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
)

TRAINING_ARGS=(
    --training-phase pretrain
    --seq-length 32768
    --max-position-embeddings 1048576
    --micro-batch-size 1
    --global-batch-size 4
    --bf16
    --norm-epsilon 1e-6
    # --- roundtrip-specific ---
    --train-iters 0
    --no-load-optim
    --no-load-rng
    --load $TOKENIZER_PATH
    --save-hf-path $SAVE_HF_PATH
    --save-hf=true
)

MOE_ARGS=(
    --moe-router-load-balancing-type none
    --moe-router-topk 6
    --moe-grouped-gemm
    --moe-router-enable-expert-bias
    --moe-router-score-function sqrtsoftplus
    --moe-router-topk-scaling-factor 1.5
    --moe-router-dtype fp32
    --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
    --attention-backend fused
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-token-dispatcher-type flex
    --moe-enable-deepep
    --use-distributed-optimizer
    --distributed-backend nccl
)

echo "========================================"
echo "HF Roundtrip Test — DeepSeek-V4-Flash-Base"
echo "  Source : $TOKENIZER_PATH"
echo "  Output : $SAVE_HF_PATH"
echo "========================================"

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/tools/dist_checkpoint/checkpoint/hf_roundtrip_test.py \
    ${MODEL_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]}
