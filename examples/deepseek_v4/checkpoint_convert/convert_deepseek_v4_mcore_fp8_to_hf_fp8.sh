#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=${LOAD:-"/mnt/cluster/loongforge-ckpt/deepseek_v4/DeepSeek-V4-Flash-Base-tp8pp8ep32etp1/release"}
SAVE=${SAVE:-"/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base-hf"}

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/deepseek_v4/deepseek_v4_flash_base.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/deepseek_v4/ckpt_convert/deepseek_v4_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --expert_parallel_size=32 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --custom_pipeline_layers 5,6,6,6,6,6,6,4 \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm \
    --fp8_force_no_requant
