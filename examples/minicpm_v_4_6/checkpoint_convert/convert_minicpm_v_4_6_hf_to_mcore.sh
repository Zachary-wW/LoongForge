#! /bin/bash

set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/data/weizhihao/agent/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/data/weizhihao/agent/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=${LOAD:-"/data/weizhihao/MiniCPM-V-4.6"}
SAVE=${SAVE:-"/data/weizhihao/agent/minicpm_v_4_6_mcore"}

SAVE_LANGUAGE_MODEL=${SAVE_LANGUAGE_MODEL:-"$SAVE/tmp/language-mcore"}
SAVE_VISION_MODEL=${SAVE_VISION_MODEL:-"$SAVE/tmp/vision-model-mcore"}
SAVE_ADAPTER=${SAVE_ADAPTER:-"$SAVE/tmp/adapter-mcore"}

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/minicpm_v_4_6.yaml
FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/ckpt_convert/minicpm_v_4_6_llm_convert.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/minicpm_v_4_6_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/minicpm_v_4_6_merger_convert.yaml

ETP=${ETP:-1}
DTP=${DTP:-1}
PP=${PP:-1}

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --mtp_num_layers 0

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron.py \
    --megatron_path $MEGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_VISION_MODEL/release \
    --adapter_path $SAVE_ADAPTER/release \
    --encoder_tensor_model_parallel_size $ETP \
    --decoder_tensor_model_parallel_size $DTP \
    --pipeline_model_parallel_size $PP \
    --save_ckpt_path $SAVE/release \
    --config_file $MODEL_CONFIG_FILE

echo release > "$SAVE/latest_checkpointed_iteration.txt"
