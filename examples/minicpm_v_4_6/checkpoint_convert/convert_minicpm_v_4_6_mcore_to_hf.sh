#! /bin/bash

set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/data/weizhihao/agent/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/data/weizhihao/agent/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=${LOAD:-"/data/weizhihao/agent/minicpm_v_4_6_mcore/release"}
SAVE=${SAVE:-"/data/weizhihao/agent/minicpm_v_4_6_roundtrip_hf"}
ORIGINAL_HF_PATH=${ORIGINAL_HF_PATH:-"/data/weizhihao/MiniCPM-V-4.6"}

TMP_DIR=${TMP_DIR:-"$SAVE/tmp"}
REVERSED_MCORE=${REVERSED_MCORE:-"$TMP_DIR/reversed-mcore"}
SAVE_LANGUAGE_MODEL=${SAVE_LANGUAGE_MODEL:-"$TMP_DIR/language-hf"}
SAVE_VISION_MODEL=${SAVE_VISION_MODEL:-"$TMP_DIR/vision-model-hf"}
SAVE_ADAPTER=${SAVE_ADAPTER:-"$TMP_DIR/adapter-hf"}

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/minicpm_v_4_6.yaml
FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/ckpt_convert/minicpm_v_4_6_llm_convert.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/minicpm_v_4_6_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/minicpm_v_4_6_merger_convert.yaml

ETP=${ETP:-1}
DTP=${DTP:-1}
PP=${PP:-1}

rm -rf "$TMP_DIR" "$SAVE"
mkdir -p "$TMP_DIR"

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser.py \
    --load_omni_ckpt_path $LOAD \
    --save_original_ckpt_path $REVERSED_MCORE \
    --decoder_tensor_model_parallel_size $DTP \
    --pipeline_model_parallel_size $PP \
    --config_file $MODEL_CONFIG_FILE

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$REVERSED_MCORE \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --mtp_num_layers 0

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$REVERSED_MCORE \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$REVERSED_MCORE \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/huggingface/merge_huggingface.py \
    --megatron_path $MEGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --vision_model_path $SAVE_VISION_MODEL \
    --vision_patch $SAVE_VISION_MODEL \
    --adapter_path $SAVE_ADAPTER \
    --save_ckpt_path $SAVE

python - "$ORIGINAL_HF_PATH" "$SAVE" <<'PY'
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
skip_names = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}
for path in source.iterdir():
    if not path.is_file():
        continue
    if path.name in skip_names or path.name.startswith("model-"):
        continue
    shutil.copy2(path, target / path.name)
PY

rm -rf "$TMP_DIR"
