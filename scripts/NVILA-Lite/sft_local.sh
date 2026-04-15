#!/bin/bash
set -euo pipefail

DEFAULT_RUN_NAME="vila-qwen2-vl-7b-sft-local"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=8
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1
DEFAULT_GPUS=1

STAGE_PATH=${1:-"Efficient-Large-Model/NVILA-Lite-8B"}
DATA_MIXTURE=${2:-"reva_v1_train"}
OUTPUT_DIR=${3:-"runs/train/reva-v1-sft"}

RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}
GLOBAL_TRAIN_BATCH_SIZE=${GLOBAL_TRAIN_BATCH_SIZE:-$DEFAULT_GLOBAL_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}
GPUS=${GPUS:-$DEFAULT_GPUS}
PER_DEVICE_TRAIN_BATCH_SIZE=$((GLOBAL_TRAIN_BATCH_SIZE / GPUS / GRADIENT_ACCUMULATION_STEPS))

if [ "$PER_DEVICE_TRAIN_BATCH_SIZE" -lt 1 ]; then
    echo "PER_DEVICE_TRAIN_BATCH_SIZE must be >= 1. Lower GRADIENT_ACCUMULATION_STEPS or increase GLOBAL_TRAIN_BATCH_SIZE."
    exit 1
fi

export WANDB_PROJECT=${WANDB_PROJECT:-vila}
export WANDB_DIR=${WANDB_DIR:-$OUTPUT_DIR}
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
export WANDB_RESUME=${WANDB_RESUME:-allow}

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --data_mixture $DATA_MIXTURE \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_3x3_fix \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --vflan_no_system_prompt True \
        --report_to ${REPORT_TO:-none}
