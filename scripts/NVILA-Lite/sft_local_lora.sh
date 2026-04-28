#!/bin/bash
set -euo pipefail

DEFAULT_RUN_NAME="vila15-3b-reva-qlora"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=16
DEFAULT_GRADIENT_ACCUMULATION_STEPS=16
DEFAULT_GPUS=1
DEFAULT_NUM_VIDEO_FRAMES=4
DEFAULT_MODEL_MAX_LENGTH=2048

STAGE_PATH=${1:-"Efficient-Large-Model/VILA1.5-3b"}
DATA_MIXTURE=${2:-"reva_v1_train"}
OUTPUT_DIR=${3:-"runs/train/reva-v1-vila15-3b-qlora"}

RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}
GLOBAL_TRAIN_BATCH_SIZE=${GLOBAL_TRAIN_BATCH_SIZE:-$DEFAULT_GLOBAL_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}
GPUS=${GPUS:-$DEFAULT_GPUS}
NUM_VIDEO_FRAMES=${NUM_VIDEO_FRAMES:-$DEFAULT_NUM_VIDEO_FRAMES}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-$DEFAULT_MODEL_MAX_LENGTH}
PER_DEVICE_TRAIN_BATCH_SIZE=$((GLOBAL_TRAIN_BATCH_SIZE / GPUS / GRADIENT_ACCUMULATION_STEPS))
LOG_DIR=${LOG_DIR:-runs/log}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${TIMESTAMP}.log"

if [ "$PER_DEVICE_TRAIN_BATCH_SIZE" -lt 1 ]; then
    echo "PER_DEVICE_TRAIN_BATCH_SIZE must be >= 1. Increase GLOBAL_TRAIN_BATCH_SIZE or lower GRADIENT_ACCUMULATION_STEPS."
    exit 1
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

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
        --vision_tower google/siglip-so400m-patch14-384 \
        --mm_projector mlp2x_gelu \
        --mm_vision_select_feature patch \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --num_video_frames $NUM_VIDEO_FRAMES \
        --bf16 True \
        --bits 4 \
        --optim paged_adamw_32bit \
        --lora_enable True \
        --lora_llm True \
        --lora_vt False \
        --tune_language_model False \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 200 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length $MODEL_MAX_LENGTH \
        --gradient_checkpointing True \
        --dataloader_num_workers 2 \
        --vflan_no_system_prompt True \
        --report_to ${REPORT_TO:-none}
