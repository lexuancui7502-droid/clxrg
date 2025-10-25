#!/bin/bash

# Set the following variables correspondingly to run this script:

################## 基础配置部分 ##################
PROMPT_VERSION=v1       # 使用的提示版本

model_base=lmsys/vicuna-7b-v1.5     # 基座 LLM（Vicuna-7B）
output_dir="${1:-/media/cuilexuan/clx/results/llavarad_finetune_lora}"    # 输出目录

PROJECTOR="/PATH_TO/mm_projector.bin" # generated using pretrain.sh
################## 视觉编码器配置 ##################
vision_tower="biomedclip_cxr_518"
vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
vision_tower_checkpoint="biomedclipcxr_518_checkpoint.pt"
################## 视觉编码器配置 ##################


################## 数据配置 ##################
data_path=/media/cuilexuan/clx/data/llava-rad-mimic-cxr-annotations-1.0.0/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json
loader="mimic_train_findings"
image_folder=/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files 
################## 数据配置 ##################

# [2025-10-20] 对训练超参数配置进行修改
################## 训练超参数和运行名称 ##################
epoch="${2:-1}"     # 训练轮数，支持命令行参数
bsz="${3:-1}"      # 批次大小
grad_acc="${4:-16}"   # 梯度累积步数
lr="1e-4"           # 学习率（比预训练更小）
schedule="lora-${epoch}e"
export run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
################## 训练超参数和运行名称 ##################


# Batch size is set for 4-GPU machines.
# 核心训练命令 - LoRA 配置
WANDB_PROJECT="llava" WANDB_RUN_ID="llava-ft-$(date +%Y%m%d%H%M%S)" WANDB_RUN_GROUP=fine-tune \
    deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --model_name_or_path ${model_base} \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --loader mimic_train_findings \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
    --pretrain_mm_mlp_adapter ${PROJECTOR} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${grad_acc} \
    --evaluation_strategy "no" \
    --save_strategy "steps" --save_steps 5000 --save_total_limit 1 \
    --learning_rate ${lr} --weight_decay 0.0 \
    --warmup_ratio 0.03 --lr_scheduler_type "cosine" \
    --logging_steps 10 --tf32 True \
    --model_max_length 2048 --gradient_checkpointing True \
    --lazy_preprocess True --dataloader_num_workers 4 \
    --report_to wandb --run_name ${run_name}