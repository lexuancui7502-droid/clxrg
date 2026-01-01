#!/bin/bash
export HF_HOME=/media/cuilexuan/clx/weights/hf_home
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ==== 多视图相关环境变量 ====
# export MV_LAMBDA=0.0         # 或 0.05 / 0.1，看你想多大强度
# export MV_TOPK_PATCH=0       # 每个 study 额外选 8 个 patch token
# export CHEXPERT_LAMBDA=0.0   # 原本设置为0.1 

# ==== 多视角相关环境变量 ====
export MV_FUSION=ar_cvi

# ---- AR-CVI核心参数 ----
export AR_CVI_EVID=16
export AR_CVI_MEM=32
export AR_CVI_AUX_R=24
export AR_CVI_FFN_HIDDEN=1024
export AR_CVI_LOG_ANCHOR=1
export AR_CVI_LOG_EVERY=1024

# ---- Sanity-C 配置：打开注入，但强约束为近零 ----
export AR_CVI_DISABLE_FUSER=0      # 开启 fuser 注入
export AR_CVI_FORCE_REINIT=0       # 评测/初期训练时强制重置为恒等映射
export AR_CVI_GATE_INIT=-10        # 初始 gate 非常小
export AR_CVI_GATE_MAX=0.01        # 最大融合比例极低
export AR_CVI_MATCH_BASELINE=1     # 主视角选择与 baseline 一致
export MV_BASELINE_PICK=INDEX
export MV_BASELINE_INDEX=0
export AR_CVI_HARD=0               # eval 阶段不使用硬路由
export AR_CVI_TAU=1.0

PROMPT_VERSION=v1
RAD_BASE=/media/cuilexuan/clx/weights/llava-rad-merged

vision_tower="biomedclip_cxr_518"
vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
vision_tower_checkpoint="biomedclipcxr_518_checkpoint.pt"

data_path=/media/cuilexuan/clx/data/multiview-cxr-annotations-1.0.0/multiview_and_single_train_data_gpt4.json
loader="mimic_multiview_findings"
image_folder=/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files

epoch="${1:-0.5}"
bsz="${2:-1}"
grad_acc="${3:-16}"
lr="2e-5"
mm_lr="0"
ar_lr="${lr}"

output_root="/media/cuilexuan/clx/results/mv_light"
schedule="mv-light-${epoch}e"
export run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name

deepspeed llava/train/train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path ${RAD_BASE} \
  --version $PROMPT_VERSION \
  --data_path ${data_path} \
  --loader ${loader} \
  --image_folder ${image_folder} \
  --vision_tower ${vision_tower} \
  --vision_tower_config ${vision_tower_config} \
  --vision_tower_checkpoint ${vision_tower_checkpoint} \
  --mm_projector_type mlp2x_gelu \
  --bf16 True \
  --output_dir ${output_root}/${run_name} \
  --num_train_epochs ${epoch} \
  --per_device_train_batch_size ${bsz} \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps ${grad_acc} \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --learning_rate ${lr} \
  --mm_projector_lr ${mm_lr} \
  --ar_cvi_lr ${ar_lr} \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 --lr_scheduler_type "constant_with_warmup" \
  --freeze_backbone True \
  --freeze_mm_projector True \
  --freeze_mm_mlp_adapter True \
  --mm_vision_select_layer -2 \
  --mm_vision_select_feature patch \
  --mm_use_im_patch_token False \
  --mm_use_im_start_end False \
  --logging_steps 10 --tf32 True \
  --model_max_length 2048 --gradient_checkpointing True \
  --lazy_preprocess True --dataloader_num_workers 4 \
  --report_to none --run_name ${run_name} \
  --optim adamw_hf