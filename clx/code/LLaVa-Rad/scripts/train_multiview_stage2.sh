#!/bin/bash
export HF_HOME=/media/cuilexuan/clx/weights/hf_home
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ========= 多视角融合方式 =========
export MV_FUSION=ar_cvi

# ========= AR-CVI 结构超参（与 Stage1 一致）=========
export AR_CVI_EVID=16
export AR_CVI_MEM=32
export AR_CVI_AUX_R=24
export AR_CVI_FFN_HIDDEN=1024
export AR_CVI_LOG_ANCHOR=1
export AR_CVI_LOG_EVERY=1024

# ========= Stage2-A：固定主视角（需要你按第2节改代码后才生效）=========
export AR_CVI_MATCH_BASELINE=1
export AR_CVI_MATCH_BASELINE_TRAIN=1
export MV_BASELINE_PICK=INDEX
export MV_BASELINE_INDEX=0

# ========= 注入强度：先小 =========
export AR_CVI_DISABLE_FUSER=0
export AR_CVI_GATE_INIT=-4
export AR_CVI_GATE_MAX=0.2

# ========= 路由：先不学（固定主视角期）=========
export AR_CVI_HARD=0
export AR_CVI_TAU=1.0

PROMPT_VERSION=v1
vision_tower="biomedclip_cxr_518"
vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
vision_tower_checkpoint="biomedclipcxr_518_checkpoint.pt"

data_path=/media/cuilexuan/clx/data/multiview-cxr-annotations-1.0.0/multiview_and_single_train_data_gpt4.json
loader="mimic_multiview_findings"
image_folder=/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files

# ====== 从 Stage1 checkpoint 权重初始化 ======
STAGE1_DIR=/media/cuilexuan/clx/results/mv_light/biomedclip_cxr_518-mv-light-0.1e-2e-5-20251230223136
STAGE1_CKPT="${STAGE1_DIR}/checkpoint-<xxxx>"   # 改成你实际的最后一个 checkpoint
RAD_BASE=${STAGE1_CKPT}

epoch="${1:-0.3}"
bsz="${2:-1}"
grad_acc="${3:-16}"

base_lr="2e-5"
mm_lr="1e-5"
ar_lr="5e-5"

output_root="/media/cuilexuan/clx/results/mv_stage2"
schedule="stage2A_lockAnchor-${epoch}e"
export run_name="${vision_tower}-${schedule}-base${base_lr}-mm${mm_lr}-ar${ar_lr}-$(date +%Y%m%d%H%M%S)"
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
  --learning_rate ${base_lr} \
  --mm_projector_lr ${mm_lr} \
  --ar_cvi_lr ${ar_lr} \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 --lr_scheduler_type "constant_with_warmup" \
  --freeze_backbone True \
  --freeze_mm_projector False \
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
