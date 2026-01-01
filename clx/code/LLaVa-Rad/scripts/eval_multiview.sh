#!/usr/bin/env bash
set -euo pipefail

# =============== 基本路径 ===============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

# =============== 加载环境变量 (.env.cache) ===============
if [ -f "$ROOT_DIR/.env.cache" ]; then
  set -a
  source "$ROOT_DIR/.env.cache"
  set +a
fi

MODE="${MODE:-offline}"   # 默认离线模式

# ==== 多视图相关环境变量 ====
# export MV_LAMBDA=0.0         # 或 0.05 / 0.1，看你想多大强度
# export MV_TOPK_PATCH=0       # 每个 study 额外选 16 个 patch token
# export CHEXPERT_LAMBDA=0.0

export AR_CVI_EVID=16       # 每个视角选 16 个 patch token。K 越大，辅助视角提供的信息越多，但噪声风险和计算也会上升。
export AR_CVI_MEM=32        # 共享 memory tokens 数量 M
export AR_CVI_AUX_R=24      # 辅助视角 patch token 的下采样参数
export AR_CVI_FFN_HIDDEN=1024    # AR-CVI 内部 FFN 的隐藏维度
export AR_CVI_LOG_ANCHOR=1      # 是否打印/统计主视角选择的日志开关

export MV_FUSION=ar_cvi           # 选择多视角融合模式为 AR-CVI
export AR_CVI_MATCH_BASELINE=1    # eval 时强制 AR-CVI 的主视角选择与 baseline 策略一致
export MV_BASELINE_PICK=INDEX     # baseline 的选图策略为“按列表下标选择”
export MV_BASELINE_INDEX=0        # 强制主视角=第 0 张
export AR_CVI_DISABLE_FUSER=0     # 是否禁用 fuser（融合注入）的开关。=1时直接返回主视角 patch，不走融合注入；=0：走完整流程
export AR_CVI_FORCE_REINIT=0      # 强制把 fuser 重置到“恒等附近”，避免 checkpoint 覆盖初始化


# 关键：把注入 gate 强约束到近零（Sanity-C 核心）
export AR_CVI_GATE_INIT=-10
export AR_CVI_GATE_MAX=0.05


# 关键：不要对齐 baseline
# export AR_CVI_MATCH_BASELINE=0
# [2025-12-15] 评测时关闭 hard gumbel（避免主视角随机跳）开始
export AR_CVI_HARD=0    # 是否使用 hard gumbel的路由选择。=1：训练中可用于“更像离散选择”，但 eval 会导致随机跳；=0：使用 soft/probabilistic 或 argmax
export AR_CVI_TAU=1.0   # gumbel-softmax 的温度参数
# [2025-12-15] 评测时关闭 hard gumbel（避免主视角随机跳）结束

# =============== 模型与数据路径 ===============
BASE_ROOT="${BASE_ROOT:-/media/cuilexuan/clx}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$BASE_ROOT/weights}"

# =============== 缓存/代理设置 ===============
export HF_HOME="${HF_HOME:-$WEIGHTS_ROOT/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$WEIGHTS_ROOT/torch_home}"
export OPEN_CLIP_CACHE_DIR="${OPEN_CLIP_CACHE_DIR:-$TORCH_HOME/open_clip}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
ulimit -n 8192 || true

if [ "$MODE" = "offline" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY='127.0.0.1,localhost,.local,*.huggingface.co,huggingface.co'
  echo "[INFO] MODE=offline（严格离线模式）"
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"
  echo "[INFO] MODE=online（允许联网）"
fi

# 关闭 W&B
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true

# ✅ 现在不再分开 Vicuna + LLaVA-Rad，只用合并后的权重
# BASE=/media/cuilexuan/clx/results/mv_light/biomedclip_cxr_518-mv-light-0.5e-5e-5-20251209111936
# LORA=/media/cuilexuan/clx/results/mv_light/biomedclip_cxr_518-stage2-lora-0.5e-2e-5-20251209235305
MODEL_PATH="/media/cuilexuan/clx/results/mv_light/biomedclip_cxr_518-mv-light-0.5e-3e-5-20251226223650"
# MODEL_PATH="/media/cuilexuan/clx/weights/llava-rad-merged"
QUERY_FILE="${QUERY_FILE:-$BASE_ROOT/data/multiview-cxr-annotations-1.0.0/multiview_official_test_data.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files}"

STAMP="$(date +%m%d_%H%M)"
OUTDIR="$ROOT_DIR/results/eval_${STAMP}"
RUN_NAME="mv_eval_${STAMP}"
BATCH_SIZE="${BATCH_SIZE:-4}"

#  =============== 自检 ===============
check_path () { [[ -e "$1" ]] || { echo "[ERROR] Not found: $1" >&2; exit 1; }; }

# 检查 base 模型、LoRA 目录、数据文件、图像目录是否存在
# for p in "$BASE" "$LORA" "$QUERY_FILE" "$IMAGE_FOLDER"; do
#   check_path "$p"
# done
for p in "$MODEL_PATH" "$QUERY_FILE" "$IMAGE_FOLDER"; do
  check_path "$p"
done

mkdir -p "$OUTDIR"

# echo "[INFO] Using LoRA model: $LORA"
# echo "[INFO] Base model: $BASE"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] Logs => $LOG_DIR"


#  
#  --model_path ${LORA} \
#  --model_base ${BASE} \
# =============== 推理 ===============
set -x
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m llava.eval.model_mimic_cxr \
  --query_file "$QUERY_FILE" \
  --loader "mimic_multiview_findings" \
  --image_folder "$IMAGE_FOLDER" \
  --conv_mode "v1" \
  --prediction_file "$OUTDIR/test_0.jsonl" \
  --temperature 0 \
  --model_path "$MODEL_PATH" \
  --chunk_idx 0 --num_chunks 1 \
  --batch_size "$BATCH_SIZE" \
  --group_by_length \
  2>&1 | tee "$LOG_DIR/infer_${STAMP}.log"
set +x

cp -f "$OUTDIR/test_0.jsonl" "$OUTDIR/mimic_cxr_preds.jsonl"

# =============== 评测 ===============
pushd "$ROOT_DIR" >/dev/null
set -x
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m llava.eval.rrg_eval.run \
  "$OUTDIR/mimic_cxr_preds.jsonl" \
  --run_name "$RUN_NAME" \
  --output_dir "$OUTDIR/eval" \
  --bootstrap_ci=False \
  2>&1 | tee "$LOG_DIR/eval_${STAMP}.log"
set +x
popd >/dev/null

echo "[OK] Inference + Evaluation done."
echo "Predictions:   $OUTDIR/test_0.jsonl"
echo "Eval results:  $OUTDIR/eval/$RUN_NAME/"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] MV_FUSION=$MV_FUSION"
