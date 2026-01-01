#!/usr/bin/env bash
set -euo pipefail

# =============== 基本路径 ===============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

GLOBAL_STAMP="$(date +%m%d_%H%M%S)"
BASE_OUTDIR="$ROOT_DIR/results/eval_stage1_min_${GLOBAL_STAMP}"
mkdir -p "$BASE_OUTDIR"

MODE="${MODE:-offline}"

# =============== 模型与数据路径（按你实际修改） ===============
BASE_ROOT="${BASE_ROOT:-/media/cuilexuan/clx}"

# 训练后的权重（你要评测的）
TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:-/media/cuilexuan/clx/results/mv_light/biomedclip_cxr_518-mv-light-0.1e-2e-5-20260101005650}"

# 基线权重（llava-rad-merged）
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-/media/cuilexuan/clx/weights/llava-rad-merged}"

QUERY_FILE="${QUERY_FILE:-$BASE_ROOT/data/multiview-cxr-annotations-1.0.0/multiview_official_test_data.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files}"
BATCH_SIZE="${BATCH_SIZE:-4}"

check_path () { [[ -e "$1" ]] || { echo "[ERROR] Not found: $1" >&2; exit 1; }; }
check_path "$TRAIN_MODEL_PATH"
check_path "$BASELINE_MODEL_PATH"
check_path "$QUERY_FILE"
check_path "$IMAGE_FOLDER"

# =============== 缓存/离线设置 ===============
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$BASE_ROOT/weights}"
export HF_HOME="${HF_HOME:-$WEIGHTS_ROOT/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
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
  echo "[INFO] MODE=online（允许联网）"
fi

# 关闭 W&B
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true

# =============== 固定的 AR-CVI 结构超参（与你训练一致） ===============
export AR_CVI_EVID=16
export AR_CVI_MEM=32
export AR_CVI_AUX_R=24
export AR_CVI_FFN_HIDDEN=1024

# 避免 eval 随机跳
export AR_CVI_HARD=0
export AR_CVI_TAU=1.0

# baseline 对齐：主视角=第0张（与你 Sanity-C 一致）
export AR_CVI_MATCH_BASELINE=1
export MV_BASELINE_PICK=INDEX
export MV_BASELINE_INDEX=0

# 打印主视角统计日志（可关）
export AR_CVI_LOG_ANCHOR=1
export AR_CVI_LOG_EVERY=1024

run_one () {
  local TAG="$1"
  local MODEL_PATH="$2"
  local MV_FUSION="$3"
  local DISABLE_FUSER="$4"
  local FORCE_REINIT="$5"
  local GATE_MAX="$6"

  local OUTDIR="$BASE_OUTDIR/$TAG"
  local RUN_NAME="mv_eval_${TAG}_${GLOBAL_STAMP}"
  mkdir -p "$OUTDIR"

  # 关键：每个 case 在子进程环境里设置变量，避免串扰
  (
    export MV_FUSION="$MV_FUSION"
    export AR_CVI_DISABLE_FUSER="$DISABLE_FUSER"
    export AR_CVI_FORCE_REINIT="$FORCE_REINIT"
    export AR_CVI_GATE_INIT=-10
    export AR_CVI_GATE_MAX="$GATE_MAX"

    echo "============================================================"
    echo "[RUN] $TAG"
    echo "  MODEL_PATH=$MODEL_PATH"
    echo "  MV_FUSION=$MV_FUSION"
    echo "  DISABLE_FUSER=$AR_CVI_DISABLE_FUSER"
    echo "  FORCE_REINIT=$AR_CVI_FORCE_REINIT"
    echo "  GATE_MAX=$AR_CVI_GATE_MAX"
    echo "============================================================"

    # =============== 推理 ===============
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
      2>&1 | tee "$LOG_DIR/infer_${TAG}_${GLOBAL_STAMP}.log"

    cp -f "$OUTDIR/test_0.jsonl" "$OUTDIR/mimic_cxr_preds.jsonl"

    # =============== 评测 ===============
    pushd "$ROOT_DIR" >/dev/null
    PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -m llava.eval.rrg_eval.run \
      "$OUTDIR/mimic_cxr_preds.jsonl" \
      --run_name "$RUN_NAME" \
      --output_dir "$OUTDIR/eval" \
      --bootstrap_ci=False \
      2>&1 | tee "$LOG_DIR/eval_${TAG}_${GLOBAL_STAMP}.log"
    popd >/dev/null

    echo "[OK] $TAG done."
    echo "  OUTDIR=$OUTDIR"
    echo "  EVAL=$OUTDIR/eval/$RUN_NAME/"
  )
}

# =============== Case 0: 基线金标准（baseline 权重 + baseline 融合）==============
run_one "00_baselineWeight_baselineFusion" \
  "$BASELINE_MODEL_PATH" "baseline" 0 0 0.0

# =============== Case 1: 训练后权重是否破坏 baseline 路径（训练权重 + baseline 融合）==============
run_one "01_trainWeight_baselineFusion" \
  "$TRAIN_MODEL_PATH" "baseline" 0 0 0.0

# =============== Case 2: ar_cvi 框架但不注入是否等价 baseline（训练权重 + ar_cvi + nofuse）==============
run_one "02_trainWeight_arCvi_noFuse" \
  "$TRAIN_MODEL_PATH" "ar_cvi" 1 0 0.0

# （可选）Case 3: Sanity-C 注入（训练权重 + ar_cvi + gate 很小）
# 只有当你想确认“注入极小也不会额外坏”时再打开
# run_one "03_trainWeight_arCvi_sanityInject" \
#   "$TRAIN_MODEL_PATH" "ar_cvi" 0 0 0.05
