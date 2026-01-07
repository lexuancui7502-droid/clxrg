#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

GLOBAL_STAMP="$(date +%m%d_%H%M%S)"
BASE_OUTDIR="$ROOT_DIR/results/eval_stage2_remaining_${GLOBAL_STAMP}"
mkdir -p "$BASE_OUTDIR"

MODE="${MODE:-offline}"

# ====== 路径 ======
BASE_ROOT="${BASE_ROOT:-/media/cuilexuan/clx}"

TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:-/media/cuilexuan/clx/results/mv_stage2/biomedclip_cxr_518-stage2A_lockAnchor-0.3e-base0-mm0-ar5e-5-20260104020331}"

QUERY_FILE="${QUERY_FILE:-$BASE_ROOT/data/multiview-cxr-annotations-1.0.0/multiview_official_test_data.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files}"

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
CHUNK_IDX="${CHUNK_IDX:-0}"

check_path () { [[ -e "$1" ]] || { echo "[ERROR] Not found: $1" >&2; exit 1; }; }
check_path "$TRAIN_MODEL_PATH"
check_path "$QUERY_FILE"
check_path "$IMAGE_FOLDER"

# ====== 离线设置 ======
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
  echo "[INFO] MODE=offline"
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
  echo "[INFO] MODE=online"
fi

# 关闭 W&B
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true

# ====== AR-CVI 固定超参 ======
export AR_CVI_EVID=16
export AR_CVI_MEM=32
export AR_CVI_AUX_R=24
export AR_CVI_FFN_HIDDEN=1024

# ====== 关键：anchor 规则（与你现在 eval 代码一致）======
export AR_CVI_MATCH_BASELINE=1
export MV_BASELINE_PICK=PA_AP_FIRST

# 日志（全量测试集建议不要太密）
export AR_CVI_LOG_ANCHOR=1
export AR_CVI_LOG_EVERY=1024
export AR_CVI_DEBUG_ANCHOR=0

run_eval () {
  local TAG="$1"
  local MV_FUSION="$2"
  local DISABLE_FUSER="$3"
  local GATE_MAX="$4"
  local FORCE_REINIT="${5:-0}"
  local GATE_INIT="${6:--10}"

  local OUTDIR="$BASE_OUTDIR/$TAG"
  local RUN_NAME="mv_eval_${TAG}_${GLOBAL_STAMP}"
  mkdir -p "$OUTDIR"

  (
    export MV_FUSION="$MV_FUSION"
    export AR_CVI_DISABLE_FUSER="$DISABLE_FUSER"

    # ===== 诊断关键：可由参数覆盖 =====
    export AR_CVI_FORCE_REINIT="$FORCE_REINIT"
    export AR_CVI_GATE_INIT="$GATE_INIT"
    export AR_CVI_GATE_MAX="$GATE_MAX"

    export AR_CVI_HARD=0
    export AR_CVI_TAU=1.0

    echo "============================================================"
    echo "[RUN] $TAG"
    echo "  MODEL_PATH=$TRAIN_MODEL_PATH"
    echo "  MV_FUSION=$MV_FUSION"
    echo "  DISABLE_FUSER=$AR_CVI_DISABLE_FUSER"
    echo "  FORCE_REINIT=$AR_CVI_FORCE_REINIT"
    echo "  GATE_INIT=$AR_CVI_GATE_INIT"
    echo "  GATE_MAX=$AR_CVI_GATE_MAX"
    echo "  CHUNK_IDX=$CHUNK_IDX / NUM_CHUNKS=$NUM_CHUNKS"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -m llava.eval.model_mimic_cxr \
      --query_file "$QUERY_FILE" \
      --loader "mimic_multiview_findings" \
      --image_folder "$IMAGE_FOLDER" \
      --conv_mode "v1" \
      --prediction_file "$OUTDIR/test_${CHUNK_IDX}.jsonl" \
      --temperature 0 \
      --model_path "$TRAIN_MODEL_PATH" \
      --chunk_idx "$CHUNK_IDX" --num_chunks "$NUM_CHUNKS" \
      --batch_size "$BATCH_SIZE" \
      --group_by_length \
      2>&1 | tee "$LOG_DIR/infer_${TAG}_${GLOBAL_STAMP}.log"

    cp -f "$OUTDIR/test_${CHUNK_IDX}.jsonl" "$OUTDIR/mimic_cxr_preds.jsonl"

    # 这一步对“相同率”诊断不是必须，你可以先注释掉节省时间
    # pushd "$ROOT_DIR" >/dev/null
    # PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    # python -m llava.eval.rrg_eval.run \
    #   "$OUTDIR/mimic_cxr_preds.jsonl" \
    #   --run_name "$RUN_NAME" \
    #   --output_dir "$OUTDIR/eval" \
    #   --bootstrap_ci=False \
    #   2>&1 | tee "$LOG_DIR/eval_${TAG}_${GLOBAL_STAMP}.log"
    # popd >/dev/null

    echo "[OK] $TAG done. OUTDIR=$OUTDIR"
  )
}


# 仅跑“你还没跑的”：
# run_eval "02_stage2Train_baselineFusion"  "baseline" 1 0.0
# run_eval "03_stage2Train_arCvi_noFuse"    "ar_cvi"   1 0.0
# run_eval "04_stage2Train_arCvi_gate02"    "ar_cvi"   0 0.2
# run_eval "05_stage2Train_arCvi_gate05"    "ar_cvi"   0 0.5
# ===== A: 原始两张图 =====
export QUERY_FILE="$BASE_ROOT/data/tmp_eval_A_500.json"
run_eval "A_DIAG_reinit1_gateinit0_gateMax1" "ar_cvi" 0 50 0 -10

# ===== C: 第二张替换成随机一张 =====
export QUERY_FILE="$BASE_ROOT/data/tmp_eval_C_500_shuffle2nd.json"
run_eval "C_DIAG_reinit1_gateinit0_gateMax1_shuffle2nd" "ar_cvi" 0 50 0 -10