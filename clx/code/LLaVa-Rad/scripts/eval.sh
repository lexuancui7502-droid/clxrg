# [2025-10-19] eval.sh 脚本，支持在线/离线两种模式运行 LLaVa-Rad 推理与评测。
#!/usr/bin/env bash
set -euo pipefail

# =============== 基本路径 ===============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

# 加载 .env.cache（若有）
if [ -f "$ROOT_DIR/.env.cache" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.env.cache"
  set +a
fi

MODE="${MODE:-offline}"   # 默认离线，更稳

# =============== 缓存/代理 ===============
# 统一固定到 hub 树（与 warm_cache.sh 保持一致）
export HF_HOME="${HF_HOME:-$WEIGHTS_ROOT/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$WEIGHTS_ROOT/torch_home}"
export OPEN_CLIP_CACHE_DIR="${OPEN_CLIP_CACHE_DIR:-$TORCH_HOME/open_clip}"

# 运行开关
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
ulimit -n 8192 || true

if [ "$MODE" = "offline" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  # 断代理（防止某些库因代理存在而误判）
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY='127.0.0.1,localhost,.local,*.huggingface.co,huggingface.co'
  echo "[INFO] MODE=offline（严格离线）。若缺文件请先执行 scripts/warm_cache.sh"
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
  # 可选代理（按需）
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"
  echo "[INFO] MODE=online（允许联网探测/下载）。网络不稳可能失败，建议先 warm_cache。"
fi

# 关闭 W&B
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true

# =============== 模型/数据路径 ===============
BASE_ROOT="${BASE_ROOT:-/media/cuilexuan/clx}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$BASE_ROOT/weights}"

MODEL_BASE="${MODEL_BASE:-$WEIGHTS_ROOT/vicuna-7b-v1.5}"          # Vicuna base（本地）
MODEL_PATH="${MODEL_PATH:-$WEIGHTS_ROOT/llava-rad}"                # 权重适配器LLaVA-Rad（本地）
# QUERY_FILE="${QUERY_FILE:-$BASE_ROOT/data/llava-rad-mimic-cxr-annotations-1.0.0/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json}"
QUERY_FILE="${QUERY_FILE:-$BASE_ROOT/data/llava-rad-mimic-cxr-annotations-1.0.0/_mini_100.json}"  # 注释 JSON
IMAGE_FOLDER="${IMAGE_FOLDER:-/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files}"     # MIMIC-CXR-JPG 图像根目录

STAMP="$(date +%m%d_%H%M)"
OUTDIR="$ROOT_DIR/results/llavarad_${STAMP}"    # 输出目录
RUN_NAME="llavarad_${STAMP}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# =============== 自检 ===============
check_path () { [[ -e "$1" ]] || { echo "[ERROR] Not found: $1" >&2; exit 1; }; }
for p in "$MODEL_BASE" "$MODEL_PATH" "$QUERY_FILE" "$IMAGE_FOLDER"; do check_path "$p"; done
mkdir -p "$OUTDIR"

echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "[INFO] TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "[INFO] Logs => $LOG_DIR"

# =============== 推理 ===============
set -x
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m llava.eval.model_mimic_cxr \
  --query_file "$QUERY_FILE" \
  --loader "mimic_test_findings" \
  --image_folder "$IMAGE_FOLDER" \
  --conv_mode "v1" \
  --prediction_file "$OUTDIR/test_0.jsonl" \
  --temperature 0 \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
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
