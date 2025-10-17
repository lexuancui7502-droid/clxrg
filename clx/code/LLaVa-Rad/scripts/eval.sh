# #!/bin/bash
# set -euo pipefail

# # ==== 路径 ====
# ROOT="/media/cuilexuan/clx"

# # ==== 必要环境（强制离线） ====
# export TRANSFORMERS_OFFLINE=1
# export HF_HUB_OFFLINE=1
# export LLAVARAD_LOCAL_WEIGHTS="/media/cuilexuan/clx/weights/llava-rad"
# export RRG_LOCAL_WEIGHTS="${ROOT}/weights/rrg_scorers"
# export BERT_BASE_DIR="${ROOT}/weights/bert-base-uncased"
# export RRG_SCORES="/media/cuilexuan/clx/weights/rrg_scorers"
# export BERT_LOCAL="/media/cuilexuan/clx/weights/bert-base-uncased"


# # 基座 LLM（Vicuna-7B）——本地
# MODEL_BASE="${ROOT}/weights/vicuna-7b-v1.5"

# # LLaVA-Rad 适配器/权重——本地  （← 由“在线仓库名”改为本地目录）
# MODEL_PATH="${ROOT}/weights/llava-rad"

# # 注释 JSON（你放在 clx/data 下）
# # QUERY_FILE="${ROOT}/data/llava-rad-mimic-cxr-annotations-1.0.0/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json"
# QUERY_FILE="${ROOT}/data/llava-rad-mimic-cxr-annotations-1.0.0/_mini_50.json"


# # MIMIC-CXR-JPG 图像根目录（包含 p10/p11... 的 files 目录）
# IMAGE_FOLDER="/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files"

# # 输出
# STAMP=$(date +%m%d_%H%M)
# OUTDIR="/media/cuilexuan/clx/code/LLaVa-Rad/results/llavarad_${STAMP}"
# RUN_NAME="llavarad_${STAMP}"

# BATCH_SIZE=4

# # ==== 路径自检 ====
# check_path () {
#   local p="$1"
#   if [[ "$p" = /* ]]; then
#     [[ -e "$p" ]] || { echo "[ERROR] Not found: $p"; exit 1; }
#   fi
# }
# check_path "$MODEL_BASE"
# check_path "$MODEL_PATH"
# check_path "$QUERY_FILE"
# check_path "$IMAGE_FOLDER"
# check_path "$RRG_SCORES"
# check_path "$BERT_LOCAL"
# mkdir -p "$OUTDIR"

# # ==== 推理 ====
# CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_mimic_cxr \
#   --query_file "$QUERY_FILE" \
#   --loader "mimic_test_findings" \
#   --image_folder "$IMAGE_FOLDER" \
#   --conv_mode "v1" \
#   --prediction_file "$OUTDIR/test_0.jsonl" \
#   --temperature 0 \
#   --model_path "$MODEL_PATH" \
#   --model_base "$MODEL_BASE" \
#   --chunk_idx 0 --num_chunks 1 \
#   --batch_size "$BATCH_SIZE" \
#   --group_by_length

# # 统一命名评测输入
# cp -f "$OUTDIR/test_0.jsonl" "$OUTDIR/mimic_cxr_preds.jsonl"

# # ==== 评测（离线，CheXbert & RadGraph；先关闭 bootstrap，样本更大时再打开）====
# pushd "/media/cuilexuan/clx/code/LLaVa-Rad/llava/eval/rrg_eval" >/dev/null
# CUDA_VISIBLE_DEVICES=0 python run.py \
#   "$OUTDIR/mimic_cxr_preds.jsonl" \
#   --run_name "$RUN_NAME" \
#   --output_dir "$OUTDIR/eval" \
#   --bootstrap_ci=False
# popd >/dev/null

# echo "[OK] Inference + Evaluation done."
# echo "Predictions:   $OUTDIR/test_0.jsonl"
# echo "Eval results:  $OUTDIR/eval/$RUN_NAME/"

#!/bin/bash
set -euo pipefail

# ========= 基础路径 =========
ROOT="/media/cuilexuan/clx"

# 代理（可选）
export HTTP_PROXY='http://127.0.0.1:7890'
export HTTPS_PROXY='http://127.0.0.1:7890'
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

# ========= 在线模式（非常重要） =========
# 确保没有任何“强制离线”的环境变量
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE RRG_SCORES PUBMED_LOCAL BERT_LOCAL LLAVARAD_LOCAL_WEIGHTS

# 如果你已经在当前 Shell 里导出过代理，这两行可以省略
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"

# ========= 权重与数据 =========
MODEL_BASE="${ROOT}/weights/vicuna-7b-v1.5"     # 基座 LLM（本地已有）
MODEL_PATH="${ROOT}/weights/llava-rad"          # LLaVA-Rad 权重（本地已有）

# 注释 JSON（你的小样本）
QUERY_FILE="${ROOT}/data/llava-rad-mimic-cxr-annotations-1.0.0/_mini_100.json"
# QUERY_FILE="${ROOT}/data/llava-rad-mimic-cxr-annotations-1.0.0/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json"

# MIMIC-CXR-JPG 图像根目录
IMAGE_FOLDER="/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files"

# 输出
STAMP=$(date +%m%d_%H%M)
OUTDIR="${ROOT}/code/LLaVa-Rad/results/llavarad_${STAMP}"
RUN_NAME="llavarad_${STAMP}"
BATCH_SIZE=4

# ========= 自检 =========
check_path () { [[ -e "$1" ]] || { echo "[ERROR] Not found: $1"; exit 1; }; }
for p in "$MODEL_BASE" "$MODEL_PATH" "$QUERY_FILE" "$IMAGE_FOLDER"; do check_path "$p"; done
mkdir -p "$OUTDIR"

# ========= 预热 RadGraph（会自动下载到 ~/.cache/radgraph）=========
python - <<'PY'
try:
    from radgraph import RadGraph
    RadGraph(cuda_device=-1)  # CPU 触发下载构建
    print("[RadGraph] ready")
except Exception as e:
    print("[RadGraph] warning:", e)
PY

# ========= 推理 =========
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_mimic_cxr \
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
  --group_by_length

# 统一命名评测输入
cp -f "$OUTDIR/test_0.jsonl" "$OUTDIR/mimic_cxr_preds.jsonl"

# ========= 评测（在线/本地缓存均可）=========
pushd "${ROOT}/code/LLaVa-Rad/llava/eval/rrg_eval" >/dev/null
CUDA_VISIBLE_DEVICES=0 python run.py \
  "$OUTDIR/mimic_cxr_preds.jsonl" \
  --run_name "$RUN_NAME" \
  --output_dir "$OUTDIR/eval" \
  --bootstrap_ci=False
popd >/dev/null

# ==== 评测（只读已有预测）====
# pushd "/media/cuilexuan/clx/code/LLaVa-Rad" >/dev/null
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
# python -m llava.eval.rrg_eval.run \
#   "$OUTDIR/mimic_cxr_preds.jsonl" \
#   --run_name "$RUN_NAME" \
#   --output_dir "$OUTDIR/eval" \
#   --bootstrap_ci=False
# popd >/dev/null



echo "[OK] Inference + Evaluation done."
echo "Predictions:   $OUTDIR/test_0.jsonl"
echo "Eval results:  $OUTDIR/eval/$RUN_NAME/"
