# [2025-10-19] 本脚本用于预热 HuggingFace 缓存，以支持离线运行 LLaVa-Rad 项目。
#!/usr/bin/env bash
set -euo pipefail

# =============== 目录与环境 ===============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 加载你的 .env.cache（若有）
if [ -f "$ROOT_DIR/.env.cache" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.env.cache"
  set +a
fi

# 统一固定到 hub 树
mkdir -p "${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
mkdir -p "${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
mkdir -p "${OPEN_CLIP_CACHE_DIR:-$HOME/.cache/open_clip}"
mkdir -p "${BIOMEDCLIP_CACHE:-$ROOT_DIR/.biomedclip}"

# 设置缓存路径
echo "[warm] HF_HOME=$HF_HOME"
echo "[warm] HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "[warm] TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "[warm] TORCH_HOME=$TORCH_HOME"
echo "[warm] BIOMEDCLIP_CKPT=$BIOMEDCLIP_CKPT"
echo "[warm] LLAVARAD_VISION_CFG=$LLAVARAD_VISION_CFG"

# 如果你需要镜像，取消注释并改成你可用的地址
# export HF_ENDPOINT="https://hf-mirror.com"

# =============== 第 1 阶段：在线拉齐 ===============
# 先确保是在线模式（允许探测/下载）
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
print("[warm] >>> 拉取 bert-base-uncased tokenizer/config")
AutoTokenizer.from_pretrained("bert-base-uncased")
AutoConfig.from_pretrained("bert-base-uncased")

print("[warm] >>> 拉取 BiomedBERT config（open_clip 文本塔依赖）")
AutoConfig.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

print("[warm] >>> 触发 RadGraph 权重缓存（评测用）")
try:
    from radgraph import RadGraph
    RadGraph(cuda_device=-1)  # CPU 也能触发缓存
    print("[warm] radgraph cache OK")
except Exception as e:
    print("[warm] radgraph cache WARN:", repr(e))
PY

# BiomedCLIP 权重（你项目本就指向本地 ckpt），做个存在性校验
if [ ! -f "${BIOMEDCLIP_CKPT:-}" ]; then
  echo "[warm] WARN: BIOMEDCLIP_CKPT not found at ${BIOMEDCLIP_CKPT:-<unset>}"
  echo "       请把 biomedclip 的 ckpt 放到这里，或改 .env.cache 指向你的 ckpt 路径。"
else
  echo "[warm] biomedclip ckpt exists."
fi

# =============== 第 2 阶段：离线验收（强制离线） ===============
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
print("[warm-offline] 校验 bert-base-uncased（local only）")
AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
AutoConfig.from_pretrained("bert-base-uncased", local_files_only=True)

print("[warm-offline] 校验 BiomedBERT config（local only）")
AutoConfig.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", local_files_only=True)

print("[warm-offline] 全部通过")
PY

echo "[warm] ✅ 缓存准备完成，可离线运行"
