import json
import os

# ==== 1. 环境变量（离线 + 指定 HF 缓存） ====
os.environ["HF_HOME"] = "/media/cuilexuan/clx/weights/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/media/cuilexuan/clx/weights/hf_home/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/media/cuilexuan/clx/weights/hf_home/hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 只需要 RadGraphMetrics
from chexbert import RadGraphMetrics

# ==== 2. 读取 jsonl，构造 (pred, ref, id) 三元组 ====
jsonl_path = "/media/cuilexuan/clx/code/LLaVa-Rad/results/llavarad_1116_1714/test_0.jsonl"
# 如需换别的结果文件，改上面这一行路径即可

pairs = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        item = json.loads(line)

        pred_text = item.get("prediction", "")
        ref_text = item.get("reference", "")
        ann_id = item.get("id", "")

        pairs.append((pred_text, ref_text, ann_id))

print("总样本数：", len(pairs))

# ==== 3. 初始化 RadGraph 评估器 ====
radgraph_path = "/media/cuilexuan/clx/weights/radgraph/models/model_checkpoint/model.tar.gz"

print("开始加载 RadGraph 模型")
radgraph_metric = RadGraphMetrics(radgraph_path=radgraph_path, mbatch_size=16)
print("RadGraph 模型加载完成")

# ==== 4. 一次性评估全部样本 ====
radgraph_metric.pairs = pairs
print("开始计算 RadGraph 评估指标...")
radgraph_scores = radgraph_metric.compute()
print("RadGraph 评估结果：", radgraph_scores)