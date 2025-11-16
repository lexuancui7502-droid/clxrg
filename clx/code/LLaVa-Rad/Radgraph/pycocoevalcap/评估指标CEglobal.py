# import json
# import gc
# import logging

# import torch
# # from chexbert import RadGraphMetrics, F1CheXbertMetrics
# from chexbert import  F1CheXbertMetrics

# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("评估CEeval_result_0.txt", mode='a', encoding='utf-8'),
#         logging.StreamHandler()  # 同时输出到控制台
#     ]
# )

# # 1. 读取json文件
# json_path = "/media/yanxiuying/DynRefer/output/train/dynrefer/20250603184/result/val_603+610.json"
# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# logging.info(f"读取json文件: {json_path}")

# # 2. 构造三元组 (预测文本, 参考文本, id)
# pairs_region = []
# for ann in data.get("annotations", []):
#     pred_text = ann.get("extra_info", {}).get("pred_result", {}).get("caption", "")
#     ref_text = ann.get("caption", "")
#     ann_id = ann.get("id", "")
#     pairs_region.append((pred_text, ref_text, ann_id))

# pairs_global = []
# for ann in data.get("annotations", []):
#     pred_text = ann.get("extra_info", {}).get("pred_result", {}).get("global_cap", "")
#     ref_text = ann.get("global_caption", "")
#     ann_id = ann.get("id", "")
#     pairs_global.append((pred_text, ref_text, ann_id))

# logging.info(f"区域对数: {len(pairs_region)}")
# logging.info(f"全局对数: {len(pairs_global)}")

# # logging.info(f"区域对数: {len(pairs_region)}, 唯一对数: {len(set(pairs_region))}")
# # logging.info(f"全局对数: {len(pairs_global)}, 唯一对数: {len(set(pairs_global))}")

# # for i in range(0, min(300, len(pairs_region)), 100):
# #     chunk = pairs_region[i:i + 100]
# #     logging.info(f"Chunk {i//100 + 1} 前3条数据: {chunk[:3]}")


# # 3. 评估
# radgraph_path = "/media/yanxiuying/DynRefer/dynrefer/pycocoevalcap/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz"
# chexbert_path = "/media/yanxiuying/DynRefer/chexbert/chexbert.pth"
# model_path = "/media/yanxiuying/DynRefer/chexbert/bert-base-uncased"

# # 初始化评估器
# # logging.info("开始使用radgraph_path")
# # radgraph_metric = RadGraphMetrics(radgraph_path=radgraph_path, mbatch_size=20)
# # logging.info("结束使用radgraph_path")

# # logging.info("开始使用chexbert_path")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"当前设备: {device}")
# f1chexbert_metric = F1CheXbertMetrics(chexbert_path=chexbert_path, model_path=model_path, mbatch_size=2)
# # logging.info("结束使用chexbert_path")

# # 分块处理函数
# def process_in_chunks(pairs, chunk_size=100, metric_type="region",start_chunk=1):
#     radgraph_results = []
#     f1chexbert_results = []
    
#     # 计算起始索引
#     if metric_type=="区域":
#         start_index = (start_chunk - 1) * chunk_size
#         if start_index >= len(pairs):
#             logging.error(f"起始 chunk {start_chunk} 超出数据范围，总共 {len(pairs) // chunk_size + 1} 个 chunk")
#             return None, None
#     else:
#         start_index = 0

#     for i in range(start_index, len(pairs), chunk_size):
#         chunk = pairs[i:i + chunk_size]
#         chunk_id = i // chunk_size + 1

#         # # 重新初始化评估器
#         # radgraph_metric = RadGraphMetrics(radgraph_path=radgraph_path, mbatch_size=20)
#         # f1chexbert_metric = F1CheXbertMetrics(chexbert_path=chexbert_path, model_path=model_path, mbatch_size=20)
        
#         # # RadGraph评估
#         # radgraph_metric.pairs = chunk
#         # radgraph_scores = radgraph_metric.compute()
#         # radgraph_results.append(radgraph_scores)
#         # logging.info(f"{metric_type} RadGraph chunk {chunk_id} 结果: {radgraph_scores}")
        
#         # F1CheXbert评估
#         f1chexbert_metric.pairs = chunk
        
#         f1chexbert_scores = f1chexbert_metric.compute()
#         f1chexbert_results.append(f1chexbert_scores)
#         logging.info(f"{metric_type} F1CheXbert chunk {chunk_id} 结果: {f1chexbert_scores}")
        
#         # 释放内存
#         gc.collect()
    
#     # 计算平均结果
#     def compute_average(results, metric_name):
#         if not results:
#             return None
#         avg_scores = {}
#         for key in results[0].keys():
#             try:
#                 avg_scores[key] = sum(r[key] for r in results) / len(results)
#             except TypeError:
#                 logging.warning(f"{metric_name} 的 {key} 指标无法计算平均值，跳过")
#         return avg_scores
    
#     radgraph_avg = compute_average(radgraph_results, f"{metric_type} RadGraph")
#     f1chexbert_avg = compute_average(f1chexbert_results, f"{metric_type} F1CheXbert")
    
#     logging.info(f"{metric_type} RadGraph 所有块的平均结果: {radgraph_avg}")
#     logging.info(f"{metric_type} F1CheXbert 所有块的平均结果: {f1chexbert_avg}")
    
#     return radgraph_avg, f1chexbert_avg

# # 处理区域和全局对
# region_radgraph_avg, region_f1chexbert_avg = process_in_chunks(pairs_region, chunk_size=100, metric_type="区域")
# global_radgraph_avg, global_f1chexbert_avg = process_in_chunks(pairs_global, chunk_size=100, metric_type="全局")



# import json
# import logging
# import gc

# import torch
# from chexbert import RadGraphMetrics, F1CheXbertMetrics

# # -------------------------------
# # 日志配置
# log_file = "评估CEeval_result_日志记录.txt"
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_file, mode='a', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )

# # -------------------------------
# # 读取 JSON 文件
# json_path = "/media/yanxiuying/DynRefer/output/train/dynrefer/20250603184/result/val_603+610.json"
# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
# logging.info(f"读取JSON数据，共有{len(data.get('annotations', []))}条")

# # 构造三元组列表 (pred, ref, id)
# pairs = []
# for ann in data.get("annotations", []):
#     pred_text = ann.get("extra_info", {}).get("pred_result", {}).get("global_cap", "")
#     ref_text = ann.get("global_cap", "")
#     ann_id = ann.get("id", "")
#     pairs.append((pred_text, ref_text, ann_id))
# logging.info(f"构造完成pairs，共{len(pairs)}对")

# # -------------------------------
# # 模型路径
# radgraph_path = "/media/yanxiuying/DynRefer/dynrefer/pycocoevalcap/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz"
# chexbert_path = "/media/yanxiuying/DynRefer/chexbert/chexbert.pth"
# model_path = "/media/yanxiuying/DynRefer/chexbert/bert-base-uncased"

# # -------------------------------
# # 分批评估函数
# def batch_compute(metric_name, metric_class, batch_size, pairs, **kwargs):
#     logging.info(f"开始分批评估：{metric_name}")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"当前设备: {device}")
#     # metric = metric_class(**kwargs, mbatch_size=batch_size)
#     all_scores = []

#     for i in range(0, len(pairs), batch_size):
#         metric = metric_class(**kwargs, mbatch_size=batch_size)
#         batch = pairs[i:i + batch_size]
#         metric.pairs = batch
#         try:
#             score = metric.compute()
#             all_scores.append(score)
#             logging.info(f"[{metric_name}] 第{i // batch_size + 1}批结果: {score}")
#         except Exception as e:
#             logging.error(f"[{metric_name}] 第{i // batch_size + 1}批次报错: {e}")
#         gc.collect()

#     logging.info(f"{metric_name} 总共评估 {len(all_scores)} 个批次")
#     return all_scores

# # -------------------------------
# # 可选平均函数
# def compute_average(results, metric_name):
#     if not results:
#         return {}
#     avg_scores = {}
#     for key in results[0].keys():
#         try:
#             avg_scores[key] = sum(r[key] for r in results) / len(results)
#         except Exception:
#             logging.warning(f"[{metric_name}] {key} 指标无法平均，跳过")
#     logging.info(f"[{metric_name}] 所有批次平均结果：{avg_scores}")
#     return avg_scores

# # -------------------------------
# # # 评估 RadGraph
# # radgraph_scores = batch_compute(
# #     "RadGraph",
# #     RadGraphMetrics,
# #     batch_size=2,
# #     pairs=pairs,
# #     radgraph_path=radgraph_path
# # )
# # radgraph_avg = compute_average(radgraph_scores, "RadGraph")

# # -------------------------------
# # 评估 F1CheXbert
# f1chexbert_scores = batch_compute(
#     "F1CheXbert",
#     F1CheXbertMetrics,
#     batch_size=16,
#     pairs=pairs,
#     chexbert_path=chexbert_path,
#     model_path=model_path
# )
# f1chexbert_avg = compute_average(f1chexbert_scores, "F1CheXbert")

# # -------------------------------
# logging.info("=== 评估全部完成 ===")

import json
import logging
logging.getLogger("allennlp").setLevel(logging.WARNING)
import gc
import torch
from chexbert import RadGraphMetrics, F1CheXbertMetrics

# -------------------------------
# 日志配置
log_file = "评估CEeval_result_日志记录全局.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),  # 继续写入
        logging.StreamHandler()
    ]
)

# -------------------------------
# 读取 JSON 文件
json_path = "/media/yanxiuying/DynRefer/output/train/dynrefer/20250603184/result/val_603+610.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
annotations = data.get("annotations", [])
logging.info(f"读取JSON数据，共有{len(annotations)}条")

# 构造三元组列表 (pred, ref, id)
pairs = []
for ann in annotations:
    pred_text = ann.get("extra_info", {}).get("pred_result", {}).get("global_cap", "")
    ref_text = ann.get("global_cap", "")
    ann_id = ann.get("id", "")
    if pred_text.strip() and ref_text.strip():  # 忽略空报告
        pairs.append((pred_text, ref_text, ann_id))
logging.info(f"构造完成pairs，共{len(pairs)}对")

# -------------------------------
# 模型路径
radgraph_path = "/media/yanxiuying/DynRefer/dynrefer/pycocoevalcap/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz"
chexbert_path = "/media/yanxiuying/DynRefer/chexbert/chexbert.pth"
model_path = "/media/yanxiuying/DynRefer/chexbert/bert-base-uncased"

# -------------------------------
# 批量评估函数（支持多个指标）
def batch_compute_multi(batch_size, pairs, start_batch=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前设备: {device}")
    logging.info("开始分批评估：RadGraph + F1CheXbert")

    # 初始化一次评估器
    radgraph_metric = RadGraphMetrics(radgraph_path=radgraph_path, mbatch_size=batch_size)
    f1chexbert_metric = F1CheXbertMetrics(chexbert_path=chexbert_path, model_path=model_path, mbatch_size=batch_size)

    radgraph_scores = []
    chexbert_scores = []

    start_idx = (start_batch - 1) * batch_size

    for i in range(start_idx, len(pairs), batch_size):
            # 初始化一次评估器
        radgraph_metric = RadGraphMetrics(radgraph_path=radgraph_path, mbatch_size=batch_size)
        f1chexbert_metric = F1CheXbertMetrics(chexbert_path=chexbert_path, model_path=model_path, mbatch_size=batch_size)
        batch = pairs[i:i + batch_size]
        if not batch:
            continue

        # RadGraph 评估
        try:
            radgraph_metric.pairs = batch
            rad_score = radgraph_metric.compute()
            radgraph_scores.append(rad_score)
            logging.info(f"[RadGraph] 第{i // batch_size + 1}批结果: {rad_score}")
        except Exception as e:
            logging.error(f"[RadGraph] 第{i // batch_size + 1}批次报错: {e}")

        # F1CheXbert 评估
        try:
            f1chexbert_metric.pairs = batch
            chex_score = f1chexbert_metric.compute()
            chexbert_scores.append(chex_score)
            logging.info(f"[F1CheXbert] 第{i // batch_size + 1}批结果: {chex_score}")
        except Exception as e:
            logging.error(f"[F1CheXbert] 第{i // batch_size + 1}批次报错: {e}")

        gc.collect()

    return radgraph_scores, chexbert_scores

# -------------------------------
# 平均函数
def compute_average(results, metric_name):
    if not results:
        return {}
    avg_scores = {}
    for key in results[0].keys():
        try:
            avg_scores[key] = sum(r[key] for r in results) / len(results)
        except Exception:
            logging.warning(f"[{metric_name}] {key} 指标无法平均，跳过")
    logging.info(f"[{metric_name}] 所有批次平均结果：{avg_scores}")
    return avg_scores

# -------------------------------
# 执行批次评估
batch_size = 16
start_batch = 1  # 可改成 302 继续跑

rad_scores, chex_scores = batch_compute_multi(
    batch_size=batch_size,
    pairs=pairs,
    start_batch=start_batch
)

# -------------------------------
# 平均
radgraph_avg = compute_average(rad_scores, "RadGraph")
chexbert_avg = compute_average(chex_scores, "F1CheXbert")

# -------------------------------
logging.info("=== 所有评估任务完成 ===")
