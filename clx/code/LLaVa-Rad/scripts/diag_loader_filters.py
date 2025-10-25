# [2025-10-21] 新加模块，用于检测某 JSON 数据集在经过若干过滤规则后被移除的原因统计

#!/usr/bin/env python3
# scripts/diag_loader_filters.py
import json, os, sys, collections, re

JSON="/media/cuilexuan/clx/data/llava-rad-mimic-cxr-annotations-1.0.0/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json"
IMG_ROOT="/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
# 若你有 BIOMEDCLIP_CACHE 路径也可以填写（用于检测是否依赖预计算特征）
BIOMEDCLIP_CACHE="/tmp/biomed_clip"  # 或 export BIOMEDCLIP_CACHE 的值

# 加载 JSON （支持 array 或 jsonl）
with open(JSON,'r',encoding='utf-8') as f:
    txt=f.read().strip()
if txt.startswith('['):
    data=json.loads(txt)
else:
    data=[json.loads(line) for line in txt.splitlines() if line.strip()]

total=len(data)
counts=collections.Counter()
examples=collections.defaultdict(list)

def exists_image(rel):
    if not rel: return False
    full=os.path.join(IMG_ROOT, rel)
    alt = None
    if "mimic/" in rel:
        alt = os.path.join(IMG_ROOT, rel.split("mimic/",1)[1])
    return os.path.exists(full) or (alt and os.path.exists(alt))

# 假设的过滤规则（会统计每条为何被移除）
for i, entry in enumerate(data):
    reason=None
    # 0) 必须有 image 存在（你已经验证全部存在）
    img=entry.get("image")
    if not img:
        reason="NO_IMAGE_FIELD"
    elif not exists_image(img):
        reason="IMAGE_NOT_FOUND"
    # 1) 视图过滤：很多 loader 只要 frontal (AP/PA)；如果为 LATERAL 或 OTHERS 可能被跳过
    if not reason:
        v=(entry.get("view") or "").upper()
        # 你可以在这里修改为你怀疑的 loader 行为：只允许 FRONTAL/AP/PA
        if v and not any(k in v for k in ("FRONTAL","AP","PA","UPRIGHT","ERECT","STANDING")):
            reason="VIEW_FILTERED:"+v
    # 2) generate_method 或其它字段过滤（示例）
    if not reason:
        gm=(entry.get("generate_method") or "").lower()
        # 如果 loader 只选某些方法，例如 'gpt4'、'rule'，其它跳过（此处只是统计）
        if gm and gm not in ("gpt4","gpt4extract","rulebased","gpt3.5","manual"):
            reason="GENMETHOD_FILTERED:"+gm
    # 3) BiomedCLIP 预计算特征缺失（通过检查 cache dir 中是否存在对应文件名）
    if not reason and os.path.isdir(BIOMEDCLIP_CACHE):
        # 这里尝试用 image basename 去检查是否存在预计算特征文件（实现依具体cache结构而变）
        b=os.path.basename(img)
        if not any(fn.startswith(b) for fn in os.listdir(BIOMEDCLIP_CACHE)):
            # 不断言这是一定原因，只是统计可能性
            reason="NO_BIOMEDCLIP_FEATURE"
    # 4) 去重规则（示例：loader 可能只保留每 patient 最早一条）
    #    统计 patient id 出现次数（稍后再用）
    if not reason:
        reason="KEEP"

    counts[reason]+=1
    if len(examples[reason])<5:
        examples[reason].append((i, entry.get("id"), entry.get("image"), entry.get("view"), entry.get("generate_method")))

print("total entries:", total)
print("reason counts (top):")
for k,v in counts.most_common():
    print(f"  {k:30s} {v}")
print("\nexamples per reason (up to 5):")
for k,ex in examples.items():
    print("\n==",k)
    for e in ex:
        print(" ", e)

