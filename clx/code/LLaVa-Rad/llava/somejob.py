# llava/somejob.py
import os
import json

import torch
from PIL import Image

from llava.model.multimodal_encoder.open_clip_encoder.open_clip_encoder import (
    OpenCLIPVisionTower,
)


# ===== 一点点小 args，用于告诉视觉塔选哪一层特征 =====
class DummyArgs:
    def __init__(self):
        # 和原 LLaVA-Rad 一致：倒数第二层的 patch 特征
        self.mm_vision_select_layer = -2
        self.mm_vision_select_feature = "patch"


# ===== 路径清洗：去掉前缀 mimic/ 和开头的 / =====
def normalize_rel_path(p: str) -> str:
    # 原 json 里常见： "mimic/p17/p1778..."
    if p.startswith("mimic/"):
        p = p[len("mimic/"):]
    # 万一有 "/mimic/..." 或 "/p17/..." 之类
    p = p.lstrip("/")
    return p


def resolve_image_path(root: str, rel: str) -> str:
    rel_norm = normalize_rel_path(rel)
    full = os.path.join(root, rel_norm)
    if not os.path.exists(full):
        raise FileNotFoundError(f"image not found: {full}")
    return full


def main():
    # ==== 打印一下关键环境变量，确认你已经 source .env.cache ====
    print(">>> HF_HOME =", os.environ.get("HF_HOME"))
    print(">>> HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"))

    # ==== 1. 加载 BiomedCLIP 视觉塔 ====
    print(">>> Loading OpenCLIPVisionTower ...")
    vision_args = DummyArgs()

    # 这里不显式传 config/ckpt，让 open_clip_encoder 自己按：
    #   传参 > 环境变量(BIOMEDCLIP_CKPT/LLAVARAD_VISION_CFG) > 默认 .biomedclip
    # 的优先级去找。你 .env.cache 里已经设置了：
    #   BIOMEDCLIP_CKPT=/media/cuilexuan/clx/weights/biomedclip/ckpt.pt
    #   LLAVARAD_VISION_CFG=/media/cuilexuan/clx/weights/biomedclip/biomedclipcxr_518.json
    vt = OpenCLIPVisionTower(
        vision_tower="biomedclipcxr_518",  # 名字其实无所谓，真正信息在 config 里
        args=vision_args,
        delay_load=False,
    )
    print(">>> Vision tower loaded OK.")
    print(">>> image_processor =", vt.image_processor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vt = vt.to(device)

    # ==== 2. 读取一个带多视图的 JSON 样本 ====
    DATA_JSON = "/media/cuilexuan/clx/data/multiview-cxr-annotations-1.0.0/train_sample_20.json"
    IMG_ROOT = "/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files"

    with open(DATA_JSON, "r") as f:
        dataset = json.load(f)

    sample = dataset[0]  # 就拿第一个 case 测
    img_field = sample["image"]
    if isinstance(img_field, str):
        img_list = [img_field]
    else:
        img_list = list(img_field)

    print(f">>> this sample has {len(img_list)} views")

    # ==== 3. 把每个视图读成 PIL，再用 BiomedCLIP 的 processor 做预处理 ====
    pil_images = []
    for rel in img_list:
        full_p = resolve_image_path(IMG_ROOT, rel)
        print(">>> using image:", full_p)
        pil_images.append(Image.open(full_p).convert("RGB"))

    # vt.image_processor 是你刚才在 open_clip_encoder 里写的 Processor
    proc_out = vt.image_processor.preprocess(pil_images, return_tensors="pt")
    pixel_values = proc_out["pixel_values"]  # [N_view, 3, H, W]
    print(">>> image tensor shape =", pixel_values.shape)

    pixel_values = pixel_values.to(device)

    # ==== 4. 丢进 OpenCLIPVisionTower，拿出 patch 特征 ====
    with torch.no_grad():
        feats = vt(pixel_values)  # 调用的是 OpenCLIPVisionTower.forward

    print(">>> feat shape =", feats.shape)
    print(">>> Done.")


if __name__ == "__main__":
    main()
