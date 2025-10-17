# """WIP"""

# import json
# import os
# import torch

# from timm.models.vision_transformer import VisionTransformer
# from transformers.modeling_outputs import BaseModelOutput
# from huggingface_hub import hf_hub_download

# from .utils import from_pretrained, remove_transformer_pooler_weights

# LLAVARAD_HF_REPO = "microsoft/llava-rad"
# LLAVARAD_LOCAL_DIR = os.environ.get("LLAVARAD_LOCAL_WEIGHTS")


# def _load_from_local_or_hf(filename: str, repo_id: str = LLAVARAD_HF_REPO, subfolder: str | None = None) -> str:
#     """
#     返回目标文件的本地绝对路径。
#     优先从环境变量 LLAVARAD_LOCAL_WEIGHTS 指定的目录读取；若离线且本地缺失，则抛 FileNotFoundError。
#     在线时若本地缺失则回退 Hugging Face Hub 下载。
#     """
#     # 1) 本地优先
#     local_dir = os.environ.get("LLAVARAD_LOCAL_WEIGHTS")
#     if local_dir:
#         parts = [local_dir]
#         if subfolder:
#             parts.append(subfolder)
#         parts.append(filename)
#         local_path = os.path.join(*parts)
#         if os.path.exists(local_path):
#             return local_path

#     # 2) 离线：不联网，直接报本地缺失
#     offline = bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))
#     if offline:
#         where = local_dir if local_dir else "<unset LLAVARAD_LOCAL_WEIGHTS>"
#         raise FileNotFoundError(
#             f"Offline mode: '{filename}' not found under {where}. "
#             f"Set LLAVARAD_LOCAL_WEIGHTS to your llava-rad weights folder."
#         )

#     # 3) 在线：回退 Hub
#     return hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)


# class VisionTower(torch.nn.Module):
#     def __init__(self, vit: VisionTransformer) -> None:
#         super().__init__()
#         self.vit = vit
#         self._hidden_size = vit.embed_dim
#         self._num_patches = vit.patch_embed.num_patches

#     @torch.no_grad()
#     def forward(self, images, output_hidden_states=True):
#         hidden_states = self.vit.patch_embed(images)
#         hidden_states = self.vit._pos_embed(hidden_states)
#         hidden_states = self.vit.norm_pre(hidden_states)
#         block_states = [hidden_states]
#         for block in self.vit.blocks:
#             hidden_states = block(hidden_states)
#             block_states.append(hidden_states)
#         if output_hidden_states:
#             return BaseModelOutput(
#                 last_hidden_state=hidden_states, hidden_states=block_states
#             )
#         else:
#             return BaseModelOutput(last_hidden_state=hidden_states)

#     @property
#     def hidden_size(self):
#         return self._hidden_size

#     @property
#     def num_patches(self):
#         return self._num_patches


# class Processor:
#     def __init__(self, fn) -> None:
#         self.fn = fn

#     def preprocess(self, image, return_tensors="pt"):
#         if return_tensors != "pt":
#             raise NotImplementedError
#         return {"pixel_values": [self.fn(image)]}


# class OpenCLIPVisionTower(torch.nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False, vision_tower_config=None, vision_tower_checkpoint=None):
#         super().__init__()

#         self.is_loaded = False
#         self.vision_tower_name = vision_tower

#         # === 加载 Vision Tower 配置（本地优先 / 可离线） ===
#         if vision_tower_config and os.path.exists(vision_tower_config):
#             with open(vision_tower_config, "r") as f:
#                 self.vision_tower_config = json.load(f)
#         else:
#             # 使用官方提供的名称
#             json_name = "biomedclipcxr_518.json"
#             cache_file = _load_from_local_or_hf(json_name, repo_id=LLAVARAD_HF_REPO)
#             with open(cache_file, "r") as f:
#                 self.vision_tower_config = json.load(f)

#         # 记录（可能是文件名或绝对路径）
#         self.vision_tower_checkpoint = vision_tower_checkpoint
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

#         if not delay_load:
#             self.load_model()
#         else:
#             self.cfg_only = vision_tower_config

#     def load_model(self):
#         """
#         本地优先：
#           - 若 self.vision_tower_checkpoint 是本地路径且存在：直接使用；
#           - 若是文件名或路径不存在：在 LLAVARAD_LOCAL_WEIGHTS 下寻找同名文件；
#           - 离线且找不到：给出清晰错误；
#           - 在线：回退到 Hub 下载。
#         """
#         # === 处理视觉塔 checkpoint 路径 ===
#         if self.vision_tower_checkpoint:
#             if not os.path.exists(self.vision_tower_checkpoint):
#                 # 传入的很可能是文件名（如 biomedclipcxr_518_checkpoint.pt）
#                 print("Loading vision tower checkpoint from local weights (or HF Hub if online)")
#                 self.vision_tower_checkpoint = _load_from_local_or_hf(
#                     filename=self.vision_tower_checkpoint, repo_id=LLAVARAD_HF_REPO
#                 )
#             # 清理权重（项目原逻辑）
#             self.vision_tower_checkpoint = remove_transformer_pooler_weights(self.vision_tower_checkpoint)

#         # === 实例化视觉塔 ===
#         model, preprocess, _ = from_pretrained(
#             self.vision_tower_name,
#             self.vision_tower_config,     # dict 或配置文件路径
#             self.vision_tower_checkpoint  # 可为 None；由 from_pretrained 决定默认
#         )
#         self.image_processor = Processor(preprocess)
#         self.vision_tower = VisionTower(model.visual.trunk)
#         self.vision_tower.requires_grad_(False)
#         self.is_loaded = True

#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
#         if self.select_feature == "patch":
#             image_features = image_features[:, 1:]
#         elif self.select_feature == "cls_patch":
#             image_features = image_features
#         else:
#             raise ValueError(f"Unexpected select feature: {self.select_feature}")
#         return image_features

#     @torch.no_grad()
#     def forward(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(
#                     image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
#                     output_hidden_states=True,
#                 )
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else:
#             image_forward_outs = self.vision_tower(
#                 images.to(device=self.device, dtype=self.dtype),
#                 output_hidden_states=True,
#             )
#             image_features = self.feature_select(image_forward_outs).to(images.dtype)

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return next(self.vision_tower.parameters()).dtype

#     @property
#     def device(self):
#         return next(self.vision_tower.parameters()).device

#     @property
#     def config(self):
#         raise NotImplementedError

#     @property
#     def hidden_size(self):
#         return self.vision_tower.hidden_size

#     @property
#     def num_patches(self):
#         return self.vision_tower.num_patches

"""WIP"""

import json
import os
import torch

from timm.models.vision_transformer import VisionTransformer
from transformers.modeling_outputs import BaseModelOutput

from .utils import from_pretrained, remove_transformer_pooler_weights

LLAVARAD_HF_REPO = "microsoft/llava-rad"

class VisionTower(torch.nn.Module):

    def __init__(self, vit: VisionTransformer) -> None:
        super().__init__()
        self.vit = vit
        self.hidden_size = vit.embed_dim
        self.num_patches = vit.patch_embed.num_patches

    @torch.no_grad()
    def forward(self, images, output_hidden_states=True):
        hidden_states = self.vit.patch_embed(images)
        hidden_states = self.vit._pos_embed(hidden_states)
        hidden_states = self.vit.norm_pre(hidden_states)
        block_states = [hidden_states]
        for block in self.vit.blocks:
            hidden_states = block(hidden_states)
            block_states.append(hidden_states)
        if output_hidden_states:
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=block_states
            )
        else:
            return BaseModelOutput(last_hidden_state=hidden_states)


class Processor:

    def __init__(self, fn) -> None:
        self.fn = fn

    def preprocess(self, image, return_tensors="pt"):
        if return_tensors != "pt":
            raise NotImplementedError
        return {"pixel_values": [self.fn(image)]}


class OpenCLIPVisionTower(torch.nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, vision_tower_config=None, vision_tower_checkpoint=None):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        if os.path.exists(vision_tower_config):
            self.vision_tower_config = json.load(open(vision_tower_config))
        else:
            # likely from hf hub
            from huggingface_hub import hf_hub_download
            cache_file = hf_hub_download(repo_id=LLAVARAD_HF_REPO, filename='biomedclipcxr_518.json')
            self.vision_tower_config = json.load(open(cache_file))
            
        self.vision_tower_checkpoint = vision_tower_checkpoint
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = vision_tower
        
    def load_model(self):
        if self.vision_tower_checkpoint:
            if not os.path.exists(self.vision_tower_checkpoint):
                print("Loading vision tower from HF Hub")
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                
                def load_from_hf(repo_id=LLAVARAD_HF_REPO, filename="",subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return cache_file
                self.vision_tower_checkpoint = load_from_hf(filename=self.vision_tower_checkpoint)

            self.vision_tower_checkpoint = remove_transformer_pooler_weights(self.vision_tower_checkpoint)
        model, preprocess, _ = from_pretrained(
            self.vision_tower_name, self.vision_tower_config, self.vision_tower_checkpoint
        )
        self.image_processor = Processor(preprocess)

        self.vision_tower = VisionTower(model.visual.trunk)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        raise NotImplementedError

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def num_patches(self):
        return self.vision_tower.num_patches

    