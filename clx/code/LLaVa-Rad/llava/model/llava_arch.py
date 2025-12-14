#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

# [2025-12-7]新增
import math
# [2025-12-7]新增
import os
import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


# [2025-11-19] 定义一个简单的view-attention模块
class SimpleViewAttention(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)

        # === [2025-11-30] 因为随机初始化会导致视觉特征投影到一个新的特征空间，所以进行关键初始化：让一开始 scores 全是 0 ===
        # softmax(0,...,0) = 1/V，相当于等权平均
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feats):
        """
        feats: (V, D) 视图级的 global feature
        return: (V, 1) 每个视角的打分
        """
        x = self.fc1(feats)     # (V, hidden_dim)
        x = self.act(x)
        scores = self.fc2(x)    # (V, 1)
        return scores
    
# [2025-12-7] 新增Multi-slot 模块（公式里的 3~6 步：初始化槽 → Q/K/V → A=softmax → Z1=AV → FFN 残差）
class MultiSlotFusion(nn.Module):
    """
    将同一 study 下所有视图的 patch token 展平，用 M 个 latent 槽位做一次 cross-attention 聚合。

    输入:
        x: Tensor, shape (N_total, D)        # 这里我们实际用的是 [V, L, D] 或 [1, V, L, D]
    输出:
        slots: Tensor, shape (M, D)          # M 个 study-level 视觉 token
        attn:  Tensor, shape (M, N_total)    # 每个 slot 对所有 patch 的注意力权重
    """
    def __init__(
        self,
        dim: int,
        num_slots: int = 4,
        attn_dim: int = None,
        ffn_hidden_dim: int = None,
        num_view_types: int = 4,
        num_orient_types: int = 3,
    ):
        """
        dim:       视觉特征维度（mm_projector 输出 = LLM hidden_size）
        num_slots: 槽位个数 M
        attn_dim:  槽位注意力维度（默认 = dim//4，省显存）
        ffn_hidden_dim: FFN 隐藏层维度（默认 = dim，省显存）
        """
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots

        # 你现在的设定：4096 -> 1024，FFN 用 dim
        d_attn = attn_dim or (dim // 2)
        d_ffn = ffn_hidden_dim or dim

        # ===== 视角 / 体位 embedding =====
        # self.view_embed = nn.Embedding(num_view_types, dim)
        # self.orient_embed = nn.Embedding(num_orient_types, dim)

        # 初始化得很小，避免一开始把视觉特征扰乱太厉害
        # nn.init.normal_(self.view_embed.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.orient_embed.weight, mean=0.0, std=0.01)

        # 控制它们的影响力（先设成常数 0.1，后面需要可以改成 nn.Parameter）
        # self.view_scale = 0.0
        # self.orient_scale = 0.0
        # 如果想让模型自己学，可以写：
        # self.view_scale = nn.Parameter(torch.tensor(0.1))
        # self.orient_scale = nn.Parameter(torch.tensor(0.1))

        # ===== 槽位参数 =====
        self.slot_embed = nn.Parameter(torch.randn(num_slots, d_attn))

        # Q/K/V 线性层（cross-attention）
        self.q_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.k_proj = nn.Linear(dim, d_attn, bias=False)
        self.v_proj = nn.Linear(dim, d_attn, bias=False)
        self.o_proj = nn.Linear(d_attn, dim, bias=False)

        # 轻量 FFN + 残差（回到 dim 空间）
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, dim),
        )

        # 正则缓存（名字要和 llava_arch 里一致）
        self.last_div_loss = None   # slot diversity loss
        self.last_cov_loss = None   # view coverage loss

    def forward(self, x, view_ids=None, orient_ids=None):
        """
        x: Tensor, shape [V, L, D] 或 [1, V, L, D]
        view_ids / orient_ids: [V]
        """
        # 支持 batch_size=1 的 [1, V, L, D]
        if x.dim() == 4:
            # 假设当前阶段 batch_size=1
            x = x[0]
        if x.dim() != 3:
            raise ValueError(f"MultiSlotFusion expects [V,L,D] or [1,V,L,D], got {x.shape}")

        V, L, D = x.shape
        device = x.device

        # ------ 1) 加视角 / 体位 embedding ------
        # if view_ids is not None:
        #     view_ids = view_ids.to(device)
        #     if view_ids.ndim == 0:
        #         view_ids = view_ids.unsqueeze(0)
        #     if view_ids.numel() < V:
        #         view_ids = torch.cat(
        #             [view_ids,
        #              view_ids.new_full((V - view_ids.numel(),), view_ids[-1].item())],
        #             dim=0,
        #         )
        #     view_ids = view_ids[:V]
        #     ve = self.view_embed(view_ids).unsqueeze(1)  # [V,1,D]
        #     x = x + self.view_scale * ve

        # if orient_ids is not None:
        #     orient_ids = orient_ids.to(device)
        #     if orient_ids.ndim == 0:
        #         orient_ids = orient_ids.unsqueeze(0)
        #     if orient_ids.numel() < V:
        #         orient_ids = torch.cat(
        #             [orient_ids,
        #              orient_ids.new_full((V - orient_ids.numel(),), orient_ids[-1].item())],
        #             dim=0,
        #         )
        #     orient_ids = orient_ids[:V]
        #     oe = self.orient_embed(orient_ids).unsqueeze(1)  # [V,1,D]
        #     x = x + self.orient_scale * oe

        # 展平成 [N, D]，N = V * L
        x_flat = x.reshape(V * L, D)  # [N,D]

        # ------ 2) cross-attention：M 个 slot 去收集所有 patch ------
        slots0 = self.slot_embed.to(device)          # [M, d_attn]
        q = self.q_proj(slots0)                      # [M, d_attn]
        k = self.k_proj(x_flat)                      # [N, d_attn]
        v = self.v_proj(x_flat)                      # [N, d_attn]

        d_attn = k.size(-1)
        attn_logits = (q @ k.t()) / math.sqrt(d_attn)  # [M, N]
        attn = attn_logits.softmax(dim=-1)             # [M, N]

        z = attn @ v                    # [M, d_attn]
        z = self.o_proj(z)              # [M, D]
        z = z + self.ffn(self.norm(z))  # [M, D]

        # ------ 3) 计算 slot 正则（多样性 + 视图覆盖）------
        self._compute_regularizers(attn, V, L, device)

        return z, attn

    def _compute_regularizers(self, attn: torch.Tensor, V: int, L: int, device):
        """
        计算两种 slot 正则：
        - diversity：不同 slot 的注意力分布尽量不重叠
        - view coverage：鼓励所有视图都被关注到，而不是只看某一个视图

        attn: (M, N) 的注意力矩阵（softmax 后），N = V * L
        V:  视图数
        L:  每个视图的 patch 数
        """
        M, N = attn.shape

        # 形状不对就直接清空正则（不会参与 loss）
        if V <= 0 or L <= 0 or N != V * L:
            self.last_div_loss = None
            self.last_cov_loss = None
            return

        # -------- 1) Slot Diversity 正则 --------
        # A: (M, N)
        A = attn

        # AA^T: (M, M)，理想情况是接近单位阵 I
        AA = A @ A.t()
        I = torch.eye(M, device=device, dtype=A.dtype)
        self.last_div_loss = ((AA - I) ** 2).mean()

        # -------- 2) View Coverage 正则 --------
        # 先把 patch 维度按 (视图, patch) 拆开
        # A_view: (M, V, L)
        A_view = A.view(M, V, L)

        # 对每个 slot，在每个视图上累加 attention，再对 slot 求平均：
        # view_mass: (V,)
        view_mass = A_view.sum(-1).mean(0)

        # 归一化成概率分布 p
        p = torch.softmax(view_mass, dim=-1)

        # 希望它接近均匀分布 [1/V, 1/V, ..., 1/V]
        uniform = torch.full_like(p, 1.0 / V)
        self.last_cov_loss = ((p - uniform) ** 2).mean()        


# 这个类定义了多模态基础模型，结合视觉编码器和语言模型
class LlavaMetaModel:

    # 模型初始化，构建视觉编码器和投影器
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)         # 视觉编码器，提取图像特征  delay_load=True表示延迟加载，优化内存使用
            self.mm_projector = build_vision_projector(config)                      # 多模态投影器，将视觉特征映射到语言模型空间

    # 获取视觉编码器实例
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    # 动态初始化视觉模块组件
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower                              # 获取视觉编码器类型或路径
        mm_vision_select_layer = model_args.mm_vision_select_layer          # 获取视觉特征选择层
        mm_vision_select_feature = model_args.mm_vision_select_feature      # 获取视觉特征选择方式
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter        # 获取预训练投影器检查点路径

        # 配置视觉编码器参数
        self.config.mm_vision_tower = vision_tower                                      # 设置视觉编码器的类型
        self.config.mm_vision_tower_config = model_args.vision_tower_config             # 存储视觉编码器的配置参数
        self.config.mm_vision_tower_checkpoint = model_args.vision_tower_checkpoint     # 存储视觉编码器的预训练检查点路径

        # 构建视觉编码器
        vision_tower = build_vision_tower(model_args)

        # [2025-11-19] 新增代码，确保全程冻结视觉塔
        for p in vision_tower.parameters():
            p.requires_grad_(False)
        vision_tower.eval()

        # 兼容不同的分布式训练策略，确保模型在不同并行模式下正常工作
        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        # 配置投影器参数
        self.config.use_mm_proj = True                                      # 启用多模态投影器，将视觉特征投影到文本空间
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')          # 获取投影器类型
        self.config.mm_hidden_size = vision_tower.hidden_size               # 设置投影器的输入维度为视觉编码器的输出维度
        self.config.mm_vision_select_layer = mm_vision_select_layer         # 设置视觉特征选择层
        self.config.mm_vision_select_feature = mm_vision_select_feature     # 设置视觉特征选择方式

        self.mm_projector = build_vision_projector(self.config)         # 构建多模态投影器

        # 加载预训练的投影器权重
        if pretrain_mm_mlp_adapter is not None:         # 支持从检查点加载预训练权重，加速收敛或保持性能
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')                  # 从指定路径加载权重文件到 CPU
            # 从完整的模型权重中提取投影器相关的权重
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}           # 去除前缀，匹配当前投影器的状态字典结构

            # 将提取的权重加载到新创建的投影器中
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


# 抽象基类，定义多模态因果语言模型的接口
class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):                        # 抽象方法，获取基础语言模型
        pass

    def get_vision_tower(self):                 # 获取视觉编码器
        return self.get_model().get_vision_tower()

    # def encode_images(self, images):            # 编码图像特征
    #     image_features = self.get_model().get_vision_tower()(images)                # 提取图像特征
    #     image_features = self.get_model().mm_projector(image_features)              # 将图像特征投影到语言模型空间
    #     return image_features               # 返回投影后的图像特征

    # [2025-11-18] 统一处理图像 -> 视觉塔 -> 投影器 的 dtype / device
    def encode_images(self, images):
        """
        统一处理图像 -> 视觉塔 -> 投影器 的 dtype / device，
        避免 mat1 / mat2 dtype 不一致的问题。
        """
        model = self.get_model()
        vision_tower = model.get_vision_tower()
        projector = model.mm_projector

        # 1. 把图片丢到视觉塔所在的 device / dtype 上
        #    （防止出现 images 在 CPU、vision_tower 在 CUDA 之类的问题）
        vt_params = list(vision_tower.parameters())
        if len(vt_params) > 0:
            vt_device = vt_params[0].device
            vt_dtype = vt_params[0].dtype
            images = images.to(device=vt_device, dtype=vt_dtype)

        # 2. 先过视觉塔，拿到 patch 特征
        image_features = vision_tower(images)   # (B_or_sumV, N_patch, D_vt)

        # 3. 再把特征 cast 到 projector 的 dtype 上
        proj_params = list(projector.parameters())
        if len(proj_params) > 0:
            proj_dtype = proj_params[0].dtype
            if image_features.dtype != proj_dtype:
                image_features = image_features.to(dtype=proj_dtype)

        # 4. 过 mm_projector：映射到语言模型空间
        image_features = projector(image_features)   # (B_or_sumV, N_patch, D_lm)

        return image_features


    # [2025-12-8] 接收 view_ids / orient_ids 并调用 slot_fusion
    def prepare_inputs_labels_for_multimodal(
            self,
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            view_ids=None,
            orient_ids=None,
    ):
        vision_tower = self.get_vision_tower()

        # [2025-12-8] 每次调用先清空上一轮的缓存
        self._last_study_image_global = None
        self._last_slot_div_loss = None
        self._last_slot_cov_loss = None

        # 输入验证和处理
        if vision_tower is None or images is None or input_ids.shape[1] == 1:           # 如果没有视觉编码器或图像，或输入仅包含单个token，则不进行多模态处理
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:         # 仅在特定条件下调整注意力掩码
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels             # 继续进行多模态处理

        # 图像特征提取和分块
        # if type(images) is list or images.ndim == 5:        # 如果输入是图像列表或5维张量，则拼接所有图像 → 批量编码 → 按原始批次分割 → 展平特征
        #     # 处理多个图像样本
        #     concat_images = torch.cat([image for image in images], dim=0)       
        #     image_features = self.encode_images(concat_images)
        #     split_sizes = [image.shape[0] for image in images]
        #     image_features = torch.split(image_features, split_sizes, dim=0)
        #     image_features = [x.flatten(0, 1) for x in image_features]
        # else:
        #     # 处理单个图像批次
        #     image_features = self.encode_images(images)

        # [2025-11-18] 图像特征提取和分块
        #   - eval 阶段：images 是 list[Tensor]，每个 Tensor 形状 (N_view, 3, H, W)
        #   - 以后 train 阶段：也可以是 (B, N_view, 3, H, W) 的 5D Tensor
        if isinstance(images, list) or (isinstance(images, torch.Tensor) and images.ndim == 5):
            # 统一成 list[Tensor(N_view, 3, H, W)]
            if isinstance(images, torch.Tensor):
                image_list = [img for img in images]          # (B, N_view, 3, H, W) -> list 长度 B
            else:
                image_list = images                           # 本身就是 list

            # 1）先把所有视图摊平成一个大 batch，送进视觉塔
            #    sum_V = 所有 sample 的视图总数
            concat_images = torch.cat(image_list, dim=0)      # (sum_V, 3, H, W)
            all_feats = self.encode_images(concat_images)     # (sum_V, N_patch, D)

            # 2）按每个样本的视图数切回来，并在视图维度上做平均 -> study-level 表征
            split_sizes = [img.shape[0] for img in image_list]
            per_sample = torch.split(all_feats, split_sizes, dim=0)   # tuple of (V_i, N_patch, D)

            # [2025-12-13] 修改了多视图分支的逻辑
            # [2025-12-8] 多-slot study-level 融合 + slot 正则（重写）
            model = self.get_model()
            slot_module = getattr(model, "slot_fusion", None)
            if slot_module is None:
                raise RuntimeError(
                    "slot_fusion is not found on core model; "
                    "please make sure LlavaLlamaModel.__init__ initializes it."
                )

            # 视图级注意力模块和超参数
            view_attn = getattr(model, "view_attn", None)
            mv_lambda = float(os.environ.get("MV_LAMBDA", "0.0"))   # 0 表示不启用

            fused_features = []
            div_losses = []
            cov_losses = []

            # view_ids / orient_ids: list[Tensor] or None
            view_ids_list = view_ids if view_ids is not None else [None] * len(per_sample)
            orient_ids_list = orient_ids if orient_ids is not None else [None] * len(per_sample)

            # 约定：VIEW_VOCAB = {PA:0, AP:1, LATERAL:2, OTHER:3}
            PA_ID, AP_ID, LAT_ID = 0, 1, 2

            for b, x in enumerate(per_sample):
                # x: (V, N_patch, D_lm)
                V, Np, D = x.shape

                v_ids = view_ids_list[b][:V] if view_ids_list[b] is not None else None
                o_ids = orient_ids_list[b][:V] if orient_ids_list[b] is not None else None

                # (1) 视图级 gating（可选）
                if (view_attn is not None) and (V > 1) and (mv_lambda > 0.0):
                    # 每个视图做 patch 平均，得到 (V, D) 的 global 特征
                    global_feats = x.mean(dim=1)                 # (V, D)
                    scores = view_attn(global_feats)             # (V, 1)
                    weights = torch.softmax(scores, dim=0)       # (V, 1)
                    weights_ = weights.view(V, 1, 1)             # (V, 1, 1)
                    gated_x = weights_ * x                       # (V, Np, D)
                    x = (1.0 - mv_lambda) * x + mv_lambda * gated_x

                # (2) 对所有视图跑多-slot 融合，得到 M 个 slot token
                slots, attn = slot_module(x, view_ids=v_ids, orient_ids=o_ids)   # slots: (M, D_lm)

                # (3) 选主视图
                main_view_idx = 0
                if V > 1:
                    if v_ids is not None:
                        # 优先 PA > AP > LATERAL > 其他
                        for tgt in (PA_ID, AP_ID, LAT_ID):
                            mask = (v_ids == tgt)
                            if mask.any():
                                main_view_idx = int(mask.nonzero(as_tuple=False)[0])
                                break
                        # 如果都没命中，就保持默认 0
                    elif (view_attn is not None) and (mv_lambda > 0.0):
                        # 没有显式 view_ids 时，用 gating 分数选一个视图
                        with torch.no_grad():
                            global_feats = x.mean(dim=1)          # (V, D)
                            scores = view_attn(global_feats)      # (V, 1)
                            main_view_idx = int(scores.squeeze(-1).argmax())
                    # else: 留在 0（例如 V=1 时）

                # (4) 主视图全 patch + slot 拼接前，加入角色 embedding 与组级 gate
                main_patches = x[main_view_idx]  # (Np, D)
                slots = slots  # (M, D)
                
                # --- Gate 缩放 ---
                alpha_s = torch.sigmoid(model.slot_gate)
                alpha_p = torch.sigmoid(model.patch_gate)
                slots = alpha_s * slots
                main_patches = alpha_p * main_patches
                
                # --- 拼接 ---
                cur_feats = torch.cat([main_patches, slots], dim=0)  # (Np + M, D)
                
                # --- 角色 embedding ---
                slot_roles = torch.ones(slots.size(0), dtype=torch.long, device=slots.device)
                patch_roles = torch.zeros(main_patches.size(0), dtype=torch.long, device=main_patches.device)
                role_ids = torch.cat([patch_roles, slot_roles], dim=0)
                role_emb = model.role_embed(role_ids)
                cur_feats = cur_feats + role_emb  # (Np + M, D)
                
                # (5) 如果在 LlavaLlamaModel.__init__ 里定义了角色 embedding，这里加上
                patch_role_embed = getattr(model, "patch_role_embed", None)
                slot_role_embed = getattr(model, "slot_role_embed", None)
                if (patch_role_embed is not None) and (slot_role_embed is not None):
                    num_slots = slots.size(0)
                    num_tokens = cur_feats.size(0)
                    num_patches = num_tokens - num_slots
                    # 前半段是 patch，后半段是 slot
                    cur_feats[:num_patches] = cur_feats[:num_patches] + patch_role_embed
                    cur_feats[num_patches:] = cur_feats[num_patches:] + slot_role_embed

                fused_features.append(cur_feats)

                # (6) 收集 slot 正则项
                if slot_module.last_div_loss is not None:
                    div_losses.append(slot_module.last_div_loss)
                if slot_module.last_cov_loss is not None:
                    cov_losses.append(slot_module.last_cov_loss)

            # 至此，每个 study 的视觉 token：
            # - 单视图：Np 个该视图的 patch + M 个 slot
            # - 多视图：主视图 Np 个 patch + M 个 slot
            image_features = fused_features

            # 把 batch 内的 slot 正则平均一下，缓存到模型上
            self._last_slot_div_loss = (
                torch.stack(div_losses).mean() if len(div_losses) > 0 else None
            )
            self._last_slot_cov_loss = (
                torch.stack(cov_losses).mean() if len(cov_losses) > 0 else None
            )

            # 为检查级对齐 / CheXpert 缓存 study-level 全局视觉向量
            if len(image_features) > 0:
                study_global = torch.stack(
                    [feat.mean(dim=0) for feat in image_features], dim=0
                )  # (B, D_lm)
            else:
                study_global = None
            self._last_study_image_global = study_global
            # [2025-12-13] 修改了多视图分支的逻辑 结束


            # [2025-12-12] 新增 [CheXpert] study-level global feature，把每个 study 的视觉 token 平均成一个 (B, D) 的全局向量，并存到模型上，供后面计算疾病 loss 使用
            if image_features is not None:
                global_feats = []
                for feat in image_features:
                    # feat: (N_token, D)
                    global_feats.append(feat.mean(dim=0))
                global_feats = torch.stack(global_feats, dim=0)  # (B, D)
                self._last_study_image_global = global_feats
            else:
                self._last_study_image_global = None
            # [2025-12-12] 新增结束


            # 把 batch 内的 slot 正则平均一下，缓存到模型上，稍后在 forward 里加到 loss
            self._last_slot_div_loss = (
                torch.stack(div_losses).mean() if len(div_losses) > 0 else None
            )
            self._last_slot_cov_loss = (
                torch.stack(cov_losses).mean() if len(cov_losses) > 0 else None
            )

        else:
            # 单视图路径：images 为 (B, 3, H, W)
            image_features = self.encode_images(images)  # (B, N_patch, D_lm)
            # 为了跟上面统一成 list[Tensor(N_patch,D_lm)]：
            if isinstance(image_features, torch.Tensor):
                image_features = [feat for feat in image_features]


        # [2025-11-27] 新增：为检查级对齐缓存每个 study 的全局视觉向量
        # 此时 image_features 是长度为 B 的 list，每个元素形状 (N_patch, D_lm)
        # if len(image_features) > 0:
        #     # 对 patch 维度做平均，得到 (B, D_lm) 的 study-level 表征
        #     study_global = torch.stack(
        #         [feat.mean(dim=0) for feat in image_features],
        #         dim=0
        #     )  # (B, D_lm)
        #     # 直接挂在模型实例上，供 forward() 使用
        #     self._last_study_image_global = study_global
        # else:
        #     self._last_study_image_global = None

        # 准备存储新的输入嵌入和标签
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        # 多模态输入构建
        for batch_idx, cur_input_ids in enumerate(input_ids):               # 遍历每个样本的输入ID
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:             # 如果当前样本没有图像token，直接嵌入文本
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                # 当前样本不包含图像token的特殊处理
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # 查找图像token位置并替换为图像特征
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # 将文本序列中的图像标记替换为实际的图像特征嵌入
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    # 情况A：使用特殊起始/结束token的处理
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    # 情况B：标准处理，直接替换IMAGE_TOKEN_INDEX为图像特征
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    # 标签对齐处理
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # 处理剩余的文本部分
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        # 对齐批次中不同样本的长度
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            # [2025-12-13] 新增 对齐 attention_mask 时知道每个样本“真实 new_len”
            embed_lens = [x.shape[0] for x in new_input_embeds]  # 每个样本对齐前 new 序列长度
            # [2025-12-13] 新增结束

            # 对输入嵌入进行填充对齐
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            # 对标签进行填充对齐
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # [2025-12-13] 修改 调整注意力掩码以匹配新的输入长度（兼容 labels=None 的推理/generate）
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, new_len in zip(attention_mask, embed_lens):
                    old_len = cur_attention_mask.shape[0]   # 原 attention_mask 长度（等于 input_ids padding 后长度）
                    pad_left = new_len - old_len            # 新增的“左侧”有效 token 数（通常是插入的视觉 token）
                    pad_right = max_len - new_len           # 为了对齐到 max_len 的“右侧”padding 数

                    # 安全检查：理论上 new_len 不应小于 old_len
                    if pad_left < 0:
                        raise ValueError(f"pad_left < 0: new_len={new_len}, old_len={old_len}. Check multimodal concat logic.")

                    new_attn_mask_pad_left = torch.full(
                        (pad_left,), True, dtype=attention_mask.dtype, device=cur_attention_mask.device
                    )
                    new_attn_mask_pad_right = torch.full(
                        (pad_right,), False, dtype=attention_mask.dtype, device=cur_attention_mask.device
                    )
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0
                    )
                    new_attention_mask.append(cur_new_attention_mask)

                attention_mask = torch.stack(new_attention_mask, dim=0)

                # 断言：训练时对齐 labels；推理时对齐 new_input_embeds
                if labels is not None:
                    assert attention_mask.shape == new_labels.shape
                else:
                    assert attention_mask.shape == new_input_embeds.shape[:2]
            # [2025-12-13] 修改结束
        else:               # 序列长度一致时的简单处理：如果所有样本长度相同，直接堆叠
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                if labels is not None:
                    assert attention_mask.shape == new_labels.shape
                else:
                    assert attention_mask.shape == new_input_embeds.shape[:2]


        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    # 初始化视觉相关的特殊token
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # 添加图像patch token
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False