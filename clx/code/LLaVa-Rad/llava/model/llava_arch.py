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
from mamba_ssm import Mamba
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


import torch.nn.functional as F

def _downsample_grid_tokens(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    x: (N, D) where N = H*W (e.g., 37*37)
    return: (r*r, D) adaptive average pooled on the token grid
    """
    N, D = x.shape
    H = int(math.sqrt(N))
    if H * H != N or r <= 0 or r >= H:
        return x
    x2d = x.transpose(0, 1).reshape(D, H, H).unsqueeze(0)   # (1, D, H, W)
    x2d = F.adaptive_avg_pool2d(x2d, (r, r))                # (1, D, r, r)
    xds = x2d.squeeze(0).reshape(D, r * r).transpose(0, 1)  # (r*r, D)
    return xds

# [2026-1-7] 新增上采样模块，把 Z_PA^ds 回到原始 H*H，从而 保持 token 数量与 baseline 一致
def _upsample_grid_tokens(x: torch.Tensor, target_n: int) -> torch.Tensor:
    """
    x: (r*r, D)
    target_n: H*H (e.g., 37*37)
    return: (H*H, D) bilinear upsample on token grid
    """
    r2, D = x.shape
    r = int(math.sqrt(r2))
    H = int(math.sqrt(target_n))

    # invalid / no-op cases
    if r * r != r2 or H * H != target_n or r <= 0 or H <= 0 or r == H:
        return x

    x2d = x.transpose(0, 1).reshape(D, r, r).unsqueeze(0)  # (1, D, r, r)

    orig_dtype = x.dtype
    if orig_dtype == torch.bfloat16:
        # bf16 bilinear upsample may be unsupported; run interpolate in fp32
        with torch.cuda.amp.autocast(enabled=False):
            x2d = F.interpolate(x2d.float(), size=(H, H), mode="bilinear", align_corners=False)
        x2d = x2d.to(dtype=orig_dtype)
    else:
        x2d = F.interpolate(x2d, size=(H, H), mode="bilinear", align_corners=False)

    xup = x2d.squeeze(0).reshape(D, H * H).transpose(0, 1)  # (H*H, D)
    return xup

# [2026-1-7] 新增结束


class TokenLearnerLite(nn.Module):
    """
    从 (N,D) patch tokens 自适应聚合出 K 个 evidence tokens（压缩辅助视角信息，提取代表性“证据 token”)
    参考 TokenLearner 思想，但做成最轻量的“注意力加权求和”版本。
    """
    def __init__(self, dim: int, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, num_tokens, bias=False)

        # [2025-12-22] 新增，logits 全 0 -> softmax 均匀 -> K 个 token 初期都是平均池化版本
        nn.init.zeros_(self.score.weight)
        # [2025-12-22] 新增结束

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,D)
        return: (K,D)
        """
        orig_dtype = x.dtype

        xn = self.norm(x)
        target_dtype = self.score.weight.dtype

        if xn.dtype != target_dtype:
            xn = xn.to(dtype=target_dtype)
        x_cast = x if x.dtype == target_dtype else x.to(dtype=target_dtype)

        if xn.is_cuda and target_dtype == torch.float32:
            with torch.cuda.amp.autocast(enabled=False):
                logits = self.score(xn)  # (N,K)
                attn = torch.softmax(logits.transpose(0, 1), dim=-1)  # (K,N)
                tok = attn @ x_cast  # (K,D)
        else:
            logits = self.score(xn)
            attn = torch.softmax(logits.transpose(0, 1), dim=-1)
            tok = attn @ x_cast

        if tok.dtype != orig_dtype:
            tok = tok.to(dtype=orig_dtype)
        return tok


# [2026-1-7] 新增：轻量 MambaBlock + Bi-Mamba
class MambaLiteBlock(nn.Module):
    """
    LN -> (D -> Dm) -> Mamba(Dm) -> (Dm -> D)
    在 Deepspeed bf16 模式下强制整个 Mamba 流程使用 fp32 计算。
    """
    def __init__(self, dim: int, mamba_dim: int = 1024, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, mamba_dim, bias=False)

        # ======= 临时强制 CPU 初始化为 float32 =======
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            self.mamba = Mamba(
                d_model=mamba_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        finally:
            # 恢复默认 dtype（避免影响其他模块）
            torch.set_default_dtype(prev_dtype)
        # ============================================

        self.out_proj = nn.Linear(mamba_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        orig_dtype = x.dtype
        is_2d = (x.dim() == 2)
        x_in = x.unsqueeze(0) if is_2d else x  # (B,L,D)

        # ---- 强制所有计算在 fp32 下 ----
        x32 = x_in.to(torch.float32)

        # LayerNorm 手动计算
        norm_weight = self.norm.weight.to(torch.float32) if self.norm.weight is not None else None
        norm_bias = self.norm.bias.to(torch.float32) if self.norm.bias is not None else None
        x_norm = torch.nn.functional.layer_norm(x32, self.norm.normalized_shape, norm_weight, norm_bias, self.norm.eps)

        # 线性输入
        h = F.linear(x_norm, self.in_proj.weight.to(torch.float32), None)

        # --- 关键：在每次 forward 前，强制 Mamba 参数为 float32 ---
        for p in self.mamba.parameters():
            if p.dtype != torch.float32:
                p.data = p.data.float()

        # Mamba 核心计算 (float32)
        h = self.mamba(h.to(torch.float32))

        # 输出线性层 (float32)
        y32 = F.linear(h, self.out_proj.weight.to(torch.float32), None)

        y = y32.to(orig_dtype)
        return y.squeeze(0) if is_2d else y




# 跨视角信息流建模
class BiMambaLite(nn.Module):
    """
    Bi-directional Mamba: forward + backward (reverse scan) then average.
    """
    def __init__(self, dim: int, mamba_dim: int = 1024, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.fwd = MambaLiteBlock(dim, mamba_dim, d_state, d_conv, expand)
        self.bwd = MambaLiteBlock(dim, mamba_dim, d_state, d_conv, expand)
        
        # [2026-1-9] 新增：Bi-Mamba forward/backward 融合改为 concat + proj
        # 目的：避免 y_f 与 y_b 直接相加造成信息抵消；让融合方式可学习（更接近 BiLSTM 的标准做法）
        self.fuse_proj = nn.Linear(2 * dim, dim, bias=False)

        # [2026-1-9] 初始化为“等价于平均”的近似恒等融合，保证训练初期行为与原来 0.5*(y_f+y_b) 尽量一致、稳定
        with torch.no_grad():
            self.fuse_proj.weight.zero_()
            idx = torch.arange(dim)
            self.fuse_proj.weight[idx, idx] = 0.5
            self.fuse_proj.weight[idx, idx + dim] = 0.5
        # [2026-1-9] 新增结束

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (L, D)
        """
        y_f = self.fwd(x)
        x_rev = torch.flip(x, dims=[0])
        y_b_rev = self.bwd(x_rev)
        y_b = torch.flip(y_b_rev, dims=[0])
        # return 0.5 * (y_f + y_b)
        # [2026-1-9] 修改：concat + proj 融合
        y = torch.cat([y_f, y_b], dim=-1)  # (L, 2D)
        y = self.fuse_proj(y)             # (L, D)
        return y
        # [2026-1-9] 修改结束

# [2026-1-7] 新增结束

# [2026-1-7] 新增两阶段融合器
class MVGridMambaFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.r = int(os.environ.get("MV_DS_R", "24"))
        self.k = int(os.environ.get("MV_EVID_K", "16"))

        mamba_dim = int(os.environ.get("MV_MAMBA_DIM", "1024"))
        d_state = int(os.environ.get("MV_MAMBA_DSTATE", "16"))
        d_conv = int(os.environ.get("MV_MAMBA_DCONV", "4"))
        expand = int(os.environ.get("MV_MAMBA_EXPAND", "2"))

        self.local = MambaLiteBlock(dim, mamba_dim, d_state, d_conv, expand)    # 对单一视角进行局部建模，用于平滑特征并提取结构信息
        self.bi = BiMambaLite(dim, mamba_dim, d_state, d_conv, expand)          # 负责跨视角双向信息流建模，能从辅助视角向主视角蒸馏信息
        self.token_learner = TokenLearnerLite(dim, num_tokens=self.k)           # 将每个辅助视角的下采样特征压缩为少量 K 个“证据 token”，代表该视角的主要诊断特征

        # [2026-1-9] 新增：显式 segment(view/type) embedding
        # token_type: 0=aux evidence tokens, 1=anchor(PA/AP/Lat) grid tokens
        self.seg_embed = nn.Embedding(2, dim)

        # view embedding：用于区分 aux 来自哪个视角；默认给一个上限（MIMIC-CXR 多视角常见 3 类：PA/AP/Lateral）
        self.num_views = int(os.environ.get("MV_NUM_VIEWS", "8"))
        self.view_embed = nn.Embedding(self.num_views, dim)

        # 初始化为 0：不在训练初期引入额外扰动（配合你 gate/ramp 的“恒等保护”哲学）
        nn.init.zeros_(self.seg_embed.weight)
        nn.init.zeros_(self.view_embed.weight)
        # [2026-1-9] 新增结束

        # gate logits: sigmoid(-6) ~ 0.0025  (非常接近 0，利于贴近 baseline)
        # self.g1_logit = nn.Parameter(torch.tensor(-6.0))
        # self.g2_logit = nn.Parameter(torch.tensor(-6.0))
        # gate init（对齐你原来的 AR_CVI_GATE_INIT 习惯）
        gate_init = float(os.environ.get("MV_GATE_INIT", "-6.0"))       # 初始值设定，控制一开始注入信息的强度
        self.g1_logit = nn.Parameter(torch.tensor(gate_init))           # 残差门控系数，控制 Stage1（局部）信息注入的强度
        self.g2_logit = nn.Parameter(torch.tensor(gate_init))           # 残差门控系数，控制 Stage2（跨视角）信息注入的强度

        # hard cap
        self.gate_max = float(os.environ.get("MV_GATE_MAX", "1.0"))


        # ramp：训练步数线性增长函数，使融合强度从 0 缓慢上升，避免早期扰动特征
        self.warmup_steps = int(os.environ.get("MV_INJECT_WARMUP", "0"))    # 前 warmup_steps 步骤内不注入任何多视角信息
        self.ramp_steps = int(os.environ.get("MV_INJECT_RAMP", "2000"))     # 从 warmup 结束后开始线性 ramp-up，到 ramp_steps 结束时达到最大注入强度

    # 线性 ramp 函数
    def _ramp(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return 1.0
        t = (step - self.warmup_steps) / float(self.ramp_steps)
        return float(max(0.0, min(1.0, t)))

    def forward(self, x_views: torch.Tensor, view_ids=None, global_step: int = 0) -> torch.Tensor:
        """
        x_views: (V, Np, D)，V个视角的token特征
        view_ids: (V,) optional, used to pick PA as anchor if possible，视角标签（如 PA/AP/Lateral）
        return: (Np, D) fused tokens for the anchor (PA)，输出融合后主视角token
        """
        V, Np, D = x_views.shape
        H = int(math.sqrt(Np))

        # -------------------------
        # (1) 统一的 anchor 选择逻辑：完全对齐 baseline 语义
        # -------------------------
        pick = os.environ.get("MV_BASELINE_PICK", "PA_AP_FIRST").upper()

        if pick == "INDEX":
            anchor_idx = int(os.environ.get("MV_BASELINE_INDEX", "0"))
            anchor_idx = max(0, min(anchor_idx, V - 1))
        else:
            # PA_AP_FIRST：优先 PA，否则 AP，否则 LATERAL，否则 0
            anchor_idx = 0
            if view_ids is not None:
                v = view_ids.tolist() if torch.is_tensor(view_ids) else list(view_ids)
                if 0 in v:        # PA=0
                    anchor_idx = v.index(0)
                elif 1 in v:      # AP=1
                    anchor_idx = v.index(1)
                elif 2 in v:      # LATERAL=2
                    anchor_idx = v.index(2)
                else:
                    anchor_idx = 0

        # -------------------------
        # (2) gate cap：优先 env，其次 self.gate_max
        # -------------------------
        gate_cap = float(os.environ.get("MV_GATE_MAX", str(self.gate_max)))  # 保留你现在的动态覆盖习惯也行

        # -------------------------
        # (3) 检查 1 的关键：硬关死（gate==0）必须“直接返回 anchor patch”
        # -------------------------
        if gate_cap <= 0.0:
            if os.environ.get("MV_DEBUG_ANCHOR", "0") == "1":
                print(f"[MV][gate0] V={V} pick={pick} anchor_idx={anchor_idx}", flush=True)
            return x_views[anchor_idx]
        
                
        # 保证模型初期仅保留baseline行为，逐步学习多视角融合
        ramp = self._ramp(global_step)
        g1 = ramp * torch.sigmoid(self.g1_logit)
        g2 = ramp * torch.sigmoid(self.g2_logit)

        g1 = torch.clamp(g1, max=gate_cap)
        g2 = torch.clamp(g2, max=gate_cap)

        g1 = g1.to(device=x_views.device, dtype=x_views.dtype)
        g2 = g2.to(device=x_views.device, dtype=x_views.dtype)

        # [2026-1-20 强制修正] 处理 view_ids，防止 Embedding 越界崩溃
        if view_ids is not None:
            view_ids_t = torch.as_tensor(view_ids, device=x_views.device, dtype=torch.long)
            
            # 打印一次 debug 信息 (仅 rank0, 仅 step 0) 确认是否生效
            if global_step == 0 and os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[DEBUG] view_ids check: min={view_ids_t.min()}, max={view_ids_t.max()}, limit={self.num_views}")

            # ★★★ 核心修复：强制钳位到 [0, num_views-1] ★★★
            # 你的 num_views 默认是 8，如果数据里有更大的 ID，不 clamp 就会导致 CUDA 报错
            view_ids_t = view_ids_t.clamp(0, self.num_views - 1)
        else:
            view_ids_t = None

        # [2026-1-19] DEBUG & SAFETY: 确保 view_ids 不越界
        if view_ids_t is not None:
            # 打印一次 debug 信息 (仅 rank0, 仅 step 0)
            if global_step == 0 and os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[DEBUG] view_ids range: min={view_ids_t.min()}, max={view_ids_t.max()}, num_views={self.num_views}")
            
            # 强制钳位，防止 Embedding 越界崩溃
            view_ids_t = view_ids_t.clamp(0, self.num_views - 1)

        # -------- Stage1: view-local on downsampled grid --------
        # 在低分辨率空间建模单视角的内部结构，减少噪声和冗余，准备跨视角融合。
        ds_list = []
        evid_list = []
        for i in range(V):
            x = x_views[i]                                  # (Np,D)
            x_ds = _downsample_grid_tokens(x, self.r)       # (r*r,D)，调用 _downsample_grid_tokens 将每视角 H×H 网格下采样到 r×r
            delta = self.local(x_ds) - x_ds                 # (r*r,D)，对下采样特征做 local(x_ds)
            x_ds_tilde = x_ds + g1 * delta
            ds_list.append(x_ds_tilde)

        # aux evidences。从每个非主视角下采样结果中选出 K 个“证据 token”，表示该视角主要的病灶特征
        for i in range(V):
            if i == anchor_idx:
                continue
            evid_i = self.token_learner(ds_list[i])         # (K,D)

            # [2026-1-9] 新增：aux evidence tokens 加上显式 segment(view/type) embedding
            # seg=0 表示 aux evidence；view_id 用于区分来源视角（若缺失则默认为 0）
            seg_aux = self.seg_embed.weight[0].to(dtype=evid_i.dtype, device=evid_i.device)
            
            if view_ids_t is not None:
                vid = int(view_ids_t[i].item())
            else:
                vid = 0
            view_e = self.view_embed.weight[vid].to(dtype=evid_i.dtype, device=evid_i.device)
            
            evid_i = evid_i + seg_aux + view_e
            # [2026-1-9] 新增结束

            evid_list.append(evid_i)

        E_aux = torch.cat(evid_list, dim=0) if len(evid_list) > 0 else None
        X_anchor = ds_list[anchor_idx]                               # (r*r,D)

        # [2026-1-9] 新增：anchor grid tokens 加上显式 segment(view/type) embedding
        # seg=1 表示 anchor grid；view_id 用 anchor 的视角 id（PA/AP/Lateral）
        seg_pa = self.seg_embed.weight[1].to(dtype=X_anchor.dtype, device=X_anchor.device)

        # [修正] 增加 None 检查，防止 inference 时未传 view_ids 导致崩溃
        if view_ids_t is not None:
            anchor_vid = int(view_ids_t[anchor_idx].item())
        else:
            anchor_vid = 0
            
        view_pa = self.view_embed.weight[anchor_vid].to(dtype=X_anchor.dtype, device=X_anchor.device)

        X_anchor = X_anchor + seg_pa + view_pa
        # [2026-1-9] 新增结束

        # -------- Stage2: cross-view Bi-Mamba mixing --------
        if E_aux is None:
            Z_anchor_ds = X_anchor
        else:
            S = torch.cat([E_aux, X_anchor], dim=0)            # (K_aux + r*r, D)，拼接序列
            S2 = self.bi(S)                                # (same)，经 BiMambaLite 进行正反向状态混合
            Z_anchor_ds = S2[-(self.r * self.r):]              # only keep anchor grid tokens，仅保留最后 r×r 个 token（主视角的融合结果）

        # upsample back to H*H and gated-inject to keep baseline token count
        Z_up = _upsample_grid_tokens(Z_anchor_ds, target_n=H * H)  # (Np,D)，将 r×r 网格恢复为 H×H
        X_anchor_full = x_views[anchor_idx]                         # (Np,D)

        out = X_anchor_full + g2 * (Z_up - X_anchor_full)
        return out
# [2026-1-7] 新增结束


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

        # self.mm_projector = build_vision_projector(self.config)         # 构建多模态投影器
        # [2025-12-29] 重新修改，让mm_Projector不会再初始化训练
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # 保留已加载的 mm_projector 权重；确保可训练
            for p in self.mm_projector.parameters():
                p.requires_grad = True
        # [2025-12-29] 修改结束

        # 加载预训练的投影器权重
        if pretrain_mm_mlp_adapter is not None:         # 支持从检查点加载预训练权重，加速收敛或保持性能
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')                  # 从指定路径加载权重文件到 CPU
            # 从完整的模型权重中提取投影器相关的权重
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}           # 去除前缀，匹配当前投影器的状态字典结构

            # 将提取的权重加载到新创建的投影器中
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("[SUCCESS] mm_projector loaded.")

            # 2. [2026-1-23] 新增加载 Fusion Mamba (mv_grid_mamba)
            # 只有当 key 中包含 mv_grid_mamba 且模型有该模块时才加载
            if hasattr(self, "mv_grid_mamba") and any("mv_grid_mamba" in k for k in mm_projector_weights.keys()):
                print(f"[DEBUG] Found {len([k for k in mm_projector_weights.keys() if 'mv_grid_mamba' in k])} Mamba keys in checkpoint.")
                
                # --- [新增验证 1]：打印加载前的某个权重均值 ---
                old_weight_val = self.mv_grid_mamba.bi.fuse_proj.weight.data.mean().item()
                print(f"[DEBUG] Before loading, Mamba fuse_proj mean: {old_weight_val:.8f}")
                # 你的保存逻辑是直接保存 model.named_parameters()，所以 key 可能是 "model.mv_grid_mamba.xxx" 或直接 "mv_grid_mamba.xxx"
                # 需要根据保存时的 key 结构适配。通常 safe_save_model_for_hf_trainer 里的 get_mm_adapter_state_maybe_zero_3 会保留完整名
                # 假设保存的 key 是 "model.mv_grid_mamba.bi.fwd..."，我们需要去掉前缀匹配

                mamba_dict = {}
                for k, v in mm_projector_weights.items():
                    if "mv_grid_mamba" in k:
                        # 简单粗暴的处理：找到 mv_grid_mamba 之后的部分
                        # 例如 model.mv_grid_mamba.local.in_proj.weight -> local.in_proj.weight
                        suffix = k.split("mv_grid_mamba.")[-1]
                        mamba_dict[suffix] = v

                # self.mv_grid_mamba.load_state_dict(mamba_dict, strict=False)

                # --- [验证步骤 B]：执行加载并获取结果 ---
                missing, unexpected = self.mv_grid_mamba.load_state_dict(mamba_dict, strict=False)
                
                # --- [验证步骤 C]：记录加载后的权重均值 ---
                new_weight_val = self.mv_grid_mamba.bi.fuse_proj.weight.data.mean().item()
                print(f"[DEBUG] After loading,  Mamba fuse_proj mean: {new_weight_val:.8f}")

                # --- [验证步骤 D]：最终判定 ---
                if len(missing) > 0:
                    print(f"[WARNING] Mamba Missing keys: {len(missing)} (First 5: {missing[:5]})")
                
                if abs(old_weight_val - new_weight_val) < 1e-9:
                    print("[ERROR] !!! Mamba weights did NOT change! Loading might have FAILED (or init happened to be same) !!!")
                else:
                    print("[SUCCESS] Mamba weights changed and loaded successfully.")

            else:
                print("[WARNING] 'mv_grid_mamba' NOT found in checkpoint or model! Initializing randomly.")

            # 3. 新增加载 Disease Heads
            for head_name in ["visual_disease_head", "text_disease_head"]:
                # 检查模型里有没有这个头，且权重文件里有没有这个头的 key
                if hasattr(self, head_name) and any(head_name in k for k in mm_projector_weights.keys()):
                    module = getattr(self, head_name)
                    print(f"---------------- Checking {head_name} ----------------")
                    
                    # [验证 A] 旧值
                    old_h_val = module.weight.data.mean().item()
                    print(f"[DEBUG] Before: {old_h_val:.8f}")

                    # 提取
                    head_dict = {}
                    for k, v in mm_projector_weights.items():
                        if head_name in k:
                            suffix = k.split(f"{head_name}.")[-1]
                            head_dict[suffix] = v
                    
                    # [验证 B] 加载
                    missing, unexpected = module.load_state_dict(head_dict, strict=False)
                    
                    # [验证 C] 新值
                    new_h_val = module.weight.data.mean().item()
                    print(f"[DEBUG] After : {new_h_val:.8f}")

                    # [验证 D] 判定
                    if len(missing) > 0:
                        print(f"[WARNING] {head_name} Missing keys: {missing}")
                    
                    if abs(old_h_val - new_h_val) < 1e-9:
                        print(f"[ERROR] !!! {head_name} weights did NOT change! !!!")
                    else:
                        print(f"[SUCCESS] {head_name} loaded successfully.")
                    print("-------------------------------------------------------")
                else:
                    print(f"[WARNING] {head_name} skipped (not in model or not in checkpoint).")
            # --- [2026-1-23] 修改结束  ---


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
    def encode_images(self, images, view_ids=None, findings_embeds=None):
        """
        统一处理图像 -> 视觉塔 -> 投影器 的 dtype / device，
        避免 mat1 / mat2 dtype 不一致的问题。
        """
        model = self.get_model()
        vision_tower = model.get_vision_tower()
        projector = model.mm_projector

        # 1. 把图片丢到视觉塔所在的 device / dtype 上
        vt_params = list(vision_tower.parameters())
        if len(vt_params) > 0:
            vt_device = vt_params[0].device
            vt_dtype = vt_params[0].dtype
            images = images.to(device=vt_device, dtype=vt_dtype)

        # [2025-12-14] 修改 encode_images：冻结 vision_tower 时用 no_grad 降显存 开始
        import contextlib

        # 2. 先过视觉塔，拿到 patch 特征
        vt_frozen = True
        for p in vision_tower.parameters():
            if p.requires_grad:
                vt_frozen = False
                break
            
        if vt_frozen:
            with torch.no_grad():
                image_features = vision_tower(images)   # (B_or_sumV, N_patch, D_vt)
        else:
            image_features = vision_tower(images)
        # [2025-12-14] 修改 encode_images：冻结 vision_tower 时用 no_grad 降显存 结束

        # 3. 再把特征 cast 到 projector 的 dtype 上
        proj_params = list(projector.parameters())
        if len(proj_params) > 0:
            proj_dtype = proj_params[0].dtype
            if image_features.dtype != proj_dtype:
                image_features = image_features.to(dtype=proj_dtype)

        # 4. 过 mm_projector：映射到语言模型空间
        image_features = projector(image_features)   # (B_or_sumV, N_patch, D_lm)

        # [修正] 这里不再计算任何 Loss，保持函数纯净
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
            findings_embeds=None,  # 新增参数
            **kwargs # [新增] 吸收 findings_embeds 等额外参数，防止报错
    ):
        vision_tower = self.get_vision_tower()

        # [2025-12-8] 每次调用先清空上一轮的缓存
        self._last_study_image_global = None
        self._last_batch_view_feats = None      # [修正] 必须清空此缓存，防止不同 batch 间数据污染
        self._last_slot_div_loss = None
        self._last_slot_cov_loss = None
        self._last_router_entropy = None

        # 输入验证和处理
        if vision_tower is None or images is None or input_ids.shape[1] == 1:           # 如果没有视觉编码器或图像，或输入仅包含单个token，则不进行多模态处理
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:         # 仅在特定条件下调整注意力掩码
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels             # 继续进行多模态处理


        if isinstance(images, list) or (isinstance(images, torch.Tensor) and images.ndim == 5):
            # 统一成 list[Tensor(N_view, 3, H, W)]
            if isinstance(images, torch.Tensor):
                image_list = [img for img in images]   # (B, V, 3, H, W) -> list[B]
            else:
                image_list = images                    # list[B] of (V,3,H,W)

            model = self.get_model()
            mv_fusion = os.environ.get("MV_FUSION", getattr(model, "mv_fusion", "baseline")).lower()

            # =========================================================
            # [STRICT BASELINE] 先选定每个 study 的一张图，再走单视角 encode
            # =========================================================
            if mv_fusion == "baseline":
                pick = os.environ.get("MV_BASELINE_PICK", "INDEX").strip().upper()
                baseline_index = int(os.environ.get("MV_BASELINE_INDEX", "0"))

                picked = []
                view_ids_list = view_ids if view_ids is not None else [None] * len(image_list)

                for b, img in enumerate(image_list):
                    V = img.shape[0]
                    v_ids = view_ids_list[b][:V] if view_ids_list[b] is not None else None

                    a = 0
                    if pick == "INDEX":
                        a = max(0, min(baseline_index, V - 1))
                    elif pick == "FIRST":
                        a = 0
                    # [2026-1-2] 把代码改为PA 优先于 AP
                    elif pick in ("PA_AP_FIRST", "PAAPFIRST", "FRONTAL_FIRST", "FRONTALFIRST") and torch.is_tensor(v_ids):
                        # 规则：PA/AP 优先；都没有才选 LATERAL；再没有就 0
                        a = 0
                        pa  = (v_ids == 0).nonzero(as_tuple=True)[0]
                        ap  = (v_ids == 1).nonzero(as_tuple=True)[0]
                        lat = (v_ids == 2).nonzero(as_tuple=True)[0]

                        if pa.numel() > 0:
                            a = int(pa[0].item())
                        elif ap.numel() > 0:
                            a = int(ap[0].item())
                        elif lat.numel() > 0:
                            a = int(lat[0].item())
                        else:
                            a = 0    
                    # [2026-1-2] 修改结束
                    else:
                        a = 0
                    # [2026-1-2] 新增打印代码
                    debug = os.environ.get("AR_CVI_DEBUG_ANCHOR", "0") == "1"
                    is_rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0

                    if debug and is_rank0 and (not self.training):
                        v_list = None if (v_ids is None) else v_ids.detach().cpu().tolist()
                        chosen = None if (v_ids is None) else int(v_ids[a].item())
                        print(f"[ANCHOR][baseline] b={b} pick={pick} V={V} view_ids={v_list} -> a={a}, view={chosen}")
                    # [2026-1-2] 新增结束
                    picked.append(img[a])  # (3,H,W)

                picked_images = torch.stack(picked, dim=0)          # (B,3,H,W)
                image_features = self.encode_images(picked_images, view_ids=None, findings_embeds=findings_embeds)  # (B,N_patch,D)

                # 统一成 list[Tensor(N_patch,D)]
                image_features = [feat for feat in image_features]

                self._last_study_image_global = torch.stack([feat.mean(dim=0) for feat in image_features], dim=0)
                
                # [修正] Baseline 模式下也需要缓存视图级特征，否则 llava_llama 计算 view-level loss 时会失效
                # Baseline 相当于每个 study 只有一个 view (B, N_patch, D) -> list of (1, D)
                self._last_batch_view_feats = [feat.mean(dim=0, keepdim=True) for feat in image_features]
                self._last_router_entropy = None
                self._last_slot_div_loss = None
                self._last_slot_cov_loss = None

            # =========================================================
            # [MULTIVIEW PATH] 其他多视角融合
            # =========================================================
            else:
                concat_images = torch.cat(image_list, dim=0)      # (sum_V,3,H,W)
                all_feats = self.encode_images(
                    concat_images, 
                    view_ids=None, # 这里通常不需要 view_ids 用于 projector，除非你有特殊逻辑
                    findings_embeds=findings_embeds
                )  # (sum_V,N_patch,D)

                split_sizes = [img.shape[0] for img in image_list]
                per_sample = torch.split(all_feats, split_sizes, dim=0)  # tuple[(V_i,Np,D)]

                fused_features = []
                router_ent_losses = []

                # [2026-1-20] 修正：去除重复定义，统一初始化
                view_ids_list = view_ids if view_ids is not None else [None] * len(per_sample)
                orient_ids_list = orient_ids if orient_ids is not None else [None] * len(per_sample)
                
                # 用于存储每个 study 的原始视图特征 (V, D)，供 llava_llama 计算视图级一致性 loss
                batch_view_features = []

                for b, x in enumerate(per_sample):
                    V, Np, D = x.shape
                    v_ids = view_ids_list[b][:V] if view_ids_list[b] is not None else None
                    o_ids = orient_ids_list[b][:V] if orient_ids_list[b] is not None else None

                    # 1. 收集视图级特征 (取 patch 平均) -> (V, D)
                    # 此时 x 是 (V, Np, D)，mean(1) 得到每个视图的全局向量
                    batch_view_features.append(x.mean(dim=1))

                    # [2026-1-7] 新增mamba_grid分支
                    if mv_fusion == "mamba_grid":
                        mv = getattr(model, "mv_grid_mamba", None)
                        if mv is None:
                            raise RuntimeError("mv_grid_mamba is not found on core model.")

                        step = int(getattr(model, "_mv_global_step", 0))  # 注意：写在 model 上更稳
                        fused_patches = mv(x, view_ids=v_ids, global_step=step)  # x: (V,Np,D) -> (Np,D)
                        fused_features.append(fused_patches)
                    # [2026-1-7] 新增结束
                    else:
                        raise RuntimeError(f"Unknown MV_FUSION={mv_fusion}")

                image_features = fused_features
                
                # 缓存特征供 Loss 计算使用
                self._last_study_image_global = torch.stack([feat.mean(dim=0) for feat in image_features], dim=0)
                self._last_batch_view_feats = batch_view_features  # [NEW] 缓存列表 [Tensor(V1,D), Tensor(V2,D)...]
                
                self._last_router_entropy = None
                self._last_slot_div_loss = None
                self._last_slot_cov_loss = None
        else:
            # 单视图路径：images 为 (B, 3, H, W)
            image_features = self.encode_images(images)  # (B, N_patch, D_lm)
            if isinstance(image_features, torch.Tensor):
                image_features = [feat for feat in image_features]
            self._last_study_image_global = torch.stack([feat.mean(dim=0) for feat in image_features], dim=0)
            self._last_router_entropy = None


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