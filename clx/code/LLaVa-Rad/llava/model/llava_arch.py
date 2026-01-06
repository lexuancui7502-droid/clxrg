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


# [2025-12-14] 新增 AR-CVI 模块（证据token + 共享memory交互 + 锚点路由 + 门控融合）开始
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

class _CrossAttnLite(nn.Module):
    """
    Low-rank cross-attention: Q in dim -> attn_dim, K/V in dim -> attn_dim, then project back.
    Uses scaled_dot_product_attention when available (PyTorch>=2).
    """
    def __init__(self, dim: int, attn_dim: int = 1024, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(dim, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, dim, bias=False)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q:  (B, Lq, D)
        kv: (B, Lk, D)
        return: (B, Lq, D)
        """
        B, Lq, _ = q.shape
        _, Lk, _ = kv.shape

        qh = self.q_proj(q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, Lq, Hd)
        kh = self.k_proj(kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, Lk, Hd)
        vh = self.v_proj(kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, Lk, Hd)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                qh, kh, vh,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )  # (B, H, Lq, Hd)
        else:
            attn = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B,H,Lq,Lk)
            attn = attn.softmax(dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            out = attn @ vh  # (B,H,Lq,Hd)

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.attn_dim)   # (B,Lq,attn_dim)
        out = self.o_proj(out)                                              # (B,Lq,D)
        return out

class TokenLearnerLite(nn.Module):
    """
    从 (N,D) patch tokens 自适应聚合出 K 个 evidence tokens。
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
        xn = self.norm(x)
        logits = self.score(xn)                  # (N,K)
        attn = torch.softmax(logits.transpose(0, 1), dim=-1)  # (K,N)
        tok = attn @ x                           # (K,D)
        return tok

class SharedMemoryCVI(nn.Module):
    """
    共享 memory tokens 作为跨视角信息交换通道（Flamingo/Perceiver-resampler 的“latent通道”思想）。
    这里不替代 patch，只用于 evidence tokens 之间的信息交换。
    """
    def __init__(
        self,
        dim: int,
        num_memory: int = 32,
        num_layers: int = 2,
        attn_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
        ffn_mult: int = 2,  # 保留参数以兼容旧调用，但在 bottleneck 模式下不再使用
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mem = nn.Parameter(torch.randn(1, num_memory, dim) * 0.02)

        self.mem_ln1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.mem_attn = nn.ModuleList([_CrossAttnLite(dim, attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.mem_ln2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

        self.evi_ln1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.evi_attn = nn.ModuleList([_CrossAttnLite(dim, attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.evi_ln2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

        # [2025-12-14] 修改 SharedMemoryCVI：将大FFN(dim->ffn_mult*dim->dim)替换为bottleneck(dim->ffn_hidden->dim)开始
        # 说明：
        # - 不改变 token 数、不改变 cross-attn 逻辑，只减少 FFN 参数量，显著降低 AdamW optimizer state 显存开销。
        # - ffn_hidden 默认 1024；你可以通过环境变量 AR_CVI_FFN_HIDDEN 调整 (512/1024/2048)。
        import os
        ffn_hidden = int(os.environ.get("AR_CVI_FFN_HIDDEN", "1024"))

        def make_bottleneck_ffn():
            ffn = nn.Sequential(
                nn.Linear(dim, ffn_hidden),
                nn.GELU(),
                nn.Linear(ffn_hidden, dim),
            )
            # 让 FFN 初始更接近“恒等映射”，降低训练初期扰动/退化风险
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)
            return ffn

        self.mem_ffn = nn.ModuleList([make_bottleneck_ffn() for _ in range(num_layers)])
        self.evi_ffn = nn.ModuleList([make_bottleneck_ffn() for _ in range(num_layers)])
        # [2025-12-14] 修改 SharedMemoryCVI：将大FFN(dim->ffn_mult*dim->dim)替换为bottleneck(dim->ffn_hidden->dim)结束


    def forward(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        evidence: (V,K,D)
        return:
          evidence_new: (V,K,D)
          mem_new: (M,D)
        """
        # [2025-12-25] 新增：eval阶段 + 禁用fuser 时，直接跳过CVI交互，防止扰动
        if (not self.training) and os.getenv("AR_CVI_DISABLE_FUSER", "0") == "1":
            mem_out = self.mem[0] if self.mem.dim() == 3 else self.mem  # (M,D)
            return evidence, mem_out
        # [2025-12-25] 新增结束
        V, K, D = evidence.shape
        E = evidence.reshape(1, V * K, D)              # (1, VK, D)
        M = self.mem.expand(1, -1, -1)                 # (1, M, D)

        for i in range(self.num_layers):
            # (A) memory update: M attends to all evidence (跨视角汇聚)
            M = M + self.mem_attn[i](self.mem_ln1[i](M), self.evi_ln1[i](E))
            M = M + self.mem_ffn[i](self.mem_ln2[i](M))

            # (B) evidence update: evidence attends to memory (跨视角广播)
            E = E + self.evi_attn[i](self.evi_ln1[i](E), self.mem_ln1[i](M))
            E = E + self.evi_ffn[i](self.evi_ln2[i](E))

        return E.view(V, K, D), M.squeeze(0)

class MemoryConditionedAnchorRouter(nn.Module):
    """
    用 memory-conditioned 的方式给每个视角打分，选择“主视角锚点”。
    关键：打分网络最后一层初始化为 0，使训练初期不扰动；再叠加可学习的 view/orient bias 作“先验暖启动”。
    """
    def __init__(self, dim: int, num_view_types: int = 4, num_orient_types: int = 3):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim // 4, bias=False)
        self.k_proj = nn.Linear(dim, dim // 4, bias=False)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        # 训练初期 scores≈0，避免随机扰动导致锚点乱选
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # 可学习的 metadata bias（不是硬规则，参数可被学习推翻）
        self.view_bias = nn.Embedding(num_view_types + 1, 1)    # +PAD
        self.orient_bias = nn.Embedding(num_orient_types + 1, 1)

        # 暖启动：PA/AP/LAT 更可能是“主视角”，但这是可学习参数（非硬编码）
        with torch.no_grad():
            # 假设 VIEW: PA=0, AP=1, LAT=2, OTHER=3, PAD=4
            self.view_bias.weight.zero_()
            if num_view_types >= 3:
                self.view_bias.weight[0] = 0.7  # PA
                self.view_bias.weight[1] = 0.2  # AP
                self.view_bias.weight[2] = 0.1  # LAT
            # orient bias 默认 0
            self.orient_bias.weight.zero_()

    def forward(
        self,
        evidence: torch.Tensor,   # (V,K,D)
        mem: torch.Tensor,        # (M,D)
        view_ids: torch.Tensor | None = None,
        orient_ids: torch.Tensor | None = None,
        hard: bool = True,
        tau: float = 1.0,
    ):
        V, K, D = evidence.shape

        # [2025-12-19] 新增硬fallback
        # 如果没有 view_ids（mvwrap/test 很常见），必须与 baseline 一致：选第0张
        if view_ids is None or view_ids.numel() == 0:
            scores = evidence.new_zeros(V)
            probs  = torch.zeros(V, device=evidence.device)
            probs[0] = 1.0
            entropy = torch.tensor(0.0, device=evidence.device)
            return 0, probs, entropy, scores
        # [2025-12-19] 新增结束

        mem_q = mem.mean(dim=0)                       # (D,)

        q = self.q_proj(mem_q)                        # (d,)
        scores = []
        for v in range(V):
            Ev = evidence[v]                          # (K,D)
            k = self.k_proj(Ev)                       # (K,d)
            attn = (k @ q) / math.sqrt(k.size(-1))    # (K,)
            attn = attn.softmax(dim=0)
            gv = (attn.unsqueeze(-1) * Ev).sum(dim=0) # (D,)
            sv = self.mlp(gv).squeeze(-1)             # scalar

            if view_ids is not None and v < view_ids.numel():
                sv = sv + self.view_bias(view_ids[v]).squeeze(-1)
            if orient_ids is not None and v < orient_ids.numel():
                sv = sv + self.orient_bias(orient_ids[v]).squeeze(-1)

            scores.append(sv)

        scores = torch.stack(scores, dim=0)           # (V,)
        probs = torch.softmax(scores / max(tau, 1e-6), dim=0)

        if hard and V > 1:
            onehot = F.gumbel_softmax(scores, tau=max(tau, 1e-6), hard=True)
            anchor_idx = int(onehot.argmax().item())
        else:
            anchor_idx = int(probs.argmax().item())

        entropy = -(probs * (probs + 1e-8).log()).sum()
        return anchor_idx, probs, entropy, scores

class GatedAnchorFuser(nn.Module):
    """
    用 Cross-Attn 将“辅助证据 tokens + memory tokens”注入到“主视角全 patch tokens”里。
    门控 gate 初始≈0：训练初期等价于只用主视角（不退化保护）。
    """
    def __init__(self, dim: int, attn_dim: int = 1024, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = _CrossAttnLite(dim, attn_dim=attn_dim, num_heads=num_heads, dropout=dropout)

        # [2025-12-23] 修改了：fuser 门控/尺度由环境变量控制，并支持“强制重置恒等初始化”
        self._gate_init = float(os.getenv("AR_CVI_GATE_INIT", "-6.0"))   # 训练建议先用 -2 ~ -4，eval 可用 -6
        self._gate_max  = float(os.getenv("AR_CVI_GATE_MAX", "1.0"))     # 训练早期建议 0.1~0.3
        self.gate = nn.Parameter(torch.tensor(self._gate_init, dtype=torch.float32))
        # [2025-12-23] 修改结束

        # [2025-12-22] 新增，让 CrossAttn 输出在初始化时≈0（残差分支零初始化）
        # nn.init.zeros_(self.attn.o_proj.weight)
        # [2025-12-22] 新增结束

        # [2025-12-23] 修改了：补上一个显式 reset 接口（用于防止 checkpoint 覆盖初始化）
        self._did_force_reinit = False
        # [2025-12-23] 修改结束

        # [2025-12-25] 新增：初始化时把输出投影置零，使“未训练/强约束”更接近恒等
        with torch.no_grad():
            self.attn.o_proj.weight.zero_()

    # [2025-12-23] 新增了 reset_to_identity，确保 fuser 回到恒等附近
    @torch.no_grad()
    def reset_to_identity(self):
        # nn.init.zeros_(self.attn.o_proj.weight)
        self.attn.o_proj.weight.zero_()
        self.gate.fill_(self._gate_init)
    # [2025-12-23] 新增结束

    def forward(self, anchor_patches: torch.Tensor, aux_tokens: torch.Tensor) -> torch.Tensor:
        """
        anchor_patches: (Np,D)
        aux_tokens:     (S,D)
        return:         (Np,D)
        """
        # [2025-12-23] 修改了：支持运行时强制重置（用于你跑 Step B/B2 这类“只测结构不训练”的 sanity check）
        if (not self._did_force_reinit) and os.getenv("AR_CVI_FORCE_REINIT", "0") == "1":
            self.reset_to_identity()
            self._did_force_reinit = True
        # [2025-12-23] 修改结束
        
        q = self.ln(anchor_patches).unsqueeze(0)   # (1,Np,D)
        kv = aux_tokens.unsqueeze(0)               # (1,S,D)
        delta = self.attn(q, kv).squeeze(0)        # (Np,D)

        g = (torch.sigmoid(self.gate) * self._gate_max).to(delta.dtype)               # scalar ~0 at init
        if (not self.training) and g.item() < 1e-5:
            return anchor_patches
        return anchor_patches + g * delta

class ARCVIFusion(nn.Module):
    """
    输入：x (V,Np,D)
    输出：fused (Np,D) —— 只把“融合后的主视角全patch”喂给 LLM（长度≈baseline）
    """
    def __init__(
        self,
        dim: int,
        evidence_tokens: int = 16,
        memory_tokens: int = 32,
        cvi_layers: int = 2,
        attn_dim: int = 1024,
        num_heads: int = 8,
        aux_downsample_r: int = 24,
        dropout: float = 0.0,
        num_view_types: int = 4,
        num_orient_types: int = 3,
    ):
        super().__init__()
        self.aux_downsample_r = aux_downsample_r
        self.token_learner = TokenLearnerLite(dim, evidence_tokens)
        self.cvi = SharedMemoryCVI(dim, memory_tokens, cvi_layers, attn_dim, num_heads, dropout)
        self.router = MemoryConditionedAnchorRouter(dim, num_view_types=num_view_types, num_orient_types=num_orient_types)
        self.fuser = GatedAnchorFuser(dim, attn_dim, num_heads, dropout)

        # router 温度与是否 hard 选择（可通过环境变量调）
        self.router_tau = float(os.environ.get("AR_CVI_TAU", "1.0"))
        self.router_hard = (os.environ.get("AR_CVI_HARD", "1") == "1")

    def forward(self, x: torch.Tensor, view_ids=None, orient_ids=None):
        """
        x: (V,Np,D)
        view_ids/orient_ids: (V,)
        """
        V, Np, D = x.shape
        if V <= 1:
            fused = x[0]
            info = {"anchor_idx": 0, "entropy": fused.new_zeros(()), "probs": fused.new_ones((1,))}
            return fused, info

        evid = []
        for v in range(V):
            xv = x[v]                                     # (Np,D)
            xv_ds = _downsample_grid_tokens(xv, self.aux_downsample_r)
            ev = self.token_learner(xv_ds)                # (K,D)
            evid.append(ev)
        evid = torch.stack(evid, dim=0)                   # (V,K,D)

        evid, mem = self.cvi(evid)                        # cross-view interaction

        if view_ids is not None:
            view_ids = view_ids.to(x.device)
        if orient_ids is not None:
            orient_ids = orient_ids.to(x.device)

        # anchor_idx, probs, entropy, scores = self.router(
        #     evid, mem, view_ids=view_ids, orient_ids=orient_ids,
        #     hard=self.router_hard, tau=self.router_tau
        # )

        # =========================
        # [2025-12-19] 新增eval-only: 强制 AR-CVI anchor 选择与 baseline 同构
        # =========================
        # [2026-1-1] 修改主视角的选择逻辑，目前固定主视角
        match_baseline = os.environ.get("AR_CVI_MATCH_BASELINE", "0") == "1"
        match_baseline_train = os.environ.get("AR_CVI_MATCH_BASELINE_TRAIN", "0") == "1"
        # [2026-1-2] 新增打印节流器
        # === DEBUG gate: 控制打印频率 & 只在 rank0 打印 ===
        dbg = (os.environ.get("AR_CVI_DEBUG_ANCHOR", "0") == "1")
        if dbg and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            if not hasattr(self, "_ar_cvi_dbg_step"):
                self._ar_cvi_dbg_step = 0
            self._ar_cvi_dbg_step += 1
            log_every = int(os.environ.get("AR_CVI_LOG_EVERY", "1024"))
            do_log = (log_every <= 1) or (self._ar_cvi_dbg_step % log_every == 0)
        else:
            do_log = False
        # [2026-1-2] 新增结束
        if match_baseline and ((not self.training) or match_baseline_train):
        # [2026-1-1] 修改结束
            pick = os.environ.get("MV_BASELINE_PICK", "INDEX").upper()
            baseline_index = int(os.environ.get("MV_BASELINE_INDEX", "0"))

            # 默认：与 baseline 一样的 INDEX/FIRST 规则
            if pick in ("INDEX", "FIRST"):
                anchor_idx = max(0, min(baseline_index, V - 1))

            # [2026-1-1] 固定主视角选择策略
            elif pick in ("PA_AP_FIRST", "PAAPFIRST", "FRONTAL_FIRST", "FRONTALFIRST"):
                # 规则：PA/AP（正位）优先；都没有才选 LATERAL；再没有就 0
                anchor_idx = 0
                if view_ids is not None and view_ids.numel() == V:
                    pa  = (view_ids == 0).nonzero(as_tuple=False)  # PA
                    ap  = (view_ids == 1).nonzero(as_tuple=False)  # AP
                    lat = (view_ids == 2).nonzero(as_tuple=False)  # LATERAL

                    # 你要的是“AP/PA 优先”，这里保持确定性：PA 优先于 AP（如你想 AP 优先，把两行交换即可）
                    if pa.numel() > 0:
                        anchor_idx = int(pa[0].item())
                    elif ap.numel() > 0:
                        anchor_idx = int(ap[0].item())
                    elif lat.numel() > 0:
                        anchor_idx = int(lat[0].item())
                    else:
                        anchor_idx = 0
            # [2026-1-1] 固定主视角选择策略结束

            else:
                # 未知策略：保守退化到 0（保证可复现）
                anchor_idx = 0
            # [2026-1-2] 新增打印锚点选择日志
            if do_log:
                v_list = view_ids.detach().cpu().tolist() if view_ids is not None else None
                o_list = orient_ids.detach().cpu().tolist() if orient_ids is not None else None
                anchor_view = None
                if view_ids is not None and view_ids.numel() == V:
                    anchor_view = int(view_ids[anchor_idx].item())
                print(
                    f"[AR-CVI][branch=match_baseline][train={self.training}] "
                    f"pick={pick} V={V} view_ids={v_list} orient_ids={o_list} "
                    f"-> anchor_idx={anchor_idx} anchor_view={anchor_view}"
                )
            # [2026-1-2] 新增打印锚点选择日志结束
            # 构造与 router 返回一致的占位输出
            probs = x.new_zeros((V,))
            probs[anchor_idx] = 1.0
            entropy = x.new_zeros(())
            scores = x.new_zeros((V,))

        else:
            # [2025-12-19] 修改，只在训练时允许“hard”
            hard = self.router_hard and self.training
            tau  = self.router_tau if self.training else 1.0
            anchor_idx, probs, entropy, scores = self.router(
                evid, mem, view_ids=view_ids, orient_ids=orient_ids,
                hard=hard, tau=tau
            )
            # [2025-12-19] 修改结束
            # [2026-1-2] 新增打印锚点选择日志
            if do_log:
                v_list = view_ids.detach().cpu().tolist() if view_ids is not None else None
                o_list = orient_ids.detach().cpu().tolist() if orient_ids is not None else None
                p_list = probs.detach().float().cpu().tolist() if probs is not None else None
                s_list = scores.detach().float().cpu().tolist() if scores is not None else None
                print(
                    f"[AR-CVI][branch=router][train={self.training}] "
                    f"hard={hard} tau={tau} V={V} view_ids={v_list} orient_ids={o_list} "
                    f"-> anchor_idx={int(anchor_idx)} probs={p_list} scores={s_list}"
                )
            # [2026-1-2] 新增打印锚点选择日志结束
        # [2025-12-19] 新增eval-only结束: 强制 AR-CVI anchor 选择与 baseline 同构

        # ====== [2025-12-22] 新增：eval-only，禁用融合注入，只看“选主视角”本身影响 ======
        if (not self.training) and (os.environ.get("AR_CVI_DISABLE_FUSER", "0") == "1"):
            fused = x[anchor_idx]
            info = {"anchor_idx": anchor_idx, "entropy": entropy, "probs": probs, "scores": scores}
            return fused, info
        # ====== [2025-12-22] 新增结束 ========

        aux_list = [evid[i] for i in range(V) if i != anchor_idx]
        aux_tokens = torch.cat(aux_list + [mem], dim=0)   # ( (V-1)*K + M, D )

        fused = self.fuser(x[anchor_idx], aux_tokens)     # (Np,D)

        info = {"anchor_idx": anchor_idx, "entropy": entropy, "probs": probs, "scores": scores}
        return fused, info
# [2025-12-14] 新增 AR-CVI 模块（证据token + 共享memory交互 + 锚点路由 + 门控融合）结束


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
        self._last_router_entropy = None

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

        if isinstance(images, list) or (isinstance(images, torch.Tensor) and images.ndim == 5):
            # 统一成 list[Tensor(N_view, 3, H, W)]
            if isinstance(images, torch.Tensor):
                image_list = [img for img in images]   # (B, V, 3, H, W) -> list[B]
            else:
                image_list = images                    # list[B] of (V,3,H,W)

            model = self.get_model()
            mv_fusion = os.environ.get("MV_FUSION", getattr(model, "mv_fusion", "ar_cvi")).lower()

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
                image_features = self.encode_images(picked_images)  # (B,N_patch,D)

                # 统一成 list[Tensor(N_patch,D)]
                image_features = [feat for feat in image_features]

                # cache：study-level global（只保留一份，避免重复计算）
                self._last_study_image_global = torch.stack([feat.mean(dim=0) for feat in image_features], dim=0)
                self._last_router_entropy = None
                self._last_slot_div_loss = None
                self._last_slot_cov_loss = None

            # =========================================================
            # [MULTIVIEW PATH] ar_cvi / 其他多视角融合
            # =========================================================
            else:
                concat_images = torch.cat(image_list, dim=0)      # (sum_V,3,H,W)
                all_feats = self.encode_images(concat_images)     # (sum_V,N_patch,D)

                split_sizes = [img.shape[0] for img in image_list]
                per_sample = torch.split(all_feats, split_sizes, dim=0)  # tuple[(V_i,Np,D)]

                fused_features = []
                router_ent_losses = []

                view_ids_list = view_ids if view_ids is not None else [None] * len(per_sample)
                orient_ids_list = orient_ids if orient_ids is not None else [None] * len(per_sample)

                for b, x in enumerate(per_sample):
                    V, Np, D = x.shape
                    v_ids = view_ids_list[b][:V] if view_ids_list[b] is not None else None
                    o_ids = orient_ids_list[b][:V] if orient_ids_list[b] is not None else None

                    if mv_fusion == "ar_cvi":
                        ar_cvi = getattr(model, "ar_cvi", None)
                        if ar_cvi is None:
                            raise RuntimeError("ar_cvi is not found on core model; please init it in LlavaLlamaModel.__init__.")

                        fused_patches, info = ar_cvi(x, view_ids=v_ids, orient_ids=o_ids)

                        # 统计 anchor（确保 counter 初始化）
                        if os.environ.get("AR_CVI_LOG_ANCHOR", "0") == "1" and info is not None and "anchor_idx" in info:
                            from collections import Counter
                            import torch.distributed as dist

                            if not hasattr(self, "_ar_cvi_anchor_counter"):
                                self._ar_cvi_anchor_counter = Counter()
                                self._ar_cvi_anchor_total = 0

                            anchor_idx = info["anchor_idx"]
                            if torch.is_tensor(anchor_idx):
                                anchor_idx = anchor_idx.detach().flatten()
                                a = int(anchor_idx.item()) if anchor_idx.numel() == 1 else int(torch.mode(anchor_idx).values.item())
                            elif isinstance(anchor_idx, (list, tuple)):
                                a = int(max(set(anchor_idx), key=anchor_idx.count))
                            else:
                                a = int(anchor_idx)

                            key = int(v_ids[a].item()) if v_ids is not None else a
                            self._ar_cvi_anchor_counter[key] += 1
                            self._ar_cvi_anchor_total += 1

                            log_every = int(os.environ.get("AR_CVI_LOG_EVERY", "512"))
                            is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
                            if is_rank0 and (self._ar_cvi_anchor_total % log_every == 0):
                                print(f"[AR-CVI] anchor_counter (n={self._ar_cvi_anchor_total}): {dict(self._ar_cvi_anchor_counter)}")

                        fused_features.append(fused_patches)
                        if info is not None and "entropy" in info:
                            router_ent_losses.append(info["entropy"])
                    else:
                        raise RuntimeError(f"Unknown MV_FUSION={mv_fusion}")

                image_features = fused_features
                self._last_study_image_global = torch.stack([feat.mean(dim=0) for feat in image_features], dim=0)
                self._last_router_entropy = torch.stack(router_ent_losses).mean() if len(router_ent_losses) > 0 else None
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