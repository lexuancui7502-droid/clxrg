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

from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from llava.constants import IGNORE_INDEX

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

# [2025-12-7] 修改
# from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, SimpleViewAttention
from ..llava_arch import (
    LlavaMetaModel,
    LlavaMetaForCausalLM,
    SimpleViewAttention,
    # ARCVIFusion,          # [2025-12-14] 新增
    MVGridMambaFusion,   # [2026-1-7] 新增
)


class LlavaConfig(LlamaConfig):                 # 继承LLaMA配置，定义LLaVA模型类型
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):                  # 组合视觉和语言能力，继承自LLaMA模型。LlamaModel: 提供文本理解能力；LlavaMetaModel: 提供多模态融合能力
    config_class = LlavaConfig

    # [2025-12-14] 修改 LlavaLlamaModel.__init__：移除 slot+gate，改为初始化 AR-CVI 开始
    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        dim = config.hidden_size

        # 可保留（如果你还想做 view-level 的轻量加权/对齐）
        # self.view_attn = SimpleViewAttention(dim=dim)

        # [2025-12-12] 疾病预测头保持不变
        self.num_diseases = 14
        # self.chexpert_lambda = float(os.environ.get("CHEXPERT_LAMBDA", "0.1"))

        # === AR-CVI 配置（都有默认值，保证可跑）===
        # self.mv_fusion = os.environ.get("MV_FUSION", "ar_cvi")
        # [2026-1-19] 新增：定义视觉和文本疾病预测头，用于疾病级对齐；统一align_lambda从config获取，避免环境变量重复
        self.visual_disease_head = nn.Linear(config.hidden_size, self.num_diseases)  # 视觉侧疾病预测头（study-level全局特征 -> 疾病logits）
        self.text_disease_head = nn.Linear(config.hidden_size, self.num_diseases)    # 文本侧疾病预测头（报告embedding -> 疾病logits）
        self.align_lambda = getattr(config, "align_lambda", 
                                    float(os.environ.get("CHEXPERT_LAMBDA", "0.0")))
        # [2026-1-19] 新增结束

        # [2026-1-7] 新增初始化mv_grid_mamba
        self.mv_fusion = getattr(config, "mv_fusion", os.environ.get("MV_FUSION", "baseline")).lower()
        self.register_buffer("_mv_global_step", torch.tensor(0, dtype=torch.long))

        if self.mv_fusion == "mamba_grid":
            self.mv_grid_mamba = MVGridMambaFusion(dim=dim)
        if self.mv_fusion in ["view_attn", "slot"]:   # 目前你没有这些分支，仅保留“未来可能性”
            self.view_attn = SimpleViewAttention(dim=dim)
        # [2026-1-7] 新增结束

# 语言生成模型，负责端到端的训练和推理。继承自 LlamaForCausalLM（纯文本生成模型）和 LlavaMetaForCausalLM（多模态支持）
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        # 保持与上游一致的初始化顺序
        super(LlamaForCausalLM, self).__init__(config)          # 跳过一级初始化
        self.model = LlavaLlamaModel(config)                    # 使用LLaVA模型替换原有的LLaMA模型

         # 重定义语言模型头，以适应LLaVA的需求

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)         # 语言模型头：将隐藏状态映射到词汇表大小的线性层

        # [2025-11-27] 用于检查级对齐的一些超参数 & 缓存
        # 对比损失的温度和权重，先给一个默认值，之后你可以写到 config 里去
        self.study_contrast_tau = getattr(config, "study_contrast_tau", 0.07)
        # [2025-12-2] 修改，训练“轻量版 view_attn”阶段默认不启用对比损失 => 默认 0.0
        self.study_contrast_weight = getattr(config, "study_contrast_weight", 0.0)

        #[2025-12-8] 每次 forward 里由 prepare_inputs_labels_for_multimodal 写入 (B, D) 的视觉表征
        self._last_study_image_global = None
        # 多-slot 正则项缓存（diversity / view-coverage）
        self._last_slot_div_loss = None
        self._last_slot_cov_loss = None

        # [2025-12-7] 新增

        # Initialize weights...
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        view_ids: Optional[List[torch.LongTensor]] = None,
        orient_ids: Optional[List[torch.LongTensor]] = None,
        return_dict: Optional[bool] = None,
        chexpert_labels: Optional[torch.FloatTensor] = None,  # [2026-01-18] 新增参数
        findings_embeds: Optional[torch.FloatTensor] = None,  # findings文本预计算embedding (B, D)
        impression_embeds: Optional[torch.FloatTensor] = None,  # impression文本预计算embedding (B, D)
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. 准备多模态输入 (在这里会计算并缓存 _last_study_image_global 等)
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                view_ids=view_ids,
                orient_ids=orient_ids,
                findings_embeds=findings_embeds,
            )

        # 2. 模型前向传播 (强制输出 hidden_states 用于文本对齐)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, 
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # [定位到 llava_llama.py 的 forward 函数内部，覆盖 if labels is not None: 开始的区域]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # =========================================================================
            # [2026-01-20] 修正：三层对齐 (Study-Level, Disease-Level, View-Level)
            # =========================================================================
            align_lambda = getattr(self.model, "align_lambda", 0.0)

            # 只有在训练且 align_lambda > 0 时才计算对齐损失
            if align_lambda > 0:
                align_loss = torch.tensor(0.0, device=loss.device)
                tau = getattr(self, "study_contrast_tau", 0.07)
                epsilon = 1e-6  # 防止 normalize 除以 0

                # --- 获取全局特征 ---
                # A. 视觉全局特征 (Study-Level)
                visual_global = getattr(self, "_last_study_image_global", None)

                # B. 文本全局特征 (Text-Level)
                if attention_mask is not None:
                    last_idx = attention_mask.sum(1) - 1
                    last_idx = last_idx.clamp(min=0, max=hidden_states.size(1) - 1)
                    text_global = hidden_states[
                        torch.arange(hidden_states.size(0), device=hidden_states.device), last_idx]
                else:
                    text_global = hidden_states[:, -1, :]

                # -----------------------------------------------------------
                # Layer 1: 疾病级对齐 (Disease-Level Alignment)
                # -----------------------------------------------------------
                # 确保 labels 存在且 visual_global 有效
                if chexpert_labels is not None and visual_global is not None:
                    chexpert_labels = chexpert_labels.to(hidden_states.device).float()
                    
                    # 只有当 label 维度和 head 输出维度匹配时才计算
                    if chexpert_labels.shape[-1] == self.model.num_diseases:
                        # 1.1 视觉 -> 疾病
                        vis_logits = self.model.visual_disease_head(visual_global)
                        align_loss += F.binary_cross_entropy_with_logits(vis_logits, chexpert_labels)

                        # 1.2 文本 -> 疾病
                        txt_logits = self.model.text_disease_head(text_global)
                        align_loss += F.binary_cross_entropy_with_logits(txt_logits, chexpert_labels)

                        # 1.3 视觉 <-> 文本 一致性 (MSE on Probabilities)
                        align_loss += F.mse_loss(torch.sigmoid(vis_logits), torch.sigmoid(txt_logits))

                # -----------------------------------------------------------
                # Layer 2: 检查级对齐 (Study-Level Alignment)
                # -----------------------------------------------------------
                # 确保 embed 存在且非全零 (防止 padding 样本导致 NaN)
                if findings_embeds is not None and impression_embeds is not None and visual_global is not None:
                    # 使用 epsilon 防止除零错误
                    visual_global_norm = F.normalize(visual_global, p=2, dim=-1, eps=epsilon)
                    findings_norm = F.normalize(findings_embeds.to(visual_global.device), p=2, dim=-1, eps=epsilon)
                    impression_norm = F.normalize(impression_embeds.to(visual_global.device), p=2, dim=-1, eps=epsilon)

                    # 计算 InfoNCE (Batch 内对比)
                    sim_findings = torch.matmul(visual_global_norm, findings_norm.t()) / tau
                    sim_impression = torch.matmul(visual_global_norm, impression_norm.t()) / tau

                    # 标签：对角线为正样本
                    target_idx = torch.arange(visual_global.size(0), device=visual_global.device)

                    # 检查是否有全零向量导致相似度计算出问题，如果有则 masking 掉 (这里简化处理，假设数据预处理已过滤坏样本)
                    study_loss = (F.cross_entropy(sim_findings, target_idx) + F.cross_entropy(sim_impression,
                                                                                              target_idx)) / 2
                    align_loss += study_loss

                # -----------------------------------------------------------
                # Layer 3: 视图级对齐 (View-Level Alignment)
                # -----------------------------------------------------------
                raw_view_feats_list = getattr(self, "_last_batch_view_feats", None)

                if raw_view_feats_list is not None and len(raw_view_feats_list) > 0:
                    # --- A. 准备数据 ---
                    all_views_flat = torch.cat(raw_view_feats_list, dim=0)  # (Total_Views, D)
                    all_views_norm = F.normalize(all_views_flat, p=2, dim=-1, eps=epsilon)

                    # 构建归属标签
                    view_study_ids = []
                    for study_idx, v_feat in enumerate(raw_view_feats_list):
                        view_study_ids.extend([study_idx] * v_feat.size(0))
                    view_study_ids = torch.tensor(view_study_ids, device=all_views_flat.device)

                    # --- B. 视图间一致性 (View-View Contrastive) ---
                    # 计算相似度矩阵 (Total_Views, Total_Views)
                    sim_matrix = torch.matmul(all_views_norm, all_views_norm.t()) / tau

                    # 修正：InfoNCE 必须 mask 掉对角线 (自己与自己)，否则模型会“走捷径”
                    mask_diag = torch.eye(len(view_study_ids), device=all_views_flat.device).bool()
                    # 将对角线置为负无穷，使其在 softmax 中概率为 0
                    sim_matrix.masked_fill_(mask_diag, -1e9)

                    # 正样本 Mask: 同一 Study 但不是自己
                    labels_equal = view_study_ids.unsqueeze(0) == view_study_ids.unsqueeze(1)
                    pos_mask = labels_equal & (~mask_diag)

                    if pos_mask.sum() > 0:
                        # Log-Sum-Exp Trick for numerical stability
                        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
                        sim_matrix_sub = sim_matrix - sim_max.detach()
                        exp_sim = torch.exp(sim_matrix_sub)

                        # 分母：所有样本的 exp 和 (因为对角线已经是 -inf -> exp=0，所以这里直接 sum 即可)
                        denominator = exp_sim.sum(dim=1)
                        
                        # 分子：仅正样本的 exp 和
                        numerator = (exp_sim * pos_mask.float()).sum(dim=1)

                        # 避免 log(0)
                        valid_rows = (numerator > 0) & (denominator > 0)
                        if valid_rows.sum() > 0:
                            log_prob = torch.log(numerator[valid_rows] / (denominator[valid_rows] + 1e-8))
                            align_loss += -log_prob.mean()

                    # --- C. 视图-文本对齐 (View-Text Alignment) ---
                    if findings_embeds is not None:
                        findings_norm = F.normalize(findings_embeds.to(all_views_norm.device), p=2, dim=-1, eps=epsilon)
                        
                        # (Total_Views, Batch_Size)
                        vt_sim_matrix = torch.matmul(all_views_norm, findings_norm.t()) / tau
                        
                        # Target: 每个视图属于哪个 Study Index (Batch Index)
                        vt_loss = F.cross_entropy(vt_sim_matrix, view_study_ids)
                        align_loss += vt_loss

                # 最后加上权重
                loss += align_lambda * align_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(                  # 在生成任务（如自回归生成）中动态准备输入数据。
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:                     # 使用KV缓存时，只需传递最新生成的token
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step 优先使用 inputs_embeds（首次生成），否则用 input_ids
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "view_ids": kwargs.get("view_ids", None),
                "orient_ids": kwargs.get("orient_ids", None),
            }
        )

        return model_inputs


# ========================= 注册到 Transformers Auto* =========================
# [2025-10-17 修改] 适配 transformers==4.31.0：该版本的 AutoConfig.register 不支持 exist_ok 形参。
#                  这里用 try/except 做向后兼容；若已注册过或老版本无该参数，则安全跳过。
# 注册​​指将自定义的​​模型类​​或​​配置类​​添加到全局的自动映射系统中

try:
    # 新版 transformers（4.33+）支持 exist_ok
    AutoConfig.register("llava", LlavaConfig, exist_ok=True)
except TypeError:
    # 老版（如 4.31.0）没有 exist_ok 参数
    try:
        AutoConfig.register("llava", LlavaConfig)
    except Exception:
        # 已注册等非关键异常——忽略
        pass

# [2025-10-17 修改] 为避免重复注册导致异常，这里也包一层保护。
try:
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
except Exception:
    # 已注册过等情况——忽略
    pass