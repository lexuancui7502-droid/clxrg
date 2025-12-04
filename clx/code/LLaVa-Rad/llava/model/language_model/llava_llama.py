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

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from llava.constants import IGNORE_INDEX

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, SimpleViewAttention


class LlavaConfig(LlamaConfig):                 # 继承LLaMA配置，定义LLaVA模型类型
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):                  # 组合视觉和语言能力，继承自LLaMA模型。LlamaModel: 提供文本理解能力；LlavaMetaModel: 提供多模态融合能力
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        # [2025-11-19] 这里的 dim 要和 encode_images 输出的 D 一致，等于 LLM 的 hidden_size
        self.view_attn = SimpleViewAttention(dim=config.hidden_size)

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

        # 每次 forward 里由 prepare_inputs_labels_for_multimodal 写入 (B, D) 的视觉表征
        self._last_study_image_global = None

        # Initialize weights...
        self.post_init()

    def get_model(self):
        return self.model

    def forward(                # 定义前向传播逻辑，处理多模态输入（文本 + 图像）并计算损失或生成预测。
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:                  # 定义模型的前向传播逻辑，处理多模态输入（文本 + 图像）并生成预测结果
        # 参数默认值处理​，如果参数未显式传入，则使用模型配置中的默认值。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions       # 设置是否输出注意力权重
        output_hidden_states = (                            # 设置是否输出隐藏状态
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states              # 如果未指定，则使用配置中的默认值
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict                           # 设置是否返回字典形式的输出

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(  # 将文本和图像输入融合为模型可处理的格式
            input_ids, attention_mask, past_key_values, labels, images
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)     调用模型主体计算隐藏状态、注意力权重等
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # 提取隐藏状态并通过 lm_head计算词汇表logits分数
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:              # 计算交叉熵损失，对齐预测和标签（标准语言模型训练方式）
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()         # 预测n的logits（去掉最后一个token）
            shift_labels = labels[..., 1:].contiguous()             # 真实标签（去掉第一个token）
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)


        # === [2025-12-2] 修改：检查级对齐（study-level image ↔ report text） ===
        # 条件：有标签 + 有图像 + study_contrast_weight>0 才启用对比损失，目前是关闭的
        if (
            labels is not None
            and getattr(self, "_last_study_image_global", None) is not None
            and getattr(self, "study_contrast_weight", 0.0) > 0
        ):
            img_global = self._last_study_image_global.to(hidden_states.device)  # (B, D)
            B, T, D = hidden_states.shape

            # 1) 用 labels!=IGNORE_INDEX 作为“报告 token”掩码
            label_mask = labels != IGNORE_INDEX   # (B, T)
            if attention_mask is not None:
                label_mask = label_mask & attention_mask.bool()

            # 2) 对每个样本，把对应 token 的 hidden_states 做 mean-pool，得到文本表征
            text_pooled = []
            for b in range(B):
                h_b = hidden_states[b]   # (T, D)
                m_b = label_mask[b]      # (T,)
                if m_b.any():
                    text_pooled.append(h_b[m_b].mean(dim=0))
                else:
                    text_pooled.append(h_b.mean(dim=0))
            text_pooled = torch.stack(text_pooled, dim=0)   # (B, D)

            # 3) CLIP 风格 InfoNCE 对比：image ↔ text
            img_norm = F.normalize(img_global, dim=-1)      # (B, D)
            txt_norm = F.normalize(text_pooled, dim=-1)     # (B, D)

            logits_per_img = img_norm @ txt_norm.t() / self.study_contrast_tau  # (B, B)
            targets = torch.arange(B, device=logits_per_img.device)

            loss_i2t = F.cross_entropy(logits_per_img, targets)
            loss_t2i = F.cross_entropy(logits_per_img.t(), targets)
            contrast_loss = 0.5 * (loss_i2t + loss_t2i)

            if loss is None:
                loss = 0.0
            loss = loss + self.study_contrast_weight * contrast_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(          # 返回结构化输出（损失、logits、KV缓存等）
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

        model_inputs.update(                # 更新输入字典，包括KV缓存、图像数据和其他生成参数
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
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
