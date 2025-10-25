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


from typing import List, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
import math

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .mpt.modeling_mpt import MPTConfig, MPTForCausalLM, MPTModel
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


# 定义LLaVA-MPT模型的配置类
class LlavaMPTConfig(MPTConfig):
    model_type = "llava_mpt"


# 实现多模态模型的核心结构，融合图像和文本特征
class LlavaMPTModel(LlavaMetaModel, MPTModel):
    config_class = LlavaMPTConfig

    def __init__(self, config: MPTConfig):
        config.hidden_size = config.d_model                 # 对齐隐藏层维度
        super(LlavaMPTModel, self).__init__(config)
    
    def embed_tokens(self, x):              # 直接调用MPT的token嵌入层 wte
        return self.wte(x)


class LlavaMPTForCausalLM(MPTForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMPTConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(MPTForCausalLM, self).__init__(config)                    # 跳过直接父类LlavaMetaForCausalLM的初始化，避免多模态相关的初始化逻辑冲突
        # 强制要求输入输出权重绑定
        if not config.tie_word_embeddings:
            raise ValueError('MPTForCausalLM only supports tied word embeddings')
        self.transformer = LlavaMPTModel(config)                        # 创建多模态版本的MPT模型
        # Logit缩放控制
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):         # 为LlavaMPTModel模块启用或禁用梯度检查点
        if isinstance(module, LlavaMPTModel):               # 只对LlavaMPTModel类型模块设置
            module.gradient_checkpointing = value

    def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, images=None):
        # 处理输入参数默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 多模态输入融合（文本+图像）
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        # 调用Transformer计算隐藏状态
        outputs = self.transformer(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache)
        # FIXME: this is a hack to fix the multiple gpu inference issue in https://github.com/haotian-liu/LLaVA/issues/338      计算logits（修复多GPU推理问题）
        logits = F.linear(outputs.last_hidden_state.to(self.transformer.wte.weight.device), self.transformer.wte.weight)
        # 应用logit缩放
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
            logits *= self.logit_scale
         # 计算损失（语言模型任务）
        loss = None
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)              # 标签左移
            labels[:, -1] = -100                                # 忽略最后一个token
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)

    # 生成任务输入预处理
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            raise NotImplementedError('inputs_embeds is not implemented for MPT yet')
        # 检查注意力掩码（MPT不支持右填充）
        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError('MPT does not support generation with right padding.')
        # 处理序列ID（用于训练时的注意力掩码）
        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None
        # 使用KV缓存时，只保留最后一个token
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        # 前缀语言模型（Prefix LM）的特殊处理
        if self.transformer.prefix_lm:
            prefix_mask = torch.ones_like(attention_mask)
            if kwargs.get('use_cache') == False:
                raise NotImplementedError('MPT with prefix_lm=True does not support use_cache=False.')
        else:
            prefix_mask = None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'prefix_mask': prefix_mask, 'sequence_id': sequence_id, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache', True), "images": kwargs.get("images", None)}


# 将自定义类注册到Hugging Face的自动加载系统
AutoConfig.register("llava_mpt", LlavaMPTConfig)
AutoModelForCausalLM.register(LlavaMPTConfig, LlavaMPTForCausalLM)
