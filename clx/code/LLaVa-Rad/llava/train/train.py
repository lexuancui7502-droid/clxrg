# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

# 定义并解析命令行参数（模型参数、数据参数、训练参数）。
# 根据参数构建/加载模型（支持普通 Llama、LLaVA 的多模态变体，以及 MPT 变体），并根据量化/bits/LoRA 等配置做准备（例如 4-bit/8-bit、prepare_for_kbit_training）。
# 初始化 tokenizer、视觉 tokenizer、可能的视觉塔（vision_tower）以及 mm projector 等多模态相关设置（若指定）。
# 构造训练数据集和数据 collator（LazySupervisedDataset 与 DataCollatorForSupervisedDataset），支持 multimodal（有/无 image）数据拼装。
# 创建定制化的 Trainer（LLaVATrainer），并控制 trainer.train()、检查点恢复、保存模型（对 LoRA 有特殊保存流程）。

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, open_image_with_retry
from llava.utils import data_loaders

from PIL import Image, ImageFile
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True      # 防止因为几张损坏的图像而导致整个程序崩溃


local_rank = None


# 仅在主进程（local_rank == 0）打印，用于分布式时避免重复输出
def rank0_print(*args):
    if local_rank == 0:         # 用于分布式训练判断当前进程是否是 rank0（主进程），是的话，则进行打印
        print(*args)


# 一组模型相关参数，包括 模型路径、vision_tower、projector 类型、是否冻结 backbone、视觉 token 的一些开关等
@dataclass
class ModelArguments:           # 控制模型选择与结构级别的多模态选项
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")      # 基础模型路径或名称
    version: Optional[str] = field(default="v0")                                # LLaVA 版本号
    freeze_backbone: bool = field(default=False)                                # 是否冻结预训练模型的参数，默认为False
    tune_mm_mlp_adapter: bool = field(default=False)                            # 是否只微调多模态 MLP 适配器
    vision_tower: Optional[str] = field(default=None)                           # 视觉塔模型名称或路径
    vision_tower_config: Optional[str] = field(default=None)                    # 视觉塔配置文件路径
    vision_tower_checkpoint: Optional[str] = field(default=None)                # 视觉塔检查点路径
    mm_vision_select_layer: Optional[int] = field(default=-1)                   # 多模态视觉特征选择层，默认 -1（最后一层）
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)                # 预训练的多模态 MLP 适配器路径，用于初始化，默认为None
    mm_projector_type: Optional[str] = field(default='linear')                  # 多模态投影器类型，默认 'linear'（线性投影）
    mm_use_im_start_end: bool = field(default=False)                            # 多模态是否在图像标记中使用开始和结束标记，默认为 False
    mm_use_im_patch_token: bool = field(default=True)                           # 多模态是否使用图像补丁标记，默认为 True
    mm_vision_select_feature: Optional[str] = field(default="patch")            # 多模态视觉特征选择方式，默认 "patch"（补丁特征）


# 数据/加载相关参数，包括 data_path、loader、是否 lazy_preprocess、是否 multimodal、image_folder、图像相关配置等
@dataclass
class DataArguments:                        # 控制数据集加载与预处理选项
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})     # 训练数据路径，默认为 None
    loader: str = "default"                                                     # 数据加载器类型，默认使用 "default" 加载器
    lazy_preprocess: bool = False                                               # 是否启用延迟预处理，默认为 False
    is_multimodal: bool = False                                                 # 是否为多模态数据集，默认为 False             
    image_folder: Optional[str] = field(default=None)                           # 图像文件夹路径，默认为 None
    image_aspect_ratio: str = 'square'                                          # 图像宽高比，默认 'square'（正方形）   
    image_grid_pinpoints: Optional[str] = field(default=None)                   # 图像网格关键点，默认为 None


# 扩展 HF 的训练参数，新增了 bits、lora_*、group_by_modality_length 等训练控制参数
@dataclass
class TrainingArguments(transformers.TrainingArguments):            # 扩展 HF 的 TrainingArguments，添加自定义训练参数
    cache_dir: Optional[str] = field(default=None)                  # 缓存目录，默认为 None
    optim: str = field(default="adamw_torch")                       # 优化器类型，默认 "adamw_torch"（AdamW 优化器）
    remove_unused_columns: bool = field(default=False)              # 是否移除未使用的数据列，默认为 False
    freeze_mm_mlp_adapter: bool = field(default=False)              # 是否冻结多模态 MLP 适配器，默认为 False
    mpt_attn_impl: Optional[str] = field(default="triton")          # MPT 注意力实现方式，默认 "triton"
    model_max_length: int = field(          # 模型最大序列长度，默认 512。序列会被右填充（可能被截断）
        default=512,            
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(             # 是否使用双重量化，默认 True。通过双重量化压缩量化统计信息
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(                # 量化数据类型，默认 "nf4"。可选 "fp4" 或 "nf4"
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(                      # 量化位数，默认 16。可选 4、8、16
        default=16,
        metadata={"help": "How many bits to use."}
    )
    # LoRA 相关参数
    lora_enable: bool = False               # 是否启用 LoRA（Low-Rank Adaptation），默认为 False
    lora_r: int = 64                        # LoRA 矩阵秩，默认 64
    lora_alpha: int = 16                    # LoRA alpha 参数（缩放系数），默认 16
    lora_dropout: float = 0.05              # LoRA 的 dropout 率，默认 0.05
    lora_weight_path: str = ""              # 预训练 LoRA 权重路径，默认为空字符串
    lora_bias: str = "none"                 # LoRA 偏置处理方式，默认 "none"（可选 "none"、"all"、"lora_only"）
    group_by_modality_length: bool = field(default=False)           # 是否按模态长度分组，默认为 False

# DeepSpeed ZeRO 是微软DeepSpeed框架中的一种内存优化技术，ZeRO3是其中的第三阶段，提供了最极致的内存优化。
# 辅助函数，和 DeepSpeed ZeRO3 相关的 wrapper（用于在 ZeRO 环境下安全访问参数 / state）
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):         # 判断参数是否属于 DeepSpeed ZeRO3 管理的参数，是否有 ds_id 属性（有这个属性说明参数正在使用 ZeRO 优化）
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:            # 检查参数状态是否为 NOT_AVAILABLE（表示参数当前不可用，可能在其他设备上）
            if not ignore_status:                   # 如果 ignore_status 为 False 且参数状态不是 NOT_AVAILABLE，记录警告日志
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):          # 使用 DeepSpeed 的 GatheredParameters 上下文管理器
            param = param.data.detach().cpu().clone()           # 在参数收集的上下文中，获取参数的张量数据
    else:
        param = param.detach().cpu().clone()        # 如果参数不属于 ZeRO3 管理，直接执行相同的操作：分离、移动到 CPU、克隆
    return param


# 根据不同的bias处理策略，选择性提取和处理 LoRA 相关的参数
# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


# 提取非LoRA的PEFT参数
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:                   # 如果 require_grad_only 为 True，只保留需要梯度的参数
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


# 专门处理多模态适配器的参数
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


# 分析模型结构，找出所有线性层，为LoRA应用做准备
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():          # 遍历模型的所有模块
        if isinstance(module, cls):                     # 如果是线性层，提取其名称的最后一部分
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # 如果包含语言模型头（lm_head），将其移除
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# 安全地保存训练器中的模型到磁盘。支持多模态适配器微调、DeepSpeed训练等特殊场景
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    # 多模态适配器微调保存。当只微调多模态MLP适配器时，只保存适配器相关参数，而不是整个模型
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        # 使用之前介绍的函数获取多模态适配器参数
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        # 保存多模态适配器参数到指定目录
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# 文本输入 → Tokenizer → token IDs → input_embeddings → 模型编码 → 隐藏状态 → output_embeddings → 词汇概率
# 智能调整tokenizer和模型embedding层的大小，以适配新的特殊token(除了常规文本词汇外，具有特殊功能的token，如图像token，视频token等)
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)          # 向tokenizer添加新的特殊token（tokenizer是将文本转换为模型可理解数字的工具）
    model.resize_token_embeddings(len(tokenizer))                               # 调整模型的token embedding层大小以匹配新的tokenizer大小

    if num_new_tokens > 0:                               # 只有当确实添加了新token时才进行初始化
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# 批量分词函数，统一处理多个文本字符串
def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [                      # 对每个文本字符串应用tokenizer
        tokenizer(
            text,
            return_tensors="pt",            # 返回PyTorch tensor格式
            padding="longest",              # 按最长序列进行填充，确保批次内所有序列长度一致
            max_length=tokenizer.model_max_length,          # 最大长度限制，防止过长序列
            truncation=True,                # 超过最大长度时进行截断
        ) for text in strings
    ]
    input_ids = labels = [                  # 从每个分词结果中提取第一个序列的input_ids
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()         # 创建布尔掩码，非填充token为True。统计非填充token的数量，得到实际序列长度
        for tokenized in tokenized_list
    ]
    return dict(                            # 返回结构化数据，包含分词结果和长度信息
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# 在对话数据中掩码特定的loss计算区域
def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]             # 初始化当前位置，跳过系统提示部分
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX         # 将当前位置之前的所有token设置为IGNORE_INDEX，在loss计算时忽略
    for tokenized_len, speaker in zip(tokenized_lens, speakers):            # 遍历对话轮次，根据说话者类型进行掩码
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


# 为对话数据添加说话者标识和开始/结束信号
def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


# 预处理多模态数据，处理图像token的插入和格式化
def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal                 # 从数据参数获取是否是多模态数据
    if not is_multimodal:
        return sources

    for source in sources:                  # 外层循环：遍历每个对话样本
        for sentence in source:             # 内层循环：遍历对话中的每个句子
            if DEFAULT_IMAGE_TOKEN in sentence['value']:                # 检查当前句子中是否包含默认图像token，只有在包含图像token的句子才需要进行特殊处理
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()          # 将图像token从原位置移除，使用strip()去除首尾空白字符
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']                      # 将图像token移动到句子开头，并添加换行符
                sentence['value'] = sentence['value'].strip()                                           # 去除因添加换行可能产生的多余空白
                if "mmtag" in conversation_lib.default_conversation.version:                    # 如果对话版本包含 "mmtag"，则使用特定的图像标记格式
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN                         # 准备替换图像token
            if data_args.mm_use_im_start_end:                           # 如果配置要求使用图像开始和结束标记
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)           # 将原始图像token替换为处理后的版本

    return sources


# LLaMA-2格式预处理
def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# V1格式预处理
def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# MPT格式预处理
def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# 简单格式预处理。处理最简单的对话格式，适用于基础的多模态任务。输入要求：严格的2轮对话（人类提问 + AI回答）
def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


# 统一预处理分发器。根据对话风格自动选择对应的预处理函数，提供统一的预处理接口，支持多种对话格式
def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


# 数据集类，支持“懒加载”与预处理，能把每条样本转为 input_ids、labels，并在需要时读取/处理图像（open_image_with_retry），为多模态训练准备数据结构。
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = data_loaders[data_args.loader](data_path)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = open_image_with_retry(os.path.join(image_folder, image_file))
            if image is None:
                logging.error("Use an empty image.")
                image = Image.new('RGB', (224, 224), tuple(int(x*255) for x in processor.image_mean))
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


# 用来把多条样本拼成 batch（pad input_ids、pad labels 用 IGNORE_INDEX、构建 attention_mask、处理 images 的 stack/非一致 shape 情况）
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


# 返回 dict(train_dataset, eval_dataset, data_collator)，被 train() 用来传递给 Trainer
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


# 主函数，解析 args、加载模型（包括 bits/quant 配置）、设置 LoRA / prepare_model_for_kbit_training、创建数据模块、构建 LLaVATrainer、运行训练、最后的保存逻辑
def train():
    global local_rank

    # 参数解析和基础设置
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 量化配置
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # 模型加载
    if model_args.vision_tower is not None:         # 如果有视觉塔，加载LLaVA模型
        if 'mpt' in model_args.model_name_or_path:              # 如果是MPT模型
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:                   # 否则，加载LLaVA-Llama模型
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:           # 如果没有视觉塔，加载标准的LLaMA模型
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    # 可选择冻结基础语言模型，只训练适配器
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 梯度检查点：用时间换空间，减少显存占用
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # LoRA适配器：添加低秩适配器进行参数高效微调
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # Tokenizer加载：根据模型类型加载对应的tokenizer
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    # Pad Token处理：根据不同版本设置pad token
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 视觉模块初始化：如果有视觉塔，初始化视觉模块并设置相关配置
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        # 多模态适配器配置。灵活控制哪些部分需要训练，可选择只训练多模态MLP适配器
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        # 视觉Token配置：设置图像相关的特殊token
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 精度一致性：确保所有模块在正确的精度下运行
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # 训练执行
    data_module = make_supervised_data_module(tokenizer=tokenizer,                      # 数据准备：创建数据加载模块
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,                 # 训练器初始化：创建LLaVA专用的训练器
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):           # 检查是否有已有的检查点，支持断点续训
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()            # 状态保存：保存训练状态

    # 模型保存
    model.config.use_cache = True

    if training_args.lora_enable:           # 如果使用LoRA，保存LoRA适配器和非LoRA参数
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:             # 仅主进程保存模型，避免重复写入
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
