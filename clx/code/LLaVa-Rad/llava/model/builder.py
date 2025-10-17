# #    Copyright 2023 Haotian Liu
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.

# import os
# import warnings
# import shutil

# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
# import torch
# from llava.model import *
# from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# def _torch_load(path):
#     # 统一的本地安全加载（不联网）
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"[builder.py] File not found (offline mode): {path}")
#     return torch.load(path, map_location="cpu")

# def _auto_config_local(path_or_dir, fallback=None):
#     """
#     优先用本地 path_or_dir 读取 AutoConfig；失败时（例如适配器目录缺少自定义配置）
#     若提供 fallback（通常是 model_base），则用 fallback 的本地配置。
#     """
#     try:
#         return AutoConfig.from_pretrained(
#             path_or_dir,
#             trust_remote_code=True,
#             local_files_only=True,
#         )
#     except Exception:
#         if fallback is None:
#             raise
#         return AutoConfig.from_pretrained(
#             fallback,
#             trust_remote_code=True,
#             local_files_only=True,
#         )

# def _auto_tokenizer_local(path_or_dir, use_fast=False):
#     return AutoTokenizer.from_pretrained(
#         path_or_dir, use_fast=use_fast, local_files_only=True
#     )

# def _auto_causal_lm_local(cls, path_or_dir, **kwargs):
#     # 仅从本地加载权重/结构
#     kwargs.setdefault("low_cpu_mem_usage", True)
#     kwargs.setdefault("trust_remote_code", True)
#     kwargs.setdefault("local_files_only", True)
#     return cls.from_pretrained(path_or_dir, **kwargs)

# def load_pretrained_model(
#     model_path,
#     model_base,
#     model_name,
#     load_8bit=False,
#     load_4bit=False,
#     device_map="auto",
#     device="cuda"
# ):
#     kwargs = {"device_map": device_map}

#     if load_8bit:
#         kwargs["load_in_8bit"] = True
#     elif load_4bit:
#         kwargs["load_in_4bit"] = True
#         kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#         )
#     else:
#         kwargs["torch_dtype"] = torch.float16

#     if "llava" in model_name.lower():
#         # ----- LLaVA / LLaVA-Rad 路线 -----
#         if ("lora" in model_name.lower() or "llavarad" in model_name.lower()) and model_base is None:
#             warnings.warn(
#                 "There is `lora` in model name but no `model_base` is provided. "
#                 "If you are loading a LoRA model, please provide the `model_base` argument."
#             )

#         if ("lora" in model_name.lower() or "llavarad" in model_name.lower()) and model_base is not None:
#             # 适配器 + 基座
#             lora_cfg_pretrained = _auto_config_local(model_path, fallback=model_base)
#             tokenizer = _auto_tokenizer_local(model_base, use_fast=False)

#             print("Loading LLaVA from base model...")
#             model = _auto_causal_lm_local(
#                 LlavaLlamaForCausalLM, model_base, config=lora_cfg_pretrained, **kwargs
#             )

#             token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
#             if model.lm_head.weight.shape[0] != token_num:
#                 model.lm_head.weight = torch.nn.Parameter(
#                     torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
#                 )
#                 model.model.embed_tokens.weight = torch.nn.Parameter(
#                     torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
#                 )

#             print("Loading additional LLaVA weights...")
#             non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
#             non_lora_trainables = _torch_load(non_lora_path)
#             non_lora_trainables = {
#                 (k[11:] if k.startswith("base_model.") else k): v
#                 for k, v in non_lora_trainables.items()
#             }
#             if any(k.startswith("model.model.") for k in non_lora_trainables):
#                 non_lora_trainables = {
#                     (k[6:] if k.startswith("model.") else k): v
#                     for k, v in non_lora_trainables.items()
#                 }
#             model.load_state_dict(non_lora_trainables, strict=False)

#             from peft import PeftModel

#             print("Loading LoRA weights...")
#             # 本地路径加载，无需联网
#             model = PeftModel.from_pretrained(model, model_path)
#             print("Merging LoRA weights...")
#             model = model.merge_and_unload()
#             print("Model is loaded...")

#         elif model_base is not None:
#             # 仅 mm projector（无 LoRA）
#             print("Loading LLaVA from base model...")
#             if "mpt" in model_name.lower():
#                 # mpt 的特殊处理：把基座的配置文件复制到适配器目录（纯本地）
#                 if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
#                     shutil.copyfile(
#                         os.path.join(model_base, "configuration_mpt.py"),
#                         os.path.join(model_path, "configuration_mpt.py"),
#                     )
#                 tokenizer = _auto_tokenizer_local(model_base, use_fast=True)
#                 cfg_pretrained = _auto_config_local(model_path)
#                 model = _auto_causal_lm_local(
#                     LlavaMPTForCausalLM, model_base, config=cfg_pretrained, **kwargs
#                 )
#             else:
#                 tokenizer = _auto_tokenizer_local(model_base, use_fast=False)
#                 cfg_pretrained = _auto_config_local(model_path)
#                 model = _auto_causal_lm_local(
#                     LlavaLlamaForCausalLM, model_base, config=cfg_pretrained, **kwargs
#                 )

#             mm_proj_path = os.path.join(model_path, "mm_projector.bin")
#             mm_projector_weights = _torch_load(mm_proj_path)
#             mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
#             model.load_state_dict(mm_projector_weights, strict=False)

#         else:
#             # 直接加载完整的 LLaVA 权重（本地）
#             if "mpt" in model_name.lower():
#                 tokenizer = _auto_tokenizer_local(model_path, use_fast=True)
#                 model = _auto_causal_lm_local(LlavaMPTForCausalLM, model_path, **kwargs)
#             else:
#                 tokenizer = _auto_tokenizer_local(model_path, use_fast=False)
#                 model = _auto_causal_lm_local(LlavaLlamaForCausalLM, model_path, **kwargs)

#     else:
#         # ----- 纯语言模型路线 -----
#         if model_base is not None:
#             # PEFT 本地
#             from peft import PeftModel
#             tokenizer = _auto_tokenizer_local(model_base, use_fast=False)
#             base_lm = _auto_causal_lm_local(
#                 AutoModelForCausalLM, model_base, torch_dtype=torch.float16, device_map="auto"
#             )
#             print(f"Loading LoRA weights from {model_path}")
#             model = PeftModel.from_pretrained(base_lm, model_path)
#             print("Merging weights")
#             model = model.merge_and_unload()
#             print("Convert to FP16...")
#             model.to(torch.float16)
#         else:
#             # 直接从本地目录加载一个 AutoModelForCausalLM
#             if "mpt" in model_name.lower():
#                 tokenizer = _auto_tokenizer_local(model_path, use_fast=True)
#                 model = _auto_causal_lm_local(
#                     AutoModelForCausalLM, model_path, trust_remote_code=True, **kwargs
#                 )
#             else:
#                 tokenizer = _auto_tokenizer_local(model_path, use_fast=False)
#                 model = _auto_causal_lm_local(AutoModelForCausalLM, model_path, **kwargs)

#     image_processor = None

#     if "llava" in model_name.lower():
#         mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
#         mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
#         if mm_use_im_patch_token:
#             tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
#         if mm_use_im_start_end:
#             tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
#         model.resize_token_embeddings(len(tokenizer))

#         vision_tower = model.get_vision_tower()
#         if not vision_tower.is_loaded:
#             vision_tower.load_model()
#         vision_tower.to(device=device, dtype=torch.float16)
#         image_processor = vision_tower.image_processor

#     if hasattr(model.config, "max_sequence_length"):
#         context_len = model.config.max_sequence_length
#     else:
#         context_len = 2048

#     return tokenizer, model, image_processor, context_len

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


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if ('lora' in model_name.lower() or 'llavarad' in model_name.lower()) and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if ('lora' in model_name.lower() or 'llavarad' in model_name.lower()) and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                mm_projector_weights = load_from_hf(model_path, 'mm_projector.bin')
            
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None
    
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
