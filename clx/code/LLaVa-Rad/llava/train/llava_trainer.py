import os
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional
from transformers.trainer_pt_utils import get_parameter_names
from torch.optim import AdamW


# 安全地从DeepSpeed ZeRO-3优化中提取参数
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):                     # 检查参数是否有ds_id属性（ZeRO-3标识）
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):      # 在ZeRO-3中，参数可能分布在多个GPU上，需要临时收集
            param = param.data.detach().cpu().clone()
    else:           # 如果不是ZeRO参数，直接提取到CPU
        param = param.detach().cpu().clone()
    return param


# 多模态适配器参数提取
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}       # 根据关键词匹配筛选多模态适配器相关的参数
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}        # 对每个参数应用maybe_zero_3处理
    return to_return


# 均匀分块算法
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    # 如果不能均匀分配，使用跨步采样
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]
 
    # 计算每块的目标大小，初始化空块和长度统计
    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:                   # 总是将当前样本分配给总长度最短的块，确保各块的总长度大致相等 
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:            # 当块达到目标大小时，将其长度设为无穷大，避免继续分配
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


# 多模态数据分组采样，将多模态样本（正长度）和纯语言样本（负长度）分开
def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # 确保两种模态都有样本
    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    # 对每种模态分别按长度分组并打乱顺序，将长度相似的样本放在一起，提高训练效率
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # 将最后一个多模态批次和语言批次合并，重新打乱顺序，确保每个megabatch尽可能满
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # 尾块处理：如果合并的尾块足够大，单独作为一个批次；否则附加到末尾
    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


# 长度分组索引生成器，主要用于优化分布式训练中的数据加载效率
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


# 长度分组采样器类
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:                 # 必须提供长度信息
            raise ValueError("Lengths must be provided.")

        # 保存所有配置参数
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):              # 返回样本总数
        return len(self.lengths)

    def __iter__(self):             # 根据是否按模态分组，调用不同的索引生成函数
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


# 自定义训练器
class LLaVATrainer(Trainer):

    # 重写训练采样器获取方法，支持智能长度分组
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # 如果启用group_by_modality_length参数，则使用LengthGroupedSampler
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()
    # 重写检查点保存方法，仅保存多模态适配器参数（如果启用相应选项）
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    # 重写保存方法，仅保存多模态适配器参数（如果启用相应选项）
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

from transformers.optimization import get_scheduler

# [2025-12-23] 修改了：自定义 Trainer，给 mm_projector / mv_grid_mamba 分配不同学习率
class _MVTrainer(LLaVATrainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        args = self.args
        base_lr = args.learning_rate
        mm_lr = args.mm_projector_lr if args.mm_projector_lr is not None else base_lr
        mamba_lr = args.mamba_lr if getattr(args, "mamba_lr", None) is not None else base_lr

        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [n for n in decay_parameters if "bias" not in n]
        # [2025-12-25] 修改了部分代码，更可读
        def pick_lr(n: str) -> float:
            if "mm_projector" in n:
                return mm_lr
            if "mv_grid_mamba" in n or "mamba" in n:
                # 独立的 Mamba 学习率
                return getattr(args, "mamba_lr", None) or base_lr
            return base_lr

        buckets = {}  # (lr, wd) -> param group
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            wd = args.weight_decay if name in decay_parameters else 0.0
            lr = pick_lr(name)
            key = (lr, wd)
            if key not in buckets:
                buckets[key] = {"params": [], "lr": lr, "weight_decay": wd}
            buckets[key]["params"].append(p)

        self.optimizer = AdamW(
            list(buckets.values()),
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )

        # rank0 打印 group 概览
        print("[DEBUG] Optimizer groups:")
        for (lr, wd), g in buckets.items():
            n = sum(p.numel() for p in g["params"])
            print(f"  - lr={lr:.2e} wd={wd:.2e} params={n/1e6:.2f}M")
        # [2025-12-25] 修改结束
        return self.optimizer

    # [2025-12-26] 新增学习率相关代码
    def create_scheduler(self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None):
        """确保多 param group 都跟随相同 scheduler 曲线变化"""
        from transformers.optimization import get_scheduler

        args = self.args
        num_warmup_steps = int(num_training_steps * args.warmup_ratio)

        # 如果外部传了 optimizer（例如 DeepSpeed 调用），用它；否则用当前 trainer.optimizer
        if optimizer is None:
            optimizer = self.optimizer

        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # 同步每个 param_group 的初始 lr
        for i, group in enumerate(optimizer.param_groups):
            group.setdefault("initial_lr", group["lr"])
            print(f"[DEBUG] Scheduler bound to group {i}: initial_lr={group['initial_lr']:.2e}")

        self.lr_scheduler = scheduler
        return self.lr_scheduler
    # [2025-12-26] 新增结束

    # [2025-12-27] 新增打印学习率的函数
    def log(self, logs):
        if getattr(self, "optimizer", None) is not None:
            for i, g in enumerate(self.optimizer.param_groups):
                logs[f"lr/group_{i}"] = g.get("lr", None)
        return super().log(logs)
    # [2025-12-27] 新增结束

    # [2026-1-7] 新增
    def training_step(self, model, inputs):
        # 将 global_step 写到 core model 上，供 MVGridMambaFusion 做 ramp
        core = model.module if hasattr(model, "module") else model
        if hasattr(core, "get_model"):
            core.get_model()._mv_global_step = int(self.state.global_step)
        else:
            core._mv_global_step = int(self.state.global_step)
        return super().training_step(model, inputs)
    # [2026-1-7] 新增结束