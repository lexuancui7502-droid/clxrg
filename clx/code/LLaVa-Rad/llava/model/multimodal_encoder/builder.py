import os
from .clip_encoder import CLIPVisionTower
from .open_clip_encoder import OpenCLIPVisionTower

# 视觉塔构建工厂函数，用于根据配置动态创建不同类型的视觉编码器
def build_vision_tower(vision_tower_cfg, **kwargs):             # 根据配置参数构建合适的视觉编码器
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))            # 获取视觉塔名称或路径
    vision_tower_config = getattr(vision_tower_cfg, 'mm_vision_tower_config', getattr(vision_tower_cfg, 'vision_tower_config', None))           # 获取视觉塔配置文件路径
    vision_tower_checkpoint = getattr(vision_tower_cfg, 'mm_vision_tower_checkpoint', getattr(vision_tower_cfg, 'vision_tower_checkpoint', None))           # 获取视觉塔检查点路径
    is_absolute_path_exists = os.path.exists(vision_tower)                  # 检查是否为存在的本地路径
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):            # 如果是本地路径或已知模型名称，创建标准的CLIP视觉塔
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf-hub:") or vision_tower_config and vision_tower_checkpoint:         # 如果是 Hugging Face Hub 模型，创建OpenCLIP视觉塔
        return OpenCLIPVisionTower(
            vision_tower, args=vision_tower_cfg, vision_tower_config=vision_tower_config, vision_tower_checkpoint=vision_tower_checkpoint, **kwargs
        )

    raise ValueError(f'Unknown vision tower: {vision_tower}')           # 如果无法识别视觉塔类型，抛出明确错误
