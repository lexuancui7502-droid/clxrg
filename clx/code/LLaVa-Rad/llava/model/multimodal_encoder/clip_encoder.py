import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


# 基于CLIP的视觉特征提取器，将输入的医学图像转换为模型可以理解的视觉特征表示
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):               # 初始化函数，接受视觉塔名称、配置参数和延迟加载标志
        super().__init__()

        self.is_loaded = False                  # 标志，指示模型是否已加载

        self.vision_tower_name = vision_tower   # 视觉塔的名称或路径
        self.select_layer = args.mm_vision_select_layer                 # 选择用于特征提取的层
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')                    # 选择用于特征提取的特征类型，默认为'patch'

        if not delay_load:                      # 如果不使用延迟加载机制，则立即加载模型
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    # 加载预训练的CLIP视觉模型和图像处理器
    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)                     # 冻结模型参数，防止在训练过程中更新。在微调阶段只训练投影器，不更新视觉编码器

        self.is_loaded = True

    # 特征选择
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # CLS标记聚合整个图像的全局信息，而patch补丁特征则提供更细粒度的局部信息
        if self.select_feature == 'patch':                  # 选择除CLS标记外的所有patch补丁特征
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':            # 选择包括CLS标记在内的所有特征
            image_features = image_features
        else:                                               # 如果选择的特征类型不在预定义选项中，抛出错误
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()        # 禁用梯度计算，因为视觉编码器被冻结
    def forward(self, images):              # 前向传播函数，接受输入图像并返回提取的视觉特征
        if type(images) is list:            # 如果输入是图像列表，逐个处理每张图像
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)           # 处理单张图像，添加批次维度
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:                               # 如果输入是单个图像批次，直接处理整个批次
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):            # 创建虚拟特征，用于模型初始化或测试
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype                  # 模型数据类型

    @property
    def device(self):
        return self.vision_tower.device                 # 模型所在设备

    @property
    def config(self):
        if self.is_loaded:                      # 如果模型已加载，返回视觉塔的配置；如果没有加载，则返回仅配置对象
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size          # 视觉特征的隐藏层大小，即特征维度

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2          # 计算图像中的patch补丁数量
