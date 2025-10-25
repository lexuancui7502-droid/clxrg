import torch
import torch.nn as nn
import re


# 恒等映射类，创建一个不做任何变换的投影器
class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):          # 直接返回输入，不进行任何计算
        return x

    @property
    def config(self):           # 返回投影器类型信息
        return {"mm_projector_type": 'identity'}


# 简单残差块类，包含预归一化和两层线性变换，激活函数为GELU
class SimpleResBlock(nn.Module):
    def __init__(self, channels):               # 实现带残差连接的前馈网络块
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(              # 两层线性变换，中间使用GELU激活函数
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):                   # 前向传播，先归一化再通过投影层，最后加上残差
        x = self.pre_norm(x)
        return x + self.proj(x)


# 多模态投影器构建工厂函数，根据配置动态创建不同类型的投影器
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':          # 创建单层线性投影器
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)          # 创建多层MLP投影器，层数由配置指定
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':            # 创建恒等映射投影器
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')           # 遇到不支持的投影器类型时给出明确错误
