CONTROLLER_HEART_BEAT_EXPIRATION = 30                   # 控制器心跳超时时间（秒）
WORKER_HEART_BEAT_INTERVAL = 15                         # Worker心跳间隔时间（秒）

LOGDIR = "."                                            # 日志文件存储目录（当前目录）

# Model Constants
IGNORE_INDEX = -100                                     # 用于损失计算时忽略特定Token（如填充符）
IMAGE_TOKEN_INDEX = -200                                # 图像Token的临时占位索引
DEFAULT_IMAGE_TOKEN = "<image>"                         # 图像Token的文本表示
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"                # 图像分块Token（未使用时可忽略）
DEFAULT_IM_START_TOKEN = "<im_start>"                   # 图像区域开始标记
DEFAULT_IM_END_TOKEN = "<im_end>"                       # 图像区域结束标记
