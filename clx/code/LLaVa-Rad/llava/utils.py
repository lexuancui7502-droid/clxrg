import datetime
import logging
import logging.handlers
import os
import sys
import json

import requests

from llava.constants import LOGDIR

# 统一错误消息，避免硬编码。
server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"                 # 高流量导致的网络错误提示
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."                         # 内容违规提示

handler = None


# 日志系统配置
def build_logger(logger_name, logger_filename):
    global handler

    # 定义日志格式（时间、日志级别、名称、消息）
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers   初始化根日志处理器，并设置格式
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers     将标准输出/错误重定向到日志系统，确保所有输出可追溯
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers    创建按天切割的日志文件
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(            # 每天生成一个新日志文件
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)                                # 为所有已存在的日志器（Logger）添加文件处理器

    return logger


# 流重定向类
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    # 将流（如 sys.stdout）的写入操作重定向到日志器
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout                  # 保留原始终端输出
        self.logger = logger                        # 目标日志器
        self.log_level = log_level                  # 日志级别
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    # 按行分割输入内容，完整行直接记录日志，不完整行暂存到 linebuf。确保跨平台换行符（\n）正确处理
    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    # 强制刷新缓冲区，确保所有内容被记录
    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


# 禁用 PyTorch 默认的权重初始化（如 Xavier/Glorot 初始化），减少模型加载时间
def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


# 调用 OpenAI 审核 API，检查文本是否包含违规内容（如暴力、仇恨言论）
def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


# ​​格式化输出信号量（Semaphore）的状态信息​​，以人类可读的字符串形式返回信号量的当前值和锁定状态
def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


# 默认加载器，直接加载 JSON 文件，无额外处理
def data_loader_default(data_path):
    logging.info("using the default loader.")
    dataset = json.load(open(data_path, "r"))
    logging.info(f"loaded {len(dataset)} samples.")
    return dataset


# 过滤只保留正位（AP/PA）胸片报告，跳过空内容
def data_loader_mimic_cxr_all_frontal_findings(data_path):
    logging.info("using the MIMIC-CXR loader: all frontal findings.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        if d['view'] in ('AP', 'PA'):
            ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


# 处理所有视角的胸片报告，动态生成提示词
def data_loader_mimic_cxr_all_views_findings(data_path):
    logging.info("using the MIMIC-CXR loader: all views findings.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        view = d["view"] if isinstance(d["view"], str) else "Unknown"
        d["conversations"][0]["value"] = f"<image>\nGiven the chest X-ray image from {view} view, describe the findings in the image: "
        ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


# 根据检查原因（reason字段）生成定制化提示词，适配训练/测试集
def data_loader_mimic_reason_findings(data_path, split):
    logging.info(f"using the MIMIC-CXR loader: MIMIC {split}.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        if split == 'test' and d['generate_method'] != 'rule-based':
            continue
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        if d['view'] not in ('AP', 'PA'):
            continue
        if d['image'].startswith("mimic/"):
            d['image'] = d['image'][len('mimic/'):]
        if d['reason'] is not None:
            reason = d['reason'].replace('\n', ' ')
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image given the following indication: {reason}"
        else:
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image."
        ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret

# [2025-11-17] 读取多视角json的数据加载函数
def data_loader_mimic_multiview_findings(data_path):
    """
    Loader for your new multiview JSON.

    Phase 2：真正保留多视角信息，让 Dataset 去加载多张图。
    这里把 image 统一成 list[str]，后面 __getitem__ 再把它变成 (N_view, 3, H, W)。
    """
    logging.info("使用 MIMIC-CXR multiview loader（保留所有视角）.")
    with open(data_path, "r") as f:
        dataset = json.load(f)

    ret = []
    for d in dataset:
        # 1) 跟其它 loader 一样，跳过空 findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue

        # 2) 处理 image 字段：可能是 str，也可能是 list[str]
        img_field = d.get("image", None)
        if img_field is None:
            continue

        if isinstance(img_field, list):
            img_list = img_field
        elif isinstance(img_field, str):
            img_list = [img_field]
        else:
            # 非法格式直接丢掉
            continue

        # 去掉每个路径前面的 "mimic/" 前缀（和原 loader 行为保持一致）
        clean_list = []
        for p in img_list:
            # if isinstance(p, str) and p.startswith("mimic/"):
            #     p = p[len("mimic/"):]
            # 11月17日晚赵一帆新增代码START
            if isinstance(p, str) and p.startswith("mimic/"):
                p = p[len("mimic/"):]
            elif p.startswith("/"):
                p = p[1:]
            # 11月17日晚赵一帆新增代码END
            clean_list.append(p)

        # ★ 关键：现在 d["image"] 是一个 list[str]
        d["image"] = clean_list

        # 3) prompt：可以沿用你原来的，也可以像 reason loader 那样拼。
        reason = d.get("reason", None)
        if reason is not None:
            reason = reason.replace("\n", " ")
            d["conversations"][0]["value"] = (
                "<image>\nProvide a description of the findings in the radiology image "
                f"given the following indication: {reason}"
            )
        else:
            # 如果你原 json 里的人类 prompt 已经是你想要的，也可以保留：
            d["conversations"][0]["value"] = d["conversations"][0].get(
                "value",
                "<image>\nDescribe the findings of the chest x-ray.\n",
            )

        ret.append(d)

    logging.info(
        f"loaded {len(ret)}/{len(dataset)} multiview samples "
        f"(each sample has len(d['image']) views)."
    )
    return ret


# 通过字典映射不同数据加载策略，便于动态调用
data_loaders = {
    "default": data_loader_default,
    "mimic_train_findings": lambda x: data_loader_mimic_reason_findings(x, "train"),
    "mimic_test_findings": lambda x: data_loader_mimic_reason_findings(x, "test"),
    "mimic_cxr_all_frontal_findings": data_loader_mimic_cxr_all_frontal_findings,
    "mimic_cxr_all_views_findings": data_loader_mimic_cxr_all_views_findings,
    "mimic_multiview_findings": data_loader_mimic_multiview_findings,           # [2025-11-17] 新增加载多视角数据的模块
}