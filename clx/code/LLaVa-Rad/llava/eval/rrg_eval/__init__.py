# [2024-10-16] 这个 __init__.py 文件的作用是将 rrg_eval 子包中的模块重新导出，
# re-export inner subpackage modules so both
# `rrg_eval.chexbert` and `rrg_eval.rrg_eval.chexbert` work.
from importlib import import_module as _im

for _name in ("chexbert", "rouge", "f1radgraph", "factuality_utils", "factuality_eval"):
    globals()[_name] = _im(f".rrg_eval.{_name}", __name__)

__all__ = ("chexbert", "rouge", "f1radgraph", "factuality_utils", "factuality_eval")
