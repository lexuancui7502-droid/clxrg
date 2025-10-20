# [2025-10-16] 这个 __init__.py 文件的作用是将 rrg_eval 子包中的模块重新导出，
# re-export inner subpackage modules so both
# `rrg_eval.chexbert` and `rrg_eval.rrg_eval.chexbert` work.
# from importlib import import_module as _im

# for _name in ("chexbert", "rouge", "f1radgraph", "factuality_utils", "factuality_eval"):
#     globals()[_name] = _im(f".rrg_eval.{_name}", __name__)

# __all__ = ("chexbert", "rouge", "f1radgraph", "factuality_utils", "factuality_eval")


# [2025-10-17]  作用是将 rrg_eval 子包“暴露”为当前包的下一级命名空间
# 统一把里层子包“暴露”为当前包的下一级命名空间
# 不要在这里导入 chexbert/rouge 等模块，避免循环/遮蔽

from importlib import import_module as _im

# 把里层子包作为子命名空间暴露出来
_inner = _im(".rrg_eval", __name__)
globals()["rrg_eval"] = _inner

# 可选：为了兼容历史相对导入，提供一些直达别名
# （只暴露模块对象，不在 import 时执行模块代码）
for _name in ("chexbert", "rouge", "f1radgraph", "factuality_utils"):
    globals()[_name] = _im(f".rrg_eval.{_name}", __name__)

# 干净收尾
del _im, _inner, _name