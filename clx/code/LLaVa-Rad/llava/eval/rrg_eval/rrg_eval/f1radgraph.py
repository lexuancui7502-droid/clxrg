print("[F1RadGraph] using:", __file__)
# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Tuple, Optional
import types
import json

import torch
import torch.nn as nn
from radgraph import RadGraph

# =========================
# 全局保险：ModuleDict 回退
# =========================
if not hasattr(nn.ModuleDict, "_f1radgraph_orig_getitem"):
    nn.ModuleDict._f1radgraph_orig_getitem = nn.ModuleDict.__getitem__

    def _f1radgraph_fallback_getitem(self, key):
        try:
            return nn.ModuleDict._f1radgraph_orig_getitem(self, key)
        except KeyError as e:
            # 仅针对 __ner_labels / __relation_labels 做回退
            if isinstance(key, str) and (key.endswith("__ner_labels") or key.endswith("__relation_labels")):
                ks = list(self._modules.keys())
                suf = "ner_labels" if key.endswith("__ner_labels") else "relation_labels"
                for cand in ks:
                    if cand.endswith(f"__{suf}"):
                        return nn.ModuleDict._f1radgraph_orig_getitem(self, cand)
                if suf in ks:
                    return nn.ModuleDict._f1radgraph_orig_getitem(self, suf)
                if ks:
                    return nn.ModuleDict._f1radgraph_orig_getitem(self, ks[0])
            raise e

    nn.ModuleDict.__getitem__ = _f1radgraph_fallback_getitem

# ------------------ 小工具：递归解包与安全 getattr ------------------
def _unwrap(obj):
    """递归解包 AllenNLP/自定义 TimeDistributed 等包装器，直到取得真实子模块。"""
    seen = set()
    cur = obj
    while cur is not None and hasattr(cur, "_module") and id(cur) not in seen:
        seen.add(id(cur))
        cur = getattr(cur, "_module")
    return cur

def _get(obj, name, default=None):
    try:
        val = getattr(obj, name)
    except Exception:
        return default
    return _unwrap(val)

def _has(obj, name):
    try:
        getattr(obj, name)
        return True
    except Exception:
        return False

# -------- 配置：子模块及其关注的 label 后缀 --------
_SUBSPECS = {
    "_ner":      ("radgraph__ner_labels",      ("ner_labels",)),
    "ner":       ("radgraph__ner_labels",      ("ner_labels",)),
    "_relation": ("radgraph__relation_labels", ("relation_labels", "relation", "pruner", "span_pair")),
    "relation":  ("radgraph__relation_labels", ("relation_labels", "relation", "pruner", "span_pair")),
    "_coref":    ("radgraph__coref_labels",    ("coref_labels",)),
    "coref":     ("radgraph__coref_labels",    ("coref_labels",)),
    "_event":    ("radgraph__event_labels",    ("event_labels",)),
    "event":     ("radgraph__event_labels",    ("event_labels",)),
    "_srl":      ("radgraph__srl_labels",      ("srl_labels",)),
    "srl":       ("radgraph__srl_labels",      ("srl_labels",)),
}

# 容器名候选（不同版本命名各异）
_CONTAINER_CANDIDATES = (
    "_ner_scorers", "ner_scorers",
    "_relation_scorers", "relation_scorers",
    "_mention_pruners", "mention_pruners",
    "_span_pair_scorers", "span_pair_scorers",
    "_entity_head_scorers", "entity_head_scorers",
    "_entity_tail_scorers", "entity_tail_scorers",
    "_scorers", "scorers",
)

def _keys(container):
    try:
        if isinstance(container, nn.ModuleDict):
            return list(container._modules.keys())
        return list(container.keys())
    except Exception:
        return []

def _get_item(container, k):
    if isinstance(container, nn.ModuleDict):
        return container._modules[k]
    return container[k]

def _set_item(container, k, v):
    try:
        if isinstance(container, nn.ModuleDict):
            container._modules[k] = v
        else:
            container[k] = v
    except Exception:
        pass

def _three_way_alias(container: Optional[object], suffix: str):
    """
    互为别名（存在一个就补齐另外两个）：
      - 'radgraph__{suffix}'
      - 'None__{suffix}'
      - '{suffix}'
    若容器只有一个键，也把该唯一键映射到上述三者。
    """
    if container is None or not isinstance(container, (dict, nn.ModuleDict)):
        return
    want = [f"radgraph__{suffix}", f"None__{suffix}", f"{suffix}"]
    present = _keys(container)

    # 确定源
    source_key = None
    for k in want:
        if k in present:
            source_key = k
            break
    if source_key is None:
        if len(present) == 1:
            source_key = present[0]
        else:
            for k in present:
                if k.endswith(f"__{suffix}") or k == suffix:
                    source_key = k
                    break
            if source_key is None:
                return

    try:
        val = _get_item(container, source_key)
    except Exception:
        return

    for k in want:
        if k not in present:
            _set_item(container, k, val)
            present.append(k)

def _three_way_alias_vocab(index_to_token_like: Optional[dict], suffix: str):
    """对 vocab._index_to_token 做三向别名（引用同一对象）"""
    if index_to_token_like is None or not isinstance(index_to_token_like, dict):
        return
    want = [f"radgraph__{suffix}", f"None__{suffix}", f"{suffix}"]
    present = list(index_to_token_like.keys())

    source_key = None
    for k in want:
        if k in present:
            source_key = k
            break
    if source_key is None:
        if len(present) == 1:
            source_key = present[0]
        else:
            for k in present:
                if k.endswith(f"__{suffix}") or k == suffix:
                    source_key = k
                    break
            if source_key is None:
                return

    val = index_to_token_like.get(source_key, None)
    if val is None:
        return
    for k in want:
        if k not in present:
            index_to_token_like[k] = val
            present.append(k)

def _alias_all_matching_containers(submod, suffixes: Tuple[str, ...]):
    """扫描 submod 的 dict / ModuleDict，凡键包含 suffix 或 *__suffix 就做三向别名。"""
    containers = []

    # 先拿常见容器名
    for cand in _CONTAINER_CANDIDATES:
        if _has(submod, cand):
            containers.append(_get(submod, cand))

    # 再通配扫描
    try:
        for _, val in vars(submod).items():
            if isinstance(val, (dict, nn.ModuleDict)):
                containers.append(val)
    except Exception:
        pass

    seen = set()
    for c in containers:
        if id(c) in seen:
            continue
        seen.add(id(c))
        ks = _keys(c)
        if not ks:
            continue
        for suf in suffixes:
            if any((k == suf or k.endswith(f"__{suf}")) for k in ks):
                _three_way_alias(c, suf)

def _pick_existing_key_for_suffix(container, suffix: str) -> Optional[str]:
    """为 suffix 选择一个真实存在的 key（radgraph__suffix -> *__suffix -> suffix -> 唯一键 -> 第一个）。"""
    if container is None or not isinstance(container, (dict, nn.ModuleDict)):
        return None
    ks = _keys(container)
    if not ks:
        return None
    preferred = f"radgraph__{suffix}"
    if preferred in ks:
        return preferred
    for k in ks:
        if k.endswith(f"__{suffix}"):
            return k
    if suffix in ks:
        return suffix
    if len(ks) == 1:
        return ks[0]
    return ks[0]

def _set_namespace_attrs(submod, ns_full: Optional[str], label_suffixes: Tuple[str, ...], chosen_key: Optional[str]):
    """尽可能设置 namespace / active_namespace / label_namespace（兼容不同命名）。"""
    for attr in ("_namespace", "namespace"):
        if _has(submod, attr):
            try:
                setattr(submod, attr, "radgraph")
            except Exception:
                pass
    if label_suffixes:
        label_val = label_suffixes[0]
        for attr in ("_label_namespace", "label_namespace", "_labels_namespace", "labels_namespace"):
            if _has(submod, attr) and _get(submod, attr) in (None, "", "None", "default"):
                try:
                    setattr(submod, attr, label_val)
                except Exception:
                    pass
    if chosen_key or ns_full:
        for attr in ("_active_namespace", "active_namespace"):
            try:
                setattr(submod, attr, chosen_key or ns_full)
            except Exception:
                pass

def _force_ns_and_activekey(submod, default_full_ns: str, suffixes: Tuple[str, ...]):
    """容器与 vocab 三向别名 -> 选择真实 key -> 写回 active/label/namespace。"""
    _alias_all_matching_containers(submod, suffixes)
    vocab = _get(submod, "vocab", None)
    if vocab is not None:
        try:
            idx2tok = _get(vocab, "_index_to_token", None)
            if isinstance(idx2tok, dict):
                for suf in suffixes:
                    _three_way_alias_vocab(idx2tok, suf)
        except Exception:
            pass
    chosen_key = None
    container = None
    for cand_name in ("_ner_scorers", "ner_scorers", "_scorers", "scorers", "_relation_scorers", "relation_scorers"):
        if _has(submod, cand_name):
            container = _get(submod, cand_name)
            for suf in suffixes:
                key = _pick_existing_key_for_suffix(container, suf)
                if key is not None:
                    chosen_key = key
                    break
        if chosen_key is not None:
            break
    if chosen_key is None and container is not None:
        ks = _keys(container)
        if ks:
            chosen_key = ks[0]
    _set_namespace_attrs(submod, default_full_ns, suffixes, chosen_key)
    if container is not None:
        for suf in suffixes:
            _three_way_alias(container, suf)
        if chosen_key is not None:
            base_suf = suffixes[0]
            for alias in (f"None__{base_suf}", f"radgraph__{base_suf}", base_suf):
                if alias not in _keys(container):
                    _set_item(container, alias, _get_item(container, chosen_key))

def _looks_like_target_module(m):
    u = _unwrap(m)
    for cand in ("_ner_scorers", "ner_scorers",
                 "_relation_scorers", "relation_scorers",
                 "_scorers", "scorers"):
        if hasattr(u, cand):
            return True
    return False

# =============== pre-hook，在 forward 调用前“就地修复” =================
def _install_forward_pre_hook(u, default_full_ns: str, suffixes: Tuple[str, ...]):
    if hasattr(u, "_f1radgraph_pre_hook_installed"):
        return

    def _pre_hook(module, inputs):
        _force_ns_and_activekey(module, default_full_ns, suffixes)

    u.register_forward_pre_hook(_pre_hook)
    setattr(u, "_f1radgraph_pre_hook_installed", True)

def _wrap_forward(submod, default_full_ns: str, suffixes: Tuple[str, ...]):
    if submod is None:
        return
    submod = _unwrap(submod)
    _install_forward_pre_hook(submod, default_full_ns, suffixes)

    if hasattr(submod, "_orig_forward"):
        return
    submod._orig_forward = submod.forward

    def _fix_this_module_namespace():
        _force_ns_and_activekey(submod, default_full_ns, suffixes)

    def safe_forward(*args, **kwargs):
        _fix_this_module_namespace()
        try:
            return submod._orig_forward(*args, **kwargs)
        except KeyError as e:
            if "__ner_labels" in str(e) or "__relation_labels" in str(e):
                print("[F1RadGraph] safe_forward: caught KeyError on this module; patch & retry")
                _fix_this_module_namespace()
                return submod._orig_forward(*args, **kwargs)
            raise

    submod.forward = safe_forward

def _wrap_if_needed(m):
    u = _unwrap(m)
    if hasattr(u, "_wrapped_by_f1radgraph"):
        return
    if hasattr(u, "_ner_scorers") or hasattr(u, "ner_scorers"):
        default_ns = "radgraph__ner_labels"; suffixes = ("ner_labels",)
    elif hasattr(u, "_relation_scorers") or hasattr(u, "relation_scorers"):
        default_ns = "radgraph__relation_labels"; suffixes = ("relation_labels", "relation", "pruner", "span_pair")
    else:
        default_ns = "radgraph__ner_labels"; suffixes = ("ner_labels",)
    _wrap_forward(u, default_ns, suffixes)
    setattr(u, "_wrapped_by_f1radgraph", True)

def _wrap_all_like_targets(model_root):
    if model_root is None:
        return
    for m in model_root.modules():
        try:
            if _looks_like_target_module(m):
                _wrap_if_needed(m)
        except Exception:
            pass

def _monkey_patch_all_submodules(model_root):
    if model_root is None:
        return
    for name, (default_ns, suffixes) in _SUBSPECS.items():
        sub = _get(model_root, name, None)
        if sub is not None:
            _wrap_forward(_unwrap(sub), default_ns, suffixes)
    _wrap_all_like_targets(model_root)

    # vocab 兜底（全局别名 + 安全 get_token_from_index）
    vocab = _get(model_root, "vocab", None)
    if vocab is not None:
        try:
            idx2tok = _get(vocab, "_index_to_token", None)
            if isinstance(idx2tok, dict):
                for _, (_, suffixes) in _SUBSPECS.items():
                    for suf in suffixes:
                        _three_way_alias_vocab(idx2tok, suf)
        except Exception:
            pass

# ------------------ 强兜底：直接把 active_namespace 与容器键修到能跑 ------------------
def _force_ns_and_activekey_hard(model, default_ns="entities"):
    def _ensure_for_nerlike(ner_mod):
        if ner_mod is None:
            return
        ner_mod = _unwrap(ner_mod)
        if getattr(ner_mod, "_active_namespace", None) in (None, ""):
            ner_mod._active_namespace = default_ns
        for cont_name in ("_ner_scorers", "ner_scorers", "_relation_scorers", "relation_scorers", "_scorers", "scorers"):
            if hasattr(ner_mod, cont_name):
                cont = getattr(ner_mod, cont_name)
                if cont is None:
                    continue
                ks = _keys(cont)
                if not ks:
                    continue
                pref = None
                for suf in ("ner_labels", "relation_labels"):
                    pref = _pick_existing_key_for_suffix(cont, suf)
                    if pref: break
                if pref is None:
                    pref = ks[0]
                for suf in ("ner_labels", "relation_labels"):
                    _three_way_alias(cont, suf)
                    for alias in (f"None__{suf}", f"radgraph__{suf}", suf):
                        if alias not in _keys(cont):
                            _set_item(cont, alias, _get_item(cont, pref))
                if getattr(ner_mod, "_active_namespace", None) not in _keys(cont):
                    try:
                        setattr(ner_mod, "_active_namespace", pref)
                    except Exception:
                        pass
                break

    m = _unwrap(model)
    try:
        for name in ("_ner", "ner", "_relation", "relation"):
            if hasattr(m, name):
                _ensure_for_nerlike(getattr(m, name))
        for sm in m.modules():
            if _looks_like_target_module(sm):
                _ensure_for_nerlike(sm)
    except Exception as e:
        print("[F1RadGraph] warn: hard fallback failed:", repr(e))

# =========================
# 词表级兜底：类级 & 实例级 get_token_from_index
# =========================
def _install_vocab_fallback_class():
    """
    把 AllenNLP Vocabulary 类的 get_token_from_index 全局替换为安全版本，
    确保所有实例（包括我们没拿到引用的子模块里的 vocab）都能 fallback。
    """
    try:
        from radgraph.allennlp.data.vocabulary import Vocabulary as _VocabCls
    except Exception:
        return

    if getattr(_VocabCls, "_f1radgraph_cls_patched", False):
        return

    _VocabCls._f1radgraph_orig_get_token_from_index = _VocabCls.get_token_from_index

    def _safe_get_token_from_index_cls(self, index: int, namespace: str = "tokens"):
        try:
            return _VocabCls._f1radgraph_orig_get_token_from_index(self, index, namespace)
        except KeyError:
            idx2tok = getattr(self, "_index_to_token", None)
            if not isinstance(idx2tok, dict):
                return "O"

            # 推断后缀
            if isinstance(namespace, str) and namespace.endswith("__ner_labels"):
                suf = "ner_labels"
            elif isinstance(namespace, str) and namespace.endswith("__relation_labels"):
                suf = "relation_labels"
            else:
                suf = "ner_labels" if ("ner" in str(namespace)) else ("relation_labels" if ("relation" in str(namespace)) else "tokens")

            candidates = [namespace] if isinstance(namespace, str) else []
            for ns in (f"radgraph__{suf}", f"entities__{suf}", f"None__{suf}", suf):
                if ns not in candidates:
                    candidates.append(ns)
            # 扫一遍所有 *labels 命名空间，找包含该 index 的
            for k in list(idx2tok.keys()):
                if k not in candidates and (k.endswith("__ner_labels") or k.endswith("__relation_labels") or k.endswith("__srl_labels")):
                    candidates.append(k)

            for ns in candidates:
                mapping = idx2tok.get(ns)
                if isinstance(mapping, dict) and (index in mapping):
                    return mapping[index]
                if isinstance(mapping, (list, tuple)) and 0 <= index < len(mapping):
                    return mapping[index]
            return "O"

    _VocabCls.get_token_from_index = _safe_get_token_from_index_cls
    _VocabCls._f1radgraph_cls_patched = True

def _install_vocab_fallback_instance(vocab):
    """
    实例级补丁（双保险）。如果类级补丁失败或某些实例被特殊化，仍可覆盖实例方法。
    """
    if vocab is None or getattr(vocab, "_f1radgraph_gtfi_patched", False):
        return
    idx2tok = getattr(vocab, "_index_to_token", None)
    orig = getattr(vocab, "get_token_from_index", None)
    if not callable(orig) or not isinstance(idx2tok, dict):
        return

    def _safe_get_token_from_index(index: int, namespace: str = "tokens"):
        try:
            return orig(index, namespace)
        except KeyError:
            # 同类级逻辑
            if isinstance(namespace, str) and namespace.endswith("__ner_labels"):
                suf = "ner_labels"
            elif isinstance(namespace, str) and namespace.endswith("__relation_labels"):
                suf = "relation_labels"
            else:
                suf = "ner_labels" if ("ner" in str(namespace)) else ("relation_labels" if ("relation" in str(namespace)) else "tokens")

            candidates = [namespace] if isinstance(namespace, str) else []
            for ns in (f"radgraph__{suf}", f"entities__{suf}", f"None__{suf}", suf):
                if ns not in candidates:
                    candidates.append(ns)
            for k in list(idx2tok.keys()):
                if k not in candidates and (k.endswith("__ner_labels") or k.endswith("__relation_labels") or k.endswith("__srl_labels")):
                    candidates.append(k)

            for ns in candidates:
                mapping = idx2tok.get(ns)
                if isinstance(mapping, dict) and (index in mapping):
                    return mapping[index]
                if isinstance(mapping, (list, tuple)) and 0 <= index < len(mapping):
                    return mapping[index]
            return "O"

    try:
        setattr(vocab, "get_token_from_index", _safe_get_token_from_index)
        setattr(vocab, "_f1radgraph_gtfi_patched", True)
    except Exception:
        pass

def _ensure_model_patched(radgraph_obj: RadGraph):
    # 先装类级补丁（一次即可，影响所有 vocab）
    _install_vocab_fallback_class()

    manager = _get(radgraph_obj, "_predict_manager", None)
    if manager is None:
        try:
            radgraph_obj._make_predict_manager()
            manager = _get(radgraph_obj, "_predict_manager", None)
        except Exception:
            manager = _get(radgraph_obj, "_predict_manager", None)

    predictor = None if manager is None else _get(manager, "_predictor", None)
    dygie_model = None if predictor is None else _get(predictor, "_model", None)
    dygie_model = _unwrap(dygie_model)
    if dygie_model is None:
        return

    already = False
    for probe in ("_ner", "ner"):
        sub = _get(dygie_model, probe, None)
        if sub is not None and hasattr(_unwrap(sub), "_orig_forward"):
            already = True
            break
    if not already:
        _monkey_patch_all_submodules(dygie_model)

    # ★ 再做一次硬兜底
    _force_ns_and_activekey_hard(dygie_model, default_ns="entities")

    # ★ 词表兜底：别名 + 实例级安全 getter（双保险）
    vocab = _get(dygie_model, "vocab", None)
    if vocab is not None:
        try:
            idx2tok = _get(vocab, "_index_to_token", None)
            if isinstance(idx2tok, dict):
                for _, (_, suffixes) in _SUBSPECS.items():
                    for suf in suffixes:
                        _three_way_alias_vocab(idx2tok, suf)
            _install_vocab_fallback_instance(vocab)
        except Exception:
            pass

    # 子模块若各自持有 vocab，也打补丁
    try:
        for m in dygie_model.modules():
            v = getattr(_unwrap(m), "vocab", None)
            if v is not None:
                _install_vocab_fallback_instance(v)
    except Exception:
        pass

# =========================
# 输出强规范化（关键修复）
# =========================
def _empty_graph() -> Dict[str, Any]:
    return {"entities": [], "relations": []}

def _normalize_graph_item(x: Any) -> Dict[str, Any]:
    """
    将 radgraph 返回的单个结果统一为 {entities: list, relations: list}。
    允许输入为 dict / str(JSON) / None / 其它类型。
    """
    try:
        # 已是 dict
        if isinstance(x, dict):
            ent = x.get("entities", []) if isinstance(x.get("entities", []), list) else []
            rel = x.get("relations", []) if isinstance(x.get("relations", []), list) else []
            return {"entities": ent, "relations": rel}
        # 是 JSON 字符串
        if isinstance(x, str):
            x = x.strip()
            if x.startswith("{") and x.endswith("}"):
                d = json.loads(x)
                if isinstance(d, dict):
                    return _normalize_graph_item(d)
            # 非 JSON 的字符串直接给空图
            return _empty_graph()
        # 其它类型（如 tuple 等）尝试转 dict
        if hasattr(x, "_asdict"):
            return _normalize_graph_item(x._asdict())
    except Exception:
        pass
    return _empty_graph()

def _normalize_graph_list(xs: Any, expect_len: int) -> List[Dict[str, Any]]:
    """
    将 radgraph 返回的整体结果统一为 list[dict]，长度对不齐时用空图补齐或截断。
    """
    out: List[Dict[str, Any]] = []
    if isinstance(xs, list):
        out = [_normalize_graph_item(i) for i in xs]
    elif isinstance(xs, tuple):
        out = [_normalize_graph_item(i) for i in list(xs)]
    else:
        # 单个也包成列表
        out = [_normalize_graph_item(xs)]

    if len(out) < expect_len:
        out.extend([_empty_graph()] * (expect_len - len(out)))
    elif len(out) > expect_len:
        out = out[:expect_len]
    return out

class F1RadGraphv2(nn.Module):
    def __init__(self, reward_level: str = "partial", batch_size: int = 1, device: Optional[str] = None):
        super().__init__()
        self.reward_level = reward_level
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.radgraph = RadGraph(
            batch_size=self.batch_size,
            use_gpu=(self.device.startswith("cuda")),
            return_tokens=False
        )

    @torch.no_grad()
    def forward(self, hyps: List[str], refs: List[str]):
        assert len(hyps) == len(refs), "hyps 与 refs 数量不一致"
        _ensure_model_patched(self.radgraph)
        hyp_graphs = self._infer_graphs(hyps)
        ref_graphs = self._infer_graphs(refs)
        # 二次兜底（理论上已是标准化后的）
        hyp_graphs = _normalize_graph_list(hyp_graphs, len(hyps))
        ref_graphs = _normalize_graph_list(ref_graphs, len(refs))

        p, r, f1 = self._compute_partial_f1(hyp_graphs, ref_graphs)

        # ★★ 兼容上层期望的键名（postprocess_eval 会取 "f1-radgraph"）
        scores = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "precision-radgraph": float(p),
            "recall-radgraph": float(r),
            "f1-radgraph": float(f1),
        }
        details = {"reward_level": self.reward_level}
        return scores, details

    def _infer_graphs(self, reports: List[str]) -> List[Dict[str, Any]]:
        _ensure_model_patched(self.radgraph)
        try:
            raw = self.radgraph(reports)
            return _normalize_graph_list(raw, len(reports))
        except KeyError as e:
            # ★ 只针对 label/namespace/索引问题做自动修复并重试
            es = str(e)
            if ("__ner_labels" in es) or ("__relation_labels" in es) or (es.strip() in {"'None__ner_labels'", "'None__relation_labels'", "1"}):
                print("[F1RadGraph] caught KeyError in inference; forcing vocab/aliases & retry once")
                _ensure_model_patched(self.radgraph)

                manager = _get(self.radgraph, "_predict_manager", None)
                predictor = None if manager is None else _get(manager, "_predictor", None)
                model = None if predictor is None else _get(predictor, "_model", None)
                model = _unwrap(model)

                try:
                    if model is not None:
                        # 子模块容器/active 再修一遍
                        for ner_name in ("_ner", "ner", "_relation", "relation"):
                            ner = getattr(model, ner_name, None)
                            if ner is None:
                                continue
                            ner = _unwrap(ner)
                            for cont_name in ("_ner_scorers", "ner_scorers", "_relation_scorers", "relation_scorers", "_scorers", "scorers"):
                                if hasattr(ner, cont_name):
                                    cont = getattr(ner, cont_name)
                                    if cont is None:
                                        continue
                                    ks = _keys(cont)
                                    if not ks:
                                        continue
                                    # 选一个最稳妥的 key
                                    target_suffix = "ner_labels" if "ner" in cont_name else "relation_labels"
                                    key = _pick_existing_key_for_suffix(cont, target_suffix) or ks[0]
                                    # 显式补别名
                                    for alias in (f"None__{target_suffix}", f"radgraph__{target_suffix}", target_suffix):
                                        if alias not in ks:
                                            _set_item(cont, alias, _get_item(cont, key))
                                            ks.append(alias)
                                    # 强制改 active
                                    try:
                                        setattr(ner, "_active_namespace", key)
                                    except Exception:
                                        pass
                                    break

                        # 词表兜底：类级 + 实例级
                        _install_vocab_fallback_class()
                        vocab = _get(model, "vocab", None)
                        if vocab is not None:
                            idx2tok = _get(vocab, "_index_to_token", None)
                            if isinstance(idx2tok, dict):
                                for sfx in ("ner_labels", "relation_labels", "srl_labels", "tokens"):
                                    _three_way_alias_vocab(idx2tok, sfx)
                            _install_vocab_fallback_instance(vocab)

                        # 子模块 vocab 也补
                        for m in model.modules():
                            v = getattr(_unwrap(m), "vocab", None)
                            if v is not None:
                                _install_vocab_fallback_instance(v)

                        _monkey_patch_all_submodules(model)
                        _force_ns_and_activekey_hard(model, default_ns="entities")
                except Exception as ee:
                    print("[F1RadGraph] warn: final direct alias failed:", repr(ee))

                # 重试一次
                raw = self.radgraph(reports)
                return _normalize_graph_list(raw, len(reports))
            raise

    def _compute_partial_f1(self, hyps: List[Dict[str, Any]], refs: List[Dict[str, Any]]):
        # 再保险：任何非 dict/异常项都规范化为空图，避免 AttributeError
        hyps = [_normalize_graph_item(h) for h in hyps]
        refs = [_normalize_graph_item(r) for r in refs]

        def norm(s: str) -> str:
            return " ".join(s.lower().split())
        def ent_key(e: Dict[str, Any]):
            text = e.get("text") or e.get("span") or ""
            typ  = e.get("label") or e.get("type") or ""
            return (norm(text), typ)
        def rel_key(r: Dict[str, Any]):
            head = r.get("head", {})
            tail = r.get("tail", {})
            typ  = r.get("type") or r.get("label") or ""
            head_t = head.get("text") or head.get("span") or ""
            tail_t = tail.get("text") or tail.get("span") or ""
            return (norm(head_t), norm(tail_t), typ)

        hyp_ents, ref_ents = set(), set()
        hyp_rels, ref_rels = set(), set()
        for h, g in zip(hyps, refs):
            for e in h.get("entities", []) if isinstance(h.get("entities", []), list) else []:
                if isinstance(e, dict):
                    hyp_ents.add(ent_key(e))
            for e in g.get("entities", []) if isinstance(g.get("entities", []), list) else []:
                if isinstance(e, dict):
                    ref_ents.add(ent_key(e))
            for rel in h.get("relations", []) if isinstance(h.get("relations", []), list) else []:
                if isinstance(rel, dict):
                    hyp_rels.add(rel_key(rel))
            for rel in g.get("relations", []) if isinstance(g.get("relations", []), list) else []:
                if isinstance(rel, dict):
                    ref_rels.add(rel_key(rel))

        def prf(pred: set, gold: set):
            tp = len(pred & gold)
            p = tp / max(1, len(pred))
            r = tp / max(1, len(gold))
            f1 = 2 * p * r / max(1e-8, (p + r))
            return p, r, f1

        p_e, r_e, f1_e = prf(hyp_ents, ref_ents)
        p_r, r_r, f1_r = prf(hyp_rels, ref_rels)
        p = (p_e + p_r) / 2
        r = (r_e + r_r) / 2
        f1 = (f1_e + f1_r) / 2
        return p, r, f1


















# 新增代码

# 分割线
# 2025/10/17 23:24注释代码
# ==============================================================================================================================
# # """
# Custom f1 radgraph that can output precision and recall
# """

# from radgraph import F1RadGraph
# import numpy as np
# from scipy.stats import bootstrap

# import os
# from radgraph.radgraph import CACHE_DIR
# from huggingface_hub import hf_hub_download


# class F1RadGraphv2(F1RadGraph):
#     def __init__(
#             self,
#             reward_level,
#             **kwargs
#     ):

#         self._download_radgraph()
#         super().__init__(reward_level, **kwargs)
#         assert reward_level in ["simple", "partial", "complete", "all"]

#     def _download_radgraph(self):
#         if not os.path.exists(os.path.join(CACHE_DIR, "radgraph.tar.gz")):
#             os.makedirs(CACHE_DIR, exist_ok=True)
#             hf_hub_download(
#                 repo_id="StanfordAIMI/RRG_scorers",
#                 filename="radgraph.tar.gz",
#                 revision="d97745aa136e5beb927da7e768e99de6ae807902",
#                 local_dir=CACHE_DIR,
#             )

#     # =======================  [2025-10-17 修改] 开始  =======================
#     # 最小兜底：强制把 RadGraph 模型的 dataset/namespace 设为 "radgraph"
#     def _force_dataset_namespace(self):
#         try:
#             predictor = getattr(self, "radgraph", None)
#             if predictor is None:
#                 return
#             predictor = getattr(predictor, "predictor", None)
#             if predictor is None:
#                 return
#             model = getattr(predictor, "_model", None)
#             if model is None:
#                 return

#             # 1) 模型级别：若 _dataset 缺失/为空，强制设为 "radgraph"
#             if getattr(model, "_dataset", None) in (None, "None", ""):
#                 setattr(model, "_dataset", "radgraph")

#             # 2) 子模块级别：把 _active_namespace 设为 "radgraph"
#             for sub_name in ("_ner", "_relation", "_coref", "_event", "_srl"):
#                 sub = getattr(model, sub_name, None)
#                 if sub is not None and getattr(sub, "_active_namespace", None) in (None, "None", ""):
#                     setattr(sub, "_active_namespace", "radgraph")
#         except Exception:
#             # 兜底失败也不要影响主流程
#             pass
#     # =======================  [2025-10-17 修改] 结束  =======================

#     def forward(self, refs, hyps):
#         # Checks
#         assert isinstance(hyps, str) or isinstance(hyps, list)
#         assert isinstance(refs, str) or isinstance(refs, list)

#         if isinstance(hyps, str):
#             hyps = [hyps]
#         # =======================  [2025-10-17 修改]  =======================
#         # 修正：这里原先误写成 isinstance(hyps, str)
#         if isinstance(refs, str):
#             refs = [refs]
#         # =======================  [2025-10-17 修改]  =======================

#         assert len(refs) == len(hyps)

#         # getting empty report list
#         number_of_reports = len(hyps)
#         empty_report_index_list = [i for i in range(number_of_reports) if (len(hyps[i]) == 0) or (len(refs[i]) == 0)]
#         number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)

#         # stacking all reports (hyps and refs)
#         report_list = [
#                           hypothesis_report
#                           for i, hypothesis_report in enumerate(hyps)
#                           if i not in empty_report_index_list
#                       ] + [
#                           reference_report
#                           for i, reference_report in enumerate(refs)
#                           if i not in empty_report_index_list
#                       ]

#         assert len(report_list) == 2 * number_of_non_empty_reports

#         # getting annotations
#         # =======================  [2025-10-17 修改]  =======================
#         # 兜底一次，避免 None__ner_labels
#         self._force_dataset_namespace()
#         # =======================  [2025-10-17 修改]  =======================
#         inference_dict = self.radgraph(report_list)

#         # Compute reward
#         reward_list = []
#         hypothesis_annotation_lists = []
#         reference_annotation_lists = []
#         non_empty_report_index = 0
#         for report_index in range(number_of_reports):
#             if report_index in empty_report_index_list:
#                 reward_list.append((0., 0., 0.))
#                 continue

#             hypothesis_annotation_list = inference_dict[str(non_empty_report_index)]
#             reference_annotation_list = inference_dict[
#                 str(non_empty_report_index + number_of_non_empty_reports)
#             ]

#             reward_list.append(
#                 compute_reward(
#                     hypothesis_annotation_list,
#                     reference_annotation_list,
#                     self.reward_level,
#                 )
#             )
#             reference_annotation_lists.append(reference_annotation_list)
#             hypothesis_annotation_lists.append(hypothesis_annotation_list)
#             non_empty_report_index += 1

#         assert non_empty_report_index == number_of_non_empty_reports

#         if self.reward_level == "all":
#             reward_list = ([r[0] for r in reward_list], [r[1] for r in reward_list], [r[2] for r in reward_list])
#             mean_reward = (np.mean(reward_list[0]), np.mean(reward_list[1]), np.mean(reward_list[2]))
#         else:
#             mean_reward = np.mean(reward_list, axis=0)
#             mean_reward = {'f1-radgraph': mean_reward[0], 'precision-radgraph': mean_reward[1], 'recall-radgraph': mean_reward[2]}

#         return (
#             mean_reward,
#             reward_list,
#             hypothesis_annotation_lists,
#             reference_annotation_lists,
#         )


# def compute_statistic(reward_list):
#     def _compute(indices):
#         r = [reward_list[i] for i in indices]
#         return np.mean(r, axis=0)[0]

#     return _compute


# def bootstrap_confidence_interval(
#         reward_list, n_resamples: int = 500, method: str = "percentile",
#     ):
#     bs = bootstrap(
#         data=(list(range(len(reward_list))),),
#         statistic=compute_statistic(reward_list),
#         method=method,
#         paired=False,
#         vectorized=False,
#         confidence_level=0.95,
#         random_state=3,
#         n_resamples=n_resamples
#     )
#     return bs


# def exact_entity_token_if_all_match_reward(
#         hypothesis_annotation_list, reference_annotation_list
# ):
#     candidates = []
#     for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
#         candidate = []
#         for entity in annotation_list["entities"].values():
#             if not entity["relations"]:
#                 candidate.append((entity["tokens"], entity["label"]))
#             if entity["relations"]:
#                 candidate.extend([(entity["tokens"].lower(),
#                                    entity["label"],
#                                    r[0],
#                                    annotation_list["entities"][r[1]]["tokens"].lower())
#                                   for r in entity["relations"]]
#                                  )

#         candidate = set(candidate)
#         candidates.append(candidate)

#     hypothesis_relation_token_list, reference_relation_token_list = candidates
#     precision = (
#         sum(
#             [
#                 1
#                 for x in hypothesis_relation_token_list
#                 if (x in reference_relation_token_list)
#             ]
#         )
#         / len(hypothesis_relation_token_list)
#         if len(hypothesis_relation_token_list) > 0
#         else 0.0
#     )
#     recall = (
#         sum(
#             [
#                 1
#                 for x in reference_relation_token_list
#                 if (x in hypothesis_relation_token_list)
#             ]
#         )
#         / len(reference_relation_token_list)
#         if len(reference_relation_token_list) > 0
#         else 0.0
#     )
#     f1_score = (
#         (2 * precision * recall / (precision + recall))
#         if (precision + recall) > 0
#         else 0.0
#     )

#     return f1_score, precision, recall


# def exact_entity_token_if_rel_exists_reward(
#         hypothesis_annotation_list, reference_annotation_list
# ):
#     candidates = []
#     for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
#         candidate = []
#         for entity in annotation_list["entities"].values():
#             if not entity["relations"]:
#                 candidate.append((entity["tokens"], entity["label"]))
#             if entity["relations"]:
#                 candidate.append((entity["tokens"], entity["label"], True))

#         candidate = set(candidate)
#         candidates.append(candidate)

#     hypothesis_relation_token_list, reference_relation_token_list = candidates

#     precision = (
#         sum(
#             [
#                 1
#                 for x in hypothesis_relation_token_list
#                 if (x in reference_relation_token_list)
#             ]
#         )
#         / len(hypothesis_relation_token_list)
#         if len(hypothesis_relation_token_list) > 0
#         else 0.0
#     )
#     recall = (
#         sum(
#             [
#                 1
#                 for x in reference_relation_token_list
#                 if (x in hypothesis_relation_token_list)
#             ]
#         )
#         / len(reference_relation_token_list)
#         if len(reference_relation_token_list) > 0
#         else 0.0
#     )
#     f1_score = (
#         (2 * precision * recall / (precision + recall))
#         if (precision + recall) > 0
#         else 0.0
#     )

#     return f1_score, precision, recall


# def exact_entity_token_match_reward(
#         hypothesis_annotation_list, reference_annotation_list
# ):
#     candidates = []
#     for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
#         candidate = []
#         for entity in annotation_list["entities"].values():
#             candidate.append((entity["tokens"], entity["label"]))

#         candidate = set(candidate)
#         candidates.append(candidate)

#     hypothesis_relation_token_list, reference_relation_token_list = candidates

#     precision = (
#         sum(
#             [
#                 1
#                 for x in hypothesis_relation_token_list
#                 if (x in reference_relation_token_list)
#             ]
#         )
#         / len(hypothesis_relation_token_list)
#         if len(hypothesis_relation_token_list) > 0
#         else 0.0
#     )
#     recall = (
#         sum(
#             [
#                 1
#                 for x in reference_relation_token_list
#                 if (x in hypothesis_relation_token_list)
#             ]
#         )
#         / len(reference_relation_token_list)
#         if len(reference_relation_token_list) > 0
#         else 0.0
#     )

#     f1_score = (
#         (2 * precision * recall / (precision + recall))
#         if (precision + recall) > 0
#         else 0.0
#     )

#     return f1_score, precision, recall


# def compute_reward(
#         hypothesis_annotation_list,
#         reference_annotation_list,
#         reward_level,
# ):
#     assert reward_level in ["simple", "partial", "complete", "all"]
#     if (
#             len(hypothesis_annotation_list["entities"].keys()) == 0
#             or len(reference_annotation_list["entities"].keys()) == 0
#     ):
#         return (0., 0., 0.)
#     simple = exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list)
#     partial = exact_entity_token_if_rel_exists_reward(hypothesis_annotation_list, reference_annotation_list)
#     complete = exact_entity_token_if_all_match_reward(hypothesis_annotation_list, reference_annotation_list)
#     all = (simple, partial, complete)
    
#     return eval(reward_level)

# if __name__ == '__main__':
#     import json
#     import time

#     m = F1RadGraphv2(reward_level='partial')
#     m_og = F1RadGraph(reward_level='partial')

#     t = time.time()
#     test_hyps = ['No pleural effusion. Normal heart size.',
#               'Normal heart size.',
#               'Increased mild pulmonary edema and left basal atelectasis.',
#               'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
#               'Elevated left hemidiaphragm and blunting of the left costophrenic angle although no definite evidence of pleural effusion seen on the lateral view.',
#               ]
#     test_refs = ['No pleural effusions.',
#               'Enlarged heart.',
#               'No evidence of pneumonia. Stable cardiomegaly.',
#               'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
#               'No acute cardiopulmonary process. No significant interval change. Please note that peribronchovascular ground-glass opacities at the left greater than right lung bases seen on the prior chest CT of ___ were not appreciated on prior chest radiography on the same date and may still be present. Additionally, several pulmonary nodules measuring up to 3 mm are not not well appreciated on the current study-CT is more sensitive.'
#               ]
#     f1_radgraph = m(hyps=test_hyps, refs=test_refs)
#     f1_radgraph_og = m_og(hyps=test_hyps, refs=test_refs)
    
    
#     assert f1_radgraph[0]['f1-radgraph'] == f1_radgraph_og[0]
# 分割线
# 2025/10/17 23:24注释代码
# ==============================================================================================================================



# 离线版本代码
# """
# Custom f1 radgraph that can output precision and recall
# """

# import os
# import numpy as np
# from scipy.stats import bootstrap

# from radgraph import F1RadGraph
# from radgraph.radgraph import CACHE_DIR


# class F1RadGraphv2(F1RadGraph):
#     def __init__(self, reward_level, **kwargs):
#         # 确保归档可用（优先本地，离线不联网）
#         self._ensure_radgraph_archives()
#         super().__init__(reward_level, **kwargs)
#         assert reward_level in ["simple", "partial", "complete", "all"]

#         # 关键兜底：强制把 dataset/active_namespace 设为 "radgraph"
#         self._force_dataset_namespace()

#     def _ensure_radgraph_archives(self):
#         """确保 radgraph(.tar.gz) 已在缓存目录；优先用本地 RRG_SCORES；离线不联网。"""
#         import shutil

#         os.makedirs(CACHE_DIR, exist_ok=True)

#         def _maybe_copy(name: str):
#             dst = os.path.join(CACHE_DIR, name)
#             if os.path.exists(dst):
#                 return
#             local_dir = os.environ.get("RRG_SCORES")
#             if local_dir:
#                 src = os.path.join(local_dir, name)
#                 if os.path.exists(src):
#                     shutil.copy2(src, dst)
#                     return

#             # 离线就别联网
#             offline = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
#             if offline:
#                 # 不抛异常，交给上层看 radgraph 是否还能加载到（例如用户自己放好了）
#                 return

#             # 在线才回退 HF Hub
#             try:
#                 from huggingface_hub import hf_hub_download
#                 hf_hub_download(
#                     repo_id="StanfordAIMI/RRG_scorers",
#                     filename=name,
#                     revision="d97745aa136e5beb927da7e768e99de6ae807902",
#                     local_dir=CACHE_DIR,
#                 )
#             except Exception:
#                 # 不致命，保留原状
#                 pass

#         _maybe_copy("radgraph.tar.gz")
#         _maybe_copy("radgraph-xl.tar.gz")

#     def _force_dataset_namespace(self):
#         """
#         运行时兜底：
#         某些打包版本/源码版本组合下，config 里的 dataset 不会被正确传递，
#         导致内部 namespace 变成 None__ner_labels，从而 KeyError。
#         这里强制把模型及子模块的 namespace 置为 "radgraph"。
#         """
#         try:
#             predictor = getattr(self, "radgraph", None)
#             if predictor is None:
#                 return
#             predictor = getattr(predictor, "predictor", None)
#             if predictor is None:
#                 return
#             model = getattr(predictor, "_model", None)
#             if model is None:
#                 return

#             # 1) 模型级别：若 _dataset 缺失/为空，强制设为 "radgraph"
#             if getattr(model, "_dataset", None) in (None, "None", ""):
#                 setattr(model, "_dataset", "radgraph")

#             # 2) 子模块级别：把 _active_namespace 设为 "radgraph"
#             for sub_name in ("_ner", "_relation", "_coref", "_event", "_srl"):
#                 sub = getattr(model, sub_name, None)
#                 if sub is not None:
#                     if getattr(sub, "_active_namespace", None) in (None, "None", ""):
#                         setattr(sub, "_active_namespace", "radgraph")
#         except Exception:
#             # 兜底失败也不要影响主流程；如果失败，后续 forward 可能仍会报错，
#             # 但大多数环境下这段就能把 None__ner_labels 问题修复掉
#             pass

#     def forward(self, refs, hyps):
#         # Checks
#         assert isinstance(hyps, (str, list))
#         assert isinstance(refs, (str, list))

#         if isinstance(hyps, str):
#             hyps = [hyps]
#         if isinstance(refs, str):  # 修复：这里原先误写成 isinstance(hyps, str)
#             refs = [refs]

#         assert len(refs) == len(hyps)

#         # getting empty report list
#         number_of_reports = len(hyps)
#         empty_report_index_list = [
#             i for i in range(number_of_reports)
#             if (len(hyps[i]) == 0) or (len(refs[i]) == 0)
#         ]
#         number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)

#         # stacking all reports (hyps and refs)
#         report_list = (
#             [h for i, h in enumerate(hyps) if i not in empty_report_index_list] +
#             [r for i, r in enumerate(refs) if i not in empty_report_index_list]
#         )
#         assert len(report_list) == 2 * number_of_non_empty_reports

#         # getting annotations
#         inference_dict = self.radgraph(report_list)

#         # Compute reward
#         reward_list = []
#         hypothesis_annotation_lists = []
#         reference_annotation_lists = []
#         non_empty_report_index = 0
#         for report_index in range(number_of_reports):
#             if report_index in empty_report_index_list:
#                 reward_list.append((0., 0., 0.))
#                 continue

#             hypothesis_annotation_list = inference_dict[str(non_empty_report_index)]
#             reference_annotation_list = inference_dict[
#                 str(non_empty_report_index + number_of_non_empty_reports)
#             ]

#             reward_list.append(
#                 compute_reward(
#                     hypothesis_annotation_list,
#                     reference_annotation_list,
#                     self.reward_level,
#                 )
#             )
#             reference_annotation_lists.append(reference_annotation_list)
#             hypothesis_annotation_lists.append(hypothesis_annotation_list)
#             non_empty_report_index += 1

#         assert non_empty_report_index == number_of_non_empty_reports

#         if self.reward_level == "all":
#             reward_list = (
#                 [r[0] for r in reward_list],
#                 [r[1] for r in reward_list],
#                 [r[2] for r in reward_list],
#             )
#             mean_reward = (
#                 np.mean(reward_list[0]),
#                 np.mean(reward_list[1]),
#                 np.mean(reward_list[2]),
#             )
#         else:
#             mean_reward = np.mean(reward_list, axis=0)
#             mean_reward = {
#                 'f1-radgraph': mean_reward[0],
#                 'precision-radgraph': mean_reward[1],
#                 'recall-radgraph': mean_reward[2],
#             }

#         return (
#             mean_reward,
#             reward_list,
#             hypothesis_annotation_lists,
#             reference_annotation_lists,
#         )


# def compute_statistic(reward_list):
#     def _compute(indices):
#         r = [reward_list[i] for i in indices]
#         return np.mean(r, axis=0)[0]
#     return _compute


# def bootstrap_confidence_interval(reward_list, n_resamples: int = 500, method: str = "percentile"):
#     bs = bootstrap(
#         data=(list(range(len(reward_list))),),
#         statistic=compute_statistic(reward_list),
#         method=method,
#         paired=False,
#         vectorized=False,
#         confidence_level=0.95,
#         random_state=3,
#         n_resamples=n_resamples
#     )
#     return bs


# def exact_entity_token_if_all_match_reward(hypothesis_annotation_list, reference_annotation_list):
#     candidates = []
#     for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
#         candidate = []
#         for entity in annotation_list["entities"].values():
#             if not entity["relations"]:
#                 candidate.append((entity["tokens"], entity["label"]))
#             if entity["relations"]:
#                 candidate.extend([
#                     (entity["tokens"].lower(), entity["label"], r[0],
#                      annotation_list["entities"][r[1]]["tokens"].lower())
#                     for r in entity["relations"]
#                 ])
#         candidate = set(candidate)
#         candidates.append(candidate)

#     hypothesis_relation_token_list, reference_relation_token_list = candidates
#     precision = (
#         sum(1 for x in hypothesis_relation_token_list if (x in reference_relation_token_list))
#         / len(hypothesis_relation_token_list)
#         if len(hypothesis_relation_token_list) > 0 else 0.0
#     )
#     recall = (
#         sum(1 for x in reference_relation_token_list if (x in hypothesis_relation_token_list))
#         / len(reference_relation_token_list)
#         if len(reference_relation_token_list) > 0 else 0.0
#     )
#     f1_score = ((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)
#     return f1_score, precision, recall


# def exact_entity_token_if_rel_exists_reward(hypothesis_annotation_list, reference_annotation_list):
#     candidates = []
#     for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
#         candidate = []
#         for entity in annotation_list["entities"].values():
#             if not entity["relations"]:
#                 candidate.append((entity["tokens"], entity["label"]))
#             if entity["relations"]:
#                 candidate.append((entity["tokens"], entity["label"], True))
#         candidate = set(candidate)
#         candidates.append(candidate)

#     hypothesis_relation_token_list, reference_relation_token_list = candidates
#     precision = (
#         sum(1 for x in hypothesis_relation_token_list if (x in reference_relation_token_list))
#         / len(hypothesis_relation_token_list)
#         if len(hypothesis_relation_token_list) > 0 else 0.0
#     )
#     recall = (
#         sum(1 for x in reference_relation_token_list if (x in hypothesis_relation_token_list))
#         / len(reference_relation_token_list)
#         if len(reference_relation_token_list) > 0 else 0.0
#     )
#     f1_score = ((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)
#     return f1_score, precision, recall


# def exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list):
#     candidates = []
#     for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
#         candidate = [(entity["tokens"], entity["label"]) for entity in annotation_list["entities"].values()]
#         candidate = set(candidate)
#         candidates.append(candidate)

#     hypothesis_relation_token_list, reference_relation_token_list = candidates
#     precision = (
#         sum(1 for x in hypothesis_relation_token_list if (x in reference_relation_token_list))
#         / len(hypothesis_relation_token_list)
#         if len(hypothesis_relation_token_list) > 0 else 0.0
#     )
#     recall = (
#         sum(1 for x in reference_relation_token_list if (x in hypothesis_relation_token_list))
#         / len(reference_relation_token_list)
#         if len(reference_relation_token_list) > 0 else 0.0
#     )
#     f1_score = ((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)
#     return f1_score, precision, recall


# def compute_reward(hypothesis_annotation_list, reference_annotation_list, reward_level):
#     assert reward_level in ["simple", "partial", "complete", "all"]
#     if (
#         len(hypothesis_annotation_list["entities"].keys()) == 0
#         or len(reference_annotation_list["entities"].keys()) == 0
#     ):
#         return (0., 0., 0.)
#     simple = exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list)
#     partial = exact_entity_token_if_rel_exists_reward(hypothesis_annotation_list, reference_annotation_list)
#     complete = exact_entity_token_if_all_match_reward(hypothesis_annotation_list, reference_annotation_list)
#     all_levels = (simple, partial, complete)
#     return {"simple": simple, "partial": partial, "complete": complete, "all": all_levels}[reward_level]


# if __name__ == '__main__':
#     import time
#     from radgraph import F1RadGraph as F1RG_OG

#     m = F1RadGraphv2(reward_level='partial')
#     m_og = F1RG_OG(reward_level='partial')

#     t = time.time()
#     test_hyps = [
#         'No pleural effusion. Normal heart size.',
#         'Normal heart size.',
#         'Increased mild pulmonary edema and left basal atelectasis.',
#         'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
#         'Elevated left hemidiaphragm and blunting of the left costophrenic angle although no definite evidence of pleural effusion seen on the lateral view.',
#     ]
#     test_refs = [
#         'No pleural effusions.',
#         'Enlarged heart.',
#         'No evidence of pneumonia. Stable cardiomegaly.',
#         'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
#         'No acute cardiopulmonary process. No significant interval change. Please note that peribronchovascular ground-glass opacities at the left greater than right lung bases seen on the prior chest CT of ___ were not appreciated on prior chest radiography on the same date and may still be present. Additionally, several pulmonary nodules measuring up to 3 mm are not not well appreciated on the current study-CT is more sensitive.',
#     ]
#     f1_radgraph = m(hyps=test_hyps, refs=test_refs)
#     f1_radgraph_og = m_og(hyps=test_hyps, refs=test_refs)

#     assert f1_radgraph[0]['f1-radgraph'] == f1_radgraph_og[0]
#     print("OK")