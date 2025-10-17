"""
Custom f1 radgraph that can output precision and recall
"""

from radgraph import F1RadGraph
import numpy as np
from scipy.stats import bootstrap

import os
from radgraph.radgraph import CACHE_DIR
from huggingface_hub import hf_hub_download


class F1RadGraphv2(F1RadGraph):
    def __init__(
            self,
            reward_level,
            **kwargs
    ):

        self._download_radgraph()
        super().__init__(reward_level, **kwargs)
        assert reward_level in ["simple", "partial", "complete", "all"]

    def _download_radgraph(self):
        if not os.path.exists(os.path.join(CACHE_DIR, "radgraph.tar.gz")):
            os.makedirs(CACHE_DIR, exist_ok=True)
            hf_hub_download(
                repo_id="StanfordAIMI/RRG_scorers",
                filename="radgraph.tar.gz",
                revision="d97745aa136e5beb927da7e768e99de6ae807902",
                local_dir=CACHE_DIR,
            )

    def forward(self, refs, hyps):
        # Checks
        assert isinstance(hyps, str) or isinstance(hyps, list)
        assert isinstance(refs, str) or isinstance(refs, list)

        if isinstance(hyps, str):
            hyps = [hyps]
        if isinstance(hyps, str):
            refs = [refs]

        assert len(refs) == len(hyps)

        # getting empty report list
        number_of_reports = len(hyps)
        empty_report_index_list = [i for i in range(number_of_reports) if (len(hyps[i]) == 0) or (len(refs[i]) == 0)]
        number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)

        # stacking all reports (hyps and refs)
        report_list = [
                          hypothesis_report
                          for i, hypothesis_report in enumerate(hyps)
                          if i not in empty_report_index_list
                      ] + [
                          reference_report
                          for i, reference_report in enumerate(refs)
                          if i not in empty_report_index_list
                      ]

        assert len(report_list) == 2 * number_of_non_empty_reports

        # getting annotations
        inference_dict = self.radgraph(report_list)

        # Compute reward
        reward_list = []
        hypothesis_annotation_lists = []
        reference_annotation_lists = []
        non_empty_report_index = 0
        for report_index in range(number_of_reports):
            if report_index in empty_report_index_list:
                reward_list.append((0., 0., 0.))
                
                continue

            hypothesis_annotation_list = inference_dict[str(non_empty_report_index)]
            reference_annotation_list = inference_dict[
                str(non_empty_report_index + number_of_non_empty_reports)
            ]

            reward_list.append(
                compute_reward(
                    hypothesis_annotation_list,
                    reference_annotation_list,
                    self.reward_level,
                )
            )
            reference_annotation_lists.append(reference_annotation_list)
            hypothesis_annotation_lists.append(hypothesis_annotation_list)
            non_empty_report_index += 1

        assert non_empty_report_index == number_of_non_empty_reports
        
        if self.reward_level == "all":
            reward_list = ([r[0] for r in reward_list], [r[1] for r in reward_list], [r[2] for r in reward_list])
            mean_reward = (np.mean(reward_list[0]), np.mean(reward_list[1]), np.mean(reward_list[2]))
        else:
            mean_reward =np.mean(reward_list, axis=0)
            mean_reward = {'f1-radgraph': mean_reward[0], 'precision-radgraph': mean_reward[1], 'recall-radgraph': mean_reward[2]}

        return (
            mean_reward,
            reward_list,
            hypothesis_annotation_lists,
            reference_annotation_lists,
        )


def compute_statistic(reward_list):
    def _compute(indices):
        r = [reward_list[i] for i in indices]
        return np.mean(r, axis=0)[0]

    return _compute


def bootstrap_confidence_interval(
        reward_list, n_resamples: int = 500, method: str = "percentile",
    ):
    bs = bootstrap(
        data=(list(range(len(reward_list))),),
        statistic=compute_statistic(reward_list),
        method=method,
        paired=False,
        vectorized=False,
        confidence_level=0.95,
        random_state=3,
        n_resamples=n_resamples
    )
    return bs


def exact_entity_token_if_all_match_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.extend([(entity["tokens"].lower(),
                                   entity["label"],
                                   r[0],
                                   annotation_list["entities"][r[1]]["tokens"].lower())
                                  for r in entity["relations"]]
                                 )

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates
    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, precision, recall


def exact_entity_token_if_rel_exists_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.append((entity["tokens"], entity["label"], True))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, precision, recall


def exact_entity_token_match_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            candidate.append((entity["tokens"], entity["label"]))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )

    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, precision, recall


def compute_reward(
        hypothesis_annotation_list,
        reference_annotation_list,
        reward_level,
):
    assert reward_level in ["simple", "partial", "complete", "all"]
    if (
            len(hypothesis_annotation_list["entities"].keys()) == 0
            or len(reference_annotation_list["entities"].keys()) == 0
    ):
        return (0., 0., 0.)
    simple = exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list)
    partial = exact_entity_token_if_rel_exists_reward(hypothesis_annotation_list, reference_annotation_list)
    complete = exact_entity_token_if_all_match_reward(hypothesis_annotation_list, reference_annotation_list)
    all = (simple, partial, complete)
    
    return eval(reward_level)

if __name__ == '__main__':
    import json
    import time

    m = F1RadGraphv2(reward_level='partial')
    m_og = F1RadGraph(reward_level='partial')

    t = time.time()
    test_hyps = ['No pleural effusion. Normal heart size.',
              'Normal heart size.',
              'Increased mild pulmonary edema and left basal atelectasis.',
              'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
              'Elevated left hemidiaphragm and blunting of the left costophrenic angle although no definite evidence of pleural effusion seen on the lateral view.',
              ]
    test_refs = ['No pleural effusions.',
              'Enlarged heart.',
              'No evidence of pneumonia. Stable cardiomegaly.',
              'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
              'No acute cardiopulmonary process. No significant interval change. Please note that peribronchovascular ground-glass opacities at the left greater than right lung bases seen on the prior chest CT of ___ were not appreciated on prior chest radiography on the same date and may still be present. Additionally, several pulmonary nodules measuring up to 3 mm are not not well appreciated on the current study-CT is more sensitive.'
              ]
    f1_radgraph = m(hyps=test_hyps, refs=test_refs)
    f1_radgraph_og = m_og(hyps=test_hyps, refs=test_refs)
    
    
    assert f1_radgraph[0]['f1-radgraph'] == f1_radgraph_og[0]


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