# -*- coding: utf-8 -*-
"""Kev checkpoint 推理适配：将 Turn 样本映射为 choice 问题并返回正例概率。"""

import os

from . import common, text_utils

QUESTION_FILENAME = "kev_question.yaml"


def resolve_checkpoint(model, task_cfg, override=None):
    """显式路径优先，其次 task.yaml 中的 checkpoint 路径。"""
    path = override or common.candidate_entry(model, task_cfg).get("path")
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(f"Kev checkpoint 目录不存在：{path!r}（请设置 --model-dir 或 models.candidates.{model}.path）")
    return path


def load_question(vdir):
    path = os.path.join(vdir, "standard", QUESTION_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Kev 问题配置不存在：{path}")
    cfg = common.load_yaml(path)
    question = cfg.get("question") or {}
    criteria = question.get("criteria")
    positive_key = cfg.get("positive_key")
    if (question.get("type") != "choice" or not question.get("instructions")
            or not isinstance(criteria, dict) or len(criteria) != 2
            or positive_key not in criteria):
        raise ValueError(f"{path}: 需要含 instructions、两个 criteria 和 positive_key 的 choice 问题")
    return str(cfg.get("question_id") or "positive_check"), positive_key, question


class KevScorer:
    def __init__(self, predictor, checkpoint, qid, positive_key, question, bert_turns, max_state):
        self.predictor = predictor
        self.checkpoint = checkpoint
        self.qid = qid
        self.positive_key = positive_key
        self.question = {**question, "label": positive_key, "src": "binary_label_skill"}
        self.bert_turns = bert_turns
        self.max_state = max_state

    def score(self, records):
        scores = []
        for rec in records:
            state = text_utils.build_bert_text(rec.get("current_turn"), rec.get("context"), self.bert_turns)
            request = {"state": state or "", "questions": {self.qid: self.question}}
            result = self.predictor(request)
            scores.append(float(result["probabilities"][self.qid][self.positive_key]))
        return scores

    def describe(self):
        return {"candidate_type": common.CANDIDATE_TYPE_KEV,
                "checkpoint": self.checkpoint, "kev_question": QUESTION_FILENAME,
                "question_id": self.qid, "positive_key": self.positive_key,
                "max_state": self.max_state, "bert_turns": self.bert_turns}


def build_kev_scorer(root, model, version, task_cfg=None, checkpoint=None):
    task_cfg = task_cfg or common.load_task_config(root)
    if common.candidate_type(model, task_cfg) != common.CANDIDATE_TYPE_KEV:
        raise ValueError(f"候选 {model} 不是 type: kev")
    path = resolve_checkpoint(model, task_cfg, checkpoint)
    vdir = common.version_dir(root, version, model=model, task_cfg=task_cfg)
    qid, positive_key, question = load_question(vdir)
    entry = common.candidate_entry(model, task_cfg)
    max_state = int(entry.get("max_state") or 4096)
    _, _, bert_turns = common.task_ctx(root, "bert_turns", 1)
    try:
        import torch
        from kev.model import training_context
        from kev.predictors import LocalPredictor
    except ImportError as exc:
        raise ImportError("Kev 候选需要安装 kev 包（pip install -e <kev 源码目录>）") from exc
    context = training_context(max_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = LocalPredictor(path, device, context=context)
    return KevScorer(predictor, path, qid, positive_key, question, bert_turns, max_state)
