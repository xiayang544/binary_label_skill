# -*- coding: utf-8 -*-
"""laya 候选打分（零训练检测器）：Router 封装 + laya_prob 计算（SKILL.md §17）。

约定（SKILL.md §17.3）：
- state 构造：按 task.yaml context.bert_turns 用 src/text_utils.build_bert_text
  统一拼接（context + current_turn），与 BERT 候选/LLM 标注看到同一上下文视图；
- question 配置：每版本冻结一份 <model>/<version>/standard/laya_question.yaml，
  固定一个 noul 型问题（instructions 从 label_definition.md 派生，可附 criteria）；
- 打分：Router(models={variant: path}).predict(state, question)，
  取 result["answers"][qid]["noul"] 作为正例校准概率，写入 laya_prob 字段；
- checkpoint：--model-dir / best_threshold.yaml / selected_model.yaml 中的显式引用
  > task.yaml candidates.<model>.path > Hub 默认仓库（convaiinnovations/laya）。

运行环境：额外需要 laya 包（pip install laya，或将本地源码加入 PYTHONPATH，
本地源码地址见 SKILL.md §5）。
"""

import os

from . import common, text_utils

# 每版本冻结的 question 配置（standard/ 的一部分，随版本演进、禁止回写历史版本）
QUESTION_FILENAME = "laya_question.yaml"

LAYA_SOURCE_PATH = ("/mnt/dolphinfs/hdd_pool/docker/user/<internal_mount>/"
                    "<user>/safety/laya")

_INSTALL_HINT = (
    "laya 候选需要 laya 包：pip install laya，或将本地源码加入 PYTHONPATH：\n"
    f"    export PYTHONPATH={LAYA_SOURCE_PATH}:$PYTHONPATH"
)


def _import_laya_router():
    try:
        from laya import Router  # noqa: F401
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return Router


def question_path(vdir):
    """版本目录 → standard/laya_question.yaml 路径。"""
    return os.path.join(vdir, "standard", QUESTION_FILENAME)


def load_laya_question(vdir):
    """读取 standard/laya_question.yaml → (question_id, question_dict)。

    文件格式（属于 standard，随版本冻结）::

        question_id: positive_check     # answers 键名，缺省 positive_check
        question:
          type: noul                    # 必须为 noul（正例校准概率）
          instructions: "判断当前 Turn 的 AI 输出是否属于……"
          criteria:                      # 可选：false/true 边界说明
            false: "……"
            true: "……"
    """
    path = question_path(vdir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 laya question 配置：{path}\n"
            "请按 SKILL.md §17.3 在当前版本 standard/ 下冻结 laya_question.yaml"
            "（instructions 从 label_definition.md 派生）")
    cfg = common.load_yaml(path)
    question = cfg.get("question") or {}
    if question.get("type") != "noul":
        raise ValueError(f"{path}: question.type 必须为 noul（SKILL.md §17.3），"
                         f"当前为 {question.get('type')!r}")
    if not question.get("instructions"):
        raise ValueError(f"{path}: question.instructions 不能为空")
    qid = str(cfg.get("question_id") or "positive_check")
    return qid, question


def resolve_checkpoint(model, task_cfg, override=None):
    """checkpoint 解析优先级：override（--model-dir / 产物记录）> task.yaml path > None（Hub 默认）。

    返回 (checkpoint, source)：checkpoint 为 None 表示使用 Hub 默认仓库；
    source 为来源描述，用于产物登记。
    """
    if override:
        return override, "explicit"
    entry = common.candidate_entry(model, task_cfg)
    path = entry.get("path")
    if path:
        return path, "task.yaml"
    return None, "hub-default"


class LayaScorer:
    """laya 零训练打分器：Router 批量推理，返回每条样本的正例概率（laya_prob）。

    Args:
        router: laya.Router 实例；
        variant: checkpoint 名（english / multilingual / typed-decisions）；
        question_id / question: noul 型问题及其 answers 键名；
        bert_turns: 上下文轮数（读 task.yaml context.bert_turns）；
        max_len: token 预算覆盖（长文档必须显式传，multilingual 默认 1024 截断）；
        batch_size: predict_batch 单次 forward 的样本数；
        checkpoint: checkpoint 引用（描述用，可为 None=Hub 默认）。
    """

    def __init__(self, router, variant, question_id, question, bert_turns,
                 max_len=None, batch_size=32, checkpoint=None):
        self.router = router
        self.variant = variant
        self.question_id = question_id
        self.question = question
        self.bert_turns = bert_turns
        self.max_len = max_len
        self.batch_size = batch_size
        self.checkpoint = checkpoint

    def score(self, records):
        """批量打分 → list[float]（与 records 等长的正例概率）。

        state 构造与 BERT 候选完全共用 text_utils.build_bert_text；
        缺 current_turn 的样本打分为 0.0（负例安全默认）。
        """
        if not records:
            return []
        requests = []
        for r in records:
            state = text_utils.build_bert_text(
                r.get("current_turn"), r.get("context"), self.bert_turns)
            req = {"state": state or "",
                   "questions": {self.question_id: self.question},
                   "model": self.variant}
            if self.max_len:
                req["max_len"] = int(self.max_len)
            requests.append(req)

        probs = []
        step = max(1, int(self.batch_size))
        for i in range(0, len(requests), step):
            results = self.router.predict_batch(requests[i:i + step],
                                                batch_size=self.batch_size)
            for res in results:
                answer = (res.get("answers") or {}).get(self.question_id) or {}
                probs.append(float(answer.get("noul", 0.0) or 0.0))
        return probs

    def describe(self):
        """打分器描述（写入 best_threshold / selected_model 等产物）。"""
        return {
            "candidate_type": common.CANDIDATE_TYPE_LAYA,
            "variant": self.variant,
            "checkpoint": self.checkpoint,
            "question_id": self.question_id,
            "max_len": self.max_len,
            "bert_turns": self.bert_turns,
        }


def build_laya_scorer(root, model, version=None, task_cfg=None, checkpoint=None,
                      question=None, question_id=None, batch_size=32, device=None):
    """构造 LayaScorer。

    Args:
        root / model / task_cfg: 项目根目录与候选（type: laya）；
        version: 给定时从 <model>/<version>/standard/laya_question.yaml 读问题配置；
        checkpoint: 显式 checkpoint（--model-dir / 产物记录）；缺省按 task.yaml path / Hub；
        question / question_id: version 为空时的显式问题配置（二选一路径）；
        device: "cuda" / "cpu"，缺省交给 laya 自动选择。
    """
    task_cfg = task_cfg or common.load_task_config(root)
    entry = common.candidate_entry(model, task_cfg)
    if not common.is_laya_candidate(model, task_cfg):
        raise ValueError(f"候选 {model} 不是 type: laya（SKILL.md §17）")

    variant = str(entry.get("variant") or "multilingual")
    max_len = entry.get("max_len")

    if version:
        vdir = common.version_dir(root, version, model=model, task_cfg=task_cfg)
        qid, q = load_laya_question(vdir)
    elif question:
        qid = str(question_id or "positive_check")
        q = question
        if q.get("type") != "noul" or not q.get("instructions"):
            raise ValueError("显式传入的 question 必须为含 instructions 的 noul 型问题")
    else:
        raise ValueError("需要 --version（读 standard/laya_question.yaml）或显式 question 配置")

    ckpt, _ = resolve_checkpoint(model, task_cfg, override=checkpoint)
    _, _, bert_turns = common.task_ctx(root, "bert_turns", 1)

    Router = _import_laya_router()
    # ckpt 为空 → 使用 Router 内置默认仓库（convaiinnovations/laya bundle）
    router = Router(models={variant: ckpt} if ckpt else None,
                    device=device, max_loaded=1)
    print(f"[laya] variant={variant} checkpoint={ckpt or 'hub-default'} "
          f"max_len={max_len} question_id={qid} bert_turns={bert_turns}")
    return LayaScorer(router, variant, qid, q, bert_turns,
                      max_len=max_len, batch_size=batch_size, checkpoint=ckpt)
