# -*- coding: utf-8 -*-
"""binary_label_skill 公共工具层。

所有 scripts/ 下的入口脚本共享：
- task.yaml / status.yaml 读写与版本目录定位；
- JSONL + Excel 双格式数据读写（字段遵循 templates/data_schema.yaml）；
- 产物登记（写回 status.yaml 的 artifacts，支持断点续跑时的完整性检查）。

约定：
- 项目根目录 root：包含 task.yaml、data/gold/、v_0/ v_1/ ... 的工作目录；
- 所有跨脚本传递的数据均为 list[dict]，dict 字段名以 data_schema.yaml 为准。
"""

import hashlib
import json
import os
from datetime import datetime

import pandas as pd
import yaml

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Excel 单表行数上限，超过时只导出抽样，完整数据保留在 JSONL
MAX_XLSX_ROWS = 1_000_000
XLSX_SAMPLE_ROWS = 100_000


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------- YAML / task.yaml ------------------------- #


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def load_task_config(root):
    path = os.path.join(root, "task.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 task.yaml：{path}（首次冷启动请按 templates/task.yaml 初始化）")
    return load_yaml(path)


def task_ctx(root, turns_key=None, default_turns=None):
    """读取 task.yaml 中 context 配置；turns_key: llm_label_turns / bert_turns / regex_turns。"""
    cfg = load_task_config(root)
    ctx = cfg.get("context", {}) or {}
    turns = ctx.get(turns_key, default_turns) if turns_key else None
    return cfg, ctx, turns


# ------------------------- 版本目录 ------------------------- #
#
# 目录布局：每个候选模型一条完全独立的流水线，互不干扰。
#   <root>/task.yaml           任务级配置
#   <root>/data/gold/          全局 Gold（人工核验真实数据全量累积；
#                              需求变更经 build_gold.py --update-labels 改标留审计）
#   <root>/<model>/v_<k>/      该模型的完整流水线版本目录：
#                              standard/ prefilter/ labeling/ synthesis/
#                              dataset/ training/ metrics/ scan/
#                              manifest.yaml status.yaml report.md
#
# <model> 为 task.yaml models.candidates 的 key（可用候选配置中的
# dir 字段覆盖，如 h_256）。task.yaml 可启用 1 个或多个模型，
# 多模型 = 多条独立流水线，逐模型分别执行工作流。


def model_dir_name(model, task_cfg=None):
    """候选模型 key → 目录名。candidates.<key>.dir 可覆盖（默认取 key 本身）。"""
    if task_cfg is None:
        return model
    cand = (task_cfg.get("models", {}) or {}).get("candidates", {}) or {}
    entry = cand.get(model)
    if isinstance(entry, dict) and entry.get("dir"):
        return str(entry["dir"])
    return model


def enabled_models(task_cfg):
    """task.yaml 中 enabled 的候选模型 key 列表。"""
    cand = (task_cfg.get("models", {}) or {}).get("candidates", {}) or {}
    return [k for k, v in cand.items()
            if not isinstance(v, dict) or v.get("enabled", True)]


def resolve_model(root, model=None, task_cfg=None):
    """解析当前流水线的目标模型。

    显式指定 > task.yaml 唯一 enabled 候选；
    多个 enabled 且未指定时报错（单次工作流只针对一个模型执行）。
    """
    if model:
        return model
    cfg = task_cfg or load_task_config(root)
    models = enabled_models(cfg)
    if len(models) == 1:
        return models[0]
    raise ValueError(
        f"task.yaml 启用了 {len(models)} 个候选模型（{models}），"
        "请通过 --model 显式指定本次执行的模型（单次工作流只针对一个模型）")


def version_dir(root, version, model=None, task_cfg=None):
    """版本目录定位：<root>/<model_dir>/<version>。

    model 缺省时自动解析（唯一 enabled 候选；多个则报错）。
    """
    m = resolve_model(root, model, task_cfg)
    cfg = task_cfg or load_task_config(root)
    return os.path.join(root, model_dir_name(m, cfg), version)


def find_max_version(root, model=None, task_cfg=None):
    """返回当前最大版本号（如 'v_3'）；无版本时返回 None。

    指定 model 时只扫该模型的版本链（各模型版本号独立计数）；
    未指定时扫根目录下所有 v_*（旧布局兼容）。
    """
    best, best_k = None, -1
    if model:
        bases = [os.path.join(root, model_dir_name(model, task_cfg))]
    else:
        if not os.path.isdir(root):
            return None
        bases = [root]
    for base in bases:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            if name.startswith("v_") and name[2:].isdigit():
                k = int(name[2:])
                if k > best_k:
                    best, best_k = name, k
    return best


# ------------------------- 候选类型与打分字段 ------------------------- #
#
# SKILL.md §17：候选配置含 type: laya 时为「零训练检测器」（不做合成/训练/超参网格，
# 只做 threshold 搜索）；缺省视为 BERT 类（微调流水线）。
# 两种候选的打分结果字段名不同：
#   bert → bert_score / bert_prediction / bert_threshold
#   laya → laya_prob  / laya_prediction / laya_threshold
# 下游脚本统一通过 score_fields()/get_*() 读取，禁止直接硬编码字段名。

CANDIDATE_TYPE_BERT = "bert"
CANDIDATE_TYPE_LAYA = "laya"
CANDIDATE_TYPE_KEV = "kev"

SCORE_FIELDS = {
    CANDIDATE_TYPE_BERT: {"score": "bert_score",
                          "prediction": "bert_prediction",
                          "threshold": "bert_threshold"},
    CANDIDATE_TYPE_LAYA: {"score": "laya_prob",
                          "prediction": "laya_prediction",
                          "threshold": "laya_threshold"},
    CANDIDATE_TYPE_KEV: {"score": "kev_prob",
                         "prediction": "kev_prediction",
                         "threshold": "kev_threshold"},
}


def candidate_entry(model, task_cfg):
    """task.yaml models.candidates.<model> 的配置 dict（不存在时返回 {}）。"""
    cand = (task_cfg.get("models", {}) or {}).get("candidates", {}) or {}
    entry = cand.get(model)
    return entry if isinstance(entry, dict) else {}


def candidate_type(model, task_cfg):
    """候选类型：bert（缺省）、laya 或 kev。"""
    entry = candidate_entry(model, task_cfg)
    ctype = str(entry.get("type") or CANDIDATE_TYPE_BERT).lower()
    if ctype not in SCORE_FIELDS:
        raise ValueError(f"候选 {model} 的 type 无效：{ctype!r}（支持 {', '.join(SCORE_FIELDS)}）")
    return ctype


def is_laya_candidate(model, task_cfg):
    """该候选是否为 laya 零训练检测器。"""
    return candidate_type(model, task_cfg) == CANDIDATE_TYPE_LAYA


def is_external_candidate(model, task_cfg):
    """已有 checkpoint、仅搜索阈值的候选（laya / kev）。"""
    return candidate_type(model, task_cfg) in (CANDIDATE_TYPE_LAYA, CANDIDATE_TYPE_KEV)


def score_fields(ctype):
    """候选类型 → 打分字段名映射 {score, prediction, threshold}。"""
    return SCORE_FIELDS[ctype]


def get_score(rec, ctype=None):
    """读取样本打分（正例概率）。ctype 缺省时按字段存在性识别。"""
    if ctype is None:
        ctype = next((t for t in (CANDIDATE_TYPE_KEV, CANDIDATE_TYPE_LAYA)
                      if rec.get(SCORE_FIELDS[t]["score"]) is not None), CANDIDATE_TYPE_BERT)
    return rec.get(score_fields(ctype)["score"])


def get_prediction(rec, ctype=None):
    """读取样本预测标签（positive/negative）。ctype 缺省时按字段存在性识别。"""
    if ctype is None:
        ctype = next((t for t in (CANDIDATE_TYPE_KEV, CANDIDATE_TYPE_LAYA)
                      if rec.get(SCORE_FIELDS[t]["score"]) is not None), CANDIDATE_TYPE_BERT)
    return rec.get(score_fields(ctype)["prediction"])


def get_threshold(rec, ctype=None):
    """读取样本携带的判定阈值。ctype 缺省时按字段存在性识别。"""
    if ctype is None:
        ctype = next((t for t in (CANDIDATE_TYPE_KEV, CANDIDATE_TYPE_LAYA)
                      if rec.get(SCORE_FIELDS[t]["score"]) is not None), CANDIDATE_TYPE_BERT)
    return rec.get(score_fields(ctype)["threshold"])


# ------------------------- JSONL / Excel 双格式读写 ------------------------- #


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load_records(path):
    """按扩展名自动读取 jsonl / xlsx / csv，统一返回 list[dict]（NaN → None）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=object)
    df = df.where(pd.notna(df), None)
    return df.to_dict("records")


def save_records(records, base):
    """双格式保存：<base>.jsonl（程序处理主格式）+ <base>.xlsx（人工查看）。

    - 超过 Excel 行数上限时，xlsx 只导出随机抽样，完整数据保留在 JSONL；
    - 缺 openpyxl 时降级写 .csv（JSONL 始终完整，不影响程序处理）。
    """
    base = base[:-5] if base.endswith(".jsonl") else base
    save_jsonl(records, base + ".jsonl")
    df = pd.DataFrame(records)
    if len(df) > MAX_XLSX_ROWS:
        df = df.sample(n=min(XLSX_SAMPLE_ROWS, len(df)), random_state=42)
        print(f"[WARN] {base}.xlsx 超过 {MAX_XLSX_ROWS} 行，仅导出抽样 {len(df)} 行，完整数据见 JSONL")
    try:
        df.to_excel(base + ".xlsx", index=False)
    except ImportError:
        df.to_csv(base + ".csv", index=False)
        print(f"[WARN] 缺少 openpyxl，{base}.xlsx 降级为 {base}.csv（JSONL 不受影响）")
    return base + ".jsonl", base + ".xlsx"


def iter_input_files(paths):
    """展开输入路径：文件直接返回；目录则收集其下全部 .jsonl（按文件名排序）。"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(
                os.path.join(p, f) for f in os.listdir(p) if f.endswith(".jsonl")))
        else:
            files.append(p)
    return files


def load_many(paths):
    """加载多个 jsonl/xlsx 并按 sample_id 去重（先出现者保留）。"""
    seen, out = set(), []
    for p in iter_input_files(paths):
        for r in load_records(p):
            key = r.get("sample_id") or f"{r.get('conversation_id')}#{r.get('turn_id')}"
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out


# ------------------------- status.yaml 维护 ------------------------- #


def load_status(root, version, model=None):
    path = os.path.join(version_dir(root, version, model=model), "status.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}，请先按 workflow 初始化版本目录")
    return load_yaml(path)


def update_status(root, version, *, model=None, status=None, current_step=None,
                  completed_step=None, artifact=None, event=None,
                  pause_waiting_for=None, pause_instruction=None, clear_pause=False):
    """增量更新某模型某版本的 status.yaml（<model>/<version>/status.yaml）。

    - model 缺省时自动解析（唯一 enabled 候选；多个则报错）；
    - status / current_step：直接覆盖；
    - completed_step：追加到 completed_steps（去重，保持顺序）；
    - artifact：(相对版本目录的路径, 行数或 None)：登记到 artifacts 检查表；
    - event：字符串，追加到 event_log（自动加时间戳）；
    - pause_waiting_for / pause_instruction：填写 pause_info（进入 waiting_* 状态时）；
    - clear_pause：清空 pause_info（恢复执行时）。
    """
    model = resolve_model(root, model)
    path = os.path.join(version_dir(root, version, model=model), "status.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    st = load_yaml(path) if os.path.exists(path) else {"version": version}
    st.setdefault("version", version)
    st["model"] = model
    st.setdefault("completed_steps", [])
    st.setdefault("artifacts", {})
    st.setdefault("event_log", [])
    st.setdefault("pause_info", {"waiting_for": None, "instruction": None, "paused_at": None})

    if status is not None:
        st["status"] = status
    if current_step is not None:
        st["current_step"] = current_step
    if completed_step is not None:
        if completed_step not in st["completed_steps"]:
            st["completed_steps"].append(completed_step)
    if artifact is not None:
        rel_path, rows = artifact
        entry = {"exists": True}
        if rows is not None:
            entry["rows"] = rows
        st["artifacts"][rel_path] = entry
    if clear_pause:
        st["pause_info"] = {"waiting_for": None, "instruction": None, "paused_at": None}
    else:
        if pause_waiting_for is not None:
            st["pause_info"]["waiting_for"] = pause_waiting_for
        if pause_instruction is not None:
            st["pause_info"]["instruction"] = pause_instruction
        if pause_waiting_for is not None or pause_instruction is not None:
            st["pause_info"]["paused_at"] = now_str()
    if event is not None:
        st["event_log"].append({"time": now_str(), "event": event})
    st["updated_at"] = now_str()
    save_yaml(st, path)
    return st


def check_artifacts(root, version, model=None):
    """恢复执行前校验 artifacts 检查表，返回缺失产物列表。"""
    st = load_status(root, version, model=model)
    missing = []
    for rel, entry in (st.get("artifacts") or {}).items():
        p = os.path.join(version_dir(root, version, model=model), rel)
        if not os.path.exists(p):
            missing.append(rel)
    return missing


# ------------------------- 其他工具 ------------------------- #


def file_md5(path, n=8):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:n]


def slug(name):
    """模型名/文件名安全化：LongCat-Flash-Chat → longcat_flash。"""
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def count_by(records, field):
    out = {}
    for r in records:
        k = r.get(field)
        k = "<empty>" if k in (None, "", []) else str(k)
        out[k] = out.get(k, 0) + 1
    return out
