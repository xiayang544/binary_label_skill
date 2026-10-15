# -*- coding: utf-8 -*-
"""线上全量数据解析：三类数据源统一产出 Turn 级待检测样本。

解析逻辑参考 law1/_扫描辅助_数据解析与推理.py，输出字段对齐
templates/data_schema.yaml 的 base_fields：

    sample_id        record_id（全局唯一，如 "85024495#turn3" / "agentic#2662041"）
    conversation_id  对话 ID
    turn_id          当前 Turn 序号
    business_scene   outbound_call / wecom / rider_assistant
    current_turn     待检测的 AI 输出
    context          完整历史渲染文本（各阶段再按轮数截断）
    conversation_turns  结构化历史 [[role, text], ...]（供下游按任意轮数重建）

数据源与文件模式：
    *_recruitment_full_cc2.json  招聘外呼通话记录（dialog 事件流）
    agentic_reply_*.json         企微 agentic 回复（model_input / model_output）
    *.csv                        骑手智能助手（user_query / final_response）
"""

import glob
import json
import os
import re

BIZ_SCENE_CALL = "outbound_call"
BIZ_SCENE_AGENTIC = "wecom"
BIZ_SCENE_RIDER = "rider_assistant"


# ------------------------- 通用解析工具 ------------------------- #


def robust_json_loads(s):
    """鲁棒 JSON 解析：兼容 markdown 代码块包裹、常见非法转义。"""
    if not isinstance(s, str):
        return s
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()
    s = s.lstrip("=").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return json.loads(re.sub(r"(?<!\\)\n", "\\\\n", s))
    except Exception:
        return None


def extract_reply_field(raw_output):
    """从 model_output 中提取 reply 文本；解析失败用正则兜底，最后退化为原文。"""
    parsed = robust_json_loads(raw_output)
    if isinstance(parsed, dict) and "reply" in parsed:
        return str(parsed["reply"])
    if isinstance(raw_output, str):
        m = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output, flags=re.DOTALL)
        if m:
            return m.group(1).encode("utf-8").decode("unicode_escape", errors="ignore")
        return raw_output.strip()
    return None


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
# 无连字符日期（rider_agent_20260928_20260928_distinct.json 等按日导出文件）
_DATE_RE_COMPACT = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")


def _normalize_date(s):
    """文件名 → YYYY-MM-DD；兼容 YYYY-MM-DD 与 YYYYMMDD 两种写法。"""
    m = _DATE_RE.search(s)
    if m:
        return m.group(1)
    m = _DATE_RE_COMPACT.search(s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def extract_data_date(file_path, rec=None):
    """样本数据日期（YYYY-MM-DD）。

    线上数据按日期文件增长（如 2026-05-01_recruitment_full_cc2.json）。
    优先取文件名中的日期；提取不到时回退记录内 ctime 的日期部分；均无则 None。
    """
    m = _normalize_date(os.path.basename(file_path))
    if m:
        return m
    if rec is not None:
        ctime = rec.get("ctime") or (rec.get("extra") or {}).get("ctime")
        if isinstance(ctime, str):
            m = _DATE_RE.search(ctime)
            if m:
                return m.group(1)
    return None


def _base_sample(sample_id, conv_id, turn_id, biz_scene, history_turns, current_text,
                 source_file, extra=None, data_date=None):
    return {
        "sample_id": str(sample_id),
        "conversation_id": str(conv_id),
        "turn_id": turn_id,
        "business_scene": biz_scene,
        "current_turn": current_text,
        "conversation_turns": [[r, t] for r, t in history_turns],
        "context": "\n".join(
            f"{'AI' if r == 'assistant' else '用户'}: {t}" for r, t in history_turns if t),
        "source_file": source_file,
        "data_date": data_date,
        "extra": extra or {},
    }


# ------------------------- 数据源 1：招聘外呼通话记录 ------------------------- #


def iter_call_samples(file_path):
    """dialog 事件流：每个 machine 发言作为一个 Turn 级样本，历史为其前全部轮。"""
    data_date = extract_data_date(file_path)
    with open(file_path, encoding="utf-8") as f:
        records = json.load(f)
    for rec in records:
        call_id = rec.get("id") or rec.get("record_id")
        dialog_raw = rec.get("dialog")
        if not dialog_raw:
            continue
        try:
            events = json.loads(dialog_raw) if isinstance(dialog_raw, str) else dialog_raw
        except Exception:
            continue
        history, machine_idx = [], 0
        for ev in events:
            src, msg = ev.get("source"), ev.get("message")
            if not (isinstance(msg, str) and msg.strip()):
                continue
            if src == "machine":
                yield _base_sample(
                    f"{call_id}#turn{machine_idx}", call_id, machine_idx,
                    BIZ_SCENE_CALL, history, msg.strip(),
                    os.path.basename(file_path), {"ctime": rec.get("ctime")},
                    data_date=data_date or extract_data_date(file_path, rec),
                )
                machine_idx += 1
                history.append(("assistant", msg.strip()))
            elif src == "user":
                history.append(("user", msg.strip()))


# ------------------------- 数据源 2：企微 agentic 回复 ------------------------- #


def iter_agentic_samples(file_path):
    with open(file_path, encoding="utf-8") as f:
        records = json.load(f)
    for rec in records:
        rec_id = rec.get("id")
        reply = extract_reply_field(rec.get("model_output"))
        if not reply or not str(reply).strip():
            continue
        history = []
        try:
            mi = rec.get("model_input")
            messages = json.loads(mi) if isinstance(mi, str) else mi
            user_msgs = [m for m in messages if m.get("role") == "user"] if isinstance(messages, list) else []
            if user_msgs:
                inner = robust_json_loads(user_msgs[-1].get("content"))
                if isinstance(inner, dict):
                    dh = inner.get("input", {}).get("dialogue_history")
                    if isinstance(dh, list):
                        history = [(t.get("role"), t.get("text")) for t in dh]
        except Exception:
            history = []
        yield _base_sample(
            f"agentic#{rec_id}", rec_id, None,
            BIZ_SCENE_AGENTIC, history, str(reply).strip(),
            os.path.basename(file_path),
            {"ctime": rec.get("ctime"), "model_type": rec.get("model_type")},
            data_date=extract_data_date(file_path, rec),
        )


# ------------------------- 数据源 3：骑手智能助手 CSV ------------------------- #


def iter_rider_csv_samples(file_path, chunksize=10000):
    import pandas as pd

    data_date = extract_data_date(file_path)
    row_idx = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            row_idx += 1
            final_response = row.get("final_response")
            if final_response is None or not str(final_response).strip():
                continue
            history = []
            user_query = row.get("user_query")
            if user_query is not None and str(user_query).strip():
                history.append(("user", str(user_query).strip()))
            yield _base_sample(
                f"rider_agent#row{row_idx}", row_idx, None,
                BIZ_SCENE_RIDER, history, str(final_response).strip(),
                os.path.basename(file_path), data_date=data_date,
            )


# ------------------------- 数据源 3b：骑手智能助手按日 JSON ------------------------- #


def iter_rider_json_samples(file_path):
    """rider_agent_YYYYMMDD_YYYYMMDD_distinct.json：按日导出的骑手智能助手
    会话记录（list[dict]，字段与 CSV 同构：user_query / final_response）。

    sample_id 取 message_id（全局唯一），conversation_id 取 session_id；
    旧的 rider_agent#rowN（CSV）为文件内行号，跨文件不唯一，仅历史数据沿用。
    """
    data_date = extract_data_date(file_path)
    with open(file_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{file_path}: 顶层应为 JSON 数组，实际为 {type(rows).__name__}")
    for row in rows:
        if not isinstance(row, dict):
            continue
        final_response = row.get("final_response")
        if final_response is None or not str(final_response).strip():
            continue
        history = []
        user_query = row.get("user_query")
        if user_query is not None and str(user_query).strip():
            history.append(("user", str(user_query).strip()))
        yield _base_sample(
            str(row.get("message_id") or f"rider_json#{file_path}#{id(row)}"),
            str(row.get("session_id") or ""), None,
            BIZ_SCENE_RIDER, history, str(final_response).strip(),
            os.path.basename(file_path), data_date=data_date,
        )


# ------------------------- 统一入口 ------------------------- #


def iter_online_samples(data_dir, include_patterns=None, since=None, until=None):
    """遍历数据目录，统一产出样本。

    include_patterns：文件名子串白名单。
    since/until：数据日期过滤（YYYY-MM-DD，含边界）——按文件名日期在文件级过滤，
    用于增量初筛（线上数据按日期文件增长，只处理新日期）。
    无日期的文件在指定 since/until 时跳过。
    """
    def _date_ok(fp):
        if since is None and until is None:
            return True
        d = extract_data_date(fp)
        if d is None:
            return False
        if since is not None and d < since:
            return False
        if until is not None and d > until:
            return False
        return True

    call_files = sorted(glob.glob(os.path.join(data_dir, "*_recruitment_full_cc2.json")))
    agentic_files = sorted(glob.glob(os.path.join(data_dir, "agentic_reply_*.json")))
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    rider_json_files = sorted(glob.glob(os.path.join(data_dir, "rider_agent_*.json")))

    if include_patterns:
        def _match(fp):
            return any(p in os.path.basename(fp) for p in include_patterns)
        call_files = [f for f in call_files if _match(f)]
        agentic_files = [f for f in agentic_files if _match(f)]
        csv_files = [f for f in csv_files if _match(f)]
        rider_json_files = [f for f in rider_json_files if _match(f)]

    n_all = len(call_files) + len(agentic_files) + len(csv_files) + len(rider_json_files)
    call_files = [f for f in call_files if _date_ok(f)]
    agentic_files = [f for f in agentic_files if _date_ok(f)]
    csv_files = [f for f in csv_files if _date_ok(f)]
    rider_json_files = [f for f in rider_json_files if _date_ok(f)]
    n_kept = (len(call_files) + len(agentic_files) + len(csv_files)
              + len(rider_json_files))
    range_desc = f"，日期范围 [{since or '*'}, {until or '*'}] 保留 {n_kept}/{n_all}" \
        if (since or until) else ""
    print(f"待处理文件：call {len(call_files)}，agentic {len(agentic_files)}，"
          f"csv {len(csv_files)}，rider_json {len(rider_json_files)}{range_desc}")

    for fp in call_files:
        try:
            yield from iter_call_samples(fp)
        except Exception as e:
            print(f"[WARN] 解析失败，跳过 {fp}: {e}")
    for fp in agentic_files:
        try:
            yield from iter_agentic_samples(fp)
        except Exception as e:
            print(f"[WARN] 解析失败，跳过 {fp}: {e}")
    for fp in csv_files:
        try:
            yield from iter_rider_csv_samples(fp)
        except Exception as e:
            print(f"[WARN] 解析失败，跳过 {fp}: {e}")
    for fp in rider_json_files:
        try:
            yield from iter_rider_json_samples(fp)
        except Exception as e:
            print(f"[WARN] 解析失败，跳过 {fp}: {e}")
