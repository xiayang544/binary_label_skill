# -*- coding: utf-8 -*-
"""上下文与文本构造（训练/推理/LLM 标注共用，保证拼接逻辑完全一致）。

核心约定（与参考实现 law1/_扫描辅助_数据解析与推理.py 一致）：
- 对话历史统一表示为 turns = [(role, text), ...]，role ∈ {assistant, user}；
- 渲染为 "AI: xxx\\n用户: xxx" 行式文本，每轮一行；
- BERT 输入 = current_turn 前置 + "\\n" + 截断后的 context：
  tokenizer truncation 从右侧截断，current_turn 放最前保证不被截掉；
- 不同阶段使用不同上下文轮数（regex_turns / llm_label_turns / bert_turns），
  轮数一律来自 task.yaml，由人工给定，脚本禁止修改。
"""

import re

ROLE_PREFIX = {"assistant": "AI", "user": "用户", "system": "系统"}

# 已渲染 context 文本中轮次行前缀（"AI: " / "用户: "）
_TURN_LINE_RE = re.compile(r"^(?:AI|用户|系统): ", re.M)


def render_turns(turns):
    """[(role, text), ...] → "AI: xxx\\n用户: yyy"。空轮自动跳过。"""
    lines = []
    for role, text in turns or []:
        text = (text or "").strip()
        if not text:
            continue
        lines.append(f"{ROLE_PREFIX.get(role, role)}: {text}")
    return "\n".join(lines)


def parse_turn_lines(context_text):
    """已渲染 context 文本 → [(role, text), ...]（用于从存储字段重建 turns）。"""
    turns = []
    for line in str(context_text or "").split("\n"):
        line = line.strip()
        m = re.match(r"^(AI|用户|系统): (.*)$", line)
        if not m:
            continue
        prefix = m.group(1)
        role = next((r for r, p in ROLE_PREFIX.items() if p == prefix), prefix)
        turns.append((role, m.group(2).strip()))
    return turns


def get_turns(record):
    """从数据记录中取出对话历史 turns。

    优先读结构化字段 conversation_turns（list[[role, text]]，prefilter 落盘时保存），
    否则从已渲染 context 字段解析。
    """
    raw = record.get("conversation_turns")
    if raw:
        return [(str(t[0]), str(t[1])) for t in raw if t and len(t) >= 2]
    return parse_turn_lines(record.get("context"))


def build_context(turns, n_turns=None):
    """取历史最后 n_turns 轮渲染；n_turns=None 或 0 表示完整对话。"""
    if n_turns:
        turns = (turns or [])[-int(n_turns):]
    else:
        turns = turns or []
    return render_turns(turns)


def context_for(record, n_turns):
    """按阶段轮数构造 record 的上下文文本。"""
    return build_context(get_turns(record), n_turns)


def truncate_context(context_text, max_turns):
    """已渲染 context 文本只保留最后 max_turns 轮（0/None 表示全部）。"""
    if not max_turns or not context_text:
        return context_text or ""
    starts = [m.start() for m in _TURN_LINE_RE.finditer(context_text)]
    if not starts:
        return str(context_text)[-300:]
    begin = starts[max(0, len(starts) - int(max_turns))]
    return context_text[begin:].strip()


def build_bert_text(current_turn, context_text, bert_turns=None):
    """BERT 输入构造。训练与推理必须调用同一函数（见 SKILL.md §11）。"""
    ctx = truncate_context(context_text, bert_turns) if bert_turns else (context_text or "")
    current = (current_turn or "").strip()
    ctx = (ctx or "").strip()
    if not ctx:
        return current
    # current_turn 前置：truncation 从右截断时优先保留待判定内容
    return current + "\n" + ctx


def build_llm_user_prompt(record, llm_turns=None, instruction=""):
    """LLM 打标 user prompt：上下文（按 llm_label_turns）+ 当前 Turn + 输出要求。"""
    context = context_for(record, llm_turns)
    parts = []
    if context:
        parts.append(f"【对话上下文】\n{context}")
    else:
        parts.append("【对话上下文】\n（无）")
    parts.append(f"【待判定 Turn（AI 输出）】\n{(record.get('current_turn') or '').strip()}")
    if instruction:
        parts.append(instruction)
    parts.append('请依据规范判定，只输出 JSON：{"label": "positive 或 negative", "scene_id": "正例必填，负例为空字符串", "reason": "判定理由"}')
    return "\n\n".join(parts)
