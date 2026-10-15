# -*- coding: utf-8 -*-
"""生成人工复核输入表：review_input.xlsx + 同名 JSONL 快照。

对应 cold_start.md 步骤 6 / iteration.md 步骤 4。
人工填写列与 import_human_review.py 的 COLUMN_ALIASES 一一对应
（人工最终标签 / 人工子场景 / 错误类型 / 核验备注 / 核验人）。

两种模式：
    cold_start: 全部 LLM 判定正例 + max(2×正例数, 100) 条负例（默认从 LLM 标注结果抽样）
    disagreement: vote 分歧/失败样本 + 一致样本抽检（--agree-sample-ratio，默认 5%，
                  用于估计 vote 自动采纳的错误率）

用法：
    # 冷启动
    python <skill>/scripts/export_review_input.py --version v_0 --mode cold_start \
        --input v_0/labeling/llm/longcat_flash.jsonl \
        --output v_0/labeling/human/review_input

    # 迭代（vote 合并结果 + 可选附加 BERT 预测列）
    python <skill>/scripts/export_review_input.py --version v_1 --mode disagreement \
        --input v_1/labeling/llm/vote_merged.jsonl \
        --bert-pred v_1/labeling/bert_predictions.jsonl \
        --output v_1/labeling/human/disagreement_review_input
"""

import argparse
import os
import random
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common  # noqa: E402

# 人工填写列（与 import_human_review.py COLUMN_ALIASES 对应）
HUMAN_COLUMNS = ["人工最终标签", "人工子场景", "错误类型", "核验备注", "核验人"]
HUMAN_COLUMN_HINTS = "人工最终标签填 正例 或 负例；正例须填人工子场景"


def render_row(r, bert_by_id=None):
    """构造复核表一行：参考列（自动填充）+ 人工填写列（留空）。"""
    row = {
        "sample_id": r.get("sample_id"),
        "conversation_id": r.get("conversation_id"),
        "turn_id": r.get("turn_id"),
        "business_scene": r.get("business_scene"),
        "context": r.get("context") or "",
        "current_turn": r.get("current_turn") or "",
        "llm_label": r.get("llm_label") or "",
        "llm_scene_id": r.get("llm_scene_id") or "",
        "llm_reason": r.get("llm_reason") or "",
    }
    # vote 列（迭代模式）
    if r.get("llm_labels"):
        row["llm_labels"] = " | ".join(f"{m}:{l}" for m, l in r["llm_labels"].items())
        row["llm_scene_ids"] = " | ".join(f"{m}:{s}" for m, s in (r.get("llm_scene_ids") or {}).items())
        row["vote_label"] = r.get("vote_label") or ""
        row["vote_agreement"] = f"{r.get('vote_agreement_count')}/{r.get('vote_total_count')}"
    # 模型预测列（保留输入中的 BERT / laya / Kev 字段名）
    prediction = (bert_by_id or {}).get(r.get("sample_id"))
    if prediction:
        for fields in common.SCORE_FIELDS.values():
            if fields["score"] in prediction:
                for field in fields.values():
                    row[field] = prediction.get(field)
    for col in HUMAN_COLUMNS:
        row[col] = ""
    return row


def main():
    parser = argparse.ArgumentParser(description="生成人工复核输入表")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--input", required=True, help="LLM 标注结果或 vote_merged.jsonl")
    parser.add_argument("--mode", choices=["cold_start", "disagreement", "all"], required=True)
    parser.add_argument("--bert-pred", default=None,
                        help="可选，检测器预测 jsonl（BERT/laya/Kev），按 sample_id 附加")
    parser.add_argument("--output", required=True, help="输出基础名（自动加 .xlsx/.jsonl）")
    parser.add_argument("--negative-ratio", type=float, default=2, help="cold_start 模式负例倍数")
    parser.add_argument("--min-negative", type=int, default=100, help="cold_start 模式负例下限")
    parser.add_argument("--agree-sample-ratio", type=float, default=0.05,
                        help="disagreement 模式下一致样本抽检比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_cfg = common.load_task_config(args.root)
    records = common.load_records(args.input)
    bert_by_id = None
    if args.bert_pred:
        bert_by_id = {r["sample_id"]: r for r in common.load_records(args.bert_pred)}

    rng = random.Random(args.seed)
    if args.mode == "cold_start":
        positives = [r for r in records if r.get("llm_label") == "positive"]
        negatives = [r for r in records if r.get("llm_label") == "negative"]
        target = max(int(len(positives) * args.negative_ratio), args.min_negative)
        rng.shuffle(negatives)
        selected = positives + negatives[:target]
        reason = f"LLM 正例 {len(positives)} + 负例 {min(target, len(negatives))}"
    elif args.mode == "disagreement":
        need_review = [r for r in records if r.get("need_human_review")]
        agreed = [r for r in records if not r.get("need_human_review")]
        n_sample = round(len(agreed) * args.agree_sample_ratio)
        agreed_sample = rng.sample(agreed, n_sample) if n_sample else []
        selected = need_review + agreed_sample
        reason = (f"分歧/失败 {len(need_review)} + 一致抽检 {len(agreed_sample)}"
                  f"（一致共 {len(agreed)}，比例 {args.agree_sample_ratio}）")
    else:  # all
        selected = list(records)
        reason = f"全部 {len(records)} 条"

    rows = [render_row(r, bert_by_id) for r in selected]
    common.save_records(rows, args.output)

    vdir = common.version_dir(args.root, args.version, model=args.model)
    common.update_status(
        args.root, args.version, model=args.model, status="waiting_human_review",
        artifact=(os.path.relpath(args.output + ".jsonl", vdir), len(rows)),
        pause_waiting_for=args.output + ".xlsx",
        pause_instruction=f"人工复核 {args.output}.xlsx，完成后另存为 *_done.xlsx 并执行 scripts/import_human_review.py",
        event=f"review input exported: {reason}")
    print(f"[OK] {reason}，共 {len(rows)} 条 → {args.output}.xlsx / .jsonl")
    print(f"     人工填写列：{HUMAN_COLUMNS}（{'、'.join(HUMAN_COLUMN_HINTS)}），"
          f"完成后改名为 *_done.xlsx 并执行 scripts/import_human_review.py")


if __name__ == "__main__":
    main()
