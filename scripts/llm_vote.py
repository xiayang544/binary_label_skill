# -*- coding: utf-8 -*-
"""多 LLM 投票合并：按 sample_id 合并各模型打标结果，多数投票 + 分歧标记。

对应 iteration.md 步骤 3。vote 仅作人工复核辅助信息，不作为最终 label
（final_label_source: human，见 task.yaml）。

用法：
    python <skill>/scripts/llm_vote.py --version v_1 \
        --inputs v_1/labeling/llm/longcat_2.jsonl v_1/labeling/llm/longcat_flash.jsonl \
                 v_1/labeling/llm/deepseek_v4_flash.jsonl \
        [--output v_1/labeling/llm/vote_merged]

输出：
    <output>.jsonl/.xlsx   含 llm_labels / llm_scene_ids / vote_label /
                            vote_scene_id / vote_agreement_count / need_human_review
"""

import argparse
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common  # noqa: E402


def majority(items):
    """多数投票；并列或空 → None。"""
    items = [i for i in items if i]
    if not items:
        return None
    counter = Counter(items)
    top_count = counter.most_common(1)[0][1]
    winners = [k for k, v in counter.items() if v == top_count]
    return winners[0] if len(winners) == 1 else None


def main():
    parser = argparse.ArgumentParser(description="多 LLM 打标结果投票合并")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--inputs", nargs="+", required=True, help="各模型标准化打标结果 jsonl")
    parser.add_argument("--output", default=None, help="默认 <root>/<version>/labeling/llm/vote_merged")
    args = parser.parse_args()

    # sample_id → 各模型结果
    merged = {}
    models = []
    for path in common.iter_input_files(args.inputs):
        records = common.load_records(path)
        model = records[0].get("label_model") if records else None
        model = model or os.path.basename(path).replace(".jsonl", "")
        models.append(model)
        for r in records:
            if not r.get("llm_parse_success"):
                continue
            merged.setdefault(r["sample_id"], {"base": r, "labels": {}, "scenes": {}})
            merged[r["sample_id"]]["labels"][model] = r.get("llm_label")
            merged[r["sample_id"]]["scenes"][model] = r.get("llm_scene_id")

    if not models:
        raise SystemExit("输入为空")

    out = []
    for sample_id, m in merged.items():
        base = dict(m["base"])
        labels, scenes = m["labels"], m["scenes"]
        vote_label = majority(list(labels.values()))
        vote_scene = majority(list(scenes.values()))
        agreement = sum(1 for v in labels.values() if v == vote_label) if vote_label else 0
        base.update({
            "llm_labels": labels,
            "llm_scene_ids": scenes,
            "vote_label": vote_label if vote_label else "disagreement",
            "vote_scene_id": vote_scene or "",
            "vote_agreement_count": agreement,
            "vote_total_count": len(labels),
            "need_human_review": vote_label is None or agreement < len(labels),
        })
        out.append(base)

    stats = Counter(r["vote_label"] for r in out)
    n_review = sum(1 for r in out if r["need_human_review"])
    print(f"合并 {len(out)} 条样本，{len(models)} 个模型：{models}")
    print(f"vote 分布：{dict(stats)}，需人工复核 {n_review} 条")

    vdir = common.version_dir(args.root, args.version, model=args.model)
    output = args.output or os.path.join(vdir, "labeling", "llm", "vote_merged")
    common.save_records(out, output)
    common.update_status(args.root, args.version, model=args.model,
                         artifact=(os.path.relpath(output + ".jsonl", vdir), len(out)),
                         event=f"llm vote: {len(out)} merged, {n_review} need review")
    print(f"[OK] 输出 {output}.jsonl / .xlsx")


if __name__ == "__main__":
    main()
