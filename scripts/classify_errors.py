# -*- coding: utf-8 -*-
"""FP/FN 归因：LLM 辅助对 evaluate 产出的 FP/FN 明细分类 error_type。

对应 iteration.md 步骤 5。错误类型读 task.yaml error_analysis：
    FN: new_scene / boundary_definition / insufficient_context / bert_capacity /
        regex_miss / other
    FP: keyword_false_trigger / negation / quoted_content / context_reversal /
        boundary_definition / other

LLM 归因仅作初稿，最终以人工复核（human_error_type）为准。

用法：
    python <skill>/scripts/classify_errors.py --version v_1 \
        --input v_1/labeling/analysis/fp.jsonl v_1/labeling/analysis/fn.jsonl \
        --model LongCat-Flash-Chat \
        [--output-dir v_1/labeling/analysis]

输出（与输入同名的 *_classified.jsonl / .xlsx，每条带 error_type / error_comment）
"""

import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, llm_client  # noqa: E402

PROMPT_TMPL = """你是检测模型错误归因专家。以下是一条 Turn 级二分类检测模型的误判样本：

【错误方向】{side}（{side_desc}）
【真实标签】{gold_label}
【模型预测】{pred_label}（score={score}）
【当前 Turn】{current_turn}
【上下文】{context}

可选错误类型（只能从中选择一个）：
{type_list}

请判断该误判的根本原因，只输出 JSON：
{{"error_type": "上述类型之一", "error_comment": "一句话说明归因依据"}}"""


def build_type_list(task_cfg, side):
    ea = task_cfg.get("error_analysis", {}) or {}
    types = ea.get("fn_types" if side == "FN" else "fp_types", [])
    return "\n".join(f"- {t}" for t in types)


def main():
    parser = argparse.ArgumentParser(description="FP/FN LLM 辅助归因")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--input", nargs="+", required=True, help="fp/fn jsonl（evaluate 产出，含 error_side）")
    parser.add_argument("--llm", default="LongCat-Flash-Chat", help="执行归因的大模型显示名")
    parser.add_argument("--output-dir", default=None, help="默认取第一个输入文件所在目录")
    parser.add_argument("--rpm", type=int, default=30)
    args = parser.parse_args()

    task_cfg = common.load_task_config(args.root)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(
        common.iter_input_files(args.input)[0]))
    rate_limiter = llm_client.RateLimiter(args.rpm)

    SIDE_DESC = {"FP": "误报：真实为负例，模型判为正例",
                 "FN": "漏报：真实为正例，模型判为负例"}

    for path in common.iter_input_files(args.input):
        records = common.load_records(path)
        if not records:
            continue
        out_records = []
        type_counter = Counter()
        for r in records:
            side = r.get("error_side") or ("FP" if "fp" in os.path.basename(path) else "FN")
            # 已有人工归因的样本不覆盖
            if r.get("human_error_type"):
                out = dict(r)
                out["error_type"] = r["human_error_type"]
                type_counter[r["human_error_type"]] += 1
                out_records.append(out)
                continue
            user_prompt = PROMPT_TMPL.format(
                side=side, side_desc=SIDE_DESC.get(side, ""),
                gold_label=r.get("label"), pred_label=common.get_prediction(r),
                score=common.get_score(r),
                current_turn=r.get("current_turn") or "",
                context=r.get("context") or "（无）",
                type_list=build_type_list(task_cfg, side))
            out = dict(r)
            try:
                content = llm_client.call_llm(args.llm, "", user_prompt,
                                              task_cfg=task_cfg, rate_limiter=rate_limiter,
                                              max_tokens=256)
                obj = llm_client.extract_json(content)
                valid_types = (task_cfg.get("error_analysis", {}) or {}
                               .get("fn_types" if side == "FN" else "fp_types", []))
                error_type = str(obj.get("error_type") or "").strip()
                if error_type not in valid_types:
                    error_type = "other"
                out["error_type"] = error_type
                out["error_comment"] = str(obj.get("error_comment") or "").strip()
                out["error_source"] = "llm"
                type_counter[error_type] += 1
            except Exception as e:
                out["error_type"] = None
                out["classify_error"] = str(e)[:150]
            out_records.append(out)

        base = os.path.splitext(os.path.basename(path))[0]
        common.save_records(out_records, os.path.join(output_dir, f"{base}_classified"))
        print(f"[OK] {base}: {len(out_records)} 条，类型分布 {dict(type_counter)}")

    common.update_status(args.root, args.version, model=args.model,
                         event=f"error classification done by [{args.llm}]: "
                               + ", ".join(os.path.basename(p) for p in common.iter_input_files(args.input)))
    print(f"输出目录：{output_dir}（*_classified.jsonl，供人工复核确认 error_type）")


if __name__ == "__main__":
    main()
