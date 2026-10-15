# -*- coding: utf-8 -*-
"""导入人工复核结果：review_done.xlsx → 规范化 review_result.jsonl/.xlsx。

对应 cold_start.md 步骤 6 / iteration.md 步骤 4 的复核完成环节。

输入 Excel 期望列（兼容中文别名，见 COLUMN_ALIASES）：
    sample_id, human_label, human_scene_id, human_error_type, human_comment, reviewer

校验规则（不合法行进入 review_errors，不进入正式结果）：
    - human_label ∈ {positive, negative}；
    - positive 必须带 scenes.yaml 中存在的 scene_id；
    - negative 的 scene_id 必须为空；
    - sample_id 必须能在待复核样本（--expected）中找到。

用法：
    python <skill>/scripts/import_human_review.py --version v_0 \
        --input v_0/labeling/human/review_done.xlsx \
        [--expected v_0/labeling/human/review_input.jsonl] \
        [--output v_0/labeling/human/review_result]
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, text_utils  # noqa: E402

# 人工填写列的中文别名（与 law1 人工核验表格式兼容）
COLUMN_ALIASES = {
    "sample_id": ["sample_id", "样本ID", "样本id"],
    "human_label": ["human_label", "人工最终标签", "最终标签", "人工标签"],
    "human_scene_id": ["human_scene_id", "人工子场景", "最终子场景", "人工scene_id"],
    "human_error_type": ["human_error_type", "错误类型", "人工错误类型"],
    "human_comment": ["human_comment", "核验备注", "备注"],
    "reviewer": ["reviewer", "核验人"],
}
LABEL_ALIASES = {
    "positive": "positive", "正例": "positive", "pos": "positive", "1": "positive", "是": "positive",
    "negative": "negative", "负例": "negative", "neg": "negative", "0": "negative", "否": "negative",
}


def norm_label(v):
    if v is None:
        return None
    return LABEL_ALIASES.get(str(v).strip().lower()) or LABEL_ALIASES.get(str(v).strip())


def pick(row, field):
    for col in COLUMN_ALIASES[field]:
        if col in row and row[col] not in (None, ""):
            return row[col]
    return None


def main():
    parser = argparse.ArgumentParser(description="导入并校验人工复核结果")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--input", required=True, help="人工完成的 review_done.xlsx")
    parser.add_argument("--expected", default=None, help="对应的 review_input.jsonl，用于校验 sample_id")
    parser.add_argument("--scenes", default=None, help="scenes.yaml（默认当前版本 standard 下）")
    parser.add_argument("--output", default=None, help="默认与输入同目录的 review_result")
    args = parser.parse_args()

    vdir = common.version_dir(args.root, args.version, model=args.model)
    scenes_path = args.scenes or os.path.join(vdir, "standard", "scenes.yaml")
    scene_ids = set()
    if os.path.exists(scenes_path):
        scenes = common.load_yaml(scenes_path)
        for s in scenes.get("scenes", []) or []:
            if isinstance(s, dict) and s.get("scene_id"):
                scene_ids.add(str(s["scene_id"]))
    else:
        print(f"[WARN] scenes.yaml 不存在（{scenes_path}），跳过 scene_id 合法性校验")

    rows = common.load_records(args.input)
    expected = None
    if args.expected and os.path.exists(args.expected):
        expected = {r["sample_id"]: r for r in common.load_records(args.expected)}
    elif args.expected:
        print(f"[WARN] 待复核清单不存在（{args.expected}），跳过 sample_id 存在性校验")

    results, errors = [], []
    for i, row in enumerate(rows):
        sample_id = pick(row, "sample_id")
        line_no = i + 2  # Excel 含表头
        if not sample_id:
            errors.append({"line": line_no, "error": "缺少 sample_id", "row": row})
            continue
        if expected is not None and str(sample_id) not in expected:
            errors.append({"line": line_no, "sample_id": sample_id, "error": "sample_id 不在待复核清单中"})
            continue
        label = norm_label(pick(row, "human_label"))
        if label is None:
            errors.append({"line": line_no, "sample_id": str(sample_id),
                           "error": f"非法 human_label: {pick(row, 'human_label')}"})
            continue
        scene_id = str(pick(row, "human_scene_id") or "").strip()
        if label == "positive":
            if not scene_id:
                errors.append({"line": line_no, "sample_id": str(sample_id), "error": "positive 缺少 scene_id"})
                continue
            if scene_ids and scene_id not in scene_ids:
                errors.append({"line": line_no, "sample_id": str(sample_id),
                               "error": f"scene_id 不在 scenes.yaml 中: {scene_id}"})
                continue
        else:
            scene_id = ""

        base = dict(expected[str(sample_id)]) if expected else {"sample_id": str(sample_id)}
        base.update({
            "sample_id": str(sample_id),
            "human_label": label,
            "human_scene_id": scene_id,
            "human_error_type": str(pick(row, "human_error_type") or "") or None,
            "human_comment": str(pick(row, "human_comment") or "") or None,
            "reviewer": str(pick(row, "reviewer") or "") or None,
            "reviewed_at": common.now_str(),
            "label": label,
            "scene_id": scene_id or None,
            "annotation_source": "human",
            "human_verified": True,
        })
        results.append(base)

    output = args.output or os.path.splitext(args.input)[0].rsplit("_done", 1)[0] + "_result"
    common.save_records(results, output)
    if errors:
        common.save_records(errors, output + "_errors")
        print(f"[WARN] {len(errors)} 行不合法，已写入 {output}_errors.jsonl（修正后重跑本脚本）")

    n_pos = sum(1 for r in results if r["label"] == "positive")
    common.update_status(args.root, args.version, model=args.model,
                         artifact=(os.path.relpath(output + ".jsonl", vdir), len(results)),
                         event=f"human review imported: {len(results)} rows ({n_pos} positive, {len(errors)} errors)")
    print(f"[OK] 有效 {len(results)} 条（正例 {n_pos}），错误 {len(errors)} 条，输出 {output}.jsonl / .xlsx")


if __name__ == "__main__":
    main()
