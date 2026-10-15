# -*- coding: utf-8 -*-
"""迭代采样：基于全量扫描结果构造下一轮标注候选池。

对应 iteration.md 步骤 2。采样范围：
    BERT 预测的全部正例
    +
    negative_ratio × 正例数的负例（优先 threshold 附近 / 低置信，其余随机）

用法：
    python <skill>/scripts/sample_iteration_data.py --version v_1 \
        --input v_0/scan/selected_model_predictions.jsonl \
        [--output-dir v_1/labeling] [--negative-ratio 2]

输出：
    <output-dir>/positive_pool.jsonl/.xlsx    BERT 预测正例
    <output-dir>/negative_sample.jsonl/.xlsx  按优先级采样的负例
    <output-dir>/candidate_pool.jsonl/.xlsx   两者合并（下一轮 LLM 打标输入）
"""

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="迭代样本采样")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--input", required=True, help="上一轮 scan/selected_model_predictions.jsonl")
    parser.add_argument("--output-dir", default=None, help="默认 <root>/<version>/labeling")
    parser.add_argument("--negative-ratio", type=float, default=None, help="默认读 task.yaml")
    parser.add_argument("--near-threshold-window", type=float, default=0.10,
                        help="|score - threshold| ≤ 该值视为 threshold 附近")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.6,
                        help="score ≥ 该值的负例视为低置信（接近翻转）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_cfg = common.load_task_config(args.root)
    if args.negative_ratio is None:
        args.negative_ratio = (task_cfg.get("iteration_sampling", {}) or {}).get("negative_ratio", 2)

    records = common.load_records(args.input)
    model = common.resolve_model(args.root, args.model, task_cfg)
    ctype = common.candidate_type(model, task_cfg)
    threshold = next((float(t) for r in records
                      if (t := common.get_threshold(r, ctype)) is not None), None)

    positives = [r for r in records if common.get_prediction(r, ctype) == "positive"]
    negatives = [r for r in records if common.get_prediction(r, ctype) != "positive"]

    # 负例优先级分层：threshold 附近 > 低置信 > 随机
    near, low_conf, rest = [], [], []
    for r in negatives:
        score = common.get_score(r, ctype)
        if score is None:
            rest.append(r)
            continue
        if threshold is not None and abs(float(score) - threshold) <= args.near_threshold_window:
            near.append(r)
        elif float(score) >= args.low_confidence_threshold:
            low_conf.append(r)
        else:
            rest.append(r)

    target = int(len(positives) * args.negative_ratio)
    rng = random.Random(args.seed)
    rng.shuffle(near)
    rng.shuffle(low_conf)
    rng.shuffle(rest)
    sampled = near[:target]
    if len(sampled) < target:
        sampled += low_conf[: target - len(sampled)]
    if len(sampled) < target:
        sampled += rest[: target - len(sampled)]

    output_dir = args.output_dir or os.path.join(
        common.version_dir(args.root, args.version, model=args.model), "labeling")
    common.save_records(positives, os.path.join(output_dir, "positive_pool"))
    common.save_records(sampled, os.path.join(output_dir, "negative_sample"))
    common.save_records(positives + sampled, os.path.join(output_dir, "candidate_pool"))

    common.update_status(args.root, args.version, model=args.model,
                         artifact=("labeling/candidate_pool.jsonl", len(positives) + len(sampled)),
                         event=f"iteration sampling: {len(positives)} positive + {len(sampled)} negative")
    print(f"[OK] 正例池 {len(positives)}，负例采样 {len(sampled)}"
          f"（threshold 附近 {min(len(near), target)}，低置信 "
          f"{max(0, min(len(low_conf), target - len(near)))}，随机 "
          f"{max(0, target - len(near) - len(low_conf))}），输出 {output_dir}")


if __name__ == "__main__":
    main()
