# -*- coding: utf-8 -*-
"""模型评估：对 validation / regression_eval / gold 等数据集计算总体与子场景指标。

对应 cold_start.md 步骤 12 / iteration.md 步骤 11、14。

laya 候选（type: laya，SKILL.md §17）：--model-dir 指 checkpoint 目录
（缺省 task.yaml path / Hub 默认）；question 读 --version 的
standard/laya_question.yaml（必填，否则无法定位问题配置）；
FP/FN 明细字段为 laya_prob / laya_prediction / laya_threshold。

用法（BERT）：
    python <skill>/scripts/evaluate.py \
        --model-dir v_0/training/macbert/best_model --threshold 0.35 \
        --dataset validation=v_0/dataset/validation.jsonl \
        --dataset regression_eval=v_0/dataset/regression_eval.jsonl \
        --dataset gold=data/gold/gold_positive.jsonl,data/gold/gold_negative.jsonl \
        --output-dir v_0/metrics --acceptance

用法（laya）：
    python <skill>/scripts/evaluate.py --model laya_multilingual --version v_0 \
        --threshold 0.35 [--model-dir <checkpoint>] --dataset gold=... \
        --output-dir laya_multilingual/v_0/metrics --acceptance

输出：
    <output-dir>/model_metrics.csv      各数据集总体指标（行=数据集）
    <output-dir>/scene_metrics.csv      各数据集 × scene 指标
    <output-dir>/<dataset>_fp.jsonl/.xlsx、<dataset>_fn.jsonl/.xlsx   FP/FN 明细
    <output-dir>/acceptance.yaml        对照 task.yaml 验收标准（--acceptance 时）
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, kev_scorer, laya_scorer  # noqa: E402

import torch  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from train_bert import predict_scores  # noqa: E402


def evaluate_dataset(name, records, scores, threshold, ctype=common.CANDIDATE_TYPE_BERT):
    """单数据集评估：总体指标 + 子场景 recall + FP/FN 明细（字段名随候选类型）。"""
    fields = common.score_fields(ctype)
    labels = [1 if r.get("label") == "positive" else 0 for r in records]
    preds = [1 if s >= threshold else 0 for s in scores]

    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)

    def _enrich(r, s, p, side):
        out = dict(r)
        out.update({fields["score"]: round(s, 6), fields["threshold"]: threshold,
                    fields["prediction"]: "positive" if p else "negative",
                    "error_side": side})
        return out

    fp_rows = [_enrich(r, s, p, "FP")
               for r, s, l, p in zip(records, scores, labels, preds) if l == 0 and p == 1]
    fn_rows = [_enrich(r, s, p, "FN")
               for r, s, l, p in zip(records, scores, labels, preds) if l == 1 and p == 0]

    overall = {
        "dataset": name,
        "rows": len(records),
        "positive_rows": sum(labels),
        "precision": round(precision_score(labels, preds, zero_division=0), 6),
        "recall": round(recall_score(labels, preds, zero_division=0), 6),
        "f1": round(f1_score(labels, preds, zero_division=0), 6),
        "tp": tp,
        "fp": len(fp_rows),
        "fn": len(fn_rows),
    }

    # 子场景 recall（以正例 scene_id 分组）
    scene_stats = defaultdict(lambda: {"total": 0, "hit": 0})
    for r, l, p in zip(records, labels, preds):
        if l != 1:
            continue
        scene = r.get("scene_id") or "<empty>"
        scene_stats[scene]["total"] += 1
        if p == 1:
            scene_stats[scene]["hit"] += 1
    scene_rows = [{
        "dataset": name,
        "scene_id": scene,
        "positive_rows": v["total"],
        "recall": round(v["hit"] / v["total"], 6) if v["total"] else None,
    } for scene, v in sorted(scene_stats.items())]

    return overall, scene_rows, fp_rows, fn_rows


def main():
    parser = argparse.ArgumentParser(description="模型评估（总体 + 子场景 + FP/FN）")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", default=None,
                        help="目标模型 key（laya 候选必填或唯一 enabled；"
                             "BERT 候选可缺省，由 --model-dir 定位）")
    parser.add_argument("--version", default=None,
                        help="laya 候选用：读 <model>/<version>/standard/laya_question.yaml")
    parser.add_argument("--model-dir", default=None,
                        help="BERT：train_bert 产出的 best_model 目录（必填）；"
                             "laya：checkpoint 目录（缺省 task.yaml path / Hub 默认）")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--dataset", action="append", required=True,
                        help="name=path[,path2...]，可多次指定")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acceptance", action="store_true", help="对照 task.yaml 验收标准")
    parser.add_argument("--prev-acceptance", default=None,
                        help="上一版本 acceptance.yaml（迭代时用于 min_improvement 停止保护）")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    task_cfg, _, bert_turns = common.task_ctx(args.root, "bert_turns", 1)

    # 候选类型解析：--model 显式/唯一 enabled；多 enabled 且未指定时按 BERT 兼容旧用法
    ctype = common.CANDIDATE_TYPE_BERT
    model = args.model
    if model:
        ctype = common.candidate_type(model, task_cfg)
    else:
        enabled = common.enabled_models(task_cfg)
        if len(enabled) == 1:
            model = enabled[0]
            ctype = common.candidate_type(model, task_cfg)
        elif args.model_dir is None:
            raise SystemExit("请通过 --model-dir 指定 BERT 模型目录，或 --model 指定目标候选")

    laya_info = {}
    if ctype == common.CANDIDATE_TYPE_LAYA:
        if not model:
            raise SystemExit("laya 候选需要 --model 显式指定（或 task.yaml 只启用一个候选）")
        if not args.version:
            raise SystemExit("laya 候选需要 --version 定位 standard/laya_question.yaml")
        scorer = laya_scorer.build_laya_scorer(
            args.root, model, version=args.version, task_cfg=task_cfg,
            checkpoint=args.model_dir, batch_size=args.batch_size)
        laya_info = scorer.describe()

        def _score_fn(records):
            return scorer.score(records)
    elif ctype == common.CANDIDATE_TYPE_KEV:
        if not args.version:
            raise SystemExit("Kev 候选需要 --version 定位 standard/kev_question.yaml")
        scorer = kev_scorer.build_kev_scorer(
            args.root, model, args.version, task_cfg=task_cfg, checkpoint=args.model_dir)
        laya_info = scorer.describe()

        def _score_fn(records):
            return scorer.score(records)
    else:
        if not args.model_dir:
            raise SystemExit("BERT 候选必须提供 --model-dir（train_bert 产出的 best_model 目录）")
        max_length = (task_cfg.get("training", {}) or {}).get("max_length", 512)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model_obj = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

        def _score_fn(records):
            return predict_scores(model_obj, tokenizer, records, bert_turns, max_length,
                                  args.batch_size, device)

    os.makedirs(args.output_dir, exist_ok=True)
    all_overall, all_scene = [], []
    for spec in args.dataset:
        name, _, paths = spec.partition("=")
        records = common.load_many(paths.split(","))
        if not records:
            print(f"[WARN] 数据集 {name} 为空，跳过")
            continue
        scores = _score_fn(records)
        overall, scene_rows, fp_rows, fn_rows = evaluate_dataset(
            name, records, scores, args.threshold, ctype=ctype)
        all_overall.append(overall)
        all_scene.extend(scene_rows)
        if fp_rows:
            common.save_records(fp_rows, os.path.join(args.output_dir, f"{name}_fp"))
        if fn_rows:
            common.save_records(fn_rows, os.path.join(args.output_dir, f"{name}_fn"))
        print(f"[{name}] rows={overall['rows']} P={overall['precision']} "
              f"R={overall['recall']} F1={overall['f1']} FP={overall['fp']} FN={overall['fn']}")

    if all_overall:
        with open(os.path.join(args.output_dir, "model_metrics.csv"), "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_overall[0].keys()))
            writer.writeheader()
            writer.writerows(all_overall)
    if all_scene:
        with open(os.path.join(args.output_dir, "scene_metrics.csv"), "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_scene[0].keys()))
            writer.writeheader()
            writer.writerows(all_scene)

    if args.acceptance:
        acc_cfg = task_cfg.get("acceptance", {}) or {}
        primary = next((o for o in all_overall if o["dataset"] == "gold"), all_overall[0])
        result = {
            "evaluated_on": primary["dataset"],
            "precision_target": acc_cfg.get("precision_min", 0.90),
            "recall_target": acc_cfg.get("recall_min", 0.80),
            "precision": primary["precision"],
            "recall": primary["recall"],
            "f1": primary["f1"],
            "precision_pass": primary["precision"] >= acc_cfg.get("precision_min", 0.90),
            "recall_pass": primary["recall"] >= acc_cfg.get("recall_min", 0.80),
        }
        # min_improvement 停止保护：与上一版本 F1 比较
        prev = None
        if args.prev_acceptance and os.path.exists(args.prev_acceptance):
            prev = common.load_yaml(args.prev_acceptance)
        if prev and prev.get("f1") is not None:
            min_impr = acc_cfg.get("min_improvement", 0.0)
            result["prev_f1"] = prev.get("f1")
            result["improvement"] = round(result["f1"] - prev["f1"], 6)
            result["improvement_pass"] = result["improvement"] >= min_impr
        # regression_eval 不退化检查
        reg = next((o for o in all_overall if o["dataset"] == "regression_eval"), None)
        if reg:
            fn_rate = round(reg["fn"] / reg["positive_rows"], 6) if reg["positive_rows"] else 0
            result["regression_fn_rate"] = fn_rate
            result["regression_pass"] = fn_rate <= 0.1  # 默认：regression FN 率 ≤ 10%
        result["accepted"] = result["precision_pass"] and result["recall_pass"] \
            and result.get("regression_pass", True) and result.get("improvement_pass", True)
        if ctype in (common.CANDIDATE_TYPE_LAYA, common.CANDIDATE_TYPE_KEV):
            result.update(laya_info)
        common.save_yaml(result, os.path.join(args.output_dir, "acceptance.yaml"))
        print(f"acceptance: {result}")

    print(f"[OK] 指标输出 {args.output_dir}")


if __name__ == "__main__":
    main()
