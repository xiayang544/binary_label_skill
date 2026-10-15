# -*- coding: utf-8 -*-
"""Threshold 搜索：在 validation 上扫描阈值，选出满足 precision 目标下 recall 最大的点。

对应 cold_start.md 步骤 11 / iteration.md 步骤 10 的 threshold 搜索。
搜索准则（与验收标准对齐，避免 max-F1 把 threshold 带向 precision 不达标的点）：
    objective = max_recall_given_precision（默认）：
        在 precision >= precision_min 的候选中选 recall 最大者；并列取 precision 更高者；
        无可行解时回退 max_f1 并标记 feasible=false。
    objective = max_f1：直接取 F1 最大点。

laya 候选（type: laya，SKILL.md §17）：零训练检测器无 model-dir 推理，打分来源二选一：
    1. --validation 数据已含 laya_prob 字段（laya_score.py 产物）→ 直接复用，零推理；
    2. 否则用 Router 现场打分（question 读 --version 的 standard/laya_question.yaml）。

用法（BERT）：
    python <skill>/scripts/tune_threshold.py \
        --model-dir v_0/training/macbert/lr2e-5_bs32/best_model \
        --validation v_0/dataset/validation.jsonl \
        [--output-dir v_0/training/macbert/lr2e-5_bs32]

用法（laya）：
    python <skill>/scripts/tune_threshold.py --model laya_multilingual --version v_0 \
        --validation laya_multilingual/v_0/dataset/validation.jsonl
        [--model-dir <checkpoint 目录，缺省 task.yaml path / Hub>]

输出：
    <output-dir>/threshold_search.csv     全网格 (threshold, precision, recall, f1)
    <output-dir>/best_threshold.yaml      选定阈值与对应指标
"""

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, kev_scorer, laya_scorer  # noqa: E402

import torch  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from train_bert import predict_scores  # noqa: E402


def laya_checkpoint_ref(args, model, task_cfg):
    """precomputed 分支的 checkpoint 引用登记（供 select_model / full_scan 解析三元组）。"""
    ckpt, source = laya_scorer.resolve_checkpoint(model, task_cfg, override=args.model_dir)
    return {"checkpoint": ckpt, "checkpoint_source": source,
            "variant": (common.candidate_entry(model, task_cfg).get("variant")
                         or "multilingual")}


def main():
    parser = argparse.ArgumentParser(description="Validation 上的 threshold 搜索")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--version", default=None,
                        help="laya 候选用：读 <model>/<version>/standard/laya_question.yaml；"
                             "并作为 --output-dir 缺省（<model>/<version>/training/）")
    parser.add_argument("--model-dir", default=None,
                        help="BERT：train_bert 产出的 best_model 目录（必填）；"
                             "laya：checkpoint 目录（缺省 task.yaml path / Hub 默认）")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", default=None,
                        help="默认：BERT 与 model-dir 同级；laya 取 <model>/<version>/training/")
    parser.add_argument("--objective", choices=["max_recall_given_precision", "max_f1"],
                        default="max_recall_given_precision")
    parser.add_argument("--precision-min", type=float, default=None,
                        help="precision 目标，默认读 task.yaml acceptance.precision_min")
    parser.add_argument("--min", type=float, default=0.10)
    parser.add_argument("--max", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    task_cfg, _, bert_turns = common.task_ctx(args.root, "bert_turns", 1)
    precision_min = args.precision_min
    if precision_min is None:
        precision_min = (task_cfg.get("acceptance", {}) or {}).get("precision_min", 0.90)

    model = common.resolve_model(args.root, args.model, task_cfg)
    ctype = common.candidate_type(model, task_cfg)

    records = common.load_records(args.validation)
    labels = [1 if r["label"] == "positive" else 0 for r in records]

    extra = {}
    if ctype == common.CANDIDATE_TYPE_LAYA:
        # laya 零训练候选：优先复用已落盘 laya_prob，否则 Router 现场打分
        if records and all(r.get("laya_prob") is not None for r in records):
            scores = [float(r["laya_prob"]) for r in records]
            print(f"[INFO] validation 已含 laya_prob（{len(records)} 条），直接复用，跳过推理")
            extra = {"score_source": "precomputed-laya_prob"}
            if args.model_dir:
                print("[WARN] 数据已含 laya_prob，--model-dir 仅登记不参与打分")
            extra.update(laya_checkpoint_ref(args, model, task_cfg))
        else:
            if not args.version:
                raise SystemExit(
                    "laya 候选需要 --version 定位 standard/laya_question.yaml"
                    "（或先用 laya_score.py 落盘含 laya_prob 的 validation）")
            scorer = laya_scorer.build_laya_scorer(
                args.root, model, version=args.version, task_cfg=task_cfg,
                checkpoint=args.model_dir, batch_size=args.batch_size)
            scores = scorer.score(records)
            extra = {"score_source": "laya-router"}
            extra.update(scorer.describe())
        if args.version:
            extra.setdefault("laya_question", laya_scorer.QUESTION_FILENAME)
        output_dir = args.output_dir or (
            os.path.join(common.version_dir(args.root, args.version, model=model,
                                            task_cfg=task_cfg), "training")
            if args.version else None)
        if not output_dir:
            raise SystemExit("laya 候选需要 --output-dir 或 --version 之一来定位输出目录")
    elif ctype == common.CANDIDATE_TYPE_KEV:
        if not args.version:
            raise SystemExit("Kev 候选需要 --version 定位 standard/kev_question.yaml")
        checkpoint = kev_scorer.resolve_checkpoint(model, task_cfg, args.model_dir)
        if records and all(r.get("kev_prob") is not None for r in records):
            scores = [float(r["kev_prob"]) for r in records]
            extra = {"score_source": "precomputed-kev_prob", "checkpoint": checkpoint,
                     "kev_question": kev_scorer.QUESTION_FILENAME}
        else:
            scorer = kev_scorer.build_kev_scorer(
                args.root, model, args.version, task_cfg=task_cfg, checkpoint=checkpoint)
            scores = scorer.score(records)
            extra = {"score_source": "kev-local", **scorer.describe()}
        output_dir = args.output_dir or os.path.join(
            common.version_dir(args.root, args.version, model=model, task_cfg=task_cfg), "training")
    else:
        if not args.model_dir:
            raise SystemExit("BERT 候选必须提供 --model-dir（train_bert 产出的 best_model 目录）")
        output_dir = args.output_dir or os.path.dirname(args.model_dir.rstrip("/"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model_obj = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()
        max_length = (task_cfg.get("training", {}) or {}).get("max_length", 512)
        scores = predict_scores(model_obj, tokenizer, records, bert_turns, max_length,
                                args.batch_size, device)
        extra = {"candidate_type": common.CANDIDATE_TYPE_BERT, "model_dir": args.model_dir}

    grid = []
    t = args.min
    while t <= args.max + 1e-9:
        preds = [1 if s >= t else 0 for s in scores]
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        grid.append({
            "threshold": round(t, 4),
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(f1_score(labels, preds, zero_division=0), 6),
        })
        t += args.step

    if args.objective == "max_f1":
        best = max(grid, key=lambda g: (g["f1"], g["precision"]))
        feasible = True
    else:
        eligible = [g for g in grid if g["precision"] >= precision_min]
        if eligible:
            best = max(eligible, key=lambda g: (g["recall"], g["precision"]))
            feasible = True
        else:
            best = max(grid, key=lambda g: (g["f1"], g["precision"]))
            feasible = False
            print(f"[WARN] 无 threshold 满足 precision >= {precision_min}，回退 max_f1")

    csv_path = os.path.join(output_dir, "threshold_search.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(grid)

    result = {
        "model": model,
        "candidate_type": ctype,
        "model_dir": args.model_dir if ctype == common.CANDIDATE_TYPE_BERT else extra.get("checkpoint"),
        "validation": args.validation,
        "validation_rows": len(records),
        "objective": args.objective,
        "precision_min": precision_min,
        "feasible": feasible,
        "threshold": best["threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        **extra,
    }
    common.save_yaml(result, os.path.join(output_dir, "best_threshold.yaml"))
    print(f"[OK] threshold={best['threshold']} precision={best['precision']} "
          f"recall={best['recall']} f1={best['f1']}（feasible={feasible}）")


if __name__ == "__main__":
    main()
