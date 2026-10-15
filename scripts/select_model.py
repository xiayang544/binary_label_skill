# -*- coding: utf-8 -*-
"""模型评估：对单一目标模型的最优配置统一评估（validation + regression_eval + gold）。

对应 cold_start.md 步骤 12 / iteration.md 步骤 11。
每个候选模型是一条独立流水线，本脚本一次只评估一个模型（--model）；
不存在跨模型比较——各模型各自迭代，最终通过各自 report.md 对比。

评估内容：
    * validation（调优集）；
    * regression_eval（历史 hard case 不退化检查）；
    * gold（人工核验评测集，验收指标以此为准；需求变更改标后需基于新 gold 重评）。

laya 候选（type: laya，SKILL.md §17）：threshold-only——只读 training/best_threshold.yaml
（不要求 best_hparams.yaml / best_model/）；打分用 Router，question 读当前版本
standard/laya_question.yaml；selected_model.yaml 记录三元组（checkpoint 引用 +
laya_question.yaml 相对路径 + best_threshold）。

用法：
    python <skill>/scripts/select_model.py --version v_0 --model h256 \
        [--validation h256/v_0/dataset/validation.jsonl] \
        [--regression h256/v_0/dataset/regression_eval.jsonl] \
        [--gold data/gold/gold_positive.jsonl,data/gold/gold_negative.jsonl] \
        [--prev-acceptance h256/v_0/metrics/acceptance.yaml]

输出（全部在该模型的版本目录下）：
    <model>/<v>/metrics/model_metrics.csv / scene_metrics.csv
    <model>/<v>/metrics/regression_metrics.yaml
    <model>/<v>/metrics/selected_model.yaml    最终配置（BERT：模型目录、threshold；
                                               laya：checkpoint+question+threshold 三元组）
    <model>/<v>/metrics/<model>_{fp,fn}.jsonl  gold FP/FN 明细（供归因分析）
"""

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, kev_scorer, laya_scorer  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from evaluate import evaluate_dataset  # noqa: E402
from train_bert import predict_scores  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="单模型统一评估（validation/regression/gold）")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--validation", default=None, help="默认 <model>/<version>/dataset/validation.jsonl")
    parser.add_argument("--regression", default=None, help="默认 <model>/<version>/dataset/regression_eval.jsonl")
    parser.add_argument("--gold", default=None,
                        help="逗号分隔的 gold jsonl，默认 <root>/data/gold/gold_{positive,negative}.jsonl")
    parser.add_argument("--prev-acceptance", default=None,
                        help="上一版本 acceptance.yaml（迭代时用于 regression 退化与 min_improvement 判断）")
    parser.add_argument("--output-dir", default=None, help="默认 <model>/<version>/metrics")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    task_cfg = common.load_task_config(args.root)
    model = common.resolve_model(args.root, args.model, task_cfg)
    ctype = common.candidate_type(model, task_cfg)
    vdir = common.version_dir(args.root, args.version, model=model, task_cfg=task_cfg)
    output_dir = args.output_dir or os.path.join(vdir, "metrics")

    # 该模型的最优配置（hparam_search 产出；laya 候选为 threshold-only）
    tdir = os.path.join(vdir, "training")
    th_path = os.path.join(tdir, "best_threshold.yaml")
    if not os.path.exists(th_path):
        raise SystemExit(f"未找到阈值产物 {th_path}（先运行 scripts/hparam_search.py "
                         f"--model {model}，laya 候选亦可直接调 scripts/tune_threshold.py）")
    th = common.load_yaml(th_path)
    threshold = th.get("threshold", 0.5)

    laya_desc = {}
    if ctype == common.CANDIDATE_TYPE_LAYA:
        # §17.3：training/ 只有 threshold_search.csv + best_threshold.yaml（无 cfg<i>/、无 best_model/）
        checkpoint = th.get("checkpoint")
        scorer = laya_scorer.build_laya_scorer(
            args.root, model, version=args.version, task_cfg=task_cfg,
            checkpoint=checkpoint, batch_size=args.batch_size)
        laya_desc = scorer.describe()
        model_dir = checkpoint
        print(f"目标模型 {model}（laya），threshold={threshold}，checkpoint={checkpoint or 'hub-default'}")

        def _score_fn(records):
            return scorer.score(records)
    elif ctype == common.CANDIDATE_TYPE_KEV:
        scorer = kev_scorer.build_kev_scorer(
            args.root, model, args.version, task_cfg=task_cfg, checkpoint=th.get("checkpoint"))
        model_dir = scorer.checkpoint
        laya_desc = scorer.describe()

        def _score_fn(records):
            return scorer.score(records)
    else:
        hp_path = os.path.join(tdir, "best_hparams.yaml")
        if not os.path.exists(hp_path):
            raise SystemExit(f"未找到训练产物 {hp_path}（先运行 scripts/hparam_search.py --model {model}）")
        hp = common.load_yaml(hp_path)
        model_dir = hp.get("best_model_dir")
        if not model_dir or not os.path.isdir(model_dir):
            raise SystemExit(f"best_model 目录不存在：{model_dir}")
        print(f"目标模型 {model}，threshold={threshold}，best_model={model_dir}")

        _, _, bert_turns = common.task_ctx(args.root, "bert_turns", 1)
        max_length = (task_cfg.get("training", {}) or {}).get("max_length", 512)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        bert_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

        def _score_fn(records):
            return predict_scores(bert_model, tokenizer, records, bert_turns, max_length,
                                  args.batch_size, device)

    # 数据集（gold 为全局资产，其余在该模型版本目录下）
    datasets = {}
    val_path = args.validation or os.path.join(vdir, "dataset", "validation.jsonl")
    if os.path.exists(val_path):
        datasets["validation"] = common.load_records(val_path)
    reg_path = args.regression or os.path.join(vdir, "dataset", "regression_eval.jsonl")
    if os.path.exists(reg_path):
        datasets["regression_eval"] = common.load_records(reg_path)
    if args.gold:
        gold_paths = args.gold.split(",")
    else:
        gold_paths = [os.path.join(args.root, "data", "gold", f"gold_{p}.jsonl")
                      for p in ("positive", "negative")]
    gold_paths = [p for p in gold_paths if os.path.exists(p)]
    if gold_paths:
        datasets["gold"] = common.load_many(gold_paths)
    if not datasets:
        raise SystemExit("无任何可评估数据集")

    all_overall, all_scene = [], []
    reg_overall = None
    gold_overall = None
    for name, records in datasets.items():
        if not records:
            continue
        scores = _score_fn(records)
        overall, scene_rows, fp_rows, fn_rows = evaluate_dataset(
            f"{model}@{name}", records, scores, threshold, ctype=ctype)
        all_overall.append(overall)
        all_scene.extend(scene_rows)
        if name == "regression_eval":
            reg_overall = overall
        if name == "gold":
            gold_overall = overall
            for tag, rows in (("fp", fp_rows), ("fn", fn_rows)):
                if rows:
                    common.save_records(rows, os.path.join(output_dir, f"{model}_{tag}"))
        print(f"[{model}@{name}] P={overall['precision']} R={overall['recall']} "
              f"F1={overall['f1']} FP={overall['fp']} FN={overall['fn']}")

    os.makedirs(output_dir, exist_ok=True)
    if all_overall:
        with open(os.path.join(output_dir, "model_metrics.csv"), "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_overall[0].keys()))
            writer.writeheader()
            writer.writerows(all_overall)
    if all_scene:
        with open(os.path.join(output_dir, "scene_metrics.csv"), "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_scene[0].keys()))
            writer.writeheader()
            writer.writerows(all_scene)

    # ---- regression 不退化检查 ----
    prev = common.load_yaml(args.prev_acceptance) if args.prev_acceptance and \
        os.path.exists(args.prev_acceptance) else None
    prev_fn_rate = (prev or {}).get("regression_fn_rate")
    regression_pass, fn_rate = True, None
    if reg_overall and reg_overall["positive_rows"]:
        fn_rate = round(reg_overall["fn"] / reg_overall["positive_rows"], 6)
        if prev_fn_rate is not None and fn_rate > prev_fn_rate:
            regression_pass = False
            print(f"[WARN] regression FN 率 {fn_rate:.4f} 高于上一版本 {prev_fn_rate:.4f}")
    common.save_yaml({"regression_eval": reg_overall, "fn_rate": fn_rate,
                      "prev_fn_rate": prev_fn_rate, "no_degradation": regression_pass},
                     os.path.join(output_dir, "regression_metrics.yaml"))

    # ---- min_improvement 检查（迭代停止保护，以 gold 为准） ----
    primary_overall = gold_overall or all_overall[0]
    eval_split = "gold" if gold_overall else all_overall[0]["dataset"].split("@")[1]
    improvement, improvement_pass = None, True
    if prev and prev.get("f1") is not None and primary_overall.get("f1") is not None:
        improvement = round(primary_overall["f1"] - prev["f1"], 6)
        min_impr = (task_cfg.get("acceptance", {}) or {}).get("min_improvement", 0.0)
        improvement_pass = improvement >= min_impr
        print(f"与上一版本 F1 差异：{improvement:+.4f}（min_improvement={min_impr}）")

    selected = {
        "model": model,
        "candidate_type": ctype,
        "best_model_dir": model_dir,
        "threshold": threshold,
        "best_hparams": {k: hp.get(k) for k in
                         ("learning_rate", "batch_size", "epochs", "weight_decay", "best_cfg_idx")}
        if ctype == common.CANDIDATE_TYPE_BERT else None,
        "evaluated_on": eval_split,
        **{k: primary_overall[k] for k in ("precision", "recall", "f1", "fp", "fn")},
        "regression_pass": regression_pass,
        "improvement_vs_prev": improvement,
        "improvement_pass": improvement_pass,
    }
    if ctype == common.CANDIDATE_TYPE_LAYA:
        # §17.3 三元组：checkpoint 引用 + laya_question.yaml 相对路径 + best_threshold
        selected.update(laya_desc)
        selected["laya_question"] = laya_scorer.QUESTION_FILENAME
        selected.pop("best_hparams", None)
    if ctype == common.CANDIDATE_TYPE_KEV:
        selected.update(laya_desc)
        selected.pop("best_hparams", None)
    common.save_yaml(selected, os.path.join(output_dir, "selected_model.yaml"))
    common.update_status(args.root, args.version, model=model,
                         artifact=("metrics/model_metrics.csv", len(all_overall)),
                         event=f"model evaluated [{model}]: "
                               f"P={selected['precision']} R={selected['recall']} F1={selected['f1']}")
    print(f"[OK] {model} 评估完成（{eval_split}），threshold={threshold}，"
          f"输出 {output_dir}/selected_model.yaml")


if __name__ == "__main__":
    main()
