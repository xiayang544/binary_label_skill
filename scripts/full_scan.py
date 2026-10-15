# -*- coding: utf-8 -*-
"""全量扫描：Regex 初筛 → 模型精筛（BERT / laya）→ 选定模型预测与 union 正例。

对应 cold_start.md 步骤 13 / iteration.md 步骤 13。

流程（Pipeline：Regex 召回 → 检测模型对召回样本打分）：
    1. 遍历线上数据（可分片：--shard i --num-shards n 多卡并行）；
    2. Regex 命中样本进入模型推理（--skip-regex 时全量直接过模型）；
    3. union_positive = 模型判正 ∪ Regex 命中（Regex 宽松召回时供人工兜底）。

laya 候选（type: laya，SKILL.md §17）：精筛用 Router 打分（laya_prob 判定），
--model-dir 指 checkpoint 目录，--model-dir/--threshold 均可缺省——
从当前版本 metrics/selected_model.yaml 记录的三元组（checkpoint + question + threshold）
解析，其次 task.yaml candidates path。

用法（BERT）：
    CUDA_VISIBLE_DEVICES=0 python <skill>/scripts/full_scan.py --version v_0 \
        --model-dir v_0/training/macbert/best_model --threshold 0.35 \
        [--regex-config v_0/standard/regex/regex_config.yaml] \
        [--output-dir v_0/scan] [--shard 0 --num-shards 1]

用法（laya）：
    python <skill>/scripts/full_scan.py --version v_0 --model laya_multilingual \
        [--model-dir <checkpoint>] [--threshold <t>]   # 缺省读 selected_model.yaml 三元组

输出：
    <output-dir>/selected_model_predictions.jsonl/.xlsx   Regex 召回样本 + 模型打分
    <output-dir>/union_positive.jsonl/.xlsx               模型判正 ∪ Regex 命中
    <output-dir>/scan_report.yaml
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, kev_scorer, laya_scorer, online_data, regex_engine, text_utils  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from train_bert import predict_scores  # noqa: E402


class BufferedScorer:
    """流式分批打分：攒满 buffer 推理一次，避免全量驻留内存（BERT）。"""

    def __init__(self, model, tokenizer, bert_turns, max_length, batch_size, device):
        self.model, self.tokenizer = model, tokenizer
        self.bert_turns, self.max_length = bert_turns, max_length
        self.batch_size, self.device = batch_size, device
        self.buffer = []

    def add(self, record):
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size * 8:
            return self.flush()
        return []

    def flush(self):
        if not self.buffer:
            return []
        scores = predict_scores(self.model, self.tokenizer, self.buffer, self.bert_turns,
                                self.max_length, self.batch_size, self.device)
        out = []
        for r, s in zip(self.buffer, scores):
            rec = dict(r)
            rec.update({"bert_score": round(s, 6), "score": round(s, 6)})
            out.append(rec)
        self.buffer = []
        return out


class BufferedLayaScorer:
    """laya 流式分批打分（与 BufferedScorer 同接口）：add/flush，产出 laya_prob 字段。"""

    def __init__(self, scorer, flush_size=256):
        self.scorer = scorer
        self.score_field = common.score_fields(scorer.describe()["candidate_type"])["score"]
        self.flush_size = flush_size
        self.buffer = []

    def add(self, record):
        self.buffer.append(record)
        if len(self.buffer) >= self.flush_size:
            return self.flush()
        return []

    def flush(self):
        if not self.buffer:
            return []
        scores = self.scorer.score(self.buffer)
        out = []
        for r, s in zip(self.buffer, scores):
            rec = dict(r)
            rec.update({self.score_field: round(s, 6), "score": round(s, 6)})
            out.append(rec)
        self.buffer = []
        return out


def main():
    parser = argparse.ArgumentParser(description="Regex 初筛 + 模型精筛（BERT / laya）全量扫描")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model-dir", default=None,
                        help="BERT：train_bert 产出的 best_model 目录（必填）；"
                             "laya：checkpoint 目录（缺省读 selected_model.yaml / task.yaml）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="判定阈值（laya 候选可缺省，读 selected_model.yaml；BERT 必填）")
    parser.add_argument("--regex-config", default=None, help="默认当前版本 standard/regex 下；--skip-regex 可关闭")
    parser.add_argument("--skip-regex", action="store_true", help="跳过 Regex，全量直接过模型")
    parser.add_argument("--model", default=None,
                        help="目标模型 key（task.yaml models.candidates）；缺省取唯一 enabled 候选")
    parser.add_argument("--online-data", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shard", type=int, default=0, help="分片编号（0 起）")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--since", default=None, help="数据日期下界（YYYY-MM-DD，含），增量扫描用")
    parser.add_argument("--until", default=None, help="数据日期上界（YYYY-MM-DD，含）")
    parser.add_argument("--append", action="store_true",
                        help="增量模式：把本次结果按 sample_id 去重后合并进已有扫描产物"
                             "（本次覆盖重复项）。数据增长后在同一版本上重扫用；"
                             "Regex/模型/threshold 变化时应全量覆盖重扫，不用本参数")
    args = parser.parse_args()

    task_cfg, ctx, regex_turns = common.task_ctx(args.root, "regex_turns", 10)
    _, _, bert_turns = common.task_ctx(args.root, "bert_turns", 1)
    max_length = (task_cfg.get("training", {}) or {}).get("max_length", 512)
    online_dir = args.online_data or task_cfg.get("input", {}).get("online_data_path")
    if not online_dir or not os.path.isdir(online_dir):
        raise FileNotFoundError(f"线上数据目录不存在：{online_dir}")

    model = common.resolve_model(args.root, args.model, task_cfg)
    ctype = common.candidate_type(model, task_cfg)
    fields = common.score_fields(ctype)
    vdir = common.version_dir(args.root, args.version, model=model, task_cfg=task_cfg)
    output_dir = args.output_dir or os.path.join(vdir, "scan")

    engine = None
    if not args.skip_regex:
        config_path = args.regex_config or os.path.join(vdir, "standard", "regex", "regex_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Regex 配置不存在：{config_path}（或使用 --skip-regex）")
        engine = regex_engine.load_engine(config_path)

    laya_desc = {}
    if ctype == common.CANDIDATE_TYPE_LAYA:
        # 三元组解析：--model-dir/--threshold > selected_model.yaml > task.yaml path（§17.3）
        sel_path = os.path.join(vdir, "metrics", "selected_model.yaml")
        sel = common.load_yaml(sel_path) if os.path.exists(sel_path) else {}
        if not args.model_dir:
            args.model_dir = sel.get("best_model_dir") or sel.get("checkpoint")
        if args.threshold is None:
            args.threshold = sel.get("threshold")
        if args.threshold is None:
            raise SystemExit(
                "laya 候选需要 --threshold，或先运行 select_model.py 生成 "
                f"metrics/selected_model.yaml（{sel_path}）")
        scorer = laya_scorer.build_laya_scorer(
            args.root, model, version=args.version, task_cfg=task_cfg,
            checkpoint=args.model_dir, batch_size=args.batch_size)
        laya_desc = scorer.describe()
        buffered = BufferedLayaScorer(scorer, flush_size=max(32, args.batch_size * 2))
        print(f"[laya] checkpoint={args.model_dir or 'hub-default'} threshold={args.threshold}")
    elif ctype == common.CANDIDATE_TYPE_KEV:
        sel_path = os.path.join(vdir, "metrics", "selected_model.yaml")
        selected = common.load_yaml(sel_path) if os.path.exists(sel_path) else {}
        args.model_dir = args.model_dir or selected.get("checkpoint") or selected.get("best_model_dir")
        if args.threshold is None:
            args.threshold = selected.get("threshold")
        if args.threshold is None:
            raise SystemExit("Kev 候选需要 --threshold 或 metrics/selected_model.yaml")
        scorer = kev_scorer.build_kev_scorer(
            args.root, model, args.version, task_cfg=task_cfg, checkpoint=args.model_dir)
        args.model_dir = scorer.checkpoint
        laya_desc = scorer.describe()
        buffered = BufferedLayaScorer(scorer, flush_size=max(1, args.batch_size))
    else:
        if not args.model_dir:
            raise SystemExit("BERT 候选必须提供 --model-dir（train_bert 产出的 best_model 目录）")
        if args.threshold is None:
            raise SystemExit("BERT 候选必须提供 --threshold")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model_obj = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()
        buffered = BufferedScorer(model_obj, tokenizer, bert_turns, max_length,
                                  args.batch_size, device)

    total = shard_total = 0
    predictions = []   # Regex 召回（或全量）+ 模型打分
    union_positive = []

    def _process_scored(records):
        for rec in records:
            rec[fields["threshold"]] = args.threshold
            rec[fields["prediction"]] = "positive" \
                if rec[fields["score"]] >= args.threshold else "negative"
            predictions.append(rec)
            # union_positive = 模型判正 ∪ Regex 命中（skip-regex 时即模型判正）
            if rec[fields["prediction"]] == "positive" or bool(rec.get("regex_hit")):
                union_positive.append(rec)

    for sample in online_data.iter_online_samples(online_dir, args.include,
                                                  since=args.since, until=args.until):
        total += 1
        if total % args.num_shards != args.shard:
            continue
        shard_total += 1
        context = text_utils.build_context(sample["conversation_turns"], regex_turns)
        regex_result = engine.match(sample["current_turn"], context) if engine else \
            {"hit": False, "rule_ids": [], "scene_ids": [], "names": []}
        if engine and not regex_result["hit"]:
            continue  # Regex 未召回，不进入 BERT
        record = {
            "sample_id": sample["sample_id"],
            "conversation_id": sample["conversation_id"],
            "turn_id": sample["turn_id"],
            "business_scene": sample["business_scene"],
            "current_turn": sample["current_turn"],
            "context": context,
            "conversation_turns": sample["conversation_turns"],
            "version": args.version,
            "data_date": sample.get("data_date"),
            "regex_hit": regex_result["hit"],
            "regex_rule_ids": regex_result["rule_ids"],
            "regex_scene_ids": regex_result["scene_ids"],
            "regex_version": (engine.regex_version if engine else None) or args.version,
            "source_file": sample["source_file"],
        }
        _process_scored(buffered.add(record))
        if shard_total % 20000 == 0:
            print(f"已处理 {shard_total}（全局 {total}），召回后 {len(predictions)}", flush=True)

    _process_scored(buffered.flush())

    # 增量模式：与已有扫描产物合并去重（本次结果优先覆盖重复项），
    # union_positive 从合并后的 predictions 按既有预测字段/regex_hit 重派生（按候选类型）
    if args.append:
        pred_path = os.path.join(output_dir, "selected_model_predictions.jsonl")
        if os.path.exists(pred_path):
            old = common.load_jsonl(pred_path)
            new_ids = {r["sample_id"] for r in predictions}
            dup = sum(1 for r in old if r["sample_id"] in new_ids)
            if dup:
                print(f"[WARN] 增量合并：{dup} 个 sample_id 与已有扫描结果重复，以本次结果覆盖")
            old = [r for r in old if r["sample_id"] not in new_ids]
            predictions = old + predictions
            union_positive = [r for r in predictions
                              if common.get_prediction(r, ctype) == "positive"
                              or bool(r.get("regex_hit"))]
            print(f"增量合并：已有 {len(old)} + 本次新增 {len(predictions) - len(old)} = {len(predictions)}")

    common.save_records(predictions, os.path.join(output_dir, "selected_model_predictions"))
    common.save_records(union_positive, os.path.join(output_dir, "union_positive"))

    report = {
        "version": args.version,
        "model": model,
        "candidate_type": ctype,
        "model_dir": args.model_dir,
        "threshold": args.threshold,
        "shard": f"{args.shard}/{args.num_shards}",
        "date_range": {"since": args.since, "until": args.until},
        "append_mode": args.append,
        "total_samples_seen": total,
        "shard_samples": shard_total,
        "regex_recalled": len(predictions),
        "model_positive": sum(1 for r in predictions
                               if common.get_prediction(r, ctype) == "positive"),
        "union_positive": len(union_positive),
        "scene_distribution": common.count_by(union_positive, "regex_scene_ids"),
        "data_date_counts": common.count_by(predictions, "data_date"),
        "created_at": common.now_str(),
        **laya_desc,
    }
    # 类型特定正例计数字段（兼容既有下游读取 bert_positive 的习惯）
    report[f"{ctype}_positive"] \
        = report["model_positive"]
    common.save_yaml(report, os.path.join(output_dir, "scan_report.yaml"))
    common.update_status(args.root, args.version, model=model,
                         artifact=("scan/selected_model_predictions.jsonl", len(predictions)),
                         event=f"full scan [{ctype}]: {len(predictions)} recalled, "
                               f"{report['model_positive']} model positive")
    print(f"[OK] Regex 召回 {len(predictions)}，模型判正 {report['model_positive']}，"
          f"union {len(union_positive)}，输出 {output_dir}")


if __name__ == "__main__":
    main()
