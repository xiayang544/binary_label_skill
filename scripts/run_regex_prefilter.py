# -*- coding: utf-8 -*-
"""Regex 初筛：对线上全量数据执行当前版本 Regex，产出召回/未召回样本与报告。

对应 cold_start.md 步骤 3 / iteration.md 步骤 13 的初筛环节。

用法（项目根目录执行）：
    python <skill>/scripts/run_regex_prefilter.py --version v_0 \
        [--online-data <dir>] [--config <regex_config.yaml>] \
        [--output-dir v_0/prefilter] [--rejected-sample-size 500]

输出：
    <output-dir>/recalled.jsonl / .xlsx          Regex 命中样本（候选池）
    <output-dir>/rejected_sample.jsonl / .xlsx   未命中抽样（用于漏召回检查）
    <output-dir>/prefilter_report.yaml           规模、规则/场景分布
    <regex所在目录>/regex_report.yaml             历史 case 回归测试结果
"""

import argparse
import os
import random
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, online_data, regex_engine, text_utils  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Regex 初筛线上全量数据")
    parser.add_argument("--root", default=".", help="项目根目录（含 task.yaml）")
    parser.add_argument("--version", required=True, help="当前版本，如 v_0")
    parser.add_argument("--model", default=None,
                        help="目标模型 key（task.yaml models.candidates）；缺省取唯一 enabled 候选")
    parser.add_argument("--online-data", default=None, help="线上数据目录（默认读 task.yaml）")
    parser.add_argument("--config", default=None, help="regex_config.yaml 路径")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认 <root>/<version>/prefilter")
    parser.add_argument("--rejected-sample-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="最多处理样本数（调试用，0=全部）")
    parser.add_argument("--include", nargs="*", default=None, help="文件名子串白名单")
    parser.add_argument("--since", default=None, help="数据日期下界（YYYY-MM-DD，含），增量初筛用")
    parser.add_argument("--until", default=None, help="数据日期上界（YYYY-MM-DD，含）")
    parser.add_argument("--append", action="store_true",
                        help="增量模式：把本次召回结果按 sample_id 去重后合并进已有 recalled.jsonl，"
                             "而不是整体重写（配合 --since 处理新增日期数据）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_cfg, ctx, regex_turns = common.task_ctx(args.root, "regex_turns", 10)
    online_dir = args.online_data or task_cfg.get("input", {}).get("online_data_path")
    if not online_dir or not os.path.isdir(online_dir):
        raise FileNotFoundError(f"线上数据目录不存在：{online_dir}")

    vdir = common.version_dir(args.root, args.version, model=args.model, task_cfg=task_cfg)
    config_path = args.config or os.path.join(vdir, "standard", "regex", "regex_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Regex 配置不存在：{config_path}")
    engine = regex_engine.load_engine(config_path)
    print(f"Regex 版本 {engine.regex_version}，共 {len(engine.rules)} 条规则，上下文轮数 {regex_turns}")

    output_dir = args.output_dir or os.path.join(vdir, "prefilter")
    rejected_size = args.rejected_sample_size
    if rejected_size is None:
        rejected_size = (task_cfg.get("regex", {}).get("rejected_sample", {}) or {}).get("sample_size", 500)

    recalled, rejected = [], []
    rule_counter, scene_counter, date_counter = Counter(), Counter(), Counter()
    total = 0
    for sample in online_data.iter_online_samples(online_dir, args.include,
                                                  since=args.since, until=args.until):
        total += 1
        if args.limit and total > args.limit:
            total -= 1
            break
        context = text_utils.build_context(sample["conversation_turns"], regex_turns)
        result = engine.match(sample["current_turn"], context)
        record = {
            "sample_id": sample["sample_id"],
            "conversation_id": sample["conversation_id"],
            "turn_id": sample["turn_id"],
            "business_scene": sample["business_scene"],
            "current_turn": sample["current_turn"],
            "context": context,
            "conversation_turns": sample["conversation_turns"],
            "label": None,
            "scene_id": None,
            "data_source": "real",
            "annotation_source": "unknown",
            "human_verified": False,
            "version": args.version,
            "data_date": sample.get("data_date"),
            "regex_hit": result["hit"],
            "regex_rule_ids": result["rule_ids"],
            "regex_scene_ids": result["scene_ids"],
            "regex_version": engine.regex_version or args.version,
            "source_file": sample["source_file"],
        }
        if sample.get("data_date"):
            date_counter[sample["data_date"]] += 1
        if result["hit"]:
            recalled.append(record)
            rule_counter.update(result["rule_ids"])
            scene_counter.update(result["scene_ids"])
        else:
            rejected.append(record)
        if total % 20000 == 0:
            print(f"已处理 {total}，召回 {len(recalled)}", flush=True)

    rng = random.Random(args.seed)
    rejected_sample = rng.sample(rejected, min(rejected_size, len(rejected)))

    # 增量模式：与已有 recalled 合并去重（本次结果优先，即最新 Regex 的命中字段覆盖旧值）
    existing_recalled = []
    if args.append:
        recalled_path = os.path.join(output_dir, "recalled.jsonl")
        if os.path.exists(recalled_path):
            existing_recalled = common.load_jsonl(recalled_path)
            old_ids = {r["sample_id"] for r in existing_recalled}
            new_ids = {r["sample_id"] for r in recalled}
            dup = old_ids & new_ids
            if dup:
                print(f"[WARN] 增量合并：{len(dup)} 个 sample_id 与已有记录重复"
                      f"（如本次日期范围与历史重叠），以本次结果覆盖")
            existing_recalled = [r for r in existing_recalled if r["sample_id"] not in new_ids]
            recalled = existing_recalled + recalled
            print(f"增量合并：已有 {len(existing_recalled)} + 本次新增 "
                  f"{len(recalled) - len(existing_recalled)} = {len(recalled)}")

    common.save_records(recalled, os.path.join(output_dir, "recalled"))
    common.save_records(rejected_sample, os.path.join(output_dir, "rejected_sample"))

    # 历史 case 回归测试（SKILL.md §11：Regex 更新后必须重跑）
    regression = None
    cases_path = os.path.join(os.path.dirname(config_path), "regex_cases.jsonl")
    if os.path.exists(cases_path):
        regression = regex_engine.run_regression(engine, common.load_jsonl(cases_path))
        common.save_yaml(regression, os.path.join(os.path.dirname(config_path), "regex_report.yaml"))
        print(f"回归测试：{regression['passed']}/{regression['total']} 通过")
        if regression["failed"]:
            for f in regression["failures"][:10]:
                print(f"  [FAIL] {f['case_id']}: expect={f['expect_hit']}/{f['expect_rule_ids']} "
                      f"actual={f['actual_hit']}/{f['actual_rule_ids']}")

    report = {
        "version": args.version,
        "regex_config": os.path.relpath(config_path, args.root),
        "regex_turns": regex_turns,
        "online_data_dir": online_dir,
        "date_range": {"since": args.since, "until": args.until},
        "append_mode": args.append,
        "total_samples": total,
        "recalled": len(recalled),
        "recall_rate": round(len(recalled) / total, 6) if total else None,
        "rejected_total": len(rejected),
        "rejected_sample_size": len(rejected_sample),
        "rule_hit_counts": dict(rule_counter.most_common()),
        "scene_hit_counts": dict(scene_counter.most_common()),
        "data_date_counts": dict(sorted(date_counter.items())),
        "regression_test": {k: v for k, v in (regression or {}).items() if k != "failures"},
    }
    common.save_yaml(report, os.path.join(output_dir, "prefilter_report.yaml"))
    common.update_status(args.root, args.version, model=args.model,
                         artifact=("prefilter/recalled.jsonl", len(recalled)),
                         event=f"regex prefilter: {len(recalled)}/{total} recalled"
                               + (f" (since {args.since}, append)" if args.since or args.append else ""))
    print(f"[OK] 召回 {len(recalled)}/{total}，输出 {output_dir}")
    if date_counter:
        dates = sorted(date_counter)
        print(f"     数据日期：{dates[0]} ~ {dates[-1]}，共 {len(dates)} 天"
              f"（分布见 prefilter_report.yaml data_date_counts）")


if __name__ == "__main__":
    main()
