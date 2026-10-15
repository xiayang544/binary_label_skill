# -*- coding: utf-8 -*-
"""LLM 打标：按当前版本 llm_label_prompt.md 对样本池打标。

对应 cold_start.md 步骤 5（单模型）/ iteration.md 步骤 3（多模型，分别调用）。

用法（项目根目录执行）：
    python <skill>/scripts/label_llm.py --version v_0 --model h256 \
        --input h256/v_0/prefilter/recalled.jsonl \
        --llm LongCat-Flash-Chat \
        [--out-dir h256/v_0/labeling/llm] [--sample-size 2000] \
        [--workers 4] [--rpm 20]

参数约定：--model 为检测器（BERT 候选 key，定位流水线目录）；
--llm 为执行打标的大模型显示名。

输出：
    <out-dir>/raw/<slug>.jsonl     原始输出（含 llm_raw_response，只追加）
    <out-dir>/<slug>.jsonl/.xlsx   标准化结果（schema llm_annotation_fields）
"""

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, llm_client, text_utils  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="LLM 打标（冷启动/迭代通用）")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key（task.yaml models.candidates）；缺省取唯一 enabled 候选")
    parser.add_argument("--input", required=True, help="样本池 jsonl/xlsx（如 prefilter/recalled.jsonl）")
    parser.add_argument("--llm", required=True, help="执行打标的大模型显示名，如 LongCat-Flash-Chat")
    parser.add_argument("--prompt", default=None, help="llm_label_prompt.md（默认当前版本 standard 下）")
    parser.add_argument("--out-dir", default=None, help="默认 <model>/<version>/labeling/llm")
    parser.add_argument("--sample-size", type=int, default=0, help="先随机采样 N 条（0=全部）")
    parser.add_argument("--save-sample", action="store_true",
                        help="采样后先落盘样本快照（配合 --sample-size，冷启动用）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    task_cfg, ctx, llm_turns = common.task_ctx(args.root, "llm_label_turns", None)
    vdir = common.version_dir(args.root, args.version, model=args.model, task_cfg=task_cfg)
    prompt_path = args.prompt or os.path.join(vdir, "standard", "llm_label_prompt.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"打标 Prompt 不存在：{prompt_path}")
    with open(prompt_path, encoding="utf-8") as f:
        system_prompt = f.read()
    prompt_version = f"md5_{common.file_md5(prompt_path)}"

    records = common.load_records(args.input)
    if args.sample_size and len(records) > args.sample_size:
        records = random.Random(args.seed).sample(records, args.sample_size)
        print(f"随机采样 {len(records)} 条（seed={args.seed}）")
    if args.save_sample:
        sample_path = os.path.join(vdir, "labeling", f"coldstart_sample_{len(records)}")
        common.save_records(records, sample_path)
        common.update_status(args.root, args.version, model=args.model,
                             artifact=(f"labeling/coldstart_sample_{len(records)}.jsonl",
                                       len(records)),
                             event=f"coldstart sample saved: {len(records)}")
        print(f"采样快照：{sample_path}.jsonl / .xlsx")

    out_dir = args.out_dir or os.path.join(vdir, "labeling", "llm")
    slug = common.slug(args.llm)
    raw_path = os.path.join(out_dir, "raw", f"{slug}.jsonl")

    def build_user(row):
        return text_utils.build_llm_user_prompt(row, llm_turns)

    finished = llm_client.label_records(
        records, args.llm, system_prompt, build_user, raw_path,
        task_cfg=task_cfg, workers=args.workers, rpm=args.rpm,
        max_retries=args.max_retries,
    )

    # 标准化结果（去掉超长 raw response，补 prompt_version）
    normalized = []
    for r in finished:
        out = dict(r)
        out["label_prompt_version"] = prompt_version
        raw = out.pop("llm_raw_response", None)
        normalized.append(out)
    common.save_records(normalized, os.path.join(out_dir, slug))

    n_ok = sum(1 for r in normalized if r.get("llm_parse_success"))
    n_pos = sum(1 for r in normalized if r.get("llm_label") == "positive")
    common.update_status(
        args.root, args.version, model=args.model,
        artifact=(f"labeling/llm/{slug}.jsonl", len(normalized)),
        event=f"llm label [{args.llm}]: {len(normalized)} labeled, {n_ok} parsed, {n_pos} positive")
    print(f"[OK] {args.llm}: {len(normalized)} 条（解析成功 {n_ok}，正例 {n_pos}），输出 {out_dir}/{slug}.jsonl")


if __name__ == "__main__":
    main()
