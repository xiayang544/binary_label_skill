# -*- coding: utf-8 -*-
"""构建数据集：LLM 标注 + 人工复核 + 合成数据 → train / validation / regression_eval。

对应 cold_start.md 步骤 10 / iteration.md 步骤 9。

数据流（详见 SKILL.md §12 数据集角色）：
    人工复核样本（human_verified） → validation 候选（conversation 级切分）+ train
    LLM 标注（未经人工复核）        → train（annotation_source=llm）
    合成数据                        → train（data_source=synthetic）
    真实 FP/FN（--regression-delta）→ regression_eval（禁止进 train）
    上一版本 train / regression_eval → 完整继承（--prev-version）
    gold 样本                       → 全局剔除，任何 split 不得包含

用法：
    python <skill>/scripts/build_dataset.py --version v_0 \
        --llm v_0/labeling/llm/longcat_flash.jsonl \
        --human v_0/labeling/human/review_result.jsonl \
        --synthetic v_0/synthesis/merged_synthetic.jsonl \
        [--prev-version v_0] [--regression-delta v_1/labeling/analysis/fp_fn.jsonl] \
        [--output-dir v_1/dataset] [--validation-ratio 0.2]
"""

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common  # noqa: E402


def to_final(r, version):
    """统一整理为正式样本（base_fields 对齐）。"""
    out = dict(r)
    out["version"] = version
    out.setdefault("data_source", "real")
    out.setdefault("human_verified", bool(out.get("human_verified")))
    out.pop("dataset_split", None)
    return out


def split_conversation_level(records, val_ratio, seed):
    """conversation 级切分：同一 conversation 只进一个 split，按正负例分别分层。"""
    rng = random.Random(seed)
    by_label = {"positive": {}, "negative": {}}
    for r in records:
        conv = str(r.get("conversation_id"))
        by_label.setdefault(r.get("label"), {}).setdefault(conv, []).append(r)

    val, train = [], []
    for label, convs in by_label.items():
        conv_ids = sorted(convs)
        rng.shuffle(conv_ids)
        # 每个 label 组内按 conversation 数量比例切分；至少留 1 个 conv 给 train
        n_val = max(1, min(len(conv_ids) - 1, round(len(conv_ids) * val_ratio))) if len(conv_ids) > 1 else 0
        for i, cid in enumerate(conv_ids):
            (val if i < n_val else train).extend(convs[cid])
    return train, val


def main():
    parser = argparse.ArgumentParser(description="构建 train/validation/regression_eval 数据集")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--llm", nargs="*", default=[], help="LLM 标注标准化结果（jsonl/xlsx/目录）")
    parser.add_argument("--vote", default=None,
                        help="vote_merged.jsonl：多 LLM 一致样本自动采纳为最终 label "
                             "(annotation_source=llm)；分歧样本须已通过 --human 复核")
    parser.add_argument("--human", default=None, help="人工复核 review_result.jsonl")
    parser.add_argument("--synthetic", nargs="*", default=[], help="合成数据（jsonl/xlsx/目录）")
    parser.add_argument("--prev-version", default=None, help="继承上一版本的 train / regression_eval")
    parser.add_argument("--regression-delta", nargs="*", default=[],
                        help="本轮人工确认的真实 FP/FN（只进 regression_eval）")
    parser.add_argument("--gold-dir", default=None, help="默认 <root>/data/gold，其样本全局剔除")
    parser.add_argument("--output-dir", default=None, help="默认 <root>/<version>/dataset")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="人工复核样本中划入 validation 的比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_cfg = common.load_task_config(args.root)
    model = common.resolve_model(args.root, args.model, task_cfg)
    is_laya = common.is_external_candidate(model, task_cfg)

    vdir = common.version_dir(args.root, args.version, model=model, task_cfg=task_cfg)
    output_dir = args.output_dir or os.path.join(vdir, "dataset")
    gold_dir = args.gold_dir or os.path.join(args.root, "data", "gold")
    gold_ids = set()
    if os.path.isdir(gold_dir):
        for f in os.listdir(gold_dir):
            if f.endswith(".jsonl"):
                gold_ids.update(r.get("sample_id") for r in common.load_jsonl(os.path.join(gold_dir, f)))
    print(f"gold 样本 {len(gold_ids)} 条（从所有 split 中剔除）")

    # ---- 1. 人工复核样本：最终标签，切分 train / validation（laya 候选全量进 validation）----
    human_recs = common.load_many([args.human]) if args.human else []
    human_recs = [to_final(r, args.version) for r in human_recs if r.get("sample_id") not in gold_ids]
    if is_laya:
        # laya 零训练候选无训练集：人工复核样本全部进 validation（仍属于该模型版本资产）
        human_train, validation = [], human_recs
    else:
        human_train, validation = split_conversation_level(human_recs, args.validation_ratio, args.seed)
    print(f"人工复核 {len(human_recs)} 条 → {'validation（laya 全量）' if is_laya else f'train {len(human_train)} / validation {len(validation)}'}")

    # ---- 2. LLM 标注（未经人工复核）→ train（laya 候选跳过，无训练集）----
    human_ids = {r["sample_id"] for r in human_recs}
    llm_recs = []
    if is_laya:
        if args.llm:
            print("[WARN] laya 零训练候选无训练集，--llm 输入忽略（仅构建 validation / regression_eval）")
    else:
        for r in common.load_many(args.llm or []):
            if r.get("sample_id") in human_ids or r.get("sample_id") in gold_ids:
                continue
            if not r.get("llm_parse_success"):
                continue  # 解析失败样本不进训练集
            out = to_final(r, args.version)
            out.update({
                "label": r.get("llm_label"),
                "scene_id": r.get("llm_scene_id") or None,
                "annotation_source": "llm",
                "label_model": r.get("label_model"),
                "label_prompt_version": r.get("label_prompt_version"),
            })
            llm_recs.append(out)

    # ---- 2b. 多 LLM 一致样本：vote_label 为正/负（非 disagreement）时自动采纳（laya 跳过）----
    vote_recs, n_disagreement = [], 0
    if is_laya:
        if args.vote:
            print("[WARN] laya 零训练候选无训练集，--vote 输入忽略")
    elif args.vote:
        for r in common.load_many([args.vote]):
            if r.get("sample_id") in human_ids or r.get("sample_id") in gold_ids:
                continue
            vote_label = r.get("vote_label")
            if vote_label not in ("positive", "negative"):
                n_disagreement += 1
                continue  # 分歧样本不自动采纳，须走人工复核（--human）
            out = to_final(r, args.version)
            out.update({
                "label": vote_label,
                "scene_id": r.get("vote_scene_id") or None if vote_label == "positive" else None,
                "annotation_source": "llm",
                "label_model": "vote",
            })
            if out["label"] == "positive" and not out.get("scene_id"):
                print(f"[WARN] vote 一致正例缺少 scene_id，跳过: {out.get('sample_id')}")
                continue
            vote_recs.append(out)
    print(f"LLM 标注进 train {len(llm_recs)} 条；vote 一致采纳 {len(vote_recs)} 条"
          + (f"（分歧 {n_disagreement} 条待人工复核）" if args.vote else ""))

    # ---- 3. 合成数据 → train（laya 候选跳过）----
    syn_recs = []
    if is_laya:
        if args.synthetic:
            print("[WARN] laya 零训练候选无训练集，--synthetic 输入忽略")
    else:
        for r in common.load_many(args.synthetic or []):
            if r.get("sample_id") in gold_ids:
                continue
            out = to_final(r, args.version)
            out.update({
                "label": r.get("label") or "positive",
                "scene_id": r.get("scene_id") or r.get("synthesis_scene_id"),
                "data_source": "synthetic",
                "annotation_source": r.get("annotation_source") or "llm",
            })
            if out["label"] == "positive" and not out.get("scene_id"):
                print(f"[WARN] 合成正例缺少 scene_id，丢弃: {out.get('sample_id')}")
                continue
            syn_recs.append(out)
        print(f"合成数据进 train {len(syn_recs)} 条")

    # ---- 4. 上一版本继承（laya 候选只继承 regression_eval，不继承 train）----
    prev_train, prev_regression = [], []
    if args.prev_version:
        pdir = common.version_dir(args.root, args.prev_version, model=model, task_cfg=task_cfg)
        for name, target in (("train.jsonl", "prev_train"), ("regression_eval.jsonl", "prev_regression")):
            if is_laya and target == "prev_train":
                continue  # laya 无训练集，不继承上一版本 train
            p = os.path.join(pdir, "dataset", name)
            if os.path.exists(p):
                recs = [r for r in common.load_jsonl(p) if r.get("sample_id") not in gold_ids]
                if target == "prev_train":
                    prev_train.extend(recs)
                else:
                    prev_regression.extend(recs)
        print(f"继承 {args.prev_version}: train {len(prev_train)} / regression_eval {len(prev_regression)}")

    # ---- 5. regression_eval：历史 + 本轮 FP/FN delta（真实 hard case 禁止进 train） ----
    regression = list(prev_regression)
    seen_reg = {r.get("sample_id") for r in regression}
    reg_delta = []
    for r in common.load_many(args.regression_delta or []):
        out = to_final(r, args.version)
        out["human_verified"] = True
        out["source_version"] = out.get("source_version") or args.version
        if out["sample_id"] in seen_reg or out["sample_id"] in gold_ids:
            continue
        seen_reg.add(out["sample_id"])
        reg_delta.append(out)
        regression.append(out)
    print(f"regression_eval：历史 {len(prev_regression)} + 本轮新增 {len(reg_delta)}")

    # ---- 6. 汇总 train（去重：人工 > 上一版本 > LLM > vote 一致 > 合成；laya 候选为空）----
    train, seen = [], set()
    for group in (human_train, prev_train, llm_recs, vote_recs, syn_recs):
        for r in group:
            if r["sample_id"] in seen:
                continue
            seen.add(r["sample_id"])
            train.append(r)

    for r in train:
        r["dataset_split"] = "train"
    for r in validation:
        r["dataset_split"] = "validation"
    for r in regression:
        r["dataset_split"] = "regression_eval"

    if is_laya:
        # laya 零训练候选：不产出 train / train_delta（仍照常产出 validation / regression_eval）
        common.save_records(validation, os.path.join(output_dir, "validation"))
        common.save_records(regression, os.path.join(output_dir, "regression_eval"))
        common.save_records(reg_delta, os.path.join(output_dir, "regression_eval_delta"))
        statistics = {
            "version": args.version,
            "candidate_type": common.candidate_type(model, task_cfg),
            "train": {"skipped": "laya 零训练候选无训练集"},
            "validation": {
                "total": len(validation),
                "label": common.count_by(validation, "label"),
                "scene": common.count_by(validation, "scene_id"),
            },
            "regression_eval": {
                "total": len(regression),
                "delta": len(reg_delta),
                "error_type": common.count_by(regression, "error_type"),
            },
            "gold_excluded": len(gold_ids),
        }
        common.save_yaml(statistics, os.path.join(output_dir, "statistics.yaml"))
        common.update_status(args.root, args.version, model=model,
                             artifact=("dataset/validation.jsonl", len(validation)),
                             event=f"dataset built [laya]: val={len(validation)} reg={len(regression)} (no train)")
        print(f"[OK] laya 零训练候选：validation {len(validation)} / regression_eval {len(regression)}，"
              f"输出 {output_dir}（无 train）")
        return

    common.save_records(train, os.path.join(output_dir, "train"))
    common.save_records(validation, os.path.join(output_dir, "validation"))
    common.save_records(regression, os.path.join(output_dir, "regression_eval"))
    common.save_records(reg_delta, os.path.join(output_dir, "regression_eval_delta"))

    statistics = {
        "version": args.version,
        "train": {
            "total": len(train),
            "label": common.count_by(train, "label"),
            "scene": common.count_by(train, "scene_id"),
            "data_source": common.count_by(train, "data_source"),
            "annotation_source": common.count_by(train, "annotation_source"),
            "from_prev_version": len(prev_train),
        },
        "validation": {
            "total": len(validation),
            "label": common.count_by(validation, "label"),
            "scene": common.count_by(validation, "scene_id"),
        },
        "regression_eval": {
            "total": len(regression),
            "delta": len(reg_delta),
            "error_type": common.count_by(regression, "error_type"),
        },
        "gold_excluded": len(gold_ids),
    }
    common.save_yaml(statistics, os.path.join(output_dir, "statistics.yaml"))
    common.update_status(args.root, args.version, model=model,
                         artifact=("dataset/train.jsonl", len(train)),
                         event=f"dataset built: train={len(train)} val={len(validation)} reg={len(regression)}")
    print(f"[OK] train {len(train)} / validation {len(validation)} / regression_eval {len(regression)}，"
          f"输出 {output_dir}")


if __name__ == "__main__":
    main()
