# -*- coding: utf-8 -*-
"""维护全局 Gold 数据：全量合入人工核验样本 / 需求变更改标。

对应 cold_start.md 步骤 6（初版构建）、iteration.md 步骤 4（每轮追加）与
步骤 6（需求变更改标）。完整规则见 SKILL.md §8。

收纳策略：
- 只使用 human_verified=true 且非合成（data_source != synthetic）的真实样本；
- 默认 --mode all：本轮全部人工核验样本合入 gold；
  --mode sample：正例按 scene_id 分层抽样（--per-scene），
  负例 = 明确负例 + 易混淆负例（LLM 曾误判为正，--confusable-ratio），
  用于需要控制 gold 规模的场景；
- 已存在 gold 时按 sample_id 去重：label 相同的样本跳过；
- 需求变更改标：显式 --update-labels --reason "<变更说明>"，
  以输入中的最新人工结论覆盖旧记录，旧 label 连同原版本/审核人/变更版本/
  原因/时间追加进 label_history（完整审计链），样本自动在
  gold_positive / gold_negative 之间迁移；
  未开启 --update-labels 时，label 冲突样本跳过并逐条告警；
- 进入 gold 的样本由 build_dataset.py 自动从 train/validation 候选池剔除。

用法：
    # 每轮人工复核后：全量追加（按 sample_id 去重）
    python <skill>/scripts/build_gold.py --version v_1 \
        --input <model>/v_1/labeling/human/review_result.jsonl

    # 需求变更改标（必须给出 reason，审计要求）
    python <skill>/scripts/build_gold.py --version v_1 \
        --input <需求变更人工重标后的 review_result.jsonl> \
        --update-labels --reason "需求变更：健康证话术一律判负例"
"""

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="构建 / 追加 / 改标全局 Gold 数据")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选（gold 本身为全局资产）")
    parser.add_argument("--input", required=True, nargs="+",
                        help="review_result.jsonl（可多个，后出现者对同一样本生效）")
    parser.add_argument("--gold-dir", default=None, help="默认 <root>/data/gold")
    parser.add_argument("--mode", choices=["all", "sample"], default="all",
                        help="all=全量采纳人工核验样本（默认，SKILL.md §8）；"
                             "sample=分层抽样控制 gold 规模")
    parser.add_argument("--per-scene", type=int, default=10,
                        help="[sample 模式] 每个子场景抽取正例数")
    parser.add_argument("--negative-ratio", type=float, default=2.0,
                        help="[sample 模式] 负例数 = 正例数 × 该比例")
    parser.add_argument("--confusable-ratio", type=float, default=0.5,
                        help="[sample 模式] 负例中 LLM 曾误判为正的比例（易混淆负例）")
    parser.add_argument("--update-labels", action="store_true",
                        help="允许覆盖已存在 sample_id 的 label（需求变更改标，须配 --reason）")
    parser.add_argument("--reason", default=None,
                        help="改标原因（--update-labels 时必填），写入 label_history 审计")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.update_labels and not args.reason:
        parser.error("--update-labels 需要同时通过 --reason 给出改标原因（审计要求）")

    gold_dir = os.path.abspath(args.gold_dir or os.path.join(args.root, "data", "gold"))
    pos_path = os.path.join(gold_dir, "gold_positive.jsonl")
    neg_path = os.path.join(gold_dir, "gold_negative.jsonl")

    # ---- 读取输入：只取人工核验的真实样本，输入内按 sample_id 去重（后出现者为准）----
    records = []
    for p in args.input:
        records.extend(common.load_records(p))
    verified = [r for r in records
                if r.get("human_verified")
                and r.get("label") in ("positive", "negative")
                and r.get("data_source") != "synthetic"]
    by_id = {}
    for r in verified:
        by_id[r.get("sample_id")] = r
    candidates = list(by_id.values())
    if not candidates:
        print("[WARN] 输入中没有 human_verified=true 的真实样本，gold 未变更")
        return

    # ---- 选择本轮合入的样本 ----
    if args.mode == "sample":
        positives = [r for r in candidates if r["label"] == "positive"]
        negatives = [r for r in candidates if r["label"] == "negative"]
        # 易混淆负例：人工判负但 LLM 曾判正
        confusable = [r for r in negatives if r.get("llm_label") == "positive"]

        rng = random.Random(args.seed)
        by_scene = {}
        for r in positives:
            by_scene.setdefault(r.get("scene_id") or "<empty>", []).append(r)

        selected_pos = []
        for scene in sorted(by_scene):
            pool = by_scene[scene]
            take = min(args.per_scene, len(pool))
            selected_pos.extend(rng.sample(pool, take))

        n_neg = round(len(selected_pos) * args.negative_ratio)
        n_conf = round(n_neg * args.confusable_ratio)
        rng.shuffle(confusable)
        rng.shuffle(negatives)
        selected_neg = confusable[:n_conf]
        remaining = n_neg - len(selected_neg)
        if remaining > 0:
            selected_neg += [r for r in negatives if r not in selected_neg][:remaining]
        selected = selected_pos + selected_neg
    else:
        selected = candidates

    # ---- 读取已有 gold ----
    existing_pos = common.load_jsonl(pos_path) if os.path.exists(pos_path) else []
    existing_neg = common.load_jsonl(neg_path) if os.path.exists(neg_path) else []
    final_pos, final_neg = list(existing_pos), list(existing_neg)
    pos_ids = {r.get("sample_id") for r in final_pos}
    neg_ids = {r.get("sample_id") for r in final_neg}

    n_new_pos = n_new_neg = n_dup = 0
    pos_missing_scene = 0
    relabeled = []    # [(sample_id, old, new)]
    conflicts = []    # [(sample_id, old, new)]

    def _old_record(sid, in_pos, in_neg):
        for r in (final_pos if in_pos else final_neg):
            if r.get("sample_id") == sid:
                return r
        return None

    for r in selected:
        sid = r.get("sample_id")
        in_pos, in_neg = sid in pos_ids, sid in neg_ids
        if not in_pos and not in_neg:
            # 新增样本
            rec = dict(r)
            rec["dataset_split"] = "gold"
            if rec["label"] == "positive":
                final_pos.append(rec)
                pos_ids.add(sid)
                n_new_pos += 1
                if not rec.get("scene_id"):
                    pos_missing_scene += 1
            else:
                final_neg.append(rec)
                neg_ids.add(sid)
                n_new_neg += 1
            continue

        old_label = "positive" if in_pos else "negative"
        if old_label == r.get("label"):
            n_dup += 1
            continue
        if not args.update_labels:
            conflicts.append((sid, old_label, r.get("label")))
            continue

        # 需求变更改标：以最新人工结论覆盖，旧记录信息进入 label_history 审计链
        old_rec = _old_record(sid, in_pos, in_neg)
        rec = dict(r)
        rec["dataset_split"] = "gold"
        history = list((old_rec or {}).get("label_history") or [])
        history.append({
            "label": (old_rec or {}).get("label") or old_label,
            "version": (old_rec or {}).get("version"),
            "reviewer": (old_rec or {}).get("reviewer"),
            "changed_in_version": args.version,
            "reason": args.reason,
            "changed_at": common.now_str(),
        })
        rec["label_history"] = history
        if in_pos:
            final_pos = [x for x in final_pos if x.get("sample_id") != sid]
            pos_ids.discard(sid)
        if in_neg:
            final_neg = [x for x in final_neg if x.get("sample_id") != sid]
            neg_ids.discard(sid)
        if rec["label"] == "positive":
            final_pos.append(rec)
            pos_ids.add(sid)
            if not rec.get("scene_id"):
                pos_missing_scene += 1
        else:
            final_neg.append(rec)
            neg_ids.add(sid)
        relabeled.append((sid, old_label, rec["label"]))

    # ---- 落盘（jsonl + xlsx 双格式）----
    common.save_records(final_pos, os.path.join(gold_dir, "gold_positive"))
    common.save_records(final_neg, os.path.join(gold_dir, "gold_negative"))

    # gold 是全局资产，不属于单版本，只记 event 不登记版本 artifacts
    event = (f"gold updated: +{n_new_pos} positive / +{n_new_neg} negative"
             + (f", relabeled {len(relabeled)}" if relabeled else "")
             + f" (total {len(final_pos)}/{len(final_neg)})")
    common.update_status(args.root, args.version, model=args.model, event=event)

    print(f"[OK] Gold 新增 {n_new_pos} 正例 / {n_new_neg} 负例"
          f"（模式 {args.mode}，输入候选 {len(candidates)} 条），"
          f"改标 {len(relabeled)} 条，重复跳过 {n_dup} 条")
    if relabeled:
        print("     改标明细（label_history 已留审计）：")
        for sid, old, new in relabeled:
            print(f"       - {sid}: {old} -> {new}")
    print(f"     当前总量：positive {len(final_pos)} / negative {len(final_neg)}")
    scenes = sorted({r.get("scene_id") for r in final_pos if r.get("scene_id")})
    print(f"     正例场景覆盖：{scenes}")
    if pos_missing_scene:
        print(f"[WARN] {pos_missing_scene} 条新增/改标正例 scene_id 为空（正例必须归属子场景，请补充）")
    if conflicts:
        print(f"[WARN] {len(conflicts)} 条样本 label 与 gold 现值冲突，"
              f"未开启 --update-labels 已跳过（确认属于需求变更请加 --update-labels --reason）：")
        for sid, old, new in conflicts[:20]:
            print(f"       - {sid}: gold={old}, input={new}")
        if len(conflicts) > 20:
            print(f"       ... 其余 {len(conflicts) - 20} 条略")
    print("     注意：进入 gold 的样本已由 build_dataset.py 自动从 train/validation 剔除")


if __name__ == "__main__":
    main()
