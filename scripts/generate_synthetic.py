# -*- coding: utf-8 -*-
"""数据合成：按子场景补充正例（冷启动）/ 按错误类型分层生成 hard case（迭代）。

对应 cold_start.md 步骤 9 / iteration.md 步骤 8。
Prompt 取自 standard/synthesis_prompts/<scene_id>.md，模型列表读 task.yaml
synthesis.models。每条合成样本落盘字段对齐 data_schema.yaml 的 synthetic_fields。

两种模式：
    scene：冷启动场景补充，--scenes scene_001 scene_002，
           每模型每场景 --samples-per-scene（默认读 task.yaml）条
    hardcase：迭代 hard case，--hardcase fp.jsonl fn.jsonl（evaluate.py 产出，
           含 error_side），总量 = 确认 hard case 数 × --multiplier（默认读 task.yaml），
           按 scene_id × error_type 分层

LLM 输出要求（每条一次调用，返回 JSON）：
    {"current_turn": "...", "context": "用户: ...\\nAI: ...", "scene_id": "...", "note": "..."}

用法：
    python <skill>/scripts/generate_synthetic.py --version v_0 --mode scene \
        --scenes scene_001 scene_003 --model LongCat-2.0 \
        [--output-dir v_0/synthesis]

    python <skill>/scripts/generate_synthetic.py --version v_1 --mode hardcase \
        --hardcase v_1/labeling/analysis/fp.jsonl v_1/labeling/analysis/fn.jsonl \
        --model LongCat-Flash-Chat [--multiplier 3]
"""

import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, llm_client  # noqa: E402

SYN_USER_TMPL = """请生成 1 条尽可能接近真实线上表达的样本。

要求：
- current_turn 是 AI 的回复，命中目标子场景的说话方式和业务语境；
- context 是该回复之前的对话历史，格式为每轮一行："用户: xxx" 或 "AI: xxx"，可为空字符串；
- 表达多样化，不要套用固定句式；
- 只输出 JSON：{{"current_turn": "...", "context": "...", "scene_id": "{scene_id}", "note": "生成说明"}}"""

HARDCASE_USER_TMPL = """以下是一条人工确认的错误样本（模型误判）：

【错误类型】{error_type}（{error_side}）
【当前 Turn】{current_turn}
【上下文】{context}

请参考该样本，生成 1 条同类型、易触发相同误判、但语义表达不同的相似样本。

要求：
- 若错误类型为 FP（误报），生成应被正确判为负例的相似表达；
- 若错误类型为 FN（漏报），生成应被正确判为正例的相似表达；
- 正例必须标注 scene_id；
- 只输出 JSON：{{"current_turn": "...", "context": "...", "scene_id": "{scene_id}", "note": "生成说明"}}"""


def parse_sample_response(content, scene_id):
    obj = llm_client.extract_json(content)
    current = str(obj.get("current_turn") or "").strip()
    if not current:
        raise ValueError("current_turn 为空")
    return {
        "current_turn": current,
        "context": str(obj.get("context") or "").strip(),
        "scene_id": str(obj.get("scene_id") or scene_id or "").strip(),
        "note": str(obj.get("note") or "").strip(),
    }


def load_prompt(vdir, scene_id):
    path = os.path.join(vdir, "standard", "synthesis_prompts", f"{scene_id}.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"合成 Prompt 不存在：{path}")
    with open(path, encoding="utf-8") as f:
        return f.read(), common.file_md5(path)


def make_synthetic_record(gen, scene_id, prompt_md5, version, model, seq, source=None):
    rec = {
        "sample_id": f"syn_{version.lstrip('v_')}_{common.slug(model)}_{seq:06d}",
        "conversation_id": f"synconv_{seq:06d}",
        "turn_id": 0,
        "current_turn": gen["current_turn"],
        "context": gen["context"],
        "conversation_turns": [],
        "label": "positive",
        "scene_id": scene_id or gen["scene_id"],
        "data_source": "synthetic",
        "annotation_source": "llm",
        "human_verified": False,
        "version": version,
        "generator_model": model,
        "synthesis_prompt_version": f"md5_{prompt_md5}",
        "synthesis_scene_id": scene_id or gen["scene_id"],
        "note": gen["note"],
    }
    if source:
        rec.update({
            "source_hardcase_id": source.get("sample_id"),
            "source_error_type": source.get("error_type"),
            "error_type": source.get("error_type"),
        })
    return rec


def main():
    parser = argparse.ArgumentParser(description="合成数据生成（scene 补充 / hardcase）")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--mode", choices=["scene", "hardcase"], required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--scenes", nargs="*", default=[], help="scene 模式：需要补充的子场景")
    parser.add_argument("--hardcase", nargs="*", default=[],
                        help="hardcase 模式：fp/fn jsonl（evaluate 产出）")
    parser.add_argument("--llm", required=True, help="执行合成的大模型显示名")
    parser.add_argument("--samples-per-scene", type=int, default=None)
    parser.add_argument("--multiplier", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rpm", type=int, default=20)
    args = parser.parse_args()

    task_cfg = common.load_task_config(args.root)
    syn_cfg = task_cfg.get("synthesis", {}) or {}
    vdir = common.version_dir(args.root, args.version, model=args.model, task_cfg=task_cfg)
    output_dir = args.output_dir or os.path.join(vdir, "synthesis")
    model_out = os.path.join(output_dir, common.slug(args.llm))

    prompts = {}  # scene_id → (system_prompt, md5)
    jobs = []     # (scene_id, source_record or None)

    if args.mode == "scene":
        if not args.scenes:
            raise SystemExit("scene 模式需要 --scenes")
        n_per = args.samples_per_scene
        if n_per is None:
            n_per = (syn_cfg.get("cold_start", {}) or {}).get("samples_per_model_per_scene", 100)
        for scene_id in args.scenes:
            prompts[scene_id] = load_prompt(vdir, scene_id)
            jobs.extend([(scene_id, None)] * n_per)
    else:
        hardcases = common.load_many(args.hardcase)
        # 人工确认的 FP/FN 才作为合成源
        confirmed = [h for h in hardcases if h.get("human_verified")]
        if not confirmed:
            print("[WARN] 无 human_verified 的 hard case，退回使用全部输入")
            confirmed = hardcases
        multiplier = args.multiplier
        if multiplier is None:
            multiplier = (syn_cfg.get("iteration", {}) or {}).get("hardcase_multiplier", 3)
        # 按 scene_id × error_type 分层：每条源样本生成 ceil(multiplier) 条
        import math
        for h in confirmed:
            scene_id = h.get("scene_id") or h.get("human_scene_id") or ""
            if scene_id and scene_id not in prompts:
                prompts[scene_id] = load_prompt(vdir, scene_id)
            n = math.ceil(multiplier)
            jobs.extend([(scene_id, h)] * n)

    if not jobs:
        raise SystemExit("无合成任务（检查 --scenes / --hardcase）")

    rate_limiter = llm_client.RateLimiter(args.rpm)
    print(f"生成模型 {args.llm}，模式 {args.mode}，共 {len(jobs)} 条合成任务")

    results = []
    for seq, (scene_id, source) in enumerate(jobs):
        system_prompt, prompt_md5 = prompts.get(
            scene_id, (f"你是业务对话安全数据合成助手。目标子场景：{scene_id}", "na"))
        if args.mode == "hardcase" and source is not None:
            user_prompt = HARDCASE_USER_TMPL.format(
                error_type=source.get("error_type") or "other",
                error_side=source.get("error_side") or "FP",
                current_turn=source.get("current_turn") or "",
                context=source.get("context") or "（无）",
                scene_id=scene_id)
        else:
            user_prompt = SYN_USER_TMPL.format(scene_id=scene_id)
        try:
            content = llm_client.call_llm(args.llm, system_prompt, user_prompt,
                                          task_cfg=task_cfg, rate_limiter=rate_limiter)
            gen = parse_sample_response(content, scene_id)
            rec = make_synthetic_record(gen, scene_id, prompt_md5, args.version,
                                        args.llm, seq, source)
            # hardcase FP 类（应判负例）改 label
            if source is not None and source.get("error_side") == "FP":
                rec["label"] = "negative"
                rec["scene_id"] = None
                rec["synthesis_scene_id"] = scene_id
            results.append(rec)
        except Exception as e:
            print(f"[WARN] 第 {seq} 条合成失败: {str(e)[:150]}")
        if (seq + 1) % 20 == 0:
            print(f"[{seq + 1}/{len(jobs)}] 已生成 {len(results)}", flush=True)

    os.makedirs(model_out, exist_ok=True)
    out_name = "hardcase_synthetic" if args.mode == "hardcase" else "scene_synthetic"
    common.save_records(results, os.path.join(model_out, out_name))
    common.update_status(args.root, args.version, model=args.model,
                         artifact=(os.path.relpath(os.path.join(model_out, out_name + ".jsonl"), vdir), len(results)),
                         event=f"synthesis [{args.llm}] {args.mode}: {len(results)} generated")
    print(f"[OK] {args.llm} 生成 {len(results)} 条 → {model_out}/{out_name}.jsonl / .xlsx")
    print(f"     scene 分布：{dict(Counter(r.get('synthesis_scene_id') for r in results))}")


if __name__ == "__main__":
    main()
