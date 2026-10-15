# -*- coding: utf-8 -*-
"""超参搜索编排：按 task.yaml 网格对单个目标模型逐组调用 train_bert + tune_threshold。

对应 cold_start.md 步骤 11 / iteration.md 步骤 10 的超参数搜索。
网格读 task.yaml training.hparam_search（lr × batch_size × epochs × weight_decay）。
每个候选模型是一条独立流水线，本脚本一次只处理一个模型（--model）。

laya 候选（type: laya，SKILL.md §17）：自动降级为 threshold-only——跳过网格与训练，
只对 validation 调一次 tune_threshold（question 读该版本 standard/laya_question.yaml），
产物仅 <model>/<v>/training/threshold_search.csv + best_threshold.yaml
（无 cfg<i>/、无 best_model/、无 best_hparams.yaml），--train 参数可省。

多卡并行（参考 law1 超参搜索模式）：
    总任务 = 网格组数，按 --config-idx 取模分配到各 GPU 分片，每片一个进程：
    for i in 0..N-1: CUDA_VISIBLE_DEVICES=$g python hparam_search.py --model h256 --version v_0 --config-idx $i --num-shards $N

用法（单进程串行）：
    python <skill>/scripts/hparam_search.py --version v_0 --model h256 \
        --train h256/v_0/dataset/train.jsonl --validation h256/v_0/dataset/validation.jsonl

用法（laya，threshold-only）：
    python <skill>/scripts/hparam_search.py --version v_0 --model laya_multilingual \
        --validation laya_multilingual/v_0/dataset/validation.jsonl

输出（全部在该模型的版本目录下）：
    <model>/<v>/training/cfg<i>/              train_bert 单组产物（BERT）
    <model>/<v>/training/tuning_trials.csv    全网格汇总（BERT）
    <model>/<v>/training/best_hparams.yaml    最优组超参（BERT；laya 无此产物）
    <model>/<v>/training/best_threshold.yaml  最优组阈值（两种候选均有）
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common  # noqa: E402

PY = sys.executable


def build_grid(hp_cfg):
    return list(itertools.product(
        hp_cfg.get("learning_rate", [2e-5]),
        hp_cfg.get("batch_size", [32]),
        hp_cfg.get("epochs", [5]),
        hp_cfg.get("weight_decay", [0.01]),
    ))


def main():
    parser = argparse.ArgumentParser(description="超参网格搜索编排（train_bert + tune_threshold）")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选")
    parser.add_argument("--train", default=None,
                        help="训练集（BERT 候选必填；laya 候选无训练集可省）")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--config-idx", type=int, default=None, help="分片编号（与 --num-shards 配合多卡并行）")
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    task_cfg = common.load_task_config(args.root)
    model = common.resolve_model(args.root, args.model, task_cfg)

    # ---- laya 候选：自动降级为 threshold-only（SKILL.md §17.2/§17.3）----
    if common.is_external_candidate(model, task_cfg):
        vdir = common.version_dir(args.root, args.version, model=model, task_cfg=task_cfg)
        out_dir = os.path.join(vdir, "training")
        marker = os.path.join(out_dir, "best_threshold.yaml")
        if os.path.exists(marker):
            print(f"[skip] {model} threshold 搜索已完成（{marker} 存在）")
        else:
            print(f"[{common.candidate_type(model, task_cfg)}] {model} 为已有 checkpoint 候选：跳过网格与训练，只做 threshold 搜索")
            tune_cmd = [PY, os.path.join(_HERE, "tune_threshold.py"),
                        "--root", args.root,
                        "--model", model,
                        "--version", args.version,
                        "--validation", args.validation,
                        "--output-dir", out_dir]
            subprocess.run(tune_cmd, check=True)
        common.update_status(args.root, args.version, model=model,
                             artifact=("training/best_threshold.yaml", None),
                             event=f"threshold-only search [{model}] ({common.candidate_type(model, task_cfg)} 候选)")
        print(f"[OK] {model} threshold-only 完成，输出 {out_dir}（无 cfg<i>/、无 best_hparams.yaml）")
        return

    hp_cfg = (task_cfg.get("training", {}) or {}).get("hparam_search", {}) or {}
    grid = build_grid(hp_cfg)
    if not grid:
        raise SystemExit("task.yaml 中无超参网格（training.hparam_search）")
    if not args.train:
        raise SystemExit("BERT 候选必须提供 --train")

    entry = (task_cfg.get("models", {}).get("candidates", {}) or {}).get(model, {})
    model_path = entry.get("path") if isinstance(entry, dict) else None
    if not model_path:
        raise SystemExit(f"task.yaml 中未配置 {model} 的模型路径（models.candidates.{model}.path）")

    # 任务分片：全局任务序号 = 网格序号
    tasks = [(model, i, cfg) for i, cfg in enumerate(grid)]
    if args.config_idx is not None:
        tasks = [t for gi, t in enumerate(tasks) if gi % args.num_shards == args.config_idx]

    print(f"目标模型 {model}，网格 {len(grid)} 组，本进程任务 {len(tasks)} 个")
    paths = {model: model_path}
    results = []
    for model, cfg_idx, (lr, bs, epochs, wd) in tasks:
        out_dir = os.path.join(
            common.version_dir(args.root, args.version, model=model, task_cfg=task_cfg),
            "training", f"cfg{cfg_idx}")
        marker = os.path.join(out_dir, "best_threshold.yaml")
        if os.path.exists(marker):
            print(f"[skip] {model} cfg{cfg_idx} 已完成")
        else:
            print(f"[run] {model} cfg{cfg_idx}: lr={lr} bs={bs} epochs={epochs} wd={wd}")
            train_cmd = [PY, os.path.join(_HERE, "train_bert.py"),
                         "--root", args.root,
                         "--train", args.train, "--validation", args.validation,
                         "--model-path", paths[model], "--output-dir", out_dir,
                         "--lr", str(lr), "--batch-size", str(bs),
                         "--epochs", str(epochs), "--weight-decay", str(wd)]
            subprocess.run(train_cmd, check=True)
            tune_cmd = [PY, os.path.join(_HERE, "tune_threshold.py"),
                        "--root", args.root,
                        "--model-dir", os.path.join(out_dir, "best_model"),
                        "--validation", args.validation,
                        "--output-dir", out_dir]
            subprocess.run(tune_cmd, check=True)

        th = common.load_yaml(marker) if os.path.exists(marker) else {}
        results.append({
            "model": model, "cfg_idx": cfg_idx,
            "learning_rate": lr, "batch_size": bs, "epochs": epochs, "weight_decay": wd,
            "threshold": th.get("threshold"), "precision": th.get("precision"),
            "recall": th.get("recall"), "f1": th.get("f1"),
            "feasible": th.get("feasible"),
        })

    # 汇总 trials（按模型分文件；分片并行时各进程只写自己的行，最后由 select_model 聚合）
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    for model, rows in by_model.items():
        mdir = os.path.join(
            common.version_dir(args.root, args.version, model=model, task_cfg=task_cfg),
            "training")
        os.makedirs(mdir, exist_ok=True)
        csv_path = os.path.join(mdir, "tuning_trials.csv")
        existing = []
        if os.path.exists(csv_path):
            with open(csv_path, encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        # CSV 读回的 cfg_idx 是字符串，与新结果的 int 统一后去重
        seen = {(str(r.get("model")), str(r.get("cfg_idx"))) for r in existing}
        merged = existing + [r for r in rows
                             if (str(r.get("model")), str(r.get("cfg_idx"))) not in seen]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
            writer.writeheader()
            writer.writerows(merged)

        valid = [r for r in merged if r.get("f1") is not None]
        if valid:
            best = max(valid, key=lambda r: (r["f1"], r.get("precision") or 0))
            best_dir = os.path.join(mdir, f"cfg{best['cfg_idx']}")
            common.save_yaml({
                "model": model, "best_cfg_idx": best["cfg_idx"], **best,
                "best_model_dir": os.path.join(best_dir, "best_model"),
            }, os.path.join(mdir, "best_hparams.yaml"))
            # 最优组 threshold 复制到模型根目录，供 select_model 使用
            src_th = os.path.join(best_dir, "best_threshold.yaml")
            if os.path.exists(src_th):
                common.save_yaml(common.load_yaml(src_th), os.path.join(mdir, "best_threshold.yaml"))
        common.update_status(args.root, args.version, model=model,
                             artifact=("training/tuning_trials.csv", len(merged)),
                             event=f"hparam search [{model}]: {len(merged)} trials, best f1={max((r.get('f1') or 0 for r in merged), default=None)}")
        print(f"[OK] {model}: {len(merged)} 组试验写入 tuning_trials.csv")


if __name__ == "__main__":
    main()
