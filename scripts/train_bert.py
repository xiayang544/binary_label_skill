# -*- coding: utf-8 -*-
"""BERT 二分类微调（单组超参），供超参搜索循环调用或单独运行。

对应 cold_start.md 步骤 11 / iteration.md 步骤 10。
训练与推理的文本构造统一走 src/text_utils.build_bert_text（SKILL.md §11：
两者必须使用完全相同的上下文拼接逻辑，轮数严格读 task.yaml 的 bert_turns）。

标签映射：0 = negative，1 = positive。

用法：
    python <skill>/scripts/train_bert.py \
        --train v_0/dataset/train.jsonl --validation v_0/dataset/validation.jsonl \
        --model-path <候选模型目录> --output-dir v_0/training/macbert/lr2e-5_bs32 \
        [--lr 2e-5] [--batch-size 32] [--epochs 5] [--weight-decay 0.01]

输出：
    <output-dir>/best_model/                 最优 checkpoint（按 validation f1）
    <output-dir>/best_hparams.yaml           本组超参
    <output-dir>/manifest.yaml               数据/模型/配置记录
    <output-dir>/validation_predictions.jsonl/.xlsx   validation 打分（供 tune_threshold）
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src import common, text_utils  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class TurnDataset(Dataset):
    def __init__(self, records, tokenizer, max_length, bert_turns):
        texts = [text_utils.build_bert_text(r.get("current_turn"), r.get("context"), bert_turns)
                 for r in records]
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.labels = [LABEL2ID[r["label"]] for r in records]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds, zero_division=0)}


@torch.no_grad()
def predict_scores(model, tokenizer, records, bert_turns, max_length, batch_size, device):
    """返回每条样本的正例概率（按阈值前的原始分）。"""
    texts = [text_utils.build_bert_text(r.get("current_turn"), r.get("context"), bert_turns)
             for r in records]
    scores = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt").to(device)
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
        scores.extend(probs.cpu().tolist())
    return scores


def main():
    parser = argparse.ArgumentParser(description="BERT 二分类微调（单组超参）")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", default=None,
                        help="目标模型 key；缺省取唯一 enabled 候选（laya 候选会直接报错退出）")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model-path", required=True, help="候选基座模型目录（读 task.yaml models.candidates 亦可）")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    args = parser.parse_args()

    task_cfg, _, bert_turns = common.task_ctx(args.root, "bert_turns", 1)
    model_key = common.resolve_model(args.root, args.model, task_cfg)
    if common.is_external_candidate(model_key, task_cfg):
        raise SystemExit(
            f"候选 {model_key} 为 type: {common.candidate_type(model_key, task_cfg)} 已有 checkpoint 候选，无 BERT 训练步骤：\n"
            "  - threshold 搜索请直接运行 scripts/tune_threshold.py（或 hparam_search.py 自动降级）\n"
            "  - 打分/评估请运行 scripts/evaluate.py 或 scripts/select_model.py（由对应 scorer 直接支持）")
    max_length = (task_cfg.get("training", {}) or {}).get("max_length", args.max_length)
    set_seed(args.seed)

    train_records = common.load_records(args.train)
    val_records = common.load_records(args.validation)
    for name, recs in (("train", train_records), ("validation", val_records)):
        bad = [r.get("sample_id") for r in recs if r.get("label") not in LABEL2ID]
        if bad:
            raise ValueError(f"{name} 中存在非法 label（前 5 个）: {bad[:5]}")
        if not recs:
            raise SystemExit(f"{name} 数据为空（{args.train if name == 'train' else args.validation}），"
                             f"无法训练。请先通过 build_dataset.py 构建足够的 {name} 样本。")
    print(f"train {len(train_records)} / validation {len(val_records)}，bert_turns={bert_turns}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    train_ds = TurnDataset(train_records, tokenizer, max_length, bert_turns)
    val_ds = TurnDataset(val_records, tokenizer, max_length, bert_turns)

    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()

    # 保存最优模型
    best_model_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # validation 打分（供 tune_threshold / 模型选择）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scores = predict_scores(model, tokenizer, val_records, bert_turns,
                            max_length, args.eval_batch_size, device)
    val_pred = []
    for r, s in zip(val_records, scores):
        out = dict(r)
        out.update({"bert_model": os.path.basename(args.model_path.rstrip("/")),
                    "bert_score": round(s, 6),
                    "bert_threshold": 0.5,  # 占位，正式阈值由 tune_threshold 决定
                    "bert_prediction": "positive" if s >= 0.5 else "negative"})
        val_pred.append(out)
    common.save_records(val_pred, os.path.join(args.output_dir, "validation_predictions"))

    hparams = {
        "model_path": args.model_path,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": max_length,
        "bert_turns": bert_turns,
        "seed": args.seed,
    }
    common.save_yaml(hparams, os.path.join(args.output_dir, "best_hparams.yaml"))
    common.save_yaml({
        "train_data": args.train, "train_rows": len(train_records),
        "validation_data": args.validation, "validation_rows": len(val_records),
        **hparams,
        "created_at": common.now_str(),
    }, os.path.join(args.output_dir, "manifest.yaml"))
    print(f"[OK] 最优模型保存至 {best_model_dir}，validation 打分 {len(val_pred)} 条")


if __name__ == "__main__":
    main()
