"""Phase 3 — HF Trainer fine-tuning loop for 3-way sentiment classification,
parameterized by model_name (mmBERT, dictabert) and training_condition
(A: pure Hebrew, B: full mixed data). Hyperparameters are fixed across all
runs (see HYPERPARAMS) for comparability. Saves each model + logs under
results/finetune_runs/{model}_{condition}/.

Run: .venv/bin/python src/finetune.py --model mmBERT --condition A
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = REPO_ROOT / "data" / "final"
RUNS_DIR = REPO_ROOT / "results" / "finetune_runs"

MODEL_IDS = {
    "mmBERT": "jhu-clsp/mmBERT-base",
    "dictabert": "dicta-il/dictabert",
}
MODEL_KWARGS = {
    "mmBERT": {},
    "dictabert": {"trust_remote_code": True},
}

SENTIMENT_LABELS = ["positive", "neutral", "negative"]

SEED = 42
MAX_LENGTH = 64
HYPERPARAMS = dict(
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    seed=SEED,
)


def load_split(name: str) -> list[dict]:
    with open(FINAL_DIR / f"{name}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_dataset(rows: list[dict], tokenizer, label2id: dict) -> Dataset:
    texts = [r["text"] for r in rows]
    labels = [label2id[r["sentiment"]] for r in rows]
    ds = Dataset.from_dict({"text": texts, "label": labels})
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH),
        batched=True,
    )


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train(model_name: str, training_condition: str, train_csv: str = "train", val_csv: str = "val") -> dict:
    set_seed(SEED)
    model_id = MODEL_IDS[model_name]
    kwargs = MODEL_KWARGS[model_name]
    label2id = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(SENTIMENT_LABELS), label2id=label2id, id2label=id2label, **kwargs
    )

    train_rows = load_split(train_csv)
    if training_condition == "A":
        train_rows = [r for r in train_rows if r["mixing_level"] == "0"]
    val_rows = load_split(val_csv)

    train_ds = to_dataset(train_rows, tokenizer, label2id)
    val_ds = to_dataset(val_rows, tokenizer, label2id)

    run_dir = RUNS_DIR / f"{model_name}_{training_condition}"
    args = TrainingArguments(
        output_dir=str(run_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to=[],
        **HYPERPARAMS,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = trainer.evaluate()
    metrics["n_train"] = len(train_rows)
    metrics["n_val"] = len(val_rows)
    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"{model_name}/{training_condition}: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 fine-tuning (one model/condition per run)")
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS))
    parser.add_argument("--condition", required=True, choices=["A", "B"])
    args = parser.parse_args()
    train(args.model, args.condition)


if __name__ == "__main__":
    main()
