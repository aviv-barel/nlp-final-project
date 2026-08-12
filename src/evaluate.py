"""Phase 3 — evaluate fine-tuned models on the held-out test set, broken out by
mixing level. Writes results/finetune_runs/summary.csv, plus the matching
accuracy-vs-mixing-level line chart.

Run after all finetune.py runs complete:
    .venv/bin/python src/evaluate.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import MAX_LENGTH, MODEL_IDS, SENTIMENT_LABELS, load_split

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "finetune_runs"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

CONDITIONS = ["A", "B"]
MIXING_LEVELS = ["0", "30", "60", "100"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def evaluate_model(model_dir: str, label2id: dict, test_csv: str = "test") -> list[dict]:
    """Returns one dict per mixing level: {mixing_level, accuracy, f1, n}."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()

    rows = load_split(test_csv)
    texts = [r["text"] for r in rows]
    labels = [label2id[r["sentiment"]] for r in rows]
    levels = [r["mixing_level"] for r in rows]

    preds = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())

    results = []
    for level in MIXING_LEVELS:
        idx = [i for i, lvl in enumerate(levels) if lvl == level]
        if not idx:
            continue
        y_true = [labels[i] for i in idx]
        y_pred = [preds[i] for i in idx]
        results.append(
            {
                "mixing_level": level,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "n": len(idx),
            }
        )
    return results


def build_summary_table(results: list[dict]) -> Path:
    out_path = RUNS_DIR / "summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "condition", "mixing_level", "accuracy", "f1", "n"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_path}")
    return out_path


def plot_accuracy_vs_mixing_level(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name in MODEL_IDS:
        for condition in CONDITIONS:
            subset = [r for r in results if r["model"] == model_name and r["condition"] == condition]
            if not subset:
                continue
            subset.sort(key=lambda r: int(r["mixing_level"]))
            x = [int(r["mixing_level"]) for r in subset]
            y = [r["accuracy"] for r in subset]
            style = "-o" if condition == "B" else "--s"
            ax.plot(x, y, style, label=f"{model_name} (cond. {condition})")
    ax.set_xlabel("mixing level (%)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Test accuracy vs. mixing level")
    ax.set_xticks([0, 30, 60, 100])
    ax.legend()
    name = "accuracy_vs_mixing_level.png"
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / name}")


def main() -> None:
    label2id = {label: i for i, label in enumerate(SENTIMENT_LABELS)}

    all_results = []
    for model_name in MODEL_IDS:
        for condition in CONDITIONS:
            run_dir = RUNS_DIR / f"{model_name}_{condition}" / "final"
            if not run_dir.exists():
                print(f"skipping {model_name}/{condition}: no trained model at {run_dir}")
                continue
            print(f"=== evaluating {model_name}/{condition} ===")
            per_level = evaluate_model(str(run_dir), label2id)
            for r in per_level:
                r["model"] = model_name
                r["condition"] = condition
            all_results.extend(per_level)
            print(f"  {[(r['mixing_level'], round(r['accuracy'], 3), round(r['f1'], 3)) for r in per_level]}")

    build_summary_table(all_results)
    plot_accuracy_vs_mixing_level(all_results)


if __name__ == "__main__":
    main()
