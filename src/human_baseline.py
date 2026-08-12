"""Phase 4 — human/LLM performance baseline on test.csv.

Reuses the existing Phase 1.3 annotations (reviewer1 = human, reviewer2 =
Claude), which already cover every id in test.csv and were collected blind
to mixing_level and to each other's answers. Scores against the full 64-row
test set rather than a subsample, since full coverage already exists.

Gold label is intended_sentiment from data/generated/dataset_raw.csv (the
label each sentence was generated to express), not dataset_final.csv's
`sentiment` column — the latter is derived from the reviewers' own answers,
so scoring against it would be circular.

Run: .venv/bin/python src/human_baseline.py
"""

import csv
import random
from pathlib import Path

from sklearn.metrics import accuracy_score, cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = REPO_ROOT / "data" / "final"
ANNOTATED_DIR = REPO_ROOT / "data" / "annotated"
RAW_PATH = REPO_ROOT / "data" / "generated" / "dataset_raw.csv"
RESULTS_DIR = REPO_ROOT / "results"

LABELS = ["positive", "neutral", "negative"]
SAMPLE_SIZE = 50
SEED = 42


def sample_stratified(test_csv: str = "test", n: int = SAMPLE_SIZE) -> list[dict]:
    rows = list(csv.DictReader(open(FINAL_DIR / f"{test_csv}.csv", encoding="utf-8")))
    by_level: dict[str, list[dict]] = {}
    for r in rows:
        by_level.setdefault(r["mixing_level"], []).append(r)

    rng = random.Random(SEED)
    per_level = n // len(by_level)
    sample = []
    for level, level_rows in by_level.items():
        rng.shuffle(level_rows)
        sample.extend(level_rows[:per_level])
    return sample


def score_against_gold(annotations: dict[str, str], gold: dict[str, str]) -> float:
    ids = list(annotations)
    return accuracy_score([gold[i] for i in ids], [annotations[i] for i in ids])


def load_reviewer(path: Path, col: str) -> dict[str, str]:
    return {row["id"]: row[col] for row in csv.DictReader(open(path, encoding="utf-8"))}


def main() -> None:
    sample_rows = list(csv.DictReader(open(FINAL_DIR / "test.csv", encoding="utf-8")))
    sample_ids = [r["id"] for r in sample_rows]
    intended = {row["id"]: row["intended_sentiment"] for row in csv.DictReader(open(RAW_PATH, encoding="utf-8"))}
    gold = {r["id"]: intended[r["id"]] for r in sample_rows}

    r1_all = load_reviewer(ANNOTATED_DIR / "reviewer1_annotations.csv", "reviewer1_sentiment")
    r2_all = load_reviewer(ANNOTATED_DIR / "reviewer2_annotations.csv", "reviewer2_sentiment")
    r1 = {i: r1_all[i] for i in sample_ids}
    r2 = {i: r2_all[i] for i in sample_ids}

    acc_r1 = score_against_gold(r1, gold)
    acc_r2 = score_against_gold(r2, gold)
    kappa = cohen_kappa_score([r1[i] for i in sample_ids], [r2[i] for i in sample_ids], labels=LABELS)

    print(f"Evaluated on: full test.csv ({len(sample_ids)} rows)")
    print(f"reviewer1 (human) accuracy vs gold: {acc_r1:.3f}")
    print(f"reviewer2 (LLM)   accuracy vs gold: {acc_r2:.3f}")
    print(f"Cohen's kappa, reviewer1 vs reviewer2, on this sample: {kappa:.4f}")

    by_level: dict[str, list[str]] = {}
    for r in sample_rows:
        by_level.setdefault(r["mixing_level"], []).append(r["id"])

    out_rows = []
    for level, ids in sorted(by_level.items(), key=lambda kv: int(kv[0])):
        out_rows.append(
            {
                "mixing_level": level,
                "n": len(ids),
                "reviewer1_human_accuracy": score_against_gold({i: r1[i] for i in ids}, gold),
                "reviewer2_llm_accuracy": score_against_gold({i: r2[i] for i in ids}, gold),
            }
        )

    out_path = RESULTS_DIR / "human_baseline.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mixing_level", "n", "reviewer1_human_accuracy", "reviewer2_llm_accuracy"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {out_path}")
    for row in out_rows:
        print(f"  mixing={row['mixing_level']:>3}  n={row['n']}  human={row['reviewer1_human_accuracy']:.3f}  llm={row['reviewer2_llm_accuracy']:.3f}")


if __name__ == "__main__":
    main()
