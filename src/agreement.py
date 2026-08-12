"""Phase 1.4 — Cohen's kappa between reviewers, and disagreement export.

Reads the two reviewer annotation CSVs, reports Cohen's kappa, and writes
data/annotated/disagreements.csv for manual adjudication.

Run: .venv/bin/python src/agreement.py
"""

import csv
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATED_DIR = REPO_ROOT / "data" / "annotated"
LABELS = ["positive", "neutral", "negative"]


def load_reviewer_csv(path: Path, sentiment_col: str) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {
            row["id"]: {"text": row["text"], "sentiment": row[sentiment_col]}
            for row in csv.DictReader(f)
        }


def compute_cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    return cohen_kappa_score(labels_a, labels_b, labels=LABELS)


def adjudicate_final_labels(annotations_a: dict, annotations_b: dict) -> tuple[dict, list[str]]:
    """Final label = agreement where reviewers match. Disagreements are NOT
    resolved here — they're returned separately so a human can adjudicate
    per the documented plan (Step 1.4), rather than picking a rule that
    silently favors one reviewer or a third source.
    """
    final = {}
    disagreement_ids = []
    for id_ in annotations_a:
        a, b = annotations_a[id_]["sentiment"], annotations_b[id_]["sentiment"]
        if a == b:
            final[id_] = a
        else:
            disagreement_ids.append(id_)
    return final, disagreement_ids


def main() -> None:
    r1 = load_reviewer_csv(ANNOTATED_DIR / "reviewer1_annotations.csv", "reviewer1_sentiment")
    r2 = load_reviewer_csv(ANNOTATED_DIR / "reviewer2_annotations.csv", "reviewer2_sentiment")
    assert r1.keys() == r2.keys(), "reviewer files cover different id sets"

    ids = list(r1.keys())
    kappa = compute_cohens_kappa([r1[i]["sentiment"] for i in ids], [r2[i]["sentiment"] for i in ids])
    print(f"Cohen's kappa (reviewer1 vs reviewer2): {kappa:.4f}")

    final, disagreement_ids = adjudicate_final_labels(r1, r2)
    print(f"Agreed: {len(final)}/{len(ids)} ({len(final) / len(ids):.1%})")
    print(f"Disagreements needing adjudication: {len(disagreement_ids)}")

    out_path = ANNOTATED_DIR / "disagreements.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "text", "reviewer1_sentiment", "reviewer2_sentiment", "final_sentiment"]
        )
        writer.writeheader()
        for id_ in disagreement_ids:
            writer.writerow(
                {
                    "id": id_,
                    "text": r1[id_]["text"],
                    "reviewer1_sentiment": r1[id_]["sentiment"],
                    "reviewer2_sentiment": r2[id_]["sentiment"],
                    "final_sentiment": "",
                }
            )
    print(f"Wrote {out_path} — fill in final_sentiment for each row, then rerun build_final_dataset.py")


if __name__ == "__main__":
    main()
