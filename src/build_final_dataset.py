"""Phase 1.4 (part 2) — build the canonical dataset and train/val/test splits.

Combines the two reviewer files with the manually-adjudicated disagreements
(data/annotated/disagreements.csv) into data/final/dataset_final.csv, then
writes a stratified 70/15/15 train/val/test split (by mixing_level x
sentiment).

Run: .venv/bin/python src/build_final_dataset.py
"""

import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATED_DIR = REPO_ROOT / "data" / "annotated"
FINAL_DIR = REPO_ROOT / "data" / "final"
RAW_PATH = REPO_ROOT / "data" / "generated" / "dataset_raw.csv"

FINAL_FIELDNAMES = ["id", "mixing_level", "text", "sentiment"]


def load_final_labels() -> dict[str, str]:
    r1 = {row["id"]: row["reviewer1_sentiment"] for row in csv.DictReader(open(ANNOTATED_DIR / "reviewer1_annotations.csv", encoding="utf-8"))}
    r2 = {row["id"]: row["reviewer2_sentiment"] for row in csv.DictReader(open(ANNOTATED_DIR / "reviewer2_annotations.csv", encoding="utf-8"))}
    disagreements_path = ANNOTATED_DIR / "disagreements.csv"
    overrides = {}
    if disagreements_path.exists():
        for row in csv.DictReader(open(disagreements_path, encoding="utf-8")):
            if not row["final_sentiment"].strip():
                raise ValueError(f"disagreements.csv row {row['id']} has no final_sentiment — adjudicate it first")
            overrides[row["id"]] = row["final_sentiment"].strip()

    final = {}
    for id_ in r1:
        if id_ in overrides:
            final[id_] = overrides[id_]
        else:
            if r1[id_] != r2[id_]:
                raise ValueError(f"{id_} disagrees but has no override in disagreements.csv")
            final[id_] = r1[id_]
    return final


def build_final_rows(final_labels: dict[str, str]) -> list[dict]:
    raw = {row["id"]: row for row in csv.DictReader(open(RAW_PATH, encoding="utf-8"))}
    rows = []
    for id_, sentiment in final_labels.items():
        rows.append(
            {
                "id": id_,
                "mixing_level": raw[id_]["mixing_level"],
                "text": raw[id_]["text"],
                "sentiment": sentiment,
            }
        )
    rows.sort(key=lambda r: r["id"])
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Splits by seed_id, not by row: all 4 mixing-level variants of a given
    seed always land in the same split. Splitting at the row level would put
    e.g. seed_005_0 in train and seed_005_60 in test — since all 4 variants
    of a seed carry near-identical content, that leaks the sentiment signal
    across the split and inflates cross-mixing-level accuracy artificially.
    Stratified by sentiment at the seed level (sentiment is constant across
    a seed's 4 mixing-level variants by construction).
    """
    by_seed: dict[str, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(r["id"].rsplit("_", 1)[0], []).append(r)

    seed_ids = list(by_seed)
    seed_sentiments = [by_seed[s][0]["sentiment"] for s in seed_ids]

    train_seeds, rest_seeds, _, rest_sent = train_test_split(
        seed_ids, seed_sentiments, test_size=0.30, stratify=seed_sentiments, random_state=42
    )
    val_seeds, test_seeds = train_test_split(rest_seeds, test_size=0.50, stratify=rest_sent, random_state=42)

    def expand(seeds: list[str]) -> list[dict]:
        out = []
        for s in seeds:
            out.extend(by_seed[s])
        return out

    return expand(train_seeds), expand(val_seeds), expand(test_seeds)


def main() -> None:
    final_labels = load_final_labels()
    rows = build_final_rows(final_labels)
    write_csv(FINAL_DIR / "dataset_final.csv", rows, FINAL_FIELDNAMES)
    print(f"Wrote {FINAL_DIR / 'dataset_final.csv'} ({len(rows)} rows)")

    train, val, test = stratified_split(rows)
    for name, split in (("train", train), ("val", val), ("test", test)):
        write_csv(FINAL_DIR / f"{name}.csv", split, FINAL_FIELDNAMES)
        print(f"Wrote {FINAL_DIR / f'{name}.csv'} ({len(split)} rows)")


if __name__ == "__main__":
    main()
