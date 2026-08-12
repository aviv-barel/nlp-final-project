"""Phase 1.3 — CLI that shows annotators one sentence at a time, blind to the
intended label, and records sentiment (+ naturalness). Progress saves after
every answer, so a session can be interrupted (Ctrl-C or "q") and resumed.

Usage:
    .venv/bin/python src/annotation_tools.py --reviewer reviewer1
    .venv/bin/python src/annotation_tools.py --reviewer reviewer2

Writes data/annotated/annotations_{reviewer}.csv
"""

import argparse
import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "generated" / "dataset_raw.csv"
ANNOTATED_DIR = REPO_ROOT / "data" / "annotated"

FIELDNAMES = ["id", "seed_id", "mixing_level", "text", "naturalness", "sentiment_label", "reviewer_id"]
SENTIMENT_CODES = {
    "p": "positive", "positive": "positive",
    "n": "negative", "negative": "negative",
    "u": "neutral", "neutral": "neutral",
}


def load_dataset(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_completed_ids(out_path: str) -> set[str]:
    if not Path(out_path).exists():
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def prompt_naturalness() -> str:
    while True:
        raw = input("  naturalness, would an Israeli engineer actually say this? (1-5): ").strip()
        if raw.lower() in ("q", "quit"):
            raise KeyboardInterrupt
        if raw in {"1", "2", "3", "4", "5"}:
            return raw
        print("  please enter a number from 1 to 5")


def print_guidelines() -> None:
    print("""
Label the sentiment the writer is EXPRESSING, not whether the sentence is
grammatically a statement of fact. If a teammate posted this in Slack, would
you read it as good news, bad news, or just information?

  p = positive   went well, or expresses satisfaction/relief/praise
  n = negative   went badly, or expresses frustration/complaint/blame
  u = neutral    NO evaluative content at all - pure logistics, scheduling,
                 or description, where you can't tell if the writer is
                 pleased or annoyed

Most common mistake: labeling neutral because a sentence SOUNDS factual, when
it reports a clearly good or bad outcome. "The build failed because of a
dependency that changed" is calm in tone but negative - something broke.

If it describes an outcome, judge that outcome. Only if it describes no
outcome at all is it neutral. Language mix is irrelevant to the label.

Full guidelines: data/annotated/ANNOTATION_GUIDELINES.md
""")


def prompt_sentiment() -> str:
    while True:
        raw = input("  sentiment (p=positive / n=negative / u=neutral, ?=help): ").strip().lower()
        if raw in ("q", "quit"):
            raise KeyboardInterrupt
        if raw == "?":
            print_guidelines()
            continue
        if raw in SENTIMENT_CODES:
            return SENTIMENT_CODES[raw]
        print("  please enter p, n, or u (or ? for help)")


def run_session(reviewer_id: str, dataset_path: str, out_path: str) -> None:
    rows = load_dataset(dataset_path)
    completed = load_completed_ids(out_path)
    remaining = [r for r in rows if r["id"] not in completed]

    rng = random.Random(reviewer_id)  # stable per-reviewer shuffle, different between reviewers
    rng.shuffle(remaining)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_exists = Path(out_path).exists()
    f = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not out_exists:
        writer.writeheader()
        f.flush()

    total = len(rows)
    done_count = len(completed)
    print(f"Reviewer: {reviewer_id}")
    print(f"{done_count}/{total} already annotated, {len(remaining)} remaining.")
    print("Rate naturalness 1-5, then label sentiment. Type 'q' anytime to save and quit.")
    if done_count == 0:
        print_guidelines()
    else:
        print("Type '?' at the sentiment prompt to reread the guidelines.\n")

    try:
        for row in remaining:
            print("-" * 60)
            print(row["text"])
            naturalness = prompt_naturalness()
            sentiment = prompt_sentiment()
            writer.writerow(
                {
                    "id": row["id"],
                    "seed_id": row["seed_id"],
                    "mixing_level": row["mixing_level"],
                    "text": row["text"],
                    "naturalness": naturalness,
                    "sentiment_label": sentiment,
                    "reviewer_id": reviewer_id,
                }
            )
            f.flush()
            done_count += 1
    except KeyboardInterrupt:
        print(f"\nSaved. {done_count}/{total} annotated so far. Resume anytime with the same --reviewer id.")
    else:
        print(f"\nAll done! {done_count}/{total} annotated.")
    finally:
        f.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind annotation CLI (Phase 1.3)")
    parser.add_argument("--reviewer", required=True, help="reviewer id, e.g. reviewer1 or reviewer2")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or str(ANNOTATED_DIR / f"annotations_{args.reviewer}.csv")
    run_session(args.reviewer, args.dataset, out_path)


if __name__ == "__main__":
    main()
