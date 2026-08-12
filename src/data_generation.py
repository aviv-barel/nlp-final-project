"""Phase 1.2 — constrained generation of mixing-level variants from seed sentences.

Reads data/seeds/seeds.csv, prompts Claude conditioned on each seed to produce
30/60/100% English-mixing variants, and writes data/generated/dataset_raw.csv.

Usage:
    ANTHROPIC_API_KEY=... .venv/bin/python src/data_generation.py [--limit N] [--model MODEL]
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = REPO_ROOT / "data" / "seeds" / "seeds.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "generated" / "dataset_raw.csv"

DEFAULT_MODEL = "claude-sonnet-5"
FIELDNAMES = ["id", "seed_id", "mixing_level", "text", "intended_sentiment"]

PROMPT_TEMPLATE = """You are helping build a linguistics dataset of natural Hebrew-English \
code-switching, as used by Israeli high-tech employees in Slack messages, standups, and \
retro notes.

You will be given ONE Hebrew seed sentence (0% English — pure Hebrew) and its sentiment \
label. Produce THREE variants of the SAME sentence — same meaning, same sentiment — at \
increasing levels of English mixing:

- "30": ~30% English by word count. Keep the sentence's grammatical backbone in Hebrew. \
Swap in English for tech nouns (e.g. "deployment", "server", "bug") written in Latin \
script, often attached to a Hebrew "ה-" prefix (e.g. "ה-deployment"), and sprinkle in \
short English filler/intensifier words Israeli engineers actually use mid-Hebrew-sentence, \
like "literally", "totally", "basically". Everything else stays Hebrew.
- "60": ~60% English by word count. More of the sentence's content is in English — more \
nouns, verbs, and phrases — but Hebrew grammatical connectors, prepositions, or short \
clauses still appear naturally, the way someone speaking quickly and code-switching \
mid-thought would.
- "100": 100% English. A fully natural English sentence with the same meaning and \
sentiment. Not a literal translation — phrase it the way a native English speaker on the \
same team would.

Rules:
- Do NOT do mechanical word-for-word substitution — mixing should follow how real Israeli \
engineers actually code-switch (tech jargon and short filler phrases in English, sentence \
structure and connectors in Hebrew, especially at 30%).
- Preserve the exact sentiment ({sentiment}) and meaning of the seed at every level.
- Keep each variant to roughly one sentence, similar length to the seed.

Examples of the pattern (for calibration — do not copy):

Seed (0%): "הישיבה הייתה ממש מתישה, בזבוז זמן מוחלט"
30%: "ה-meeting הייתה literally מתישה, בזבוז זמן"
60%: "ה-meeting הייתה literally בזבוז זמן, אני totally מותש"
100%: "The meeting was literally a waste of time, totally exhausted"

Seed (0%): "הדיפלוי לייצור עבר חלק, בלי שום תקלה."
30%: "ה-deployment ל-production עבר חלק, totally בלי תקלות."
60%: "ה-deployment to production was חלק, totally no issues."
100%: "The deployment to production went smoothly, totally no issues."

Now generate variants for this seed:

Seed (0%): "{seed_text}"
Sentiment: {sentiment}

Return ONLY a JSON object with exactly these keys: "30", "60", "100" — each mapping to the \
corresponding variant string. No extra commentary, no markdown code fences.
"""


def build_prompt(seed_text: str, sentiment: str) -> str:
    return PROMPT_TEMPLATE.format(seed_text=seed_text, sentiment=sentiment)


def _strip_code_fence(raw: str) -> str:
    return re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()


def generate_variants(
    client: anthropic.Anthropic,
    model: str,
    seed_text: str,
    sentiment: str,
    max_retries: int = 2,
) -> dict[str, str]:
    prompt = build_prompt(seed_text, sentiment)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _strip_code_fence(response.content[0].text)
            data = json.loads(raw)
            return {"30": data["30"], "60": data["60"], "100": data["100"]}
        except Exception as e:  # noqa: BLE001 — external API call, retry any failure
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to generate variants for seed {seed_text!r}: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=str(SEEDS_PATH))
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N seeds (for testing)"
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    with open(args.seeds, newline="", encoding="utf-8") as f:
        seeds = list(csv.DictReader(f))
    if args.limit:
        seeds = seeds[: args.limit]

    rows = []
    failed = []
    for i, seed in enumerate(seeds, start=1):
        seed_id = seed["id"]
        sentiment = seed["intended_sentiment"]
        rows.append(
            {
                "id": f"{seed_id}_0",
                "seed_id": seed_id,
                "mixing_level": 0,
                "text": seed["hebrew_text"],
                "intended_sentiment": sentiment,
            }
        )
        try:
            variants = generate_variants(client, args.model, seed["hebrew_text"], sentiment)
        except RuntimeError as e:
            print(f"[{i}/{len(seeds)}] FAILED {seed_id}: {e}", file=sys.stderr)
            failed.append(seed_id)
            continue
        for level in ("30", "60", "100"):
            rows.append(
                {
                    "id": f"{seed_id}_{level}",
                    "seed_id": seed_id,
                    "mixing_level": int(level),
                    "text": variants[level],
                    "intended_sentiment": sentiment,
                }
            )
        print(f"[{i}/{len(seeds)}] done {seed_id}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {args.out}")
    if failed:
        print(f"{len(failed)} seed(s) failed and were skipped: {failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
