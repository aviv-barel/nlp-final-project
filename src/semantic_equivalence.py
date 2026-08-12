"""Semantic equivalence check: does the model recognize two mixing-level
variants of a seed as the same sentence? Reuses the Phase 2 embeddings, no
new training. POSITIVE pairs = a seed's 0% variant vs. its own L% variant;
NEGATIVE pairs = a seed's 0% variant vs. a different seed's L% variant.
Cosine similarity is the same/different score; ROC-AUC (0.5=chance,
1.0=perfect) measures separation.

Run: .venv/bin/python src/semantic_equivalence.py
"""

import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from embeddings import ALL_MODELS, MIXING_LEVELS, extract_embeddings, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_COLOR = {"mmBERT": "#0072B2", "dictabert": "#D55E00"}
MUTED = "#9A9A9A"
SEED = 42


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_pairs(embeddings_by_seed: dict, level: str, rng: random.Random) -> tuple[list[float], list[int]]:
    """Returns (similarities, labels) — labels: 1=same seed (paraphrase), 0=different seed."""
    seed_ids = [s for s, by_level in embeddings_by_seed.items() if "0" in by_level and level in by_level]
    sims, labels = [], []

    for s in seed_ids:
        sims.append(cosine_sim(embeddings_by_seed[s]["0"], embeddings_by_seed[s][level]))
        labels.append(1)

    for s in seed_ids:
        other = rng.choice([o for o in seed_ids if o != s])
        sims.append(cosine_sim(embeddings_by_seed[s]["0"], embeddings_by_seed[other][level]))
        labels.append(0)

    return sims, labels


def main() -> None:
    rows = load_dataset()
    texts = [r["text"] for r in rows]
    seed_ids = [r["seed_id"] for r in rows]
    mixing_levels = [r["mixing_level"] for r in rows]

    results = []
    for model_name in ALL_MODELS:
        print(f"=== {model_name} ===")
        embeddings = extract_embeddings(texts, model_name)
        embeddings_by_seed: dict[str, dict[str, np.ndarray]] = {}
        for seed_id, level, vec in zip(seed_ids, mixing_levels, embeddings):
            embeddings_by_seed.setdefault(seed_id, {})[level] = vec

        for level in ("30", "60", "100"):
            rng = random.Random(SEED)
            sims, labels = build_pairs(embeddings_by_seed, level, rng)
            auc = roc_auc_score(labels, sims)
            mean_pos = np.mean([s for s, l in zip(sims, labels) if l == 1])
            mean_neg = np.mean([s for s, l in zip(sims, labels) if l == 0])
            results.append(
                {
                    "model": model_name,
                    "mixing_level": level,
                    "auc": auc,
                    "mean_similarity_same_meaning": mean_pos,
                    "mean_similarity_different_meaning": mean_neg,
                    "n_pairs": len(labels),
                }
            )
            print(f"  mix={level:>3}%  AUC={auc:.3f}  same-meaning sim={mean_pos:.3f}  different-meaning sim={mean_neg:.3f}")

    out_path = RESULTS_DIR / "semantic_equivalence.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "mixing_level", "auc", "mean_similarity_same_meaning",
                        "mean_similarity_different_meaning", "n_pairs"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_path}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # Direct-labelled at the right endpoint; several AUCs land within a few
    # thousandths of each other at 100%, so labels are nudged apart in
    # endpoint order to avoid overlapping.
    final = {r["model"]: r["auc"] for r in results if r["mixing_level"] == "100"}
    order = sorted(ALL_MODELS, key=lambda m: final[m])
    nudge = {m: 0.0 for m in ALL_MODELS}
    for prev, cur in zip(order, order[1:]):
        gap = final[cur] - final[prev] + nudge[prev]
        if gap < 0.035:
            nudge[cur] = nudge[prev] + (0.035 - gap)

    for model_name in ALL_MODELS:
        highlighted = model_name in MODEL_COLOR
        color = MODEL_COLOR.get(model_name, MUTED)
        sub = [r for r in results if r["model"] == model_name]
        xs = [int(r["mixing_level"]) for r in sub]
        ys = [r["auc"] for r in sub]
        ax.plot(xs, ys, marker="o", markersize=6 if highlighted else 4,
                linewidth=2.2 if highlighted else 1.2, color=color,
                alpha=1.0 if highlighted else 0.5, zorder=3 if highlighted else 2)
        ax.annotate(model_name, (100, ys[-1] + nudge[model_name]), textcoords="offset points", xytext=(7, 0),
                    color=color, fontweight="bold" if highlighted else "normal",
                    va="center", fontsize=8.5)
    ax.axhline(0.5, color="#555555", linewidth=1, linestyle=":", zorder=0)
    ax.annotate("chance (AUC=0.5)", (30, 0.5), textcoords="offset points", xytext=(0, 5),
                fontsize=8, color="#555555")
    ax.set_xlabel("mixing level (%) of the compared variant")
    ax.set_ylabel("AUC — can the model tell same vs. different meaning?")
    ax.set_title("Recognizing paraphrases gets harder as mixing increases")
    ax.set_xticks([30, 60, 100])
    ax.set_xlim(22, 168)
    top = max(1.02, max(final[m] + nudge[m] for m in ALL_MODELS) + 0.025)
    ax.set_ylim(0.45, top)
    fig.savefig(FIGURES_DIR / "fig4_semantic_equivalence.png", bbox_inches="tight", dpi=150)
    print(f"Wrote {FIGURES_DIR / 'fig4_semantic_equivalence.png'}")


if __name__ == "__main__":
    main()
