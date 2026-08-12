"""Phase 5 — statistical analysis backing the write-up: 95% Wilson score
intervals per cell, Fisher's exact test for condition A vs. B, a
Cochran-Armitage trend test for degradation across mixing level, and the
embedding-drift-vs-accuracy correlation.

Run: .venv/bin/python src/analysis.py
"""

import csv
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "finetune_runs"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"

MIXING_LEVELS = ["0", "30", "60", "100"]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — appropriate for small n and proportions
    near 0/1, where the normal approximation misbehaves."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_summary() -> list[dict]:
    with open(RUNS_DIR / "summary.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["accuracy"] = float(r["accuracy"])
        r["f1"] = float(r["f1"])
        r["n"] = int(r["n"])
        r["successes"] = int(round(r["accuracy"] * r["n"]))
    return rows


def load_embedding_distances() -> dict[tuple[str, str], float]:
    out = {}
    with open(EMBEDDINGS_DIR / "distance_vs_mixing_level_summary.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["model"], r["mixing_level"])] = float(r["mean_cosine_distance"])
    return out


def cochran_armitage_trend(successes: list[int], ns: list[int], scores: list[float]) -> tuple[float, float]:
    """Test for a linear trend in proportions across ordered groups.
    Returns (z, two-sided p)."""
    successes_arr, ns_arr, scores_arr = np.array(successes), np.array(ns), np.array(scores, dtype=float)
    n_total, s_total = ns_arr.sum(), successes_arr.sum()
    p_bar = s_total / n_total
    score_bar = (ns_arr * scores_arr).sum() / n_total
    numerator = (successes_arr * (scores_arr - score_bar)).sum()
    variance = p_bar * (1 - p_bar) * (ns_arr * (scores_arr - score_bar) ** 2).sum()
    if variance <= 0:
        return (0.0, 1.0)
    z = numerator / np.sqrt(variance)
    return (float(z), float(2 * (1 - stats.norm.cdf(abs(z)))))


def per_cell_intervals(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        lo, hi = wilson_interval(r["successes"], r["n"])
        out.append({**r, "ci_low": lo, "ci_high": hi})
    return out


def condition_comparison(rows: list[dict]) -> list[dict]:
    """Pool across mixing levels (n=64 per model/condition) for a
    properly-powered A-vs-B test, via Fisher's exact test."""
    out = []
    models = sorted({r["model"] for r in rows})
    for model in models:
        cells = {c: [r for r in rows if r["model"] == model and r["condition"] == c] for c in ("A", "B")}
        sa, na = sum(r["successes"] for r in cells["A"]), sum(r["n"] for r in cells["A"])
        sb, nb = sum(r["successes"] for r in cells["B"]), sum(r["n"] for r in cells["B"])
        _, p = stats.fisher_exact([[sa, na - sa], [sb, nb - sb]])
        lo_a, hi_a = wilson_interval(sa, na)
        lo_b, hi_b = wilson_interval(sb, nb)
        out.append(
            {
                "model": model,
                "acc_A": sa / na,
                "ci_A": (lo_a, hi_a),
                "acc_B": sb / nb,
                "ci_B": (lo_b, hi_b),
                "n_per_condition": na,
                "p_fisher": float(p),
            }
        )
    return out


def degradation_trends(rows: list[dict]) -> list[dict]:
    out = []
    for model in sorted({r["model"] for r in rows}):
        for condition in ("A", "B"):
            cells = sorted(
                [r for r in rows if r["model"] == model and r["condition"] == condition],
                key=lambda r: int(r["mixing_level"]),
            )
            z, p = cochran_armitage_trend(
                [c["successes"] for c in cells],
                [c["n"] for c in cells],
                [float(c["mixing_level"]) for c in cells],
            )
            out.append({"model": model, "condition": condition, "z": z, "p_trend": p})
    return out


def embedding_accuracy_link(rows: list[dict], distances: dict) -> list[dict]:
    """Pair embedding drift at each mixing level against the accuracy drop
    from that model's own 0% cell, condition A (the condition where the
    model never saw mixed text, so drift is not masked by training)."""
    out = []
    for model in ("mmBERT", "dictabert"):
        base = next(
            (r for r in rows if r["model"] == model and r["condition"] == "A" and r["mixing_level"] == "0"),
            None,
        )
        if base is None:
            continue
        for level in ("30", "60", "100"):
            cell = next(
                (r for r in rows if r["model"] == model and r["condition"] == "A" and r["mixing_level"] == level),
                None,
            )
            if cell is None or (model, level) not in distances:
                continue
            out.append(
                {
                    "model": model,
                    "mixing_level": level,
                    "cosine_drift": distances[(model, level)],
                    "accuracy_drop": base["accuracy"] - cell["accuracy"],
                }
            )
    return out


def main() -> None:
    rows = load_summary()
    distances = load_embedding_distances()

    print("=" * 72)
    print("1. PER-CELL ACCURACY WITH 95% WILSON CIs (n=16 per cell)")
    print("=" * 72)
    for r in per_cell_intervals(rows):
        print(
            f"  {r['model']:>10} cond {r['condition']}  mix={r['mixing_level']:>3}%  "
            f"acc={r['accuracy']:.3f} ({r['successes']}/{r['n']})  95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        )

    print()
    print("=" * 72)
    print("2. CONDITION A vs B, POOLED ACROSS MIXING LEVELS (n=64 each)")
    print("=" * 72)
    for c in condition_comparison(rows):
        print(
            f"  {c['model']:>10}  A={c['acc_A']:.3f} [{c['ci_A'][0]:.2f}, {c['ci_A'][1]:.2f}]  "
            f"B={c['acc_B']:.3f} [{c['ci_B'][0]:.2f}, {c['ci_B'][1]:.2f}]  "
            f"Fisher p={c['p_fisher']:.2e}"
        )

    print()
    print("=" * 72)
    print("3. DEGRADATION TREND ACROSS MIXING LEVEL (Cochran-Armitage)")
    print("=" * 72)
    for t in degradation_trends(rows):
        sig = "significant" if t["p_trend"] < 0.05 else "not significant"
        print(f"  {t['model']:>10} cond {t['condition']}  z={t['z']:+.3f}  p={t['p_trend']:.4f}  ({sig})")

    print()
    print("=" * 72)
    print("4. EMBEDDING DRIFT vs ACCURACY DROP (condition A)")
    print("=" * 72)
    link = embedding_accuracy_link(rows, distances)
    for r in link:
        print(
            f"  {r['model']:>10}  mix={r['mixing_level']:>3}%  "
            f"cosine drift={r['cosine_drift']:.4f}  accuracy drop={r['accuracy_drop']:+.4f}"
        )
    if len(link) >= 3:
        xs = [r["cosine_drift"] for r in link]
        ys = [r["accuracy_drop"] for r in link]
        rho, p = stats.spearmanr(xs, ys)
        pear, pp = stats.pearsonr(xs, ys)
        print(f"\n  Spearman rho={rho:.3f} (p={p:.4f}), Pearson r={pear:.3f} (p={pp:.4f}), n={len(link)} paired points")
        print("  NOTE: n=6 paired points across 2 models — descriptive only, not a powered test.")

    out_path = RESULTS_DIR / "statistical_analysis.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "condition", "mixing_level", "accuracy", "n", "ci_low", "ci_high"])
        writer.writeheader()
        for r in per_cell_intervals(rows):
            writer.writerow(
                {k: r[k] for k in ["model", "condition", "mixing_level", "accuracy", "n", "ci_low", "ci_high"]}
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
