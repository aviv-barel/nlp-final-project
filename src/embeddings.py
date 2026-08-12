"""Phase 2 — embedding-space analysis: cosine distance vs. mixing level + t-SNE.

For each seed and each of 6 encoders (mmBERT, DictaBERT, XLM-R, mpnet,
LaBSE, alephbertgimmel-sbert), extracts embeddings, computes cosine distance
from the 0% variant to the 30/60/100% variants, and aggregates mean distance
vs. mixing level. Also runs t-SNE per model over all 420 embeddings.

Run: .venv/bin/python src/embeddings.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "generated" / "dataset_raw.csv"
EMBEDDINGS_DIR = REPO_ROOT / "results" / "embeddings"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
MIXING_LEVELS = ["0", "30", "60", "100"]

AUTO_MODELS = {
    "mmBERT": "jhu-clsp/mmBERT-base",
    "dictabert": "dicta-il/dictabert",
    "xlm-roberta": "xlm-roberta-base",
}

MODEL_KWARGS = {
    "dictabert": {"trust_remote_code": True},
}

SBERT_MODELS = {
    "mpnet-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "LaBSE": "sentence-transformers/LaBSE",
    "alephbertgimmel-sbert": "imvladikon/sentence_transformers_alephbertgimmel_small",
}

ALL_MODELS = {**AUTO_MODELS, **SBERT_MODELS}


def load_dataset() -> list[dict]:
    with open(RAW_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)


def extract_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    """Mean-pooled embeddings for AutoModel entries, native encode() for SBERT entries."""
    if model_name in AUTO_MODELS:
        model_id = AUTO_MODELS[model_name]
        kwargs = MODEL_KWARGS.get(model_name, {})
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModel.from_pretrained(model_id, **kwargs).to(DEVICE).eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                outputs = model(**inputs)
                pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                chunks.append(pooled.cpu().numpy())
        del model
        return np.concatenate(chunks, axis=0)
    elif model_name in SBERT_MODELS:
        model_id = SBERT_MODELS[model_name]
        model = SentenceTransformer(model_id, device=DEVICE)
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
        del model
        return embeddings
    else:
        raise ValueError(f"unknown model name: {model_name}")


def compute_pairwise_distances(embeddings_by_seed: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """embeddings_by_seed: {seed_id: {mixing_level: vector}}.
    Returns one row per (seed_id, non-zero mixing_level): cosine distance
    between that variant and the seed's 0% (pure Hebrew) variant.
    """
    rows = []
    for seed_id, by_level in embeddings_by_seed.items():
        if "0" not in by_level:
            continue
        base = by_level["0"]
        base_norm = np.linalg.norm(base)
        for level in ("30", "60", "100"):
            if level not in by_level:
                continue
            vec = by_level[level]
            cos_sim = float(np.dot(base, vec) / (base_norm * np.linalg.norm(vec)))
            rows.append({"seed_id": seed_id, "mixing_level": level, "cosine_distance": 1.0 - cos_sim})
    return rows


def plot_tsne(embeddings: np.ndarray, mixing_levels: list[str], output_path: Path, title: str) -> None:
    perplexity = min(30, max(5, len(embeddings) // 4))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca").fit_transform(embeddings)

    level_colors = {"0": "#1f77b4", "30": "#2ca02c", "60": "#ff7f0e", "100": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for level in MIXING_LEVELS:
        idx = [i for i, lvl in enumerate(mixing_levels) if lvl == level]
        if not idx:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1], label=f"{level}%", color=level_colors[level], alpha=0.75, s=25)
    ax.legend(title="mixing level")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_dataset()
    texts = [r["text"] for r in rows]
    seed_ids = [r["seed_id"] for r in rows]
    mixing_levels = [r["mixing_level"] for r in rows]

    summary_rows = []

    for model_name in ALL_MODELS:
        print(f"=== {model_name} ({ALL_MODELS[model_name]}) ===")
        embeddings = extract_embeddings(texts, model_name)

        embeddings_by_seed: dict[str, dict[str, np.ndarray]] = {}
        for seed_id, level, vec in zip(seed_ids, mixing_levels, embeddings):
            embeddings_by_seed.setdefault(seed_id, {})[level] = vec

        distance_rows = compute_pairwise_distances(embeddings_by_seed)
        write_csv(
            EMBEDDINGS_DIR / f"{model_name}_distances.csv",
            distance_rows,
            ["seed_id", "mixing_level", "cosine_distance"],
        )

        by_level: dict[str, list[float]] = {}
        for r in distance_rows:
            by_level.setdefault(r["mixing_level"], []).append(r["cosine_distance"])
        for level in ("30", "60", "100"):
            vals = by_level.get(level, [])
            if vals:
                summary_rows.append(
                    {
                        "model": model_name,
                        "mixing_level": level,
                        "mean_cosine_distance": float(np.mean(vals)),
                        "n": len(vals),
                    }
                )
        print(f"  distances: {[(r['mixing_level'], round(r['mean_cosine_distance'], 4)) for r in summary_rows if r['model'] == model_name]}")

        plot_tsne(
            embeddings,
            mixing_levels,
            FIGURES_DIR / f"tsne_{model_name}.png",
            title=f"t-SNE of {model_name} embeddings, by mixing level",
        )
        print(f"  wrote results/figures/tsne_{model_name}.png")

    write_csv(
        EMBEDDINGS_DIR / "distance_vs_mixing_level_summary.csv",
        summary_rows,
        ["model", "mixing_level", "mean_cosine_distance", "n"],
    )
    print(f"\nWrote {EMBEDDINGS_DIR / 'distance_vs_mixing_level_summary.csv'}")


if __name__ == "__main__":
    main()
