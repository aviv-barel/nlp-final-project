"""Phase 0 — one-off check that every reference model loads via
AutoModel/AutoTokenizer or SentenceTransformer, before Phase 2/3 depend on it.

Run manually: .venv/bin/python src/smoke_test_models.py
"""

from transformers import AutoModel, AutoTokenizer

AUTO_MODELS = [
    ("jhu-clsp/mmBERT-base", {}),
    ("dicta-il/dictabert", {"trust_remote_code": True}),
    ("dicta-il/alephbertgimmel-base", {}),
    ("xlm-roberta-base", {}),
    ("avichr/heBERT", {}),
]

SBERT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "imvladikon/sentence-transformers-alephbert",
]


def check_automodel(model_id: str, kwargs: dict) -> tuple[bool, str]:
    try:
        AutoTokenizer.from_pretrained(model_id, **kwargs)
        AutoModel.from_pretrained(model_id, **kwargs)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_sbert(model_id: str) -> tuple[bool, str]:
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_id)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    results = []
    for model_id, kwargs in AUTO_MODELS:
        ok, msg = check_automodel(model_id, kwargs)
        results.append((model_id, ok, msg))
        print(f"{'OK  ' if ok else 'FAIL'} {model_id}: {msg}")

    for model_id in SBERT_MODELS:
        ok, msg = check_sbert(model_id)
        results.append((model_id, ok, msg))
        print(f"{'OK  ' if ok else 'FAIL'} {model_id}: {msg}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} models loaded successfully")


if __name__ == "__main__":
    main()
