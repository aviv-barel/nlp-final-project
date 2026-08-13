# Hebrew-English Code-Switching Sentiment

Measuring how Hebrew-English code-switching (Israeli tech workplace text)
breaks multilingual NLP models, and fixing it via fine-tuning.

The full report is not in this repo (submitted separately as a graded PDF);
this repo holds the code, data, results, and executed analysis notebook
behind it.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Requires Python 3.11+.

## Repo structure

```
data/
  seeds/       hand-written seed sentences
  generated/   LLM-generated mixing-level variants
  annotated/   post-annotation reviewer files + disagreements
  final/       final dataset + train/val/test splits
src/
  data_generation.py     constrained generation of mixing-level variants
  annotation_tools.py    blind annotator CLI
  agreement.py           Cohen's kappa + disagreement export
  build_final_dataset.py builds dataset_final.csv + splits
  embeddings.py          embedding-space analysis (cosine distance + t-SNE)
  finetune.py            fine-tuning loop, --model {mmBERT,dictabert} --condition {A,B}
  evaluate.py            test-set evaluation + summary table
  human_baseline.py      human/LLM performance baseline
  analysis.py            Wilson CIs, Fisher's exact test, Cochran-Armitage trend test
  semantic_equivalence.py  does the model recognize a paraphrase across mixing levels?
  smoke_test_models.py   one-off model load check
results/
  analysis.ipynb   executed notebook: all figures, tables, and statistics
  embeddings/      cosine-distance tables per encoder
  finetune_runs/   fine-tuned model metrics, summary.csv
  figures/         report figures
```

## Reproducing the results

Run in order from the repo root:

```bash
.venv/bin/python src/agreement.py
.venv/bin/python src/build_final_dataset.py
.venv/bin/python src/embeddings.py
.venv/bin/python src/finetune.py --model mmBERT    --condition A
.venv/bin/python src/finetune.py --model mmBERT    --condition B
.venv/bin/python src/finetune.py --model dictabert --condition A
.venv/bin/python src/finetune.py --model dictabert --condition B
.venv/bin/python src/evaluate.py
.venv/bin/python src/semantic_equivalence.py
.venv/bin/python src/human_baseline.py
.venv/bin/python src/analysis.py
```

Fixed hyperparameters for all fine-tuning runs: 5 epochs, batch 16,
lr 2e-5, weight decay 0.01, warmup ratio 0.1, max length 64, seed 42;
best checkpoint by validation macro-F1.

## Headline results

| Model | Cond. A (pure-Hebrew training) | Cond. B (mixed training) |
|---|---|---|
| mmBERT | 0.594 [0.47, 0.71] | **0.938** [0.85, 0.98] |
| DictaBERT | 0.422 [0.31, 0.54] | 0.844 [0.74, 0.91] |

Pooled across mixing levels, n=64/condition, 95% Wilson CIs.
Fisher exact A vs B: p=5.4e-6 (mmBERT), p=1.1e-6 (DictaBERT).

Degradation across mixing level is significant under condition A
(Cochran-Armitage p=0.042 mmBERT, p=0.019 DictaBERT) and disappears under B.
The Hebrew-specific model is the less robust one, and drifts ~2x further
in embedding space; drift tracks accuracy loss (Spearman rho=0.93, n=6).

See `results/analysis.ipynb` for the full set of figures and statistics.
