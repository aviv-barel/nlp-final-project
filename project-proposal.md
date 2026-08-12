# Lost in Translation: How Hebrew-English Code-Switching Breaks Multilingual NLP — and How to Fix It

**Student:** Aviv Barel
**Student ID:** 209055722
**Email:** avivitzhak.barelbal@post.runi.ac.il

---

## Problem Statement

Israeli social media and high-tech workplace communication naturally mix
Hebrew and English within the same sentence — a phenomenon known as
code-switching. Sentences like "ה-deployment ל-production literally קרס,
אנחנו totally blocked" are completely normal in Israeli tech culture, yet
current multilingual NLP models (mBERT, XLM-R) were trained predominantly
on clean, single-language corpora. As a result, their performance on
real-world Israeli text is poorly understood and likely degraded. This
project aims to systematically measure exactly how and when code-switching
breaks model performance, and then fix it through fine-tuning on a novel
dataset I construct.

## Detailed Description of Idea

I propose a three-part study. First, I construct a novel Hebrew-English
code-switching dataset for sentiment analysis, with sentences annotated by
mixing level (0%, 30%, 60%, 100% English), focused on high-tech workplace
language that reflects how Israelis actually write. Second, I analyze how
multilingual embedding models represent these sentences — specifically,
whether sentences with identical meaning but different mixing levels are
placed close together or pushed apart in embedding space. Third, I
fine-tune models on my dataset and measure whether this fixes the
degradation, proposing a practical solution for building NLP systems that
work on real Israeli text.

## Dataset Examples

| Sentence | Mixing Level | Sentiment |
|---|---|---|
| הישיבה הייתה ממש מתישה, בזבוז זמן מוחלט | 0% | Negative |
| ה-meeting הייתה literally מתישה, בזבוז זמן | 30% | Negative |
| ה-meeting הייתה literally בזבוז זמן, אני totally מותש | 60% | Negative |
| The meeting was literally a waste of time, totally exhausted | 100% | Negative |
| עשינו deploy לסביבת הייצור ושום דבר לא נשבר | 0% | Positive |
| ה-deploy ל-production הצליח, literally אפס issues | 30% | Positive |
| ה-deployment ל-production היה totally חלק, literally אפס bugs | 60% | Positive |
| The deployment to production was flawless, literally zero bugs | 100% | Positive |
| יש לנו באג קריטי בייצור, המערכת קורסת | 0% | Negative |
| יש לנו critical bug ב-production, זה literally urgent, צריך hotfix עכשיו | 60% | Negative |
| ה-team literally מחץ את ה-OKRs השנה, totally proud של כולם | 60% | Positive |
| יש לנו serious bottleneck ב-server, ה-latency literally עלתה פי שלוש | 60% | Negative |
| ה-pipeline literally קרס, ה-deployment תקוע ואנחנו totally blocked | 60% | Negative |
| ה-onboarding של ה-new hire הלך literally חלק, הוא totally התאקלם מהר | 60% | Positive |
| ה-PR קיבל literally עשרים comments, ה-reviewer totally לא היה satisfied | 60% | Negative |
| literally סגרנו את כל ה-tickets ב-sprint, ה-velocity שלנו totally השתפר | 60% | Positive |

## Implementation Steps

### 1. Dataset Construction

Generate ~300 sentences across 4 mixing levels (0% / 30% / 60% / 100%
English) with sentiment labels (positive / negative / neutral), through a
three-stage curation process rather than direct generation:

1. **Real seed sentences** — 30–50 hand-written sentences reflecting
   actual Israeli tech workplace phrasing (Slack messages, standups),
   anonymized so nothing is identifiable.
2. **Constrained generation** — each seed is expanded into its 30/60/100%
   variants by prompting an LLM conditioned on the seed, which preserves
   natural matrix-language structure instead of mechanical word-by-word
   translation.
3. **Human review** — 2 bilingual annotators rate each sentence for
   naturalness and independently relabel sentiment, blind to the intended
   label and to each other. Inter-annotator agreement (Cohen's kappa) is
   computed and reported; disagreements are adjudicated with a documented
   rule.

### 2. Embedding Analysis

Extract sentence embeddings for all sentences using both raw multilingual
model embeddings and SBERT-style sentence embeddings (which are trained
specifically for semantic similarity and are a better test of whether
same-meaning sentences cluster together):

- Raw CLS embeddings: mmBERT (primary multilingual model)
- SBERT-style: `paraphrase-multilingual-mpnet-base-v2`, LaBSE, and a
  Hebrew-native sentence-embedding model

Measure cosine distance between semantically equivalent sentences at
different mixing levels, and visualize the embedding space using t-SNE to
identify clustering patterns by mixing level.

### 3. Classification & Fine-tuning

Fine-tune models on sentiment classification under two conditions: (a)
training on pure Hebrew only, (b) training on a mix that includes
code-switched examples. Evaluate all models on all four mixing levels and
compare accuracy/F1 curves.

**Model selection prioritizes newer, stronger models over older ones**,
per instructor feedback:

- **Primary:** mmBERT (modern multilingual successor to XLM-R, pretrained
  on 3T tokens across 1800+ languages) and DictaBERT (modern Hebrew-specific
  model, reported to outperform existing Hebrew models on most benchmarks) —
  or AlephBERTGimmel as an alternative Hebrew model if needed.
- **Optional, time permitting:** XLM-R and HeBERT as an older-baseline
  comparison pair, to test whether "newer/stronger on standard benchmarks"
  also means "more robust to code-switching," or whether these are
  separate properties.

### 4. Human Performance Baseline

Sample ~50 sentences stratified across mixing levels; have the same 2
annotators label sentiment blind to mixing level and blind to each other.
This produces a human accuracy/agreement ceiling to contextualize model
performance — a model accuracy number is only meaningful relative to how
well humans do on the same sentences.

### 5. Analysis & Write-up

Identify at which mixing level performance degrades most sharply, which
model is most robust, whether fine-tuning on mixed data closes the
performance gap, how model accuracy compares to the human baseline, and
whether embedding-space distance from the 0% version correlates with the
accuracy drop.

## Methodology & Datasets

- **Dataset:** Self-constructed (novel contribution) — no existing
  Hebrew-English code-switching dataset for sentiment exists.
- **Models:** mmBERT, DictaBERT (primary); XLM-R, HeBERT (optional
  older-baseline comparison).
- **Evaluation metrics:** Accuracy and F1-score per mixing level; Cohen's
  kappa for annotator/human-baseline agreement.
- **Visualization:** t-SNE on sentence embeddings, accuracy-vs-mixing-level
  curves.
- **Tools:** HuggingFace Transformers, Sentence-Transformers,
  scikit-learn, matplotlib.

## Related Work

The closest prior work is **SemEval-2020 Task 9 (SentiMix)**, which
released Hindi-English and Spanish-English code-mixed sentiment corpora
(20K and 19K sentences respectively) with word-level language ID and
sentence-level sentiment labels. The best systems reached ~75% F1 on
Hinglish, with BERT-based approaches and ensembles performing best. This
project extends that line of work to Hebrew-English, and adds a mixing-level
framing (0/30/60/100% English) and geometric embedding-space analysis that
goes beyond classification accuracy alone.

- Patwa et al., "SemEval-2020 Task 9: Overview of Sentiment Analysis of
  Code-Mixed Tweets," SemEval 2020.
- Garain, Mahata & Das, "JUNLP at SemEval-2020 Task 9," ACL Anthology 2020.

## Innovation Highlights

- **Novel dataset:** The first annotated Hebrew-English code-switching
  dataset for sentiment analysis — a resource that does not currently
  exist and can benefit future NLP research on Israeli text.
- **Rigorous data curation:** Seed-constrained generation plus blind human
  review and inter-annotator agreement scoring, rather than unvalidated
  LLM generation.
- **Novel research question:** Systematic measurement of how mixing level
  affects the multilingual embedding space — not just classification
  accuracy, but geometric analysis of where meaning "breaks down."
- **Human-grounded evaluation:** A human performance baseline on the same
  data, so model accuracy is judged against a real ceiling rather than in
  isolation.
- **Practical solution:** Fine-tuning on even a small code-switched
  training set is tested directly against a pure-Hebrew baseline to show
  measurable robustness gains.
- **Cultural relevance:** Directly addresses the gap between how NLP
  models are trained and how Israeli high-tech professionals actually
  write online.

---

**Student ID:** 209055722 | **Email:** avivitzhak.barelbal@post.runi.ac.il
