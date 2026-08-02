## Neuro-Symbolic LLM Reliability Engine

A hybrid fact-verification system that combines a fine-tuned transformer with a symbolic knowledge base and rule engine to score the factual reliability of text claims from raw data through to a deployable FastAPI backend.

Overview

This project builds an end-to-end pipeline that takes free text, extracts factual claims from it, and produces a **reliability score (0–1)** by cross-checking those claims against both a neural fact-verification model and a symbolic knowledge base. The neuro-symbolic design means claims aren't just classified by a black-box model — they're also checked against explicit, inspectable logic rules, which makes contradictions and unsupported claims easier to explain than a pure ML approach would allow.

**Author:** Kommera Shanmukhi IFHE University, Faculty of Science & Technology

## Pipeline

```
FEVER Dataset (185K claims)
   → Data Preprocessing & Tokenization
   → Knowledge Base Construction (JSON + RDF)
   → Claim Extraction (spaCy NER + Dependency Parsing)
   → Symbolic Representation Layer
   → RoBERTa Fine-tuning (3-class Fact Verification)
   → Rule-Based Verification Engine
   → Reliability Scoring
   → Evaluation & Confusion Matrix
   → FastAPI Backend
```

## Key Components

- **Dataset:** [FEVER v1.0](https://fever.ai/) — ~185,455 annotated claims from Wikipedia, split into SUPPORTS / REFUTES / NOT ENOUGH INFO
- **Model:** `roberta-base` fine-tuned for 3-class sequence classification on FEVER claim–evidence pairs
- **Knowledge Base:** A two-layer store — a JSON lookup table for O(1) predicate access, and an RDF graph (with SPARQL query support) built from curated facts plus high-confidence SUPPORTS claims from FEVER
- **Claim Extraction:** Converts free text into `(subject, relation, object)` triples using both regex pattern matching and spaCy dependency parsing
- **Symbolic Layer:** Canonicalizes triples into logical predicates (e.g. `located_in(eiffel_tower, paris)`) with entity alias resolution and relation normalization
- **Rule Engine:** Verifies predicates against the KB (forward/reverse lookup + contradiction detection) and enforces uniqueness constraints (e.g. a country has exactly one capital)
- **Reliability Scoring:** A weighted formula that penalizes contradicted claims heavily (0.8) and unverifiable claims lightly (0.1):

  ```
  Score = clip((V − 0.8·C − 0.1·U) / T, 0, 1)
  ```
  where V = verified claims, C = contradicted claims, U = unknown claims, T = total claims

  | Score Range | Label |
  |---|---|
  | 0.85 – 1.00 | 🟢 Highly Reliable |
  | 0.65 – 0.84 | 🟡 Mostly Reliable |
  | 0.40 – 0.64 | 🟠 Partially Reliable |
  | 0.00 – 0.39 | 🔴 Unreliable |

- **Serving:** A FastAPI backend that exposes the full pipeline (extraction → verification → scoring) as an API endpoint

## Tech Stack

`Python` · `PyTorch` · `Transformers (HuggingFace)` · `spaCy` · `RDFLib` · `scikit-learn` · `FastAPI` · `pandas` / `numpy` · `matplotlib` / `seaborn`

## Training Details

- Base model: `roberta-base` (125M params)
- Batch size: 32 effective (8 per device × 4 gradient accumulation steps)
- Learning rate: 2e-5 with linear warmup (6%)
- 3 epochs with early stopping (patience 2), mixed precision (FP16)
- Trained/evaluated on a stratified sample (30K train / 5K val) of the full FEVER set for tractability on limited GPU resources; the full corpus (145K claims) is supported by the same pipeline

## Getting Started

This project is packaged as a single Jupyter/Colab notebook (`Neuro_Symbolic_LLM_Reliability_Engine_Final.ipynb`) designed to run top-to-bottom on Google Colab (GPU runtime recommended).

1. Open the notebook in Google Colab
2. Run all cells sequentially — dependencies install automatically
3. The notebook downloads FEVER via HuggingFace `datasets`, trains the model, builds the KB, and evaluates the full pipeline
4. At the end, it packages the fine-tuned model + knowledge base for download and includes a ready-to-run FastAPI app (`app.py`) for serving

## Example

**Input:** `"Paris is the capital of Germany. Einstein was born in Australia."`
**Output:** 🔴 Unreliable: both claims contradict the knowledge base (verified capital and birthplace facts).

## Notes

- This is a research/academic project (final-year project submission); the reliability scoring thresholds and rule weights are heuristic and tunable.
- The knowledge base is seeded with a small set of curated facts plus an auto-augmented set mined from FEVER SUPPORTS claims — it is not exhaustive and is meant to demonstrate the neuro-symbolic architecture rather than serve as a production-grade fact database.
