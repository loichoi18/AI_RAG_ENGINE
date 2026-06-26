# Evaluation Framework

Every architecture change in this system is measurable. The evaluation framework
scores both **retrieval** (did we find the right documents?) and **generation**
(is the answer correct, grounded, and properly cited?) against a golden dataset,
and produces JSON / Markdown / CSV reports for regression tracking and
configuration comparisons.

## Design

Two independent probes, matching where ground truth lives:

- **Retrieval evaluation** (`RetrievalEvaluator`) runs a `Retriever` per golden
  question and scores the ranked results. No LLM — deterministic and CI-friendly.
- **Generation evaluation** (`GenerationEvaluator`) runs the full `AnswerService`
  and scores the answer via the **judge layer**.

Relevance is **additive**: a retrieved chunk is relevant if its text contains an
`answer_span`, OR its `document_id` is in the example's expected documents, OR its
`chunk_id` is in `expected_chunks`. The span/document basis is independent of the
chunking strategy, which keeps chunking comparisons fair.

## Golden dataset

`evaluation/datasets/golden.json` (50+ examples). Each `GoldenExample`:

| field | meaning |
|---|---|
| `id` | stable identifier |
| `query` / `question` | the question (aliases) |
| `document_id` | primary supporting document |
| `expected_documents` | optional set of relevant documents (document-level) |
| `expected_chunks` | optional finer relevance signal |
| `answer_spans` | substrings whose presence marks a chunk relevant |
| `answer` / `ground_truth` | reference answer (aliases) |
| `question_type` | `simple_lookup` · `multi_hop` · `ambiguous` · `unanswerable` |

Load with `load_golden(path)` (JSON or YAML).

## Retrieval metrics

All operate on a ranked list of boolean relevance flags; `K` is configurable.

- **Recall@K** = `|relevant ∩ top-K| / |relevant|` — coverage of relevant items.
- **Precision@K** = `|relevant ∩ top-K| / K` — purity of the top K.
- **MRR** = `mean(1 / rank_of_first_relevant)` — how high the first hit lands.
- **nDCG@K** = `DCG@K / IDCG@K`, where `DCG = Σ relᵢ / log₂(i+1)` — full ranking
  quality, normalized by the ideal ranking.
- **Hit Rate@K** = `1` if any relevant item is in the top K else `0` — coarse
  "did we surface anything?".

**Interpretation:** Recall/Hit-Rate measure coverage; Precision measures purity;
MRR/nDCG measure ranking quality. They are complementary — a change can raise
recall while hurting nDCG, so read them together.

## Generation metrics

Scored through the **judge interface** (`evaluation/judges/`):

- **Answer Correctness** — does the answer match the reference?
- **Faithfulness** — are all claims grounded in retrieved context?
- **Answer Completeness** — does the answer cover the reference's information?
- **Citation Accuracy** — `supported / (supported + unsupported + missing)`, where
  citations are classified as supported (exists + source supports the claim),
  unsupported (out of range, or unsupported by source), or missing (a claim with
  no valid citation).

Unanswerable questions are scored on whether the system correctly **refused**.

### Judge layer

`Judge` is an interface with three implementations:

- `LLMJudge` — prompts an LLM (Ollama / OpenAI / Anthropic via the shared
  `LLMClient`) for a JSON `{score, reasoning}`. **Primary** in production.
- `LexicalJudge` — deterministic lexical metrics; **fallback** and CI default.
- `FallbackJudge(LLMJudge, LexicalJudge)` — uses the LLM judge but degrades to
  lexical scoring on any error, so a run never crashes when the model is down.

Build the production judge with `build_judge(settings.llm)`.

## Running evaluation

```python
from evaluation.dataset import load_golden
from evaluation.runner import EvaluationRunner

dataset = load_golden("evaluation/datasets/golden.json")
runner = EvaluationRunner(retriever, answer_service=answer_service, k=5, name="baseline")
report = runner.run(dataset, output_dir="reports", formats=("json", "md", "csv"))
print(report.retrieval, report.generation)
```

Retrieval-only (fast, no model) — omit `answer_service`. Reports are written to
`reports/<name>.{json,md,csv}` with overall metrics, per-question rows, latency,
and citation statistics.

## Comparing configurations

```python
from evaluation.comparison import compare_retrieval, compare_chunking

# dense vs sparse vs hybrid vs multi-query (chunking fixed)
compare_retrieval(documents, golden, embedder=..., tokenizer=..., make_store=..., k=5)

# fixed vs recursive vs semantic chunking (retriever fixed = hybrid)
compare_chunking(documents, golden, embedder=..., tokenizer=..., make_store=..., k=5)
```

Each returns a table of `{variant: {metric: value}}`, rendered to Markdown via
`render_report(...)`.

## Regression testing

Persist a baseline report's JSON, re-run after a change, and diff the `overall`
metrics. Because the lexical path is fully deterministic, retrieval and
lexical-generation numbers are reproducible across runs and machines.
