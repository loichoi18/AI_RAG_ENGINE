# Production Hybrid RAG System — Internal Knowledge Search

> Architecture & design document. Authored before implementation.
> Status: **approved design (Phase 0)** · Default LLM: **Ollama (local)** · Scale: **medium, self-hosted**

---

## 1. Requirements analysis

Internal knowledge search differs from a generic RAG demo on two axes that drive the
whole design: **trust** (employees must believe answers and see citations) and
**access** (users must only retrieve what they are permitted to see).

### Functional requirements
- Ingest heterogeneous internal sources: PDF, Markdown/wiki, HTML, Office docs.
  The source set is mixed and will grow, so ingestion must be pluggable.
- Hybrid retrieval (dense + sparse). Internal corpora contain many exact tokens
  (error codes, ticket IDs, config keys, acronyms) that pure semantic search misses
  and BM25 captures.
- Grounded generation with inline citations to source document + section/page.
- Explicit refusal when retrieved context is insufficient. In an internal tool a
  confident wrong answer is worse than "I don't know — see these documents."
- Incremental re-indexing. Internal documents change constantly; stale answers
  erode trust quickly.

### Non-functional requirements
- Reproducible: clone-and-run with no API keys (local models via Ollama).
- Observable: latency per stage, retrieval statistics, token usage, request tracing.
- Evaluable: every retrieval / chunking / rerank change must move a metric.
- Self-hosted, medium scale (~10^4–10^6 chunks), deployable via Docker Compose.

### Cross-cutting concerns specific to internal search
- **Access control (ACL):** answers must respect document permissions. An `acl`
  (allowed groups/users) is stored in chunk metadata and filtered **at query time**
  in Qdrant, *before* retrieval. Retrieving and then redacting leaks data through
  ranking signals and counts.
- **Freshness:** track `source_updated_at`; support delete + re-embed on change via
  idempotent upserts keyed by a stable `chunk_id`.
- **Sensitivity / PII:** metadata flag so sensitive chunks can be excluded by policy.

### Assumptions
- Single-tenant to start; English corpus.
- Permissions are modeled as group IDs supplied by the ingestion source.
- "Medium scale" fits a single Qdrant node (sharding noted as scale-out path).

---

## 2. Production architecture

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion (offline / scheduled)"]
        L[Loaders: PDF / MD / HTML / Office] --> P[Preprocess + metadata + ACL]
        P --> CK[Chunker: fixed | recursive | semantic]
        CK --> EM[Embed: bge-base local]
        EM --> IDX[Indexer upsert]
        IDX --> VS[(Qdrant: dense + sparse + payload)]
    end

    subgraph Query["Query (online)"]
        U[Client] --> API[FastAPI /query]
        API --> ACL{ACL filter}
        ACL --> D[Dense retrieve]
        ACL --> S[Sparse BM25]
        D --> RRF[RRF fusion]
        S --> RRF
        RRF --> RER[Cross-encoder rerank]
        RER --> CONF{Confidence gate}
        CONF -->|low| REFUSE[Refuse + suggest docs]
        CONF -->|ok| GEN[Generator: grounded + citations]
        GEN --> RESP[Answer + sources + trace]
    end

    subgraph Eval["Evaluation (offline)"]
        GS[Golden set] --> RM[Recall@k / MRR / nDCG]
        GS --> GM[Faithfulness / citation acc / correctness]
    end

    VS -.-> D
    VS -.-> S
    Obs[[Structured logging / latency / token usage]] -.-> API
```

### Why each component exists
- **Two retrievers, not one.** Dense provides semantic recall; sparse (BM25) provides
  lexical precision on exact terms. Internal KBs need both.
- **RRF, not score-weighting.** Dense cosine scores and BM25 scores are on
  incomparable scales. Reciprocal Rank Fusion fuses by rank, is parameter-light
  (`k ≈ 60`), and is the robust default.
- **Cross-encoder reranker.** Fusion yields a good top-20; the reranker reads
  (query, chunk) jointly to produce a precise top-5. Classic two-stage funnel:
  cheap-and-wide retrieval, then expensive-and-narrow reranking.
- **Confidence gate.** Turns "never hallucinate" into an architectural guarantee: if
  the top rerank score is below threshold, refuse before the LLM runs.
- **ACL filter pre-retrieval.** Security boundary, enforced in the vector store.

---

## 3. Repository structure

```
rag-engine/
├── api/            # FastAPI app, routers, request/response schemas, deps
├── ingestion/      # loaders/, preprocess, chunkers/, indexer, embedder
├── retrieval/      # base.py, dense.py, sparse.py, hybrid.py (RRF)
├── reranking/      # base.py, cross_encoder.py
├── generation/     # providers/ (ollama|anthropic|openai), prompt.py, citations.py, gate.py
├── evaluation/     # metrics/, datasets/, runner.py, report.py
├── storage/        # qdrant_client.py (collection schema, upsert, search, ACL filter)
├── services/       # rag_service.py — orchestrator wiring every layer
├── configs/        # settings.py (pydantic-settings), config.yaml
├── models/         # domain.py — Document, Chunk, RetrievalResult, GenerationResult
├── utils/          # logging.py, tokenizer.py, timing.py
├── tests/          # unit/, integration/, eval/
├── scripts/        # ingest.py, evaluate.py, bootstrap_qdrant.py
├── docker/         # Dockerfile, docker-compose.yml (api + qdrant + ollama)
└── docs/           # README, architecture.md, decisions/ (ADRs)
```

Every component depends on an interface, not a concrete class, so each is
independently testable and mockable.

---

## 4. Components and interfaces

| Interface        | Method                                          | Responsibility                                   |
|------------------|-------------------------------------------------|--------------------------------------------------|
| `Loader`         | `load(path) -> Document`                        | Source file to normalized Document + metadata    |
| `Chunker`        | `chunk(doc) -> list[Chunk]`                     | Document to chunks with provenance metadata      |
| `Embedder`       | `embed(texts) -> list[Vector]`                  | Text to dense vectors                            |
| `VectorStore`    | `upsert / search / delete`                      | Qdrant operations incl. ACL payload filter       |
| `Retriever`      | `retrieve(query, k, acl) -> list[RetrievalResult]` | Dense, Sparse, Hybrid all implement this      |
| `Reranker`       | `rerank(query, results, k) -> list[RetrievalResult]` | Cross-encoder second stage                  |
| `ConfidenceGate` | `passes(results) -> bool`                       | Hallucination guardrail                          |
| `LLMProvider`    | `generate(prompt, context) -> GenerationResult` | Ollama / Anthropic / OpenAI behind one contract  |
| `Evaluator`      | `evaluate(dataset) -> MetricReport`             | Retrieval + generation metrics                   |

- `HybridRetriever` composes `DenseRetriever` + `SparseRetriever` and applies RRF
  (Strategy + Composite patterns).
- `RAGService` orchestrates the sequence; it knows the order of operations, not the
  concrete implementations.

### Core domain models (`models/domain.py`)
- `Document` — normalized source + metadata.
- `Chunk` — text plus `chunk_id, source_doc, page, section, chunk_strategy,
  token_count, acl, source_updated_at`.
- `RetrievalResult` — chunk + score + retriever origin.
- `GenerationResult` — answer + citations + confidence + token usage.

---

## 5. Trade-offs

| Decision     | Chosen                         | Alternative           | Rationale                                                                 |
|--------------|--------------------------------|-----------------------|---------------------------------------------------------------------------|
| Vector store | Qdrant                         | pgvector              | Native sparse+dense+payload filtering in one engine; ACL filter first-class. |
| Embeddings   | bge-base (local)               | Hosted API embeddings | Reproducible, free, private. Cost: slower, model download.                |
| LLM default  | Ollama (local)                 | Hosted API            | Clone-and-run, no keys. Swappable via one config line; abstraction is the signal. |
| Fusion       | RRF                            | Weighted score blend  | Scale-free, no tuning. Weighted needs per-corpus calibration.             |
| Chunking     | 3 strategies, recursive default| Pick one              | Let evaluation choose; semantic is accurate but costly so off by default. |
| ACL          | Filter pre-retrieval           | Retrieve then redact  | Redaction leaks via ranking signals/counts; filtering is the safe boundary. |

### Risks & bottlenecks
- Reranker latency — mitigate with batching and a top-k cap.
- Embedding throughput on CPU — batch; allow GPU via `device` config.
- Qdrant single-node ceiling — acceptable at medium scale; sharding is the scale-out path.
- Golden-set quality — evaluation is only as honest as the dataset.

---

## 6. Implementation roadmap

1. **Foundation** — scaffold, pydantic-settings config, domain models, layer
   interfaces, Docker Compose (api + qdrant + ollama).
   *Exit: containers up, `pytest` green on config + models.*
2. **Ingestion** — loaders, 3 chunkers with metadata + ACL, embedder, Qdrant indexer.
   *Exit: `scripts/ingest.py` populates Qdrant; unit tests per chunker.*
3. **Retrieval + reranking** — dense, sparse, RRF hybrid, cross-encoder, ACL filter.
   *Exit: hybrid beats single retrievers on a smoke set.*
4. **Generation** — provider abstraction (Ollama default), grounded prompt, citation
   extraction, confidence gate + refusal.
   *Exit: answers carry inline citations; low confidence yields refusal.*
5. **API + observability** — FastAPI endpoints, structured logging, latency/token
   tracking, request tracing.
   *Exit: `/query` end-to-end with traces.*
6. **Evaluation** — golden dataset; Recall@k / MRR / nDCG + faithfulness /
   citation-accuracy / correctness; comparison report.
   *Exit: one table comparing chunking + retrieval configs.*
7. **Documentation** — README, architecture.md, ADRs.
   *Exit: a stranger can clone and run.*

---

## Design decisions log (summary)

- **ACL:** designed-for, stubbed initially. Schema and query-time filter interface are
  real; the default policy is permissive until a permission source is wired in.
- **Default LLM provider:** Ollama, for key-free reproducibility.
- **Vector store:** Qdrant, for native hybrid + payload filtering.
