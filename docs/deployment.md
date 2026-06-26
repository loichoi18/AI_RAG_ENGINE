# Deployment

The full stack runs with one command via Docker Compose: **API**, **Qdrant**,
**Ollama**, and the **Streamlit frontend**.

## Quick start

```bash
# 1. Launch the stack (build images on first run)
docker compose -f docker/docker-compose.yml up --build

# 2. Pull the local LLM into Ollama (one-time; persisted in a volume)
docker exec -it rag-ollama ollama pull llama3.1:8b

# 3. Seed demo data so the system is queryable immediately
docker exec rag-api python -m scripts.bootstrap_demo
```

Endpoints:

| Service | URL |
|---|---|
| API (OpenAPI docs) | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Qdrant | http://localhost:6333 |
| Ollama | http://localhost:11434 |

## Services

- **qdrant** — vector store; data persists in the `qdrant_storage` volume.
- **ollama** — local LLM; model weights persist in the `ollama_models` volume.
- **api** — FastAPI app (`api.app:app`); reaches backends by service name
  (`RAG_QDRANT__HOST=qdrant`, `RAG_LLM__OLLAMA_BASE_URL=http://ollama:11434`);
  has a `/v1/health` healthcheck.
- **frontend** — Streamlit dashboard; thin HTTP client (`RAG_API_URL=http://api:8000`),
  starts after the API is healthy.

## Configuration

All configuration is environment-driven (`pydantic-settings`), nested with a
`__` delimiter and the `RAG_` prefix — e.g. `RAG_QDRANT__HOST`,
`RAG_LLM__PROVIDER`, `RAG_GENERATION__GATE_THRESHOLD`. See `.env.example`. To use
a hosted LLM instead of Ollama, set `RAG_LLM__PROVIDER=openai|anthropic` and
`RAG_LLM__API_KEY`.

## Observability

- **Logs** — structured JSON via structlog (`RAG_LOGGING__JSON_LOGS=true`); each
  request carries a `request_id`; each query logs `query_id`, latency,
  retrieval/rerank counts, token usage, and confidence. View with
  `docker logs -f rag-api`.
- **Tracing** — a `Tracer` abstraction with a structlog backend. To add Langfuse
  or OpenTelemetry, implement `utils.tracing.Tracer` and select it in
  `build_tracer(...)`; no call sites change.
- **Metrics** — `GET /v1/metrics` returns request counts, success/error rates,
  and per-operation latency (retrieval/generation/query). Swap the in-process
  `MetricsRegistry` for a Prometheus client behind the same record calls if
  scrape-based metrics are needed.

## Production notes

- The API builds its service graph once at startup (`build_services`); models
  load lazily on first use. The Qdrant collection is ensured best-effort so the
  API boots even if Qdrant is briefly unavailable.
- BM25 is held in memory and rebuilt from a Qdrant scroll after ingestion; for
  large corpora migrate sparse retrieval to Qdrant native sparse vectors.
- Run behind a reverse proxy (TLS, auth) for any non-local deployment; the mock
  user ACLs are for demonstration and should be replaced by real identity.
