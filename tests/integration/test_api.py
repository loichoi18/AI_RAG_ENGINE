"""API integration tests via FastAPI TestClient.

Uses a fake service graph injected through ``app.dependency_overrides`` so the
endpoints are exercised end-to-end without models or Qdrant. Covers health,
query (answer + refusal), ingest, list/delete documents, and metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import get_services
from models.domain import Chunk, RetrievalResult
from services.answer_service import AnswerResponse
from utils.metrics import MetricsRegistry
from utils.tracing import NoOpTracer


class _FakeAnswerService:
    def __init__(self, refuse: bool = False) -> None:
        self._refuse = refuse

    def answer(self, query: str, acl: Sequence[str] | None = None) -> AnswerResponse:
        if self._refuse:
            return AnswerResponse(
                answer="insufficient context", citations=[], confidence=0.0,
                refused=True, token_usage={}, sources=[],
            )
        chunk = Chunk(chunk_id="d-0", document_id="d", text="deploy with docker")
        src = RetrievalResult(chunk=chunk, score=0.9, retriever_name="hybrid")
        return AnswerResponse(
            answer="Deploy with docker [1].", citations=["d-0"], confidence=0.8,
            refused=False, token_usage={"prompt": 5, "completion": 2}, sources=[src],
        )


class _FakeStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def ensure_collection(self) -> None:
        pass

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        self._chunks.extend(chunks)

    def scroll_all(self) -> list[Chunk]:
        return list(self._chunks)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        ids = set(chunk_ids)
        self._chunks = [c for c in self._chunks if c.chunk_id not in ids]


class _FakeChunker:
    def chunk(self, document: object) -> list[Chunk]:
        return [Chunk(chunk_id=f"{document.document_id}-0", document_id=document.document_id,  # type: ignore[attr-defined]
                      text=document.content)]  # type: ignore[attr-defined]


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class _FakeServices:
    def __init__(self, refuse: bool = False) -> None:
        self.store = _FakeStore()
        self.chunker = _FakeChunker()
        self.embedder = _FakeEmbedder()
        self.metrics = MetricsRegistry()
        self.tracer = NoOpTracer()
        self.answer_services = {
            "hybrid": _FakeAnswerService(refuse), "dense": _FakeAnswerService(refuse),
        }

    def reindex_sparse(self) -> None:
        pass


def _client(services: _FakeServices) -> TestClient:
    app = create_app(services=services)  # type: ignore[arg-type]
    app.dependency_overrides[get_services] = lambda: services
    return TestClient(app)


def test_health() -> None:
    client = _client(_FakeServices())
    assert client.get("/v1/health").json() == {"status": "ok"}


def test_query_returns_answer_with_citations() -> None:
    client = _client(_FakeServices())
    resp = client.post("/v1/query", json={"query": "how to deploy?", "mode": "hybrid"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("Deploy with docker")
    assert body["citations"] == ["d-0"]
    assert body["confidence"] == 0.8
    assert body["retrieved_chunks"][0]["document_id"] == "d"
    assert "latency_ms" in body
    assert "X-Request-ID" in resp.headers


def test_query_refusal() -> None:
    client = _client(_FakeServices(refuse=True))
    body = client.post("/v1/query", json={"query": "obscure?"}).json()
    assert body["refused"] is True


def test_query_unknown_user_is_structured_error() -> None:
    client = _client(_FakeServices())
    resp = client.post("/v1/query", json={"query": "q", "user": "ceo"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "unknown_user"


def test_ingest_then_list_then_delete() -> None:
    services = _FakeServices()
    client = _client(services)

    ingest = client.post(
        "/v1/ingest",
        json={"documents": [{"document_id": "runbook", "text": "deploys use docker"}]},
    )
    assert ingest.status_code == 200
    assert ingest.json()["indexed_chunks"] == 1

    docs = client.get("/v1/documents").json()
    assert docs["total"] == 1
    assert docs["documents"][0]["document_id"] == "runbook"

    deleted = client.delete("/v1/documents/runbook").json()
    assert deleted["deleted_chunks"] == 1
    assert client.get("/v1/documents").json()["total"] == 0


def test_metrics_endpoint_tracks_requests() -> None:
    client = _client(_FakeServices())
    client.post("/v1/query", json={"query": "how to deploy?"})
    metrics = client.get("/v1/metrics").json()
    assert metrics["counters"]["requests_total"] >= 1
    assert "query" in metrics["latency_ms"]
