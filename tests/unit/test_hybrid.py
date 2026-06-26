"""Unit tests for RRF fusion and the hybrid retriever."""

from __future__ import annotations

from models.domain import Chunk, RetrievalResult
from retrieval.dense import DenseRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import BM25Retriever
from tests.fakes import FakeEmbedder, FakeVectorStore


def _result(chunk_id: str, score: float, name: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=chunk_id, document_id="d", text=f"text {chunk_id}"),
        score=score,
        retriever_name=name,
    )


def test_rrf_rewards_items_in_multiple_lists() -> None:
    dense = [_result("x", 0.9, "dense"), _result("y", 0.8, "dense")]
    sparse = [_result("y", 5.0, "sparse"), _result("z", 4.0, "sparse")]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    # "y" appears in both lists, so it should rank first after fusion.
    assert fused[0].chunk.chunk_id == "y"
    assert {r.chunk.chunk_id for r in fused} == {"x", "y", "z"}
    assert all(r.retriever_name == "rrf" for r in fused)


def test_rrf_respects_top_k() -> None:
    lists = [[_result(c, 1.0, "dense") for c in ("a", "b", "c", "d")]]
    fused = reciprocal_rank_fusion(lists, k=60, top_k=2)
    assert len(fused) == 2


def _hybrid() -> HybridRetriever:
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    chunks = [
        Chunk(chunk_id="a", document_id="d", text="deploy service docker", acl=[]),
        Chunk(chunk_id="b", document_id="d", text="hr vacation policy", acl=[]),
        Chunk(chunk_id="c", document_id="d", text="deployment rollout release", acl=[]),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    dense = DenseRetriever(embedder, store)
    sparse = BM25Retriever(chunks)
    return HybridRetriever(dense, sparse, rrf_k=60, candidate_k=10)


def test_hybrid_returns_fused_results() -> None:
    results = _hybrid().retrieve("docker deploy", top_k=3)
    assert results
    assert all(r.retriever_name == "hybrid" for r in results)
    assert results[0].chunk.chunk_id in {"a", "c"}
