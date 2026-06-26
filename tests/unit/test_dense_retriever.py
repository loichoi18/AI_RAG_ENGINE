"""Unit tests for the dense retriever using fakes (no model, no Qdrant)."""

from __future__ import annotations

from models.domain import Chunk
from retrieval.dense import DenseRetriever
from tests.fakes import FakeEmbedder, FakeVectorStore


def _index() -> tuple[FakeEmbedder, FakeVectorStore]:
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    chunks = [
        Chunk(chunk_id="a", document_id="doc1", text="alpha alpha alpha", acl=[]),
        Chunk(chunk_id="b", document_id="doc1", text="beta beta", acl=["group:hr"]),
        Chunk(chunk_id="c", document_id="doc2", text="gamma gamma", acl=[]),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    return embedder, store


def test_dense_ranks_by_similarity_and_tags_name() -> None:
    embedder, store = _index()
    results = DenseRetriever(embedder, store).retrieve("alpha", top_k=3)

    assert results[0].chunk.chunk_id == "a"
    assert all(r.retriever_name == "dense" for r in results)


def test_dense_applies_acl_filter() -> None:
    embedder, store = _index()
    # An engineering caller must not see the hr-only chunk "b".
    results = DenseRetriever(embedder, store).retrieve(
        "beta", top_k=5, acl=["group:engineering"]
    )
    assert all(r.chunk.chunk_id != "b" for r in results)


def test_dense_score_threshold_filters_weak_matches() -> None:
    embedder, store = _index()
    retriever = DenseRetriever(embedder, store, score_threshold=0.99)
    results = retriever.retrieve("alpha", top_k=5)
    # Only the (near-)identical "alpha" chunk clears a 0.99 cosine threshold.
    assert [r.chunk.chunk_id for r in results] == ["a"]


def test_dense_metadata_filter_passthrough() -> None:
    embedder, store = _index()
    DenseRetriever(embedder, store).retrieve(
        "gamma", top_k=5, metadata_filter={"document_id": "doc2"}
    )
    assert store.last_search["metadata_filter"] == {"document_id": "doc2"}
