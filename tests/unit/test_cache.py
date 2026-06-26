"""Unit tests for caching: in-memory cache, embedding cache, CachedRetriever."""

from __future__ import annotations

import time
from collections.abc import Sequence

from models.domain import Chunk, RetrievalResult
from retrieval.base import Retriever
from retrieval.cached import CachedRetriever
from services.cache import EmbeddingCache, InMemoryCache


def test_in_memory_cache_get_set() -> None:
    cache = InMemoryCache()
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}
    assert cache.get("missing") is None


def test_in_memory_cache_ttl_expiry() -> None:
    cache = InMemoryCache()
    cache.set("k", "v", ttl_seconds=0)
    time.sleep(0.01)
    assert cache.get("k") is None


def test_embedding_cache_roundtrip() -> None:
    ec = EmbeddingCache(InMemoryCache(), model_name="bge")
    assert ec.get_embedding("hello") is None
    ec.set_embedding("hello", [0.1, 0.2, 0.3])
    assert ec.get_embedding("hello") == [0.1, 0.2, 0.3]


class _CountingRetriever(Retriever):
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(
        self, query: str, top_k: int, acl: Sequence[str] | None = None
    ) -> list[RetrievalResult]:
        self.calls += 1
        chunk = Chunk(chunk_id="a", document_id="d", text="cached text")
        return [RetrievalResult(chunk=chunk, score=0.5, retriever_name="inner")]


def test_cached_retriever_hits_on_second_call() -> None:
    inner = _CountingRetriever()
    cached = CachedRetriever(inner, InMemoryCache())

    first = cached.retrieve("q", top_k=3)
    second = cached.retrieve("q", top_k=3)

    assert inner.calls == 1  # second call served from cache
    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]
    assert first[0].score == second[0].score


def test_cached_retriever_separates_by_acl() -> None:
    inner = _CountingRetriever()
    cached = CachedRetriever(inner, InMemoryCache())

    cached.retrieve("q", top_k=3, acl=["group:hr"])
    cached.retrieve("q", top_k=3, acl=["group:engineering"])

    # Different ACLs are different cache keys -> two underlying calls.
    assert inner.calls == 2
