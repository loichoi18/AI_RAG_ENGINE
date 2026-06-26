"""Dense (semantic) retriever.

Embeds the query with the same bge model used at ingest time and runs an
approximate-nearest-neighbour search in Qdrant. Supports ``top_k``, an optional
score threshold, exact metadata filters, and ACL filtering — all pushed into the
vector store so filtering happens before ranking. An optional embedding cache
avoids re-encoding repeated queries.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from ingestion.base import Embedder
from models.domain import RetrievalResult
from retrieval.base import Retriever
from storage.base import VectorStore
from utils.logging import get_logger

logger = get_logger(__name__)


class DenseRetriever(Retriever):
    """Semantic retriever over dense bge embeddings stored in a vector store."""

    name = "dense"

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        score_threshold: float | None = None,
        embedding_cache: "EmbeddingCacheLike | None" = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._score_threshold = score_threshold
        self._cache = embedding_cache

    def retrieve(
        self,
        query: str,
        top_k: int,
        acl: Sequence[str] | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Embed ``query`` and return the top-``k`` accessible chunks."""
        started = time.perf_counter()
        vector = self._embed(query)
        hits = self._store.search(
            query_vector=vector,
            top_k=top_k,
            acl=acl,
            metadata_filter=metadata_filter,
            score_threshold=self._score_threshold,
        )
        results = [
            RetrievalResult(chunk=h.chunk, score=h.score, retriever_name=self.name)
            for h in hits
        ]
        logger.info(
            "retrieval.dense",
            top_k=top_k,
            returned=len(results),
            top_score=results[0].score if results else None,
            acl_scoped=acl is not None,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return results

    def _embed(self, query: str) -> list[float]:
        """Embed ``query``, using the embedding cache when available."""
        if self._cache is not None:
            cached = self._cache.get_embedding(query)
            if cached is not None:
                logger.info("cache.hit", kind="embedding")
                return cached
        vector = self._embedder.embed([query])[0]
        if self._cache is not None:
            self._cache.set_embedding(query, vector)
            logger.info("cache.miss", kind="embedding")
        return vector


class EmbeddingCacheLike:
    """Structural type for an embedding cache (see services.cache)."""

    def get_embedding(self, query: str) -> list[float] | None:  # pragma: no cover
        raise NotImplementedError

    def set_embedding(self, query: str, vector: list[float]) -> None:  # pragma: no cover
        raise NotImplementedError
