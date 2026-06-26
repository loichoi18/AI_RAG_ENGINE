"""In-memory vector store.

A real, dependency-free :class:`VectorStore` implementation backed by a dict and
brute-force cosine search. It is appropriate for small corpora, local
development, and offline evaluation where standing up Qdrant is unnecessary. For
production scale, use :class:`~storage.qdrant_store.QdrantVectorStore`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from models.domain import Chunk, RetrievalResult
from retrieval.access import acl_allows
from storage.base import VectorStore


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine vector store with ACL and metadata filtering."""

    def __init__(self) -> None:
        self._points: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._points[chunk.chunk_id] = (chunk, list(vector))

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        acl: Sequence[str] | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        scored: list[RetrievalResult] = []
        for chunk, vector in self._points.values():
            if not acl_allows(chunk.acl, acl):
                continue
            if metadata_filter and not self._matches(chunk, metadata_filter):
                continue
            score = _cosine(query_vector, vector)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(
                RetrievalResult(chunk=chunk, score=score, retriever_name="vector_store")
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self._points.pop(chunk_id, None)

    def count(self) -> int:
        """Number of stored points."""
        return len(self._points)

    @staticmethod
    def _matches(chunk: Chunk, metadata_filter: Mapping[str, Any]) -> bool:
        payload = {"document_id": chunk.document_id, "chunk_id": chunk.chunk_id}
        return all(payload.get(key) == value for key, value in metadata_filter.items())
