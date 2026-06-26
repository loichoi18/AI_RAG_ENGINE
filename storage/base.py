"""Abstract interface for the vector storage layer."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any
from models.domain import Chunk, RetrievalResult


class VectorStore(ABC):
    """Persistence and similarity-search contract for chunk embeddings."""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: Sequence[float], top_k: int, acl: Sequence[str] | None = None,
               metadata_filter: Mapping[str, Any] | None = None,
               score_threshold: float | None = None) -> list[RetrievalResult]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_ids: Sequence[str]) -> None:
        raise NotImplementedError
