"""Embedding cache adapter.

Wraps a generic :class:`Cache` to memoize query embeddings, matching the small
structural interface (``get_embedding`` / ``set_embedding``) that
:class:`~retrieval.dense.dense_retriever.DenseRetriever` expects. Keys are
namespaced and content-hashed so different models never collide.
"""

from __future__ import annotations

import hashlib

from services.cache.base import Cache


class EmbeddingCache:
    """Caches query -> embedding vectors via a backing :class:`Cache`."""

    def __init__(self, cache: Cache, model_name: str, ttl_seconds: int | None = None) -> None:
        self._cache = cache
        self._model_name = model_name
        self._ttl = ttl_seconds

    def _key(self, query: str) -> str:
        digest = hashlib.sha256(f"{self._model_name}|{query}".encode()).hexdigest()
        return f"emb:{digest}"

    def get_embedding(self, query: str) -> list[float] | None:
        """Return a cached embedding for ``query``, or ``None``."""
        value = self._cache.get(self._key(query))
        return list(value) if value is not None else None

    def set_embedding(self, query: str, vector: list[float]) -> None:
        """Cache ``vector`` for ``query``."""
        self._cache.set(self._key(query), vector, self._ttl)
