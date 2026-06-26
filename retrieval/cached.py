"""Caching decorator for retrievers.

Wraps any :class:`Retriever` and memoizes whole result lists keyed by the query,
top-k, and the caller's ACL (so cached results never cross access boundaries).
Results are serialized via Pydantic and rehydrated on read. Cache hits and
misses are logged for observability.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence

from models.domain import RetrievalResult
from retrieval.base import Retriever
from services.cache.base import Cache
from utils.logging import get_logger

logger = get_logger(__name__)


class CachedRetriever(Retriever):
    """Memoizes a wrapped retriever's results in a :class:`Cache`."""

    name = "cached"

    def __init__(
        self,
        retriever: Retriever,
        cache: Cache,
        ttl_seconds: int | None = None,
        namespace: str = "retrieval",
    ) -> None:
        self._retriever = retriever
        self._cache = cache
        self._ttl = ttl_seconds
        self._namespace = namespace

    def retrieve(
        self,
        query: str,
        top_k: int,
        acl: Sequence[str] | None = None,
    ) -> list[RetrievalResult]:
        """Return cached results if present, otherwise retrieve and cache."""
        key = self._key(query, top_k, acl)
        cached = self._cache.get(key)
        if cached is not None:
            logger.info("cache.hit", kind="retrieval", query=query)
            return [RetrievalResult.model_validate(item) for item in cached]

        started = time.perf_counter()
        results = self._retriever.retrieve(query, top_k, acl)
        self._cache.set(key, [r.model_dump() for r in results], self._ttl)
        logger.info(
            "cache.miss",
            kind="retrieval",
            query=query,
            returned=len(results),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return results

    def _key(self, query: str, top_k: int, acl: Sequence[str] | None) -> str:
        acl_part = "*" if acl is None else ",".join(sorted(acl))
        raw = f"{self._namespace}|{query}|{top_k}|{acl_part}"
        return f"{self._namespace}:" + hashlib.sha256(raw.encode()).hexdigest()
