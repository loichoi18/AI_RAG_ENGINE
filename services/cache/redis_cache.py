"""Redis-backed cache.

Production backend for sharing cached embeddings and retrieval results across
API workers. Values are JSON-encoded. The ``redis`` import is deferred to
construction so the package imports cleanly when Redis is not installed (the
in-memory backend remains usable).
"""

from __future__ import annotations

import json
from typing import Any

from services.cache.base import Cache


class RedisCache(Cache):
    """Cache stored in Redis with JSON serialization and TTL."""

    def __init__(self, url: str = "redis://localhost:6379/0", default_ttl_seconds: int = 3600) -> None:
        import redis  # lazy: optional dependency

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._default_ttl = default_ttl_seconds

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        return None if raw is None else json.loads(raw)

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        self._client.set(key, json.dumps(value), ex=ttl)
