"""In-memory cache with TTL.

Used by tests and for key-free local runs. Not shared across processes; for a
multi-worker deployment use :class:`~services.cache.redis_cache.RedisCache`.
"""

from __future__ import annotations

import time
from typing import Any

from services.cache.base import Cache


class InMemoryCache(Cache):
    """Process-local dict cache with per-entry expiry."""

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self._default_ttl = default_ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        self._store[key] = (time.monotonic() + ttl, value)
