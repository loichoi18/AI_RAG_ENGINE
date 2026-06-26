"""Cache interface.

A minimal string-keyed cache of JSON-serializable values with TTL. Keeping the
contract tiny lets the retrieval layer cache both embeddings and result lists
without depending on a specific backend (Redis in production, in-memory for
tests and key-free local runs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Cache(ABC):
    """A simple TTL cache of JSON-serializable values."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or ``None`` if absent/expired."""
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL."""
        raise NotImplementedError
