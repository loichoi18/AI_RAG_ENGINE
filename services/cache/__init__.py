"""Caching: interface, in-memory and Redis backends, adapters, factory."""

from services.cache.base import Cache
from services.cache.embedding_cache import EmbeddingCache
from services.cache.factory import build_cache
from services.cache.memory import InMemoryCache

__all__ = ["Cache", "EmbeddingCache", "InMemoryCache", "build_cache"]
