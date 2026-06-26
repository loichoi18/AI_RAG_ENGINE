"""Cache construction from settings, with graceful fallback.

Builds the configured backend. If Redis is selected but unavailable (not
installed or unreachable), it falls back to the in-memory cache and logs a
warning rather than failing startup — caching is an optimization, not a
correctness requirement.
"""

from __future__ import annotations

from configs.settings import CacheBackend, CacheSettings
from services.cache.base import Cache
from services.cache.memory import InMemoryCache
from utils.logging import get_logger

logger = get_logger(__name__)


def build_cache(settings: CacheSettings) -> Cache:
    """Return a cache instance per ``settings`` (in-memory fallback on failure)."""
    if settings.backend is CacheBackend.REDIS:
        try:
            from services.cache.redis_cache import RedisCache

            cache: Cache = RedisCache(settings.redis_url, settings.ttl_seconds)
            logger.info("cache.backend", backend="redis", url=settings.redis_url)
            return cache
        except Exception as exc:  # noqa: BLE001 - fall back on any redis error
            logger.warning("cache.redis_unavailable", error=str(exc), fallback="memory")

    logger.info("cache.backend", backend="memory")
    return InMemoryCache(settings.ttl_seconds)
