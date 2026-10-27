"""Builders for the operational components. The signatures are the contract (Lane B's app
lifespan calls them); Lane A owns the bodies (cycle 4)."""

from pharmarag.app.settings import Settings
from pharmarag.cache import Cache, RedisCache
from pharmarag.caching.answers import CacheAnswerCache, NullAnswerCache
from pharmarag.caching.embeddings import CachedQueryEmbeddings
from pharmarag.caching.ratelimit import LocalRateLimiter, RedisRateLimiter
from pharmarag.models.embedder import Embedder
from pharmarag.ops import AnswerCache, QueryEmbeddingCache, RateLimiter


def build_rate_limiter(settings: Settings, cache: Cache) -> RateLimiter:
    """Redis counters when the cache is Redis (shared across API processes), process-local
    counters otherwise (CACHE_BACKEND=memory means one process)."""
    if isinstance(cache, RedisCache):
        return RedisRateLimiter(cache, settings.rate_limit_per_minute)
    return LocalRateLimiter(settings.rate_limit_per_minute)


def build_answer_cache(settings: Settings, cache: Cache) -> AnswerCache:
    if not settings.answer_cache:
        return NullAnswerCache()
    return CacheAnswerCache(cache)


def build_query_embedding_cache(embedder: Embedder, cache: Cache) -> QueryEmbeddingCache:
    return CachedQueryEmbeddings(embedder, cache)
