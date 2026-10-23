"""Builders for the operational components. The signatures are the contract (Lane B's app
lifespan calls them); Lane A owns the bodies (cycle 4)."""

from pharmarag.app.settings import Settings
from pharmarag.cache import Cache
from pharmarag.models.embedder import Embedder
from pharmarag.ops import AnswerCache, QueryEmbeddingCache, RateLimiter


def build_rate_limiter(settings: Settings, cache: Cache) -> RateLimiter:
    raise NotImplementedError("cycle 4, lane A")


def build_answer_cache(settings: Settings, cache: Cache) -> AnswerCache:
    raise NotImplementedError("cycle 4, lane A")


def build_query_embedding_cache(embedder: Embedder, cache: Cache) -> QueryEmbeddingCache:
    raise NotImplementedError("cycle 4, lane A")
