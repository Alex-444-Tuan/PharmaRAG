"""Runtime tests must not depend on whether Lane A's caching factories exist yet.

The three cycle-4 factories are replaced with fakes; a test that wants the "not implemented"
behaviour patches them itself. CACHE_BACKEND is memory so no test builds a Redis client unless it
asks for one.
"""

import pytest

from pharmarag.app import runtime as runtime_module
from tests.graph.fakes import FakeAnswerCache, FakeQueryEmbeddingCache, FakeRateLimiter


@pytest.fixture
def settings(settings):
    return settings.model_copy(update={"cache_backend": "memory"})


@pytest.fixture(autouse=True)
def _fake_caching_factories(monkeypatch):
    monkeypatch.setattr(runtime_module, "build_rate_limiter", lambda s, c: FakeRateLimiter())
    monkeypatch.setattr(runtime_module, "build_answer_cache", lambda s, c: FakeAnswerCache())
    monkeypatch.setattr(
        runtime_module, "build_query_embedding_cache", lambda e, c: FakeQueryEmbeddingCache(e)
    )
