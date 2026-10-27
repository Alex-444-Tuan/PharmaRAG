"""Runtime wiring for cycle 4: cache backend, rate limiter, answer cache, query-embedding cache."""

import logging

import fakeredis
import pytest

from pharmarag.app import runtime as runtime_module
from pharmarag.app.observability import RXNAV_LOGGER, Metrics, MissProbeEmbedder
from pharmarag.app.runtime import build_runtime, close_runtime
from pharmarag.cache import InMemoryCache, RedisCache
from tests.graph.fakes import (
    FakeAnswerCache,
    FakeQueryEmbeddingCache,
    FakeRateLimiter,
    FakeResolver,
    FakeRxFacts,
    FakeVerifier,
)

pytestmark = pytest.mark.pg


def not_implemented(*args, **kwargs):
    raise NotImplementedError("cycle 4, lane A")


@pytest.fixture
def settings_pg(settings, pg_url):
    return settings.model_copy(update={"database_url": pg_url})


@pytest.fixture
def prod(settings_pg):
    return settings_pg.model_copy(update={"eval_mode": False})


@pytest.fixture(autouse=True)
def _no_real_verifier(monkeypatch):
    monkeypatch.setattr(runtime_module, "build_verifier", not_implemented)


def stages():
    return {
        "resolver": FakeResolver({}),
        "rx_facts": FakeRxFacts(),
        "verifier": FakeVerifier([[]]),
    }


class SpyRedis(RedisCache):
    def __init__(self, url, **kw):
        super().__init__(url, client=fakeredis.FakeAsyncRedis())
        self.url = url
        self.closed = False

    async def aclose(self):
        self.closed = True
        await super().aclose()


async def test_memory_backend_builds_an_in_memory_cache_that_the_runtime_does_not_own(
    settings_pg, make_fake_llm, fake_embedder
):
    rt = await build_runtime(settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), **stages())
    try:
        assert isinstance(rt.deps.cache, InMemoryCache) and rt.owns_cache is False
    finally:
        await close_runtime(rt)


async def test_redis_backend_builds_redis_from_the_settings_url_and_closes_it(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    monkeypatch.setattr(runtime_module, "RedisCache", SpyRedis)
    s = settings_pg.model_copy(
        update={"cache_backend": "redis", "redis_url": "redis://cache.internal:6379/3"}
    )
    rt = await build_runtime(s, embedder=fake_embedder, llm=make_fake_llm([]), **stages())
    cache = rt.deps.cache
    assert isinstance(cache, SpyRedis) and cache.url == "redis://cache.internal:6379/3"
    assert rt.owns_cache is True
    await close_runtime(rt)
    assert cache.closed


async def test_an_injected_cache_is_not_closed_by_the_runtime(
    settings_pg, make_fake_llm, fake_embedder
):
    cache = SpyRedis("redis://x")
    s = settings_pg.model_copy(update={"cache_backend": "redis"})
    rt = await build_runtime(
        s, embedder=fake_embedder, llm=make_fake_llm([]), cache=cache, **stages()
    )
    await close_runtime(rt)
    assert rt.deps.cache is cache and not cache.closed  # the caller owns what it passed in


async def test_the_factories_get_the_shared_cache_and_the_query_cache_wraps_the_embedder(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    seen = {}
    limiter, answers, queries = FakeRateLimiter(), FakeAnswerCache(), FakeQueryEmbeddingCache(None)

    def build_limiter(settings, cache):
        seen["limiter"] = (settings, cache)
        return limiter

    def build_answers(settings, cache):
        seen["answers"] = (settings, cache)
        return answers

    def build_queries(embedder, cache):
        seen["queries"] = (embedder, cache)
        return queries

    monkeypatch.setattr(runtime_module, "build_rate_limiter", build_limiter)
    monkeypatch.setattr(runtime_module, "build_answer_cache", build_answers)
    monkeypatch.setattr(runtime_module, "build_query_embedding_cache", build_queries)
    rt = await build_runtime(settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), **stages())
    try:
        assert rt.rate_limiter is limiter and rt.answer_cache is answers
        assert rt.deps.query_embeddings is queries
        shared = rt.deps.cache
        assert seen["limiter"] == (settings_pg, shared) and seen["answers"] == (settings_pg, shared)
        wrapped, cache = seen["queries"]
        assert cache is shared
        assert isinstance(wrapped, MissProbeEmbedder) and wrapped._inner is fake_embedder
        assert rt.deps.embedder is fake_embedder  # the graph's own embedder is not wrapped
    finally:
        await close_runtime(rt)


async def test_answer_cache_false_does_not_build_an_answer_cache(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    monkeypatch.setattr(
        runtime_module, "build_answer_cache", lambda *a: pytest.fail("built although disabled")
    )
    s = settings_pg.model_copy(update={"answer_cache": False})
    rt = await build_runtime(s, embedder=fake_embedder, llm=make_fake_llm([]), **stages())
    try:
        assert rt.answer_cache is None
    finally:
        await close_runtime(rt)


@pytest.mark.parametrize("factory", ["build_rate_limiter", "build_answer_cache"])
async def test_outside_eval_mode_a_missing_caching_factory_stops_startup(
    prod, make_fake_llm, fake_embedder, monkeypatch, factory
):
    monkeypatch.setattr(runtime_module, factory, not_implemented)
    with pytest.raises(NotImplementedError):  # production never runs without a rate limit
        await build_runtime(prod, embedder=fake_embedder, llm=make_fake_llm([]), **stages())


async def test_outside_eval_mode_a_missing_query_cache_factory_stops_startup(
    prod, make_fake_llm, fake_embedder, monkeypatch
):
    monkeypatch.setattr(runtime_module, "build_query_embedding_cache", not_implemented)
    with pytest.raises(NotImplementedError):
        await build_runtime(prod, embedder=fake_embedder, llm=make_fake_llm([]), **stages())


async def test_a_refused_start_closes_the_pool_and_the_redis_client_it_made(
    prod, make_fake_llm, fake_embedder, monkeypatch
):
    made = []

    def make(url):
        made.append(SpyRedis(url))
        return made[-1]

    monkeypatch.setattr(runtime_module, "RedisCache", make)
    monkeypatch.setattr(runtime_module, "build_rate_limiter", not_implemented)
    with pytest.raises(NotImplementedError):
        await build_runtime(
            prod.model_copy(update={"cache_backend": "redis"}),
            embedder=fake_embedder,
            llm=make_fake_llm([]),
            **stages(),
        )
    assert len(made) == 1 and made[0].closed


async def test_in_eval_mode_missing_caching_factories_leave_the_stages_off_and_log_it(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch, caplog
):
    for name in ("build_rate_limiter", "build_answer_cache", "build_query_embedding_cache"):
        monkeypatch.setattr(runtime_module, name, not_implemented)
    with caplog.at_level(logging.WARNING, logger="pharmarag.app.runtime"):
        rt = await build_runtime(
            settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), **stages()
        )
    try:
        assert rt.rate_limiter is None and rt.answer_cache is None
        assert rt.deps.query_embeddings is None
        for what in ("rate limiter", "answer cache", "query-embedding cache"):
            assert f"{what} is not implemented" in caplog.text
    finally:
        await close_runtime(rt)


async def test_injected_stages_replace_the_caching_factories(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    for name in ("build_rate_limiter", "build_answer_cache", "build_query_embedding_cache"):
        monkeypatch.setattr(runtime_module, name, lambda *a: pytest.fail("factory called"))
    rt = await build_runtime(
        settings_pg,
        embedder=fake_embedder,
        llm=make_fake_llm([]),
        rate_limiter=None,
        answer_cache=None,
        query_embeddings=None,
        **stages(),
    )
    try:
        assert rt.rate_limiter is None and rt.answer_cache is None
        assert rt.deps.query_embeddings is None
    finally:
        await close_runtime(rt)


async def test_metrics_are_on_deps_and_the_rxnav_counter_is_attached_until_close(
    settings_pg, make_fake_llm, fake_embedder
):
    metrics = Metrics()
    rt = await build_runtime(
        settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), metrics=metrics, **stages()
    )
    log = logging.getLogger(RXNAV_LOGGER)
    assert rt.deps.metrics is metrics and rt.rxnav_error_handler in log.handlers
    log.warning("rxnav unavailable (/x): E: e")
    assert metrics.registry.get_sample_value("pharmarag_rxnav_errors_total") == 1.0
    await close_runtime(rt)
    assert rt.rxnav_error_handler not in log.handlers
    log.warning("rxnav unavailable (/x): E: e")
    assert metrics.registry.get_sample_value("pharmarag_rxnav_errors_total") == 1.0


async def test_a_runtime_without_injected_metrics_gets_its_own(
    settings_pg, make_fake_llm, fake_embedder
):
    a = await build_runtime(settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), **stages())
    b = await build_runtime(settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), **stages())
    try:
        assert a.deps.metrics is not None and a.deps.metrics is not b.deps.metrics
        assert a.deps.metrics.registry is not b.deps.metrics.registry
    finally:
        await close_runtime(a)
        await close_runtime(b)
