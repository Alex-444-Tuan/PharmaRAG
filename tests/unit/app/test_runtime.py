"""Runtime wiring: factories, degradation when a stage is not implemented, shared cache."""

import logging

import pytest

from pharmarag.app import runtime as runtime_module
from pharmarag.app.runtime import build_runtime, close_runtime
from pharmarag.cache import InMemoryCache
from pharmarag.graph.run import effective_flags
from pharmarag.graph.state import Flags
from tests.graph.fakes import FakeResolver, FakeRxFacts

pytestmark = pytest.mark.pg


def not_implemented(*args, **kwargs):
    raise NotImplementedError("cycle 2, lane A")


@pytest.fixture
def settings_pg(settings, pg_url):
    return settings.model_copy(update={"database_url": pg_url})


async def test_factories_raising_not_implemented_leave_the_stages_off_and_log_it(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch, caplog
):
    monkeypatch.setattr(runtime_module, "build_resolver", not_implemented)
    monkeypatch.setattr(runtime_module, "build_rx_facts_source", not_implemented)
    with caplog.at_level(logging.WARNING, logger="pharmarag.app.runtime"):
        rt = await build_runtime(settings_pg, embedder=fake_embedder, llm=make_fake_llm([]))
    try:
        deps = rt.deps
        assert deps.resolver is None and deps.rx_facts is None and deps.verifier is None
        assert deps.resolver_available is False and deps.verifier_available is False
        assert isinstance(deps.cache, InMemoryCache)
        # The degraded app runs every request as the baseline, and says so in `meta`.
        assert effective_flags(Flags(), deps) == Flags(resolver=False, verifier=False)
        assert "resolver is not implemented" in caplog.text
        assert "facts source is not implemented" in caplog.text
    finally:
        await close_runtime(rt)


async def test_one_factory_failing_does_not_disable_the_other(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    facts = FakeRxFacts()
    monkeypatch.setattr(runtime_module, "build_resolver", not_implemented)
    monkeypatch.setattr(runtime_module, "build_rx_facts_source", lambda *a: facts)
    rt = await build_runtime(settings_pg, embedder=fake_embedder, llm=make_fake_llm([]))
    try:
        assert rt.deps.resolver is None and rt.deps.rx_facts is facts
    finally:
        await close_runtime(rt)


async def test_factories_get_the_settings_thresholds_pool_and_one_shared_cache(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    seen: dict = {}
    resolver, facts = FakeResolver({}), FakeRxFacts()

    def fake_build_resolver(settings, thresholds, pool, cache):
        seen["resolver"] = (settings, thresholds, pool, cache)
        return resolver

    def fake_build_facts(settings, thresholds, cache):
        seen["facts"] = (settings, thresholds, cache)
        return facts

    monkeypatch.setattr(runtime_module, "build_resolver", fake_build_resolver)
    monkeypatch.setattr(runtime_module, "build_rx_facts_source", fake_build_facts)
    rt = await build_runtime(settings_pg, embedder=fake_embedder, llm=make_fake_llm([]))
    try:
        d = rt.deps
        assert d.resolver is resolver and d.rx_facts is facts and d.verifier is None
        assert d.resolver_available is True
        assert seen["resolver"] == (settings_pg, d.thresholds, d.pool, d.cache)
        assert seen["facts"][2] is d.cache is seen["resolver"][3]  # one cache for both stages
    finally:
        await close_runtime(rt)


async def test_an_injected_cache_is_used(settings_pg, make_fake_llm, fake_embedder, monkeypatch):
    monkeypatch.setattr(runtime_module, "build_resolver", not_implemented)
    monkeypatch.setattr(runtime_module, "build_rx_facts_source", not_implemented)
    cache = InMemoryCache()
    rt = await build_runtime(
        settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), cache=cache
    )
    try:
        assert rt.deps.cache is cache
    finally:
        await close_runtime(rt)
