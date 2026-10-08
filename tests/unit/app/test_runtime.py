"""Runtime wiring: factories, degradation when a stage is not implemented, shared cache."""

import logging

import pytest

from pharmarag.app import runtime as runtime_module
from pharmarag.app.runtime import build_runtime, close_runtime
from pharmarag.cache import InMemoryCache
from pharmarag.graph.run import effective_flags
from pharmarag.graph.state import Flags
from tests.graph.fakes import FakeResolver, FakeRxFacts, FakeVerifier

pytestmark = pytest.mark.pg


def not_implemented(*args, **kwargs):
    raise NotImplementedError("cycle 2, lane A")


@pytest.fixture
def settings_pg(settings, pg_url):
    return settings.model_copy(update={"database_url": pg_url})


@pytest.fixture(autouse=True)
def _no_real_verifier(monkeypatch):
    """The real factory loads an NLI model. Tests that care about it patch their own."""
    monkeypatch.setattr(runtime_module, "build_verifier", not_implemented)


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


# --- production safety: no resolver means no start (FR-3) --------------------------------------


async def test_outside_eval_mode_a_resolver_that_cannot_be_built_stops_startup(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    prod = settings_pg.model_copy(update={"eval_mode": False})
    monkeypatch.setattr(runtime_module, "build_resolver", not_implemented)
    with pytest.raises(NotImplementedError):
        await build_runtime(prod, embedder=fake_embedder, llm=make_fake_llm([]))


async def test_outside_eval_mode_an_explicit_no_resolver_stops_startup_too(
    settings_pg, make_fake_llm, fake_embedder
):
    prod = settings_pg.model_copy(update={"eval_mode": False})
    with pytest.raises(RuntimeError, match="no drug resolver"):
        await build_runtime(
            prod, embedder=fake_embedder, llm=make_fake_llm([]), resolver=None, rx_facts=None
        )


async def test_a_resolver_error_other_than_not_implemented_always_propagates(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    def broken(*a):
        raise ValueError("bad config")

    monkeypatch.setattr(runtime_module, "build_resolver", broken)
    with pytest.raises(ValueError):  # eval mode too: only NotImplementedError is tolerated
        await build_runtime(settings_pg, embedder=fake_embedder, llm=make_fake_llm([]))


async def test_in_eval_mode_a_missing_resolver_is_the_allowed_baseline_build(
    settings_pg, make_fake_llm, fake_embedder
):
    assert settings_pg.eval_mode is True
    rt = await build_runtime(
        settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), resolver=None, rx_facts=None
    )
    try:
        assert rt.deps.resolver is None
    finally:
        await close_runtime(rt)


async def test_injected_stages_replace_the_factories_and_the_real_ones_are_not_called(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    def must_not_run(*a):
        raise AssertionError("factory called although a stage was injected")

    monkeypatch.setattr(runtime_module, "build_resolver", must_not_run)
    monkeypatch.setattr(runtime_module, "build_rx_facts_source", must_not_run)
    resolver, facts = FakeResolver({}), FakeRxFacts()
    verifier = FakeVerifier([[]])
    rt = await build_runtime(
        settings_pg,
        embedder=fake_embedder,
        llm=make_fake_llm([]),
        resolver=resolver,
        rx_facts=facts,
        verifier=verifier,
    )
    try:
        assert rt.deps.resolver is resolver and rt.deps.rx_facts is facts
        assert rt.deps.verifier is verifier and rt.deps.verifier_available is True
    finally:
        await close_runtime(rt)


# --- shutdown ----------------------------------------------------------------------------------


async def test_close_runtime_closes_the_pool_and_the_shared_rxnav_client(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    closed: list = []

    async def fake_close_rxnav(settings):
        closed.append(settings)

    monkeypatch.setattr(runtime_module, "close_rxnav", fake_close_rxnav)
    rt = await build_runtime(
        settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), resolver=None, rx_facts=None
    )
    await close_runtime(rt)
    assert closed == [settings_pg] and rt.deps.pool.closed


async def test_rxnav_is_closed_even_when_closing_the_pool_fails(
    settings_pg, make_fake_llm, fake_embedder, monkeypatch
):
    closed: list = []

    async def fake_close_rxnav(settings):
        closed.append(settings)

    async def bad_close():
        raise RuntimeError("pool close failed")

    monkeypatch.setattr(runtime_module, "close_rxnav", fake_close_rxnav)
    rt = await build_runtime(
        settings_pg, embedder=fake_embedder, llm=make_fake_llm([]), resolver=None, rx_facts=None
    )
    real_close = rt.deps.pool.close
    monkeypatch.setattr(rt.deps.pool, "close", bad_close)
    with pytest.raises(RuntimeError):
        await close_runtime(rt)
    assert closed == [settings_pg]
    await real_close()


async def test_the_real_factory_client_is_released_by_close_runtime(
    settings_pg, make_fake_llm, fake_embedder
):
    """No stubs: build with the real factories, then check the client registry is emptied."""
    from pharmarag.resolver import factory

    rt = await build_runtime(settings_pg, embedder=fake_embedder, llm=make_fake_llm([]))
    assert id(settings_pg) in factory._clients
    await close_runtime(rt)
    assert id(settings_pg) not in factory._clients
