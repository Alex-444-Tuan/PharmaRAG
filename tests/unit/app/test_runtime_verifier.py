"""The verifier startup guard (FR-6): the same rule as the resolver's (FR-3)."""

import logging

import pytest

from pharmarag.app import runtime as runtime_module
from pharmarag.app.runtime import build_runtime, close_runtime
from tests.graph.fakes import FakeResolver, FakeRxFacts, FakeVerifier

pytestmark = pytest.mark.pg


def not_implemented(*args, **kwargs):
    raise NotImplementedError("cycle 3, lane A")


@pytest.fixture(autouse=True)
def _real_verifier_factory_is_not_built_yet(monkeypatch):
    monkeypatch.setattr(runtime_module, "build_verifier", not_implemented)


@pytest.fixture
def prod(settings, pg_url):
    return settings.model_copy(update={"database_url": pg_url, "eval_mode": False})


@pytest.fixture
def dev(settings, pg_url):
    return settings.model_copy(update={"database_url": pg_url, "eval_mode": True})


async def test_outside_eval_mode_a_verifier_that_cannot_be_built_stops_startup(
    prod, make_fake_llm, fake_embedder
):
    with pytest.raises(NotImplementedError):  # not swallowed: production never starts unchecked
        await build_runtime(
            prod,
            embedder=fake_embedder,
            llm=make_fake_llm([]),
            resolver=FakeResolver({}),
            rx_facts=None,
        )


async def test_outside_eval_mode_an_explicit_no_verifier_stops_startup_too(
    prod, make_fake_llm, fake_embedder
):
    with pytest.raises(RuntimeError, match="no grounding verifier"):
        await build_runtime(
            prod,
            embedder=fake_embedder,
            llm=make_fake_llm([]),
            resolver=FakeResolver({}),
            rx_facts=None,
            verifier=None,
        )


async def test_outside_eval_mode_verify_mode_off_stops_startup(prod, make_fake_llm, fake_embedder):
    off = prod.model_copy(update={"verify_mode": "off"})
    with pytest.raises(RuntimeError, match="no grounding verifier"):
        await build_runtime(
            off,
            embedder=fake_embedder,
            llm=make_fake_llm([]),
            resolver=FakeResolver({}),
            rx_facts=None,
            verifier=FakeVerifier([[]]),
        )


async def test_a_refused_start_does_not_leak_the_connection_pool(
    prod, make_fake_llm, fake_embedder, monkeypatch
):
    pools = []
    real_create = runtime_module.create_pool

    async def spy(*args, **kwargs):
        pools.append(await real_create(*args, **kwargs))
        return pools[-1]

    monkeypatch.setattr(runtime_module, "create_pool", spy)
    for kwargs in ({"verifier": None}, {}):  # explicit None, then the failing factory
        with pytest.raises((RuntimeError, NotImplementedError)):
            await build_runtime(
                prod,
                embedder=fake_embedder,
                llm=make_fake_llm([]),
                resolver=FakeResolver({}),
                rx_facts=None,
                **kwargs,
            )
    assert len(pools) == 2 and all(p.closed for p in pools)


async def test_outside_eval_mode_a_complete_build_starts(prod, make_fake_llm, fake_embedder):
    rt = await build_runtime(
        prod,
        embedder=fake_embedder,
        llm=make_fake_llm([]),
        resolver=FakeResolver({}),
        rx_facts=FakeRxFacts(),
        verifier=FakeVerifier([[]]),
    )
    try:
        assert rt.deps.verifier_available and rt.deps.resolver_available
    finally:
        await close_runtime(rt)


async def test_in_eval_mode_a_verifier_not_built_yet_stays_off_and_is_logged(
    dev, make_fake_llm, fake_embedder, caplog
):
    with caplog.at_level(logging.WARNING, logger="pharmarag.app.runtime"):
        rt = await build_runtime(
            dev, embedder=fake_embedder, llm=make_fake_llm([]), resolver=None, rx_facts=None
        )
    try:
        assert rt.deps.verifier is None and rt.deps.verifier_available is False
        assert "verifier is not implemented" in caplog.text
    finally:
        await close_runtime(rt)


async def test_a_verifier_build_error_other_than_not_implemented_always_propagates(
    dev, make_fake_llm, fake_embedder, monkeypatch
):
    def broken(*args):
        raise OSError("NLI weights missing")

    monkeypatch.setattr(runtime_module, "build_verifier", broken)
    with pytest.raises(OSError):  # eval mode too
        await build_runtime(
            dev, embedder=fake_embedder, llm=make_fake_llm([]), resolver=None, rx_facts=None
        )


async def test_the_verifier_factory_gets_settings_thresholds_and_the_embedder(
    dev, make_fake_llm, fake_embedder, monkeypatch
):
    seen = []
    verifier = FakeVerifier([[]])

    def fake_build(settings, thresholds, embedder):
        seen.append((settings, thresholds, embedder))
        return verifier

    monkeypatch.setattr(runtime_module, "build_verifier", fake_build)
    rt = await build_runtime(
        dev, embedder=fake_embedder, llm=make_fake_llm([]), resolver=None, rx_facts=None
    )
    try:
        assert rt.deps.verifier is verifier
        assert seen == [(dev, rt.deps.thresholds, fake_embedder)]
    finally:
        await close_runtime(rt)


async def test_an_injected_verifier_replaces_the_factory(
    dev, make_fake_llm, fake_embedder, monkeypatch
):
    monkeypatch.setattr(runtime_module, "build_verifier", lambda *a: pytest.fail("factory called"))
    verifier = FakeVerifier([[]])
    rt = await build_runtime(
        dev,
        embedder=fake_embedder,
        llm=make_fake_llm([]),
        resolver=None,
        rx_facts=None,
        verifier=verifier,
    )
    try:
        assert rt.deps.verifier is verifier
    finally:
        await close_runtime(rt)
