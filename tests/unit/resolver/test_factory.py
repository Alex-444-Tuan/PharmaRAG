"""The factory bodies, end to end through the real RxNav client with httpx mocked by respx."""

import logging

import httpx
import pytest
import respx

from pharmarag.cache import InMemoryCache
from pharmarag.resolver import factory
from pharmarag.resolver.base import Resolver, RxFactsSource

pytestmark = pytest.mark.pg

BASE = "https://rx.test/REST"


@pytest.fixture
def rx_settings(settings):
    return settings.model_copy(update={"rxnav_base_url": BASE})


@pytest.fixture(autouse=True)
async def _no_backoff_sleep(monkeypatch):
    async def fast(_s):
        return None

    monkeypatch.setattr("pharmarag.ingest.http.asyncio.sleep", fast)


def related(*concepts):
    groups: dict[str, list] = {}
    for rxcui, name, tty in concepts:
        groups.setdefault(tty, []).append({"rxcui": rxcui, "name": name, "tty": tty})
    return {
        "relatedGroup": {
            "conceptGroup": [{"tty": t, "conceptProperties": c} for t, c in groups.items()]
        }
    }


@respx.mock
async def test_resolver_through_http_brand_exact_hit(pool, rx_settings, thresholds):
    respx.get(f"{BASE}/rxcui.json").respond(json={"idGroup": {"rxnormId": ["1234"]}})
    respx.get(f"{BASE}/rxcui/1234/properties.json").respond(
        json={"properties": {"rxcui": "1234", "name": "Atarax", "tty": "BN"}}
    )
    respx.get(f"{BASE}/rxcui/1234/related.json").respond(
        json=related(("5553", "hydroxyzine", "IN"))
    )
    cache = InMemoryCache()
    resolver = factory.build_resolver(rx_settings, thresholds, pool, cache)
    (r,) = await resolver.resolve(["Atarax 25 mg"])
    assert (r.status, r.match, r.ingredient_key, r.product_hint) == (
        "resolved",
        "brand",
        "5553",
        "1234",
    )
    assert await cache.get("rx:resolve:atarax") is not None
    await factory.close_rxnav(rx_settings)


@respx.mock
async def test_resolver_through_http_rxnav_503_falls_back_to_local(
    pool, rx_settings, thresholds, caplog
):
    route = respx.get(url__startswith=BASE).respond(503)
    resolver = factory.build_resolver(rx_settings, thresholds, pool, InMemoryCache())
    with caplog.at_level(logging.WARNING, logger="pharmarag.resolver"):
        (r,) = await resolver.resolve(["klonipin"])  # does not raise
    assert (r.status, r.match, r.ingredient_key) == ("resolved", "approximate", "2598")
    assert route.call_count == 3  # the client's three attempts, once
    assert any("degraded" in rec.message for rec in caplog.records)

    # The breaker is open now: the next mention does not wait for RxNav again.
    calls = route.call_count
    (r2,) = await resolver.resolve(["lasik"])
    assert r2.ingredient_key == "4603"
    assert route.call_count == calls
    await factory.close_rxnav(rx_settings)


@respx.mock
async def test_resolver_through_http_timeouts_fall_back_to_local(pool, rx_settings, thresholds):
    respx.get(url__startswith=BASE).mock(side_effect=httpx.ReadTimeout("2 s"))
    resolver = factory.build_resolver(rx_settings, thresholds, pool, InMemoryCache())
    (r,) = await resolver.resolve(["klonipin"])
    assert r.ingredient_key == "2598"
    await factory.close_rxnav(rx_settings)


@respx.mock
async def test_facts_source_through_http(rx_settings, thresholds):
    respx.get(f"{BASE}/rxcui/17767/related.json").respond(
        json=related(
            ("17767", "amlodipine", "IN"),
            ("58927", "Norvasc", "BN"),
            ("317541", "Oral Tablet", "DF"),
            ("329528", "amlodipine 5 MG", "SCDC"),
        )
    )
    respx.get(f"{BASE}/rxcui/58927/related.json").respond(
        json=related(("17767", "amlodipine", "IN"))
    )
    source = factory.build_rx_facts_source(rx_settings, thresholds, InMemoryCache())
    lines = await source.facts(["17767"])
    assert lines == [
        "RxNorm brand names for amlodipine: Norvasc.",
        "RxNorm dose forms for amlodipine: Oral Tablet.",
        "RxNorm strengths for amlodipine: 5 MG.",
    ]
    assert len(lines) <= thresholds.retrieval.max_facts
    await factory.close_rxnav(rx_settings)


async def test_builders_return_the_protocol_types_and_share_one_rxnav_client(
    pool, rx_settings, thresholds
):
    cache = InMemoryCache()
    resolver: Resolver = factory.build_resolver(rx_settings, thresholds, pool, cache)
    facts: RxFactsSource = factory.build_rx_facts_source(rx_settings, thresholds, cache)
    assert callable(resolver.resolve) and callable(facts.facts)
    assert resolver._rx is facts._rx  # one rate limiter, one breaker, one concept memo
    other = rx_settings.model_copy()
    assert factory.build_resolver(other, thresholds, pool, cache)._rx is not resolver._rx
    await factory.close_rxnav(rx_settings)
    await factory.close_rxnav(other)


def test_the_facts_source_is_capped_by_the_max_facts_threshold(rx_settings, thresholds):
    capped = thresholds.model_copy(
        update={"retrieval": thresholds.retrieval.model_copy(update={"max_facts": 4})}
    )
    assert factory.build_rx_facts_source(rx_settings, capped, InMemoryCache())._max == 4
