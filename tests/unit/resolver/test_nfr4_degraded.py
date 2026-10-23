"""NFR-4 through the REAL `RedisCache` (on fakeredis): the resolver keeps answering when RxNav is
down, when Redis is down, and when both are, and a degraded or local-only result is never written
to `rx:resolve:*`.

The Redis outage is `FakeServer.connected = False`: every command raises ConnectionError, which is
what a dead Redis does to `redis.asyncio`. The positive control (an RxNav-backed result IS cached)
keeps the "nothing was cached" assertions from passing vacuously.
"""

import fakeredis
import pytest

from pharmarag.cache import RedisCache
from pharmarag.resolver.cascade import CascadeResolver
from pharmarag.resolver.local import PgLocalStore
from pharmarag.resolver.rxnav_client import ApproxCandidate

pytestmark = pytest.mark.pg

HYDRALAZINE = "5470"


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def client(server):
    return fakeredis.FakeAsyncRedis(server=server)


@pytest.fixture
def resolver(pool, rx, thresholds, client) -> CascadeResolver:
    cache = RedisCache("redis://unused", client=client)
    return CascadeResolver(PgLocalStore(pool), rx, thresholds.resolver, cache)


async def one(resolver, mention):
    (out,) = await resolver.resolve_detailed([mention])
    return out


async def rx_keys(client) -> list[bytes]:
    return await client.keys("rx:*")


async def test_rxnav_and_redis_both_down_exact_name_still_resolves_and_nothing_is_cached(
    resolver, rx, server, client
):
    rx.down = True
    server.connected = False
    out = await one(resolver, "Vistaril")  # exact brand in the local alias table
    assert (out.resolution.status, out.resolution.match) == ("resolved", "brand")
    assert out.resolution.ingredient_key == "5553"
    assert not out.degraded  # the alias table answered; RxNav was never asked
    assert rx.calls == []

    server.connected = True  # Redis comes back: it must hold nothing from the outage
    assert await rx_keys(client) == []


async def test_rxnav_and_redis_both_down_a_misspelling_falls_back_to_trigram_and_is_not_cached(
    resolver, rx, server, client
):
    rx.down = True
    server.connected = False
    out = await one(resolver, "klonipin")
    assert out.degraded and out.resolution.degraded
    assert (out.resolution.status, out.resolution.ingredient_key) == ("resolved", "2598")
    server.connected = True
    assert await rx_keys(client) == []


async def test_redis_down_rxnav_up_resolves_without_a_cache(resolver, rx, server):
    rx.approx["hydralzine"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    server.connected = False
    out = await one(resolver, "hydralzine")  # cache get = miss, cache set = no-op, no exception
    assert (out.resolution.status, out.resolution.ingredient_key) == ("resolved", HYDRALAZINE)
    assert not out.degraded and not out.cached
    assert "approximate" in rx.names_called


async def test_redis_up_rxnav_down_degraded_and_local_only_results_are_not_cached(
    resolver, rx, client
):
    rx.down = True
    await one(resolver, "klonipin")  # degraded: trigram only
    await one(resolver, "Vistaril")  # local exact
    await one(resolver, "xyzzy")  # degraded, unknown
    assert await rx_keys(client) == []


async def test_positive_control_an_rxnav_backed_result_is_cached_with_a_24h_ttl(
    resolver, rx, client
):
    rx.approx["hydralzine"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    await one(resolver, "hydralzine")
    assert await rx_keys(client) == [b"rx:resolve:hydralzine"]
    assert 0 < await client.ttl("rx:resolve:hydralzine") <= 24 * 3600


async def test_a_cached_entry_survives_an_rxnav_outage_but_a_new_mention_degrades(
    resolver, rx, client
):
    rx.approx["hydralzine"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    await one(resolver, "hydralzine")  # RxNav up: cached
    rx.down = True
    rx.calls.clear()
    hit = await one(resolver, "hydralzine")
    assert hit.cached and not hit.degraded and rx.calls == []
    miss = await one(resolver, "klonipin")
    assert miss.degraded
    assert await rx_keys(client) == [b"rx:resolve:hydralzine"]  # the degraded one was not added


async def test_rx_facts_with_rxnav_down_return_no_lines_and_cache_nothing(rx, client):
    from pharmarag.resolver.facts import RxNavFactsSource

    source = RxNavFactsSource(rx, RedisCache("redis://unused", client=client), max_facts=6)
    rx.down = True
    assert await source.facts(["5553"]) == []
    assert await client.keys("rx:*") == []
