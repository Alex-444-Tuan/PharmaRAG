"""Fixed-window rate limiter (spec 10, 12): window, limit edge, rollover, fail-open."""

import logging

import fakeredis
import pytest

from pharmarag.app.settings import Settings
from pharmarag.cache import InMemoryCache, RedisCache
from pharmarag.caching.factory import build_rate_limiter
from pharmarag.caching.ratelimit import LocalRateLimiter, RedisRateLimiter
from pharmarag.ops import RATE_WINDOW_S, rate_key


class Clock:
    def __init__(self, t: float = 6000.0) -> None:  # 6000 = start of a minute
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def server():
    return fakeredis.FakeServer()


@pytest.fixture
def redis_client(server):
    return fakeredis.FakeAsyncRedis(server=server)


@pytest.fixture(params=["redis", "local"])
def make_limiter(request, redis_client):
    def make(limit: int, clock: Clock):
        if request.param == "redis":
            cache = RedisCache("redis://unused", client=redis_client)
            return RedisRateLimiter(cache, limit, clock=clock)
        return LocalRateLimiter(limit, clock=clock)

    return make


async def test_limit_edge_request_number_limit_is_allowed_next_is_not(make_limiter):
    lim = make_limiter(3, Clock())
    decisions = [await lim.hit("1.2.3.4") for _ in range(4)]
    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert [d.remaining for d in decisions] == [2, 1, 0, 0]
    assert decisions[0].retry_after_s == 0
    assert decisions[3].retry_after_s == 60  # 0 s into the window


async def test_retry_after_counts_down_to_the_end_of_the_window(make_limiter):
    clock = Clock(6000.0)
    lim = make_limiter(1, clock)
    await lim.hit("c")
    clock.t = 6000.0 + 45.5
    d = await lim.hit("c")
    assert not d.allowed
    assert d.retry_after_s == 14  # int(60 - 45.5)
    clock.t = 6000.0 + 59.99
    assert (await lim.hit("c")).retry_after_s == 1  # never 0


async def test_window_rollover_resets_the_count(make_limiter):
    clock = Clock()
    lim = make_limiter(2, clock)
    assert [(await lim.hit("c")).allowed for _ in range(3)] == [True, True, False]
    clock.t += RATE_WINDOW_S  # next minute
    d = await lim.hit("c")
    assert d.allowed and d.remaining == 1


async def test_clients_have_separate_budgets(make_limiter):
    lim = make_limiter(1, Clock())
    assert (await lim.hit("a")).allowed
    assert not (await lim.hit("a")).allowed
    assert (await lim.hit("b")).allowed


async def test_limit_zero_disables_the_limiter(make_limiter):
    lim = make_limiter(0, Clock())
    assert all([(await lim.hit("c")).allowed for _ in range(50)])


async def test_redis_counter_uses_the_spec_key_and_sets_a_ttl_once(redis_client):
    lim = RedisRateLimiter(
        RedisCache("redis://unused", client=redis_client), 10, clock=Clock(6000.0)
    )
    await lim.hit("9.9.9.9")
    key = rate_key("9.9.9.9", 100)  # 6000 // 60
    assert await redis_client.get(key) == b"1"
    assert 0 < await redis_client.ttl(key) <= 60
    # A later hit in the same window must not extend the TTL (EXPIRE NX: first hit only).
    await redis_client.expire(key, 5)
    await lim.hit("9.9.9.9")
    assert await redis_client.get(key) == b"2"
    assert await redis_client.ttl(key) <= 5


async def test_redis_counter_is_shared_between_two_limiters(redis_client):
    cache = RedisCache("redis://unused", client=redis_client)
    clock = Clock()
    a = RedisRateLimiter(cache, 2, clock=clock)
    b = RedisRateLimiter(cache, 2, clock=clock)  # a second API process
    assert (await a.hit("c")).allowed
    assert (await b.hit("c")).allowed
    assert not (await a.hit("c")).allowed


async def test_redis_down_fails_open_and_logs(server, redis_client, caplog):
    lim = RedisRateLimiter(RedisCache("redis://unused", client=redis_client), 1, clock=Clock())
    assert (await lim.hit("secret-client")).allowed
    assert not (await lim.hit("secret-client")).allowed
    server.connected = False  # every command now raises ConnectionError
    with caplog.at_level(logging.WARNING):
        d = await lim.hit("secret-client")
    assert d.allowed and d.retry_after_s == 0
    assert any("failing open" in r.message for r in caplog.records)
    assert "secret-client" not in caplog.text  # the client id (an IP) is not logged
    server.connected = True  # Redis is back: the old counter still applies
    assert not (await lim.hit("secret-client")).allowed


def test_factory_picks_redis_or_local_by_cache_type(redis_client):
    s = Settings(_env_file=None, rate_limit_per_minute=7)
    redis_cache = RedisCache("redis://unused", client=redis_client)
    assert isinstance(build_rate_limiter(s, redis_cache), RedisRateLimiter)
    assert isinstance(build_rate_limiter(s, InMemoryCache()), LocalRateLimiter)
