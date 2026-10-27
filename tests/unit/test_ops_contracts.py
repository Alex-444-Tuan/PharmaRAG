"""Cycle 4 contract tests: RedisCache (via fakeredis) and spec section 5 key formats."""

import fakeredis
import pytest

from pharmarag.cache import RedisCache
from pharmarag.ops import answer_key, query_embedding_key, rate_key


async def test_redis_cache_roundtrip_and_ttl():
    r = fakeredis.FakeAsyncRedis()
    cache = RedisCache("redis://unused", client=r)
    await cache.set("k", b"v", ttl_s=60)
    assert await cache.get("k") == b"v"
    assert 0 < await r.ttl("k") <= 60
    assert await cache.ping() is True


class _Down:
    async def get(self, *a, **k):
        raise ConnectionError("down")

    async def set(self, *a, **k):
        raise ConnectionError("down")

    async def ping(self):
        raise ConnectionError("down")


async def test_redis_cache_is_fail_soft():
    cache = RedisCache("redis://unused", client=_Down())
    assert await cache.get("k") is None
    await cache.set("k", b"v", ttl_s=1)  # must not raise
    assert await cache.ping() is False


def test_key_formats_follow_spec_section_5():
    assert query_embedding_key("m", "q").startswith("emb:q:m:")
    k1 = answer_key("cv", "resolver=on", "Q", ["2", "1"])
    k2 = answer_key("cv", "resolver=on", "Q", ["1", "2"])
    assert k1 == k2 and k1.startswith("ans:cv:resolver=on:")
    assert answer_key("cv2", "resolver=on", "Q", ["1"]) != answer_key(
        "cv", "resolver=on", "Q", ["1"]
    )
    assert rate_key("1.2.3.4", 42) == "rl:1.2.3.4:42"


@pytest.mark.parametrize("backend", ["redis", "memory"])
def test_settings_cache_backend(settings, backend):
    assert settings.model_copy(update={"cache_backend": backend}).cache_backend == backend


async def test_incr_fixed_window_both_backends():
    from pharmarag.cache import InMemoryCache

    now = [0.0]
    mem = InMemoryCache(clock=lambda: now[0])
    red = RedisCache("redis://unused", client=fakeredis.FakeAsyncRedis())
    for c in (mem, red):
        assert [await c.incr("rl:x:1", 60) for _ in range(3)] == [1, 2, 3]
    assert 0 < await red._r.ttl("rl:x:1") <= 60
    now[0] = 61.0
    assert await mem.incr("rl:x:1", 60) == 1  # window expired


async def test_redis_cache_cooldown_skips_redis_after_a_fault():
    now = [0.0]
    calls = []

    class Flaky(_Down):
        async def get(self, *a, **k):
            calls.append(1)
            raise ConnectionError("down")

    cache = RedisCache("redis://unused", client=Flaky(), clock=lambda: now[0])
    assert await cache.get("k") is None
    assert await cache.get("k") is None  # within cooldown: Redis not called again
    assert len(calls) == 1
    assert await cache.incr("k", 60) is None  # fail-open signal
    now[0] = RedisCache.COOLDOWN_S + 0.1
    await cache.get("k")
    assert len(calls) == 2
