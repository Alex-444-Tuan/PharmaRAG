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
