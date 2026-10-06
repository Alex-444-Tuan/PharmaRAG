from pharmarag.cache import InMemoryCache


async def test_in_memory_cache_ttl():
    now = [0.0]
    cache = InMemoryCache(clock=lambda: now[0])
    await cache.set("k", b"v", ttl_s=10)
    assert await cache.get("k") == b"v"
    now[0] = 10.0
    assert await cache.get("k") is None
    assert await cache.get("missing") is None
