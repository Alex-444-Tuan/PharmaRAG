import threading

import fakeredis
import numpy as np

from pharmarag.cache import InMemoryCache, RedisCache
from pharmarag.caching.embeddings import CachedQueryEmbeddings
from pharmarag.caching.factory import build_query_embedding_cache
from pharmarag.models.embedder import FakeEmbedder
from pharmarag.ops import QUERY_EMBEDDING_TTL_S, query_embedding_key


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, int]] = []

    def embed_query(self, text):
        self.calls.append((text, threading.get_ident()))
        return super().embed_query(text)


async def test_byte_round_trip_equals_a_fresh_embed_through_real_redis_cache():
    r = fakeredis.FakeAsyncRedis()
    emb = CountingEmbedder()
    cached = CachedQueryEmbeddings(emb, RedisCache("redis://unused", client=r))
    first = await cached.embed_query("hydroxyzine dose")
    second = await cached.embed_query("hydroxyzine dose")
    fresh = FakeEmbedder().embed_query("hydroxyzine dose")
    assert first.dtype == np.float32 and second.dtype == np.float32
    assert np.array_equal(second, fresh)  # bit-identical, not merely close
    assert np.array_equal(first, fresh)
    assert len(emb.calls) == 1  # the second call was served from Redis
    key = query_embedding_key("fake-embedder", "hydroxyzine dose")
    assert len(await r.get(key)) == 384 * 4
    assert 0 < await r.ttl(key) <= QUERY_EMBEDDING_TTL_S


async def test_hit_and_miss_counting():
    cached = CachedQueryEmbeddings(CountingEmbedder(), InMemoryCache())
    await cached.embed_query("a")
    await cached.embed_query("b")
    await cached.embed_query("a")
    assert (cached.hits, cached.misses) == (1, 2)


async def test_embedding_runs_in_a_worker_thread():
    emb = CountingEmbedder()
    await CachedQueryEmbeddings(emb, InMemoryCache()).embed_query("x")
    assert emb.calls[0][1] != threading.get_ident()


async def test_returned_vector_is_writable_and_does_not_alias_the_cache():
    cached = CachedQueryEmbeddings(FakeEmbedder(), InMemoryCache())
    await cached.embed_query("q")
    v = await cached.embed_query("q")
    v[:] = 0.0  # a caller mutating its copy must not corrupt the next hit
    again = await cached.embed_query("q")
    assert np.array_equal(again, FakeEmbedder().embed_query("q"))


async def test_model_name_is_in_the_key():
    cache = InMemoryCache()
    a = CachedQueryEmbeddings(FakeEmbedder(model_name="m1"), cache)
    b = CachedQueryEmbeddings(FakeEmbedder(model_name="m2"), cache)
    await a.embed_query("q")
    await b.embed_query("q")
    assert (b.hits, b.misses) == (0, 1)


async def test_entry_of_the_wrong_size_is_a_miss_and_is_replaced():
    cache = InMemoryCache()
    key = query_embedding_key("fake-embedder", "q")
    await cache.set(key, b"\x00" * 100, 60)
    cached = CachedQueryEmbeddings(FakeEmbedder(), cache)
    v = await cached.embed_query("q")
    assert cached.misses == 1 and np.array_equal(v, FakeEmbedder().embed_query("q"))
    assert len(await cache.get(key)) == 384 * 4


async def test_redis_down_still_embeds_fail_soft():
    server = fakeredis.FakeServer()
    server.connected = False
    client = fakeredis.FakeAsyncRedis(server=server)
    cached = CachedQueryEmbeddings(FakeEmbedder(), RedisCache("redis://unused", client=client))
    v = await cached.embed_query("q")
    assert np.array_equal(v, FakeEmbedder().embed_query("q"))
    assert cached.misses == 1


def test_factory_builds_the_cache():
    assert isinstance(
        build_query_embedding_cache(FakeEmbedder(), InMemoryCache()), CachedQueryEmbeddings
    )
