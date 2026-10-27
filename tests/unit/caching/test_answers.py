import json

from pharmarag.app.settings import Settings
from pharmarag.cache import InMemoryCache
from pharmarag.caching.answers import CacheAnswerCache, NullAnswerCache
from pharmarag.caching.factory import build_answer_cache
from pharmarag.ops import ANSWER_TTL_S, answer_key


async def test_round_trip_with_unicode_and_nesting():
    cache = InMemoryCache()
    answers = CacheAnswerCache(cache)
    key = answer_key("cv1", "resolver=on", "q", ["1"])
    resp = {"status": "answered", "answer": [{"text": "5 µg", "cites": ["c1"]}], "n": 3}
    await answers.put(key, resp)
    assert await answers.get(key) == resp
    assert isinstance(await cache.get(key), bytes)


async def test_ttl_is_answer_ttl_and_expires():
    now = [0.0]
    answers = CacheAnswerCache(InMemoryCache(clock=lambda: now[0]))
    await answers.put("k", {"a": 1})
    now[0] = ANSWER_TTL_S - 1
    assert await answers.get("k") == {"a": 1}
    now[0] = ANSWER_TTL_S
    assert await answers.get("k") is None


async def test_hit_and_miss_counting():
    answers = CacheAnswerCache(InMemoryCache())
    assert await answers.get("k") is None
    await answers.put("k", {"a": 1})
    await answers.get("k")
    await answers.get("k")
    assert (answers.hits, answers.misses) == (2, 1)


async def test_corrupt_or_non_object_entry_is_a_miss():
    cache = InMemoryCache()
    answers = CacheAnswerCache(cache)
    await cache.set("bad", b"{not json", 60)
    await cache.set("list", json.dumps([1]).encode(), 60)
    assert await answers.get("bad") is None
    assert await answers.get("list") is None
    assert answers.misses == 2 and answers.hits == 0


async def test_unserialisable_response_is_skipped_not_raised():
    cache = InMemoryCache()
    answers = CacheAnswerCache(cache)
    await answers.put("k", {"x": object()})
    assert await cache.get("k") is None


async def test_corpus_version_in_the_key_isolates_answers():
    answers = CacheAnswerCache(InMemoryCache())
    await answers.put(answer_key("cv1", "f", "q", ["1"]), {"v": 1})
    assert await answers.get(answer_key("cv2", "f", "q", ["1"])) is None


async def test_answer_cache_off_returns_a_null_cache():
    off = build_answer_cache(Settings(_env_file=None, answer_cache=False), InMemoryCache())
    assert isinstance(off, NullAnswerCache)
    await off.put("k", {"a": 1})
    assert await off.get("k") is None
    on = build_answer_cache(Settings(_env_file=None, answer_cache=True), InMemoryCache())
    assert isinstance(on, CacheAnswerCache)
