"""Operations over HTTP (cycle 4): rate limit, answer cache, readiness, metrics, logs, request ID.

A stub graph and fake caches, no database: these are HTTP-behaviour tests. The graph-level
counterparts (per-node log lines and histograms, the query-embedding cache) are in
tests/graph/test_ops_graph.py.
"""

import copy
import json
import logging
from contextlib import asynccontextmanager

import fakeredis
import httpx
import psycopg
import pytest

from pharmarag.app import main as main_module
from pharmarag.app.main import create_app, flags_key
from pharmarag.app.observability import EVENT_LOGGER, JsonFormatter, Metrics
from pharmarag.app.runtime import Runtime
from pharmarag.cache import InMemoryCache, RedisCache
from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Flags
from tests.graph.fakes import FakeAnswerCache, FakeRateLimiter, FakeResolver, FakeVerifier

QUESTION = "What is the dose of hydroxyzine for anxiety?"


def answered(
    corpus_version: str = "v1",
    status: str = "answered",
    *,
    degraded: bool = False,
    facts_unavailable: bool = False,
) -> dict:
    return {
        "status": status,
        "drugs": [],
        "answer": [{"text": "ok", "citations": ["c1"], "support": None}]
        if status == "answered"
        else [],
        "sources": [],
        "notice": "From FDA drug labels via DailyMed. Not medical advice.",
        "meta": {
            "corpus_version": corpus_version,
            "latency_ms": 9999,
            "retrieval_attempts": 2,
            "generation_attempts": 1,
            "removed_sentences": 3,
            "degraded": degraded,
            "facts_unavailable": facts_unavailable,
        },
    }


class StubGraph:
    def __init__(self, response: dict | None = None):
        self.response = response or answered()
        self.calls = 0

    async def ainvoke(self, state, config=None):
        self.calls += 1
        return {"response": copy.deepcopy(self.response), "intent": "label_qa", "timings": {}}


class FakePool:
    """`connection()` like psycopg_pool's: an async context manager yielding a connection."""

    def __init__(self, error: Exception | None = None):
        self.error = error

    @asynccontextmanager
    async def connection(self):
        if self.error:
            raise self.error

        class Conn:
            async def execute(self, sql):
                assert sql == "SELECT 1"

        yield Conn()


class Down:
    async def ping(self):
        raise ConnectionError("down")

    async def get(self, *a, **k):
        raise ConnectionError("down")


@pytest.fixture
def corpus_version_box(monkeypatch):
    """`/v1/ask` reads the corpus version from the database; the tests set it here."""
    box = {"v": "v1"}

    async def fake(deps):
        return box["v"]

    monkeypatch.setattr(main_module, "corpus_version", fake)
    return box


@pytest.fixture
def make_app(settings, thresholds, fake_embedder, make_fake_llm):
    def _make(
        *,
        graph=None,
        rate_limiter=None,
        answer_cache=None,
        pool=None,
        eval_mode=False,
        cache=None,
        verifier="default",
        trust_forwarded_for=False,
        client_addr=("203.0.113.7", 5000),
        answer_cache_enabled=True,
        thresholds_override=None,
        **settings_update,
    ):
        s = settings.model_copy(
            update={
                "eval_mode": eval_mode,
                "cache_backend": "memory",
                "answer_cache": answer_cache_enabled,
                **settings_update,
            }
        )
        deps = Deps(
            llm=make_fake_llm([]),
            embedder=fake_embedder,
            pool=pool or FakePool(),
            settings=s,
            thresholds=thresholds_override or thresholds,
            resolver=FakeResolver({}),
            verifier=FakeVerifier([[]]) if verifier == "default" else verifier,
            cache=cache or InMemoryCache(),
            metrics=Metrics(),
        )
        graph = graph or StubGraph()
        app = create_app(s, trust_forwarded_for=trust_forwarded_for)
        app.state.runtime = Runtime(
            deps=deps, graph=graph, rate_limiter=rate_limiter, answer_cache=answer_cache
        )
        transport = httpx.ASGITransport(app=app, client=client_addr)
        return httpx.AsyncClient(transport=transport, base_url="http://t"), app, graph

    return _make


async def ask(client, question=QUESTION, headers=None, **extra):
    return await client.post("/v1/ask", json={"question": question, **extra}, headers=headers)


# --- rate limit ---------------------------------------------------------------------------------


async def test_over_the_limit_is_429_with_retry_after_and_never_reaches_the_graph(make_app):
    limiter = FakeRateLimiter(limit=2)
    client, app, graph = make_app(rate_limiter=limiter)
    async with client:
        codes = [(await ask(client)).status_code for _ in range(3)]
        limited = await ask(client)
    assert codes == [200, 200, 429]
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "17"
    assert limited.json() == {"detail": "rate limit exceeded"}
    assert limited.headers["x-request-id"]  # a 429 is still traceable
    assert graph.calls == 2  # the two over-limit requests never ran the graph
    metrics = app.state.runtime.deps.metrics
    assert metrics.registry.get_sample_value("pharmarag_rate_limited_total") == 2.0


async def test_retry_after_is_at_least_one_second(make_app):
    class Zero:
        async def hit(self, client_id):
            from pharmarag.ops import RateDecision

            return RateDecision(allowed=False, remaining=0, retry_after_s=0)

    client, _, _ = make_app(rate_limiter=Zero())
    async with client:
        r = await ask(client)
    assert r.status_code == 429 and r.headers["retry-after"] == "1"


@pytest.mark.parametrize("path", ["/healthz", "/readyz", "/metrics", "/docs", "/openapi.json"])
async def test_ops_and_docs_paths_are_never_rate_limited(make_app, path):
    limiter = FakeRateLimiter(limit=0)  # every counted hit is denied
    client, _, _ = make_app(rate_limiter=limiter)
    async with client:
        for _ in range(3):
            r = await client.get(path)
            assert r.status_code != 429
    assert limiter.counts == {}  # the limiter was not even consulted


async def test_v1_get_endpoints_are_limited_too(make_app):
    client, _, _ = make_app(rate_limiter=FakeRateLimiter(limit=0))
    async with client:
        r = await client.get("/v1/drugs/resolve", params={"name": "x"})
    assert r.status_code == 429


async def test_each_client_address_has_its_own_budget(make_app):
    limiter = FakeRateLimiter(limit=1)
    a, _, _ = make_app(rate_limiter=limiter, client_addr=("198.51.100.1", 1))
    b, _, _ = make_app(rate_limiter=limiter, client_addr=("198.51.100.2", 1))
    async with a, b:
        assert (await ask(a)).status_code == 200
        assert (await ask(a)).status_code == 429
        assert (await ask(b)).status_code == 200
    assert set(limiter.counts) == {"198.51.100.1", "198.51.100.2"}


async def test_x_forwarded_for_is_ignored_by_default(make_app):
    limiter = FakeRateLimiter()
    client, _, _ = make_app(rate_limiter=limiter, client_addr=("10.0.0.9", 1))
    async with client:
        await ask(client, headers={"X-Forwarded-For": "1.1.1.1"})
    assert list(limiter.counts) == ["10.0.0.9"]  # a client cannot pick its own bucket


async def test_trusting_x_forwarded_for_uses_the_rightmost_entry(make_app):
    limiter = FakeRateLimiter()
    client, _, _ = make_app(
        rate_limiter=limiter, trust_forwarded_for=True, client_addr=("10.0.0.9", 1)
    )
    async with client:
        await ask(client, headers={"X-Forwarded-For": "6.6.6.6, 1.1.1.1, 192.0.2.50"})
    assert list(limiter.counts) == ["192.0.2.50"]  # the proxy's entry, not the client-written one


async def test_trust_forwarded_for_falls_back_to_the_peer_without_the_header(make_app):
    limiter = FakeRateLimiter()
    client, _, _ = make_app(
        rate_limiter=limiter, trust_forwarded_for=True, client_addr=("10.0.0.9", 1)
    )
    async with client:
        await ask(client)
    assert list(limiter.counts) == ["10.0.0.9"]


@pytest.mark.parametrize(
    ("trust", "bucket"), [(False, "10.0.0.9"), (True, "192.0.2.50")], ids=["off", "on"]
)
async def test_trust_forwarded_for_comes_from_the_settings(settings, thresholds, trust, bucket):
    """No explicit argument: `Settings.trust_forwarded_for` decides, and it is off by default."""
    assert settings.trust_forwarded_for is False
    s = settings.model_copy(update={"trust_forwarded_for": trust})
    limiter = FakeRateLimiter()
    app = create_app(s)
    app.state.runtime = Runtime(
        deps=Deps(
            llm=None, embedder=None, pool=None, settings=s, thresholds=thresholds, metrics=Metrics()
        ),
        graph=StubGraph(),
        rate_limiter=limiter,
    )
    transport = httpx.ASGITransport(app=app, client=("10.0.0.9", 1))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await client.post(
            "/v1/ask", json={"question": QUESTION}, headers={"X-Forwarded-For": "192.0.2.50"}
        )
    assert list(limiter.counts) == [bucket]


async def test_rotating_x_forwarded_for_does_not_escape_the_limit_when_trust_is_off(
    settings, thresholds
):
    """The reviewer's bypass: with the setting off, a new XFF value per request must not buy a
    new bucket. The socket peer is the key, so the third request is limited."""
    limiter = FakeRateLimiter(limit=2)
    app = create_app(settings)
    app.state.runtime = Runtime(
        deps=Deps(
            llm=None,
            embedder=None,
            pool=None,
            settings=settings,
            thresholds=thresholds,
            metrics=Metrics(),
        ),
        graph=StubGraph(),
        rate_limiter=limiter,
    )
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        codes = [
            (
                await client.post(
                    "/v1/ask",
                    json={"question": QUESTION},
                    headers={"X-Forwarded-For": f"198.51.100.{i}"},
                )
            ).status_code
            for i in range(3)
        ]
    assert codes == [200, 200, 429] and list(limiter.counts) == ["127.0.0.1"]


async def test_a_broken_limiter_fails_open(make_app, caplog):
    client, _, graph = make_app(rate_limiter=FakeRateLimiter(error=RuntimeError("boom")))
    with caplog.at_level(logging.WARNING, logger="pharmarag.app.middleware"):
        async with client:
            r = await ask(client)
    assert r.status_code == 200 and graph.calls == 1
    assert "rate limiter failed" in caplog.text


async def test_no_limiter_in_the_build_means_no_limit(make_app):
    client, _, graph = make_app(rate_limiter=None)
    async with client:
        codes = {(await ask(client)).status_code for _ in range(5)}
    assert codes == {200} and graph.calls == 5


# --- answer cache -------------------------------------------------------------------------------


async def test_a_second_identical_question_is_served_from_the_cache(make_app, corpus_version_box):
    cache = FakeAnswerCache()
    client, app, graph = make_app(answer_cache=cache)
    async with client:
        first = (await ask(client)).json()
        second = (await ask(client)).json()
    assert graph.calls == 1
    assert first["meta"]["cache_hit"] is False and second["meta"]["cache_hit"] is True
    assert second["answer"] == first["answer"] and second["sources"] == first["sources"]
    # latency_ms is this request's own time, not the stored 9999 from the first run.
    assert second["meta"]["latency_ms"] < 9999
    assert len(cache.puts) == 1 and len(cache.data) == 1
    # The stored copy does not claim to be a hit.
    assert "cache_hit" not in next(iter(cache.data.values()))["meta"]
    sample = app.state.runtime.deps.metrics.registry.get_sample_value
    assert sample("pharmarag_cache_requests_total", {"cache": "answer", "result": "hit"}) == 1.0
    assert sample("pharmarag_cache_requests_total", {"cache": "answer", "result": "miss"}) == 1.0


async def test_a_cache_hit_cannot_be_mutated_through_the_stored_object(
    make_app, corpus_version_box
):
    cache = FakeAnswerCache()
    client, _, _ = make_app(answer_cache=cache)
    async with client:
        await ask(client)
        stored = copy.deepcopy(next(iter(cache.data.values())))
        await ask(client)  # a hit gets cache_hit=True and a new latency
    assert next(iter(cache.data.values())) == stored


@pytest.mark.parametrize(
    "status", ["declined", "insufficient_evidence", "needs_clarification", "not_covered"]
)
async def test_only_answered_responses_are_cached(make_app, corpus_version_box, status):
    cache = FakeAnswerCache()
    client, _, graph = make_app(answer_cache=cache, graph=StubGraph(answered(status=status)))
    async with client:
        await ask(client)
        again = (await ask(client)).json()
    assert cache.puts == [] and graph.calls == 2
    assert again["meta"]["cache_hit"] is False


async def test_a_new_corpus_version_invalidates_old_answers(make_app, corpus_version_box):
    cache = FakeAnswerCache()
    client, _, graph = make_app(answer_cache=cache)
    async with client:
        await ask(client)
        corpus_version_box["v"] = "v2"  # a weekly ingest changed the labels
        graph.response = answered(corpus_version="v2")
        after = (await ask(client)).json()
        corpus_version_box["v"] = "v1"
        graph.response = answered(corpus_version="v1")
        old = (await ask(client)).json()
    assert graph.calls == 2  # v2 was a miss; going back to v1 hits the v1 entry
    assert after["meta"]["cache_hit"] is False and old["meta"]["cache_hit"] is True
    assert len(cache.data) == 2 and {k.split(":")[1] for k in cache.data} == {"v1", "v2"}


async def test_an_answer_for_a_newer_corpus_than_the_key_is_not_stored(
    make_app, corpus_version_box
):
    """The corpus refreshed between the key lookup and the graph's own read."""
    cache = FakeAnswerCache()
    client, _, _ = make_app(answer_cache=cache, graph=StubGraph(answered(corpus_version="v2")))
    async with client:
        await ask(client)  # key built with v1, response says v2
    assert cache.puts == []


async def test_pinned_rxcuis_are_part_of_the_key_in_any_order(make_app, corpus_version_box):
    cache = FakeAnswerCache()
    client, _, graph = make_app(answer_cache=cache)
    async with client:
        await ask(client, pinned_rxcuis=["1", "2"])
        hit = (await ask(client, pinned_rxcuis=["2", "1"])).json()
        miss = (await ask(client, pinned_rxcuis=["1"])).json()
    assert hit["meta"]["cache_hit"] is True and miss["meta"]["cache_hit"] is False
    assert graph.calls == 2


async def test_a_different_question_is_a_miss(make_app, corpus_version_box):
    client, _, graph = make_app(answer_cache=FakeAnswerCache())
    async with client:
        await ask(client)
        other = (await ask(client, question="What is the dose of hydralazine?")).json()
    assert other["meta"]["cache_hit"] is False and graph.calls == 2


async def test_eval_flags_are_part_of_the_key(make_app, corpus_version_box):
    """A verifier-off answer must never be served to a verifier-on request."""
    cache = FakeAnswerCache()
    client, _, graph = make_app(answer_cache=cache, eval_mode=True)
    async with client:
        await ask(client)
        off = (await ask(client, headers={"X-Eval-Flags": "verifier=off"})).json()
        on_again = (await ask(client)).json()
    assert off["meta"]["cache_hit"] is False and on_again["meta"]["cache_hit"] is True
    assert graph.calls == 2


async def test_flags_key_is_canonical():
    assert (
        flags_key(Flags(resolver=True, verifier=False), "stream")
        == "resolver=on,verifier=off,verify=stream"
    )


async def test_answer_cache_setting_false_bypasses_the_cache(make_app, corpus_version_box):
    cache = FakeAnswerCache()
    client, _, graph = make_app(answer_cache=cache, answer_cache_enabled=False)
    async with client:
        await ask(client)
        again = (await ask(client)).json()
    assert graph.calls == 2 and cache.gets == [] and cache.puts == []
    assert again["meta"]["cache_hit"] is False


async def test_debug_requests_bypass_the_cache(make_app, corpus_version_box):
    cache = FakeAnswerCache()
    client, _, graph = make_app(answer_cache=cache)
    async with client:
        await ask(client, debug=True)
        await ask(client, debug=True)
    assert graph.calls == 2 and cache.gets == [] and cache.puts == []


async def test_sse_bypasses_the_answer_cache(make_app, corpus_version_box, monkeypatch):
    async def fake_stream(graph, deps, **kw):
        yield {"event": "sentence", "data": {"index": 0, "text": "ok"}}
        yield {"event": "done", "data": answered()}

    monkeypatch.setattr(main_module, "run_ask_stream", fake_stream)
    cache = FakeAnswerCache()
    cache.data["anything"] = answered()
    client, app, _ = make_app(answer_cache=cache)
    async with client:
        r = await ask(client, stream=True)
    assert r.status_code == 200 and "event: done" in r.text
    assert cache.gets == [] and cache.puts == []
    done = json.loads(r.text.split("event: done\ndata: ")[1].split("\n\n")[0])
    assert done["meta"]["cache_hit"] is False
    ask_total = app.state.runtime.deps.metrics.registry.get_sample_value(
        "pharmarag_ask_total", {"status": "answered"}
    )
    assert ask_total == 1.0  # the SSE request is metered when `done` goes out


# --- /readyz ------------------------------------------------------------------------------------


async def test_readyz_is_200_when_everything_is_up(make_app):
    client, _, _ = make_app()
    async with client:
        r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "checks": {"database": "ok", "models": "ok"}}


async def test_readyz_checks_redis_only_when_it_is_the_backend(make_app):
    redis = RedisCache("redis://unused", client=fakeredis.FakeAsyncRedis())
    client, _, _ = make_app(cache=redis, cache_backend="redis")
    async with client:
        r = await client.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"]["redis"] == "ok"


async def test_readyz_503_names_a_database_that_is_down(make_app):
    client, _, _ = make_app(pool=FakePool(psycopg.OperationalError("refused")))
    async with client:
        r = await client.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable" and body["failing"] == ["database"]
    assert body["checks"]["database"] == "unreachable" and body["checks"]["models"] == "ok"


async def test_readyz_503_names_redis_when_it_is_down(make_app):
    client, _, _ = make_app(
        cache=RedisCache("redis://unused", client=Down()), cache_backend="redis"
    )
    async with client:
        r = await client.get("/readyz")
    assert r.status_code == 503 and r.json()["failing"] == ["redis"]


async def test_readyz_503_names_missing_models_outside_eval_mode(make_app):
    client, _, _ = make_app(verifier=None, eval_mode=False)
    async with client:
        r = await client.get("/readyz")
    assert r.status_code == 503 and r.json()["failing"] == ["models"]


async def test_readyz_tolerates_a_missing_verifier_in_eval_mode(make_app):
    client, _, _ = make_app(verifier=None, eval_mode=True)
    async with client:
        r = await client.get("/readyz")
    assert r.status_code == 200


async def test_readyz_lists_every_failing_dependency(make_app):
    client, _, _ = make_app(
        pool=FakePool(psycopg.OperationalError("refused")),
        cache=RedisCache("redis://unused", client=Down()),
        cache_backend="redis",
        verifier=None,
    )
    async with client:
        r = await client.get("/readyz")
    assert r.status_code == 503 and r.json()["failing"] == ["database", "models", "redis"]


async def test_readyz_does_not_hang_on_a_stuck_database(make_app):
    import asyncio

    class Stuck(FakePool):
        @asynccontextmanager
        async def connection(self):
            await asyncio.sleep(5)
            yield None

    client, _, _ = make_app(pool=Stuck(), db_timeout_s=0.05)
    async with client:
        r = await client.get("/readyz")
    assert r.status_code == 503 and r.json()["failing"] == ["database"]


async def test_healthz_does_not_look_at_dependencies(make_app):
    client, _, _ = make_app(pool=FakePool(psycopg.OperationalError("down")))
    async with client:
        r = await client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


async def test_before_the_lifespan_has_run_readyz_and_ask_are_503(settings):
    app = create_app(settings)  # no runtime on app.state
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/readyz")).status_code == 503
        assert (await c.post("/v1/ask", json={"question": QUESTION})).status_code == 503
        assert (await c.get("/healthz")).status_code == 200


# --- /metrics -----------------------------------------------------------------------------------


async def test_metrics_exposition_has_the_expected_series(make_app, corpus_version_box):
    client, _, _ = make_app(rate_limiter=FakeRateLimiter(limit=1), answer_cache=FakeAnswerCache())
    async with client:
        await ask(client)  # 200, answered, answer-cache miss
        await ask(client)  # 429
        r = await client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    for series in (
        'pharmarag_http_requests_total{code="200",path="/v1/ask"} 1.0',
        'pharmarag_http_requests_total{code="429",path="unrouted"} 1.0',
        'pharmarag_ask_total{status="answered"} 1.0',
        'pharmarag_cache_requests_total{cache="answer",result="miss"} 1.0',
        'pharmarag_cache_requests_total{cache="answer",result="hit"} 0.0',
        'pharmarag_cache_requests_total{cache="query_embedding",result="hit"} 0.0',
        "pharmarag_rate_limited_total 1.0",
        "pharmarag_rxnav_errors_total 0.0",
    ):
        assert series in text, series
    assert "# TYPE pharmarag_node_duration_seconds histogram" in text


async def test_metrics_path_label_is_the_route_template_not_the_raw_path(make_app):
    client, _, _ = make_app()
    async with client:
        await client.get("/v1/labels/not-a-uuid")  # 422 from path validation
        await client.get("/no/such/route")
        text = (await client.get("/metrics")).text
    assert 'path="/v1/labels/{setid}"' in text
    assert 'path="unrouted"' in text and "/no/such/route" not in text


# --- logs ---------------------------------------------------------------------------------------


def request_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == EVENT_LOGGER and r.fields["event"] == "request"]


async def test_the_request_line_is_json_with_the_spec_fields(make_app, corpus_version_box, caplog):
    client, _, _ = make_app(answer_cache=FakeAnswerCache())
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER):
        async with client:
            r = await ask(client)
    (record,) = request_records(caplog)
    line = json.loads(JsonFormatter().format(record))
    assert line["event"] == "request" and line["request_id"] == r.headers["x-request-id"]
    assert line["path"] == "/v1/ask" and line["method"] == "POST" and line["http_status"] == 200
    assert line["status"] == "answered"
    assert line["retrieval_attempts"] == 2 and line["generation_attempts"] == 1
    assert line["removed_sentences"] == 3 and line["cache_hit"] is False
    assert line["duration_ms"] >= 0


async def test_a_cache_hit_is_logged_as_one(make_app, corpus_version_box, caplog):
    client, _, _ = make_app(answer_cache=FakeAnswerCache())
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER):
        async with client:
            await ask(client)
            await ask(client)
    first, second = (r.fields["cache_hit"] for r in request_records(caplog))
    assert (first, second) == (False, True)


async def test_the_raw_question_never_reaches_the_logs_by_default(
    make_app, corpus_version_box, caplog
):
    secret = "my private question about zolpidem"
    client, _, _ = make_app(answer_cache=FakeAnswerCache())
    with caplog.at_level(logging.DEBUG):
        async with client:
            await ask(client, question=secret)
            await ask(client, question=secret, stream=False, debug=True)
    rendered = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert "zolpidem" not in rendered and secret not in rendered
    (first, _) = request_records(caplog)
    assert len(first.fields["question_hash"]) == 16 and "question" not in first.fields


async def test_log_questions_true_adds_the_question(make_app, corpus_version_box, caplog):
    client, _, _ = make_app(log_questions=True)
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER):
        async with client:
            await ask(client)
    (record,) = request_records(caplog)
    assert record.fields["question"] == QUESTION and record.fields["question_hash"]


async def test_the_same_question_hashes_the_same(make_app, caplog):
    client, _, _ = make_app()
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER):
        async with client:
            await ask(client)
            await ask(client)
            await ask(client, question="something else")
    a, b, c = (r.fields["question_hash"] for r in request_records(caplog))
    assert a == b != c


async def test_non_ask_requests_are_logged_without_ask_fields(make_app, caplog):
    client, _, _ = make_app()
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER):
        async with client:
            await client.get("/healthz")
    (record,) = request_records(caplog)
    assert record.fields["path"] == "/healthz" and "cache_hit" not in record.fields


def test_json_formatter_renders_plain_records_as_json_with_exceptions():
    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "x.y", logging.ERROR, __file__, 1, "went %s", ("wrong",), sys.exc_info()
        )
    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "went wrong" and line["level"] == "ERROR" and line["logger"] == "x.y"
    assert "ValueError: bad" in line["exc"] and line["ts"].endswith("+00:00")


# --- X-Request-ID -------------------------------------------------------------------------------


async def test_a_client_request_id_is_echoed_and_reaches_the_graph(make_app):
    client, _, graph = make_app()
    states = []
    original = graph.ainvoke

    async def spy(state, config=None):
        states.append(state)
        return await original(state, config)

    graph.ainvoke = spy
    async with client:
        r = await ask(client, headers={"X-Request-ID": "trace-123_a.b"})
    assert r.headers["x-request-id"] == "trace-123_a.b"
    assert states[0]["request_id"] == "trace-123_a.b"
    assert len(r.headers.get_list("x-request-id")) == 1  # not added twice


@pytest.mark.parametrize("bad", ["has space", "x" * 65, "new\tline", "semi;colon"])
async def test_an_unsafe_request_id_is_replaced(make_app, bad):
    client, _, _ = make_app()
    async with client:
        r = await ask(client, headers={"X-Request-ID": bad})
    assert r.headers["x-request-id"] != bad and len(r.headers["x-request-id"]) == 32


async def test_every_response_has_a_distinct_generated_request_id(make_app):
    client, _, _ = make_app()
    async with client:
        ids = {(await client.get("/healthz")).headers["x-request-id"] for _ in range(3)}
    assert len(ids) == 3


async def test_the_request_id_is_on_error_responses_too(make_app):
    client, _, _ = make_app()
    async with client:
        r = await client.post("/v1/ask", json={"question": ""})  # 422
    assert r.status_code == 422 and r.headers["x-request-id"]


# --- API description ----------------------------------------------------------------------------


async def test_the_swagger_description_carries_the_nlm_statement(make_app):
    client, _, _ = make_app()
    async with client:
        spec = (await client.get("/openapi.json")).json()
    assert main_module.NLM_ATTRIBUTION in spec["info"]["description"]
    assert "National Library of Medicine" in spec["info"]["description"]


# --- degraded answers are not cached (RxNav outage) ---------------------------------------------


@pytest.mark.parametrize("flag", ["degraded", "facts_unavailable"])
async def test_an_answer_written_during_an_rxnav_outage_is_not_cached_and_recovery_is_a_miss(
    make_app, corpus_version_box, flag
):
    cache = FakeAnswerCache()
    graph = StubGraph(answered(**{flag: True}))
    client, _, _ = make_app(answer_cache=cache, graph=graph)
    async with client:
        outage = (await ask(client)).json()
        # RxNav is back: the same question must be computed again, not served from the outage.
        graph.response = answered()
        recovered = (await ask(client)).json()
        after = (await ask(client)).json()
    assert outage["meta"][flag] is True and outage["status"] == "answered"
    assert len(cache.puts) == 1  # only the healthy answer was stored
    assert recovered["meta"]["cache_hit"] is False and recovered["meta"][flag] is False
    assert after["meta"]["cache_hit"] is True and after["meta"][flag] is False
    assert graph.calls == 2


async def test_a_degraded_answer_is_still_returned_to_the_client(make_app, corpus_version_box):
    client, _, _ = make_app(
        answer_cache=FakeAnswerCache(),
        graph=StubGraph(answered(degraded=True, facts_unavailable=True)),
    )
    async with client:
        body = (await ask(client)).json()
    assert body["status"] == "answered" and body["answer"]
    assert body["meta"]["degraded"] is True and body["meta"]["facts_unavailable"] is True


# --- the answer-cache key follows the configuration ---------------------------------------------


async def test_a_retuned_threshold_or_a_new_model_never_serves_an_old_answer(
    make_app, thresholds, corpus_version_box
):
    cache = FakeAnswerCache()
    retuned = thresholds.model_copy(
        update={
            "grounding": thresholds.grounding.model_copy(
                update={"entail_min": thresholds.grounding.entail_min + 0.05}
            )
        }
    )

    async def second(**kw):
        client, _, graph = make_app(answer_cache=cache, **kw)
        async with client:
            body = (await ask(client)).json()
        return body["meta"]["cache_hit"], graph.calls

    assert await second() == (False, 1)  # first: computed and stored
    assert await second() == (True, 0)  # same configuration: served
    assert await second(thresholds_override=retuned) == (False, 1)  # new thresholds: recomputed
    assert await second(llm_model="another-model-id") == (False, 1)  # new LLM: recomputed
    assert await second(nli_model_revision="0" * 40) == (False, 1)  # new weights: recomputed
    assert len(cache.data) == 4
    digests = {k.split("cfg=")[1] for k in cache.data}
    assert all(",cfg=" in k for k in cache.data) and len(digests) == 4


def test_the_config_digest_is_short_and_stable(settings, thresholds):
    deps = Deps(llm=None, embedder=None, pool=None, settings=settings, thresholds=thresholds)
    digest = main_module.config_digest(deps)
    assert len(digest) == 10 and digest == main_module.config_digest(deps)


# --- 500 responses ------------------------------------------------------------------------------


def with_boom(app):
    @app.get("/boom")
    async def boom():
        raise RuntimeError("bug")


async def test_an_unhandled_error_is_a_500_that_still_carries_the_request_id(make_app, caplog):
    client, app, _ = make_app()
    with_boom(app)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        with caplog.at_level(logging.INFO, logger=EVENT_LOGGER):
            generated = await c.get("/boom")
            echoed = await c.get("/boom", headers={"X-Request-ID": "trace-500"})
    assert generated.status_code == 500 and generated.json() == {"detail": "internal error"}
    assert len(generated.headers["x-request-id"]) == 32
    assert echoed.headers["x-request-id"] == "trace-500"
    # The request line records the 500 under the same ID.
    lines = [r.fields for r in request_records(caplog) if r.fields["path"] == "/boom"]
    assert [ln["http_status"] for ln in lines] == [500, 500]
    assert lines[1]["request_id"] == "trace-500"


async def test_an_unhandled_error_still_reaches_the_server_log(make_app):
    """Starlette re-raises after the 500 is sent, so uvicorn logs the traceback (as JSON)."""
    client, app, _ = make_app()
    with_boom(app)
    async with client:
        with pytest.raises(RuntimeError, match="bug"):
            await client.get("/boom")


# --- end-to-end latency histogram ---------------------------------------------------------------


async def test_request_latency_is_a_histogram_by_route(make_app, corpus_version_box):
    client, app, _ = make_app()
    async with client:
        await ask(client)
        text = (await client.get("/metrics")).text
    sample = app.state.runtime.deps.metrics.registry.get_sample_value
    assert sample("pharmarag_request_duration_seconds_count", {"path": "/v1/ask"}) == 1.0
    assert "# TYPE pharmarag_request_duration_seconds histogram" in text
    # The NFR-1 targets (4 s, 8 s) are bucket edges, so p50/p95 can be read off a dashboard.
    for edge in ("4.0", "8.0"):
        assert f'pharmarag_request_duration_seconds_bucket{{le="{edge}",path="/v1/ask"}}' in text


# --- request ID format ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["abc\n", "a b", "", "x" * 65, "ünï", "a;b"])
def test_request_id_must_match_in_full(bad):
    from pharmarag.app.observability import valid_request_id

    assert valid_request_id(bad) is None


def test_request_id_accepts_the_documented_alphabet():
    from pharmarag.app.observability import valid_request_id

    assert valid_request_id("Trace-1_2.3") == "Trace-1_2.3" and valid_request_id("x" * 64)
