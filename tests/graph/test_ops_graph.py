"""Cycle 4 observability and caching inside the real graph (needs PostgreSQL)."""

import json
import logging

import pytest
import respx
from httpx import Response

from pharmarag.app.observability import (
    EVENT_LOGGER,
    JsonFormatter,
    Metrics,
    MissProbeEmbedder,
    attach_rxnav_error_counter,
    detach_rxnav_error_counter,
)
from pharmarag.cache import InMemoryCache
from pharmarag.graph.run import run_ask
from pharmarag.resolver.facts import RxNavFactsSource
from pharmarag.resolver.rxnav_client import RxNavClient, RxNavUnavailable
from tests.graph.fakes import (
    DOSE_TEXT,
    FakeAnswerCache,
    FakeQueryEmbeddingCache,
    FakeResolver,
    FakeRxFacts,
    analysis,
    resolved,
)

pytestmark = pytest.mark.pg

QUESTION = "What is the hydroxyzine dosage for anxiety in adults?"
WRITER = f"{DOSE_TEXT} [c1]"


def node_records(caplog) -> list[dict]:
    return [
        json.loads(JsonFormatter().format(r))
        for r in caplog.records
        if r.name == EVENT_LOGGER and r.fields["event"] == "node"
    ]


async def test_every_node_logs_one_json_line_with_request_id_and_duration(
    make_fake_llm, make_deps, make_graph, corpus, caplog
):
    deps = make_deps(make_fake_llm([analysis(), WRITER]))
    deps.metrics = Metrics()
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER):
        await run_ask(make_graph(deps), deps, question=QUESTION, request_id="req-1")
    lines = node_records(caplog)
    assert [ln["node"] for ln in lines if ln["node"] in ("analyze", "answer", "finalize")] == [
        "analyze",
        "answer",
        "finalize",
    ]
    assert {ln["node"] for ln in lines} == {
        "analyze",
        "resolve",
        "retrieve_dense",
        "retrieve_lexical",
        "fetch_rx_facts",
        "fuse_grade",
        "answer",
        "finalize",
    }
    for ln in lines:
        assert ln["request_id"] == "req-1" and ln["status"] == "ok"
        assert ln["duration_ms"] >= 0
        assert ln["retrieval_attempts"] == 1 and ln["generation_attempts"] == 1
    # `removed_sentences` is on the node that decides it.
    assert next(ln for ln in lines if ln["node"] == "finalize")["removed_sentences"] == 0


async def test_node_lines_never_contain_the_question(
    make_fake_llm, make_deps, make_graph, corpus, caplog
):
    deps = make_deps(make_fake_llm([analysis(), WRITER]))
    with caplog.at_level(logging.DEBUG):
        await run_ask(make_graph(deps), deps, question=QUESTION)
    rendered = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert QUESTION not in rendered and "hydroxyzine dosage for anxiety" not in rendered


async def test_a_failing_node_logs_status_error_and_is_still_timed(
    make_fake_llm, make_deps, make_graph, corpus, caplog
):
    deps = make_deps(make_fake_llm([]))  # analyze runs out of scripted replies and raises
    deps.metrics = Metrics()
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER), pytest.raises(AssertionError):
        await run_ask(make_graph(deps), deps, question=QUESTION)
    (line,) = node_records(caplog)
    assert line["node"] == "analyze" and line["status"] == "error"
    count = deps.metrics.registry.get_sample_value(
        "pharmarag_node_duration_seconds_count", {"node": "analyze"}
    )
    assert count == 1.0


async def test_the_retry_attempt_numbers_are_logged(
    make_fake_llm, make_deps, make_graph, pg_conn, caplog
):
    """Empty corpus: one rewrite, so the second pass of the branches reports attempt 2."""
    deps = make_deps(make_fake_llm([analysis(), json.dumps({"search_query": "dose of drug"})]))
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER):
        await run_ask(make_graph(deps), deps, question=QUESTION)
    dense = [ln for ln in node_records(caplog) if ln["node"] == "retrieve_dense"]
    assert [ln["retrieval_attempts"] for ln in dense] == [1, 2]


async def test_node_durations_are_exposed_as_a_histogram_per_node(
    make_fake_llm, make_deps, make_graph, corpus
):
    deps = make_deps(make_fake_llm([analysis(), WRITER]))
    deps.metrics = Metrics()
    await run_ask(make_graph(deps), deps, question=QUESTION)
    sample = deps.metrics.registry.get_sample_value
    for node in ("analyze", "retrieve_dense", "fuse_grade", "answer", "finalize"):
        assert sample("pharmarag_node_duration_seconds_count", {"node": node}) == 1.0, node
    assert sample("pharmarag_node_duration_seconds_count", {"node": "rewrite_query"}) is None


# --- query-embedding cache ----------------------------------------------------------------------


async def test_dense_retrieval_embeds_through_the_query_cache_and_counts_hit_and_miss(
    make_fake_llm, make_deps, make_graph, corpus, fake_embedder
):
    class Spy(type(fake_embedder)):
        def __init__(self):
            super().__init__()
            self.query_calls = 0

        def embed_query(self, text):
            self.query_calls += 1
            return super().embed_query(text)

    spy = Spy()
    cache = FakeQueryEmbeddingCache(MissProbeEmbedder(spy))
    replies = [analysis(), WRITER, analysis(), WRITER]
    deps = make_deps(make_fake_llm(replies), query_embeddings=cache)
    deps.metrics = Metrics()
    graph = make_graph(deps)
    await run_ask(graph, deps, question=QUESTION)
    await run_ask(graph, deps, question=QUESTION)  # same search query -> cached vector
    sample = deps.metrics.registry.get_sample_value
    miss = sample("pharmarag_cache_requests_total", {"cache": "query_embedding", "result": "miss"})
    hit = sample("pharmarag_cache_requests_total", {"cache": "query_embedding", "result": "hit"})
    assert (miss, hit) == (1.0, 1.0)
    # The graph's own embedder was not used for the query: the cache was.
    assert spy.query_calls == 1


async def test_without_a_query_cache_the_node_embeds_directly(
    make_fake_llm, make_deps, make_graph, corpus
):
    deps = make_deps(make_fake_llm([analysis(), WRITER]))
    deps.metrics = Metrics()
    out = await run_ask(make_graph(deps), deps, question=QUESTION)
    assert out["status"] == "answered"
    sample = deps.metrics.registry.get_sample_value
    assert (
        sample("pharmarag_cache_requests_total", {"cache": "query_embedding", "result": "miss"})
        == 0.0
    )


async def test_concurrent_requests_do_not_share_a_cache_probe(fake_embedder):
    """Two lookups in flight at once: the one that hits must not be marked a miss."""
    import asyncio

    from pharmarag.app.observability import probe_cache

    wrapped = MissProbeEmbedder(fake_embedder)
    cache = FakeQueryEmbeddingCache(wrapped)
    await cache.embed_query("warm")  # now cached

    async def lookup(text):
        with probe_cache() as probe:
            await asyncio.sleep(0)
            await cache.embed_query(text)
            await asyncio.sleep(0)
        return probe.missed

    results = await asyncio.gather(lookup("warm"), lookup("cold"), lookup("warm"))
    assert results == [False, True, False]


# --- RxNav error metric -------------------------------------------------------------------------


@respx.mock
async def test_a_real_rxnav_failure_increments_the_error_counter():
    """Guards the log-message contract with the RxNav client: if its wording changes, the metric
    would silently stay at zero."""
    respx.get(url__regex=r"https://rx\.test/REST/.*").mock(return_value=Response(503))
    metrics = Metrics()
    handler = attach_rxnav_error_counter(metrics)
    client = RxNavClient("https://rx.test/REST", attempts=1, backoff_s=0.0, max_rps=1000)
    try:
        for _ in range(2):
            with pytest.raises(RxNavUnavailable):
                await client.find_rxcuis("hydroxyzine")
    finally:
        detach_rxnav_error_counter(handler)
        await client.aclose()
    assert metrics.registry.get_sample_value("pharmarag_rxnav_errors_total") == 2.0


@respx.mock
async def test_a_404_from_rxnav_is_not_an_error():
    respx.get(url__regex=r"https://rx\.test/REST/.*").mock(return_value=Response(404))
    metrics = Metrics()
    handler = attach_rxnav_error_counter(metrics)
    client = RxNavClient("https://rx.test/REST", attempts=1, backoff_s=0.0, max_rps=1000)
    try:
        await client.find_rxcuis("hydroxyzine")
    finally:
        detach_rxnav_error_counter(handler)
        await client.aclose()
    assert metrics.registry.get_sample_value("pharmarag_rxnav_errors_total") == 0.0


async def test_a_detached_counter_stops_counting():
    metrics = Metrics()
    handler = attach_rxnav_error_counter(metrics)
    detach_rxnav_error_counter(handler)
    logging.getLogger("pharmarag.resolver.rxnav_client").warning("rxnav unavailable (/x): E: e")
    assert metrics.registry.get_sample_value("pharmarag_rxnav_errors_total") == 0.0


# --- answer cache with the real corpus_version --------------------------------------------------


async def test_the_answer_cache_key_follows_the_database_corpus_version(
    make_fake_llm, make_deps, make_graph, corpus, pg_conn, settings
):
    """Through the real `corpus_version` read: a refresh in corpus_meta changes the key."""
    import httpx

    from pharmarag.app.main import create_app
    from pharmarag.app.runtime import Runtime
    from tests.graph.fakes import FakeAnswerCache
    from tests.graph.seed import set_corpus_version

    deps = make_deps(make_fake_llm([analysis(), WRITER, analysis(), WRITER]))
    deps.settings = settings.model_copy(update={"cache_backend": "memory", "answer_cache": True})
    cache = FakeAnswerCache()
    app = create_app(deps.settings)
    app.state.runtime = Runtime(deps=deps, graph=make_graph(deps), answer_cache=cache)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        first = (await client.post("/v1/ask", json={"question": QUESTION})).json()
        again = (await client.post("/v1/ask", json={"question": QUESTION})).json()
        await set_corpus_version(pg_conn, "corpus-NEW")
        await pg_conn.commit()
        after = (await client.post("/v1/ask", json={"question": QUESTION})).json()

    assert (
        first["meta"]["cache_hit"] is False and first["meta"]["corpus_version"] == "corpus-abc123"
    )
    assert again["meta"]["cache_hit"] is True
    assert after["meta"]["cache_hit"] is False and after["meta"]["corpus_version"] == "corpus-NEW"
    assert sorted(k.split(":")[1] for k in cache.data) == ["corpus-NEW", "corpus-abc123"]


# --- RxNav outage flags (meta.degraded, meta.facts_unavailable) ---------------------------------

HYDROXYZINE = resolved("hydroxyzin", "1", "hydroxyzine", match="approximate")
ASK = analysis(mentions=["hydroxyzin"])
RX_URL = r"https://rx\.test/REST/.*"


def rx_client(**kw) -> RxNavClient:
    return RxNavClient("https://rx.test/REST", attempts=1, backoff_s=0.0, max_rps=1000, **kw)


async def ask_once(make_deps, make_graph, llm, **deps_kw):
    deps = make_deps(llm, **deps_kw)
    deps.metrics = Metrics()
    out = await run_ask(make_graph(deps), deps, question=QUESTION)
    return out, deps


async def test_a_healthy_request_reports_no_outage(make_fake_llm, make_deps, make_graph, corpus):
    out, _ = await ask_once(
        make_deps,
        make_graph,
        make_fake_llm([ASK, WRITER]),
        resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}),
        rx_facts=FakeRxFacts({"1": ["RxNorm brand names for hydroxyzine: Atarax."]}),
    )
    assert out["status"] == "answered"
    assert out["meta"]["degraded"] is False and out["meta"]["facts_unavailable"] is False


async def test_a_resolver_that_fell_back_to_the_local_alias_table_marks_degraded(
    make_fake_llm, make_deps, make_graph, corpus
):
    degraded = HYDROXYZINE.model_copy(update={"degraded": True})
    out, _ = await ask_once(
        make_deps,
        make_graph,
        make_fake_llm([ASK, WRITER]),
        resolver=FakeResolver({"hydroxyzin": degraded}),
    )
    assert out["status"] == "answered" and out["meta"]["degraded"] is True


async def test_a_facts_source_that_raises_marks_facts_unavailable(
    make_fake_llm, make_deps, make_graph, corpus
):
    out, _ = await ask_once(
        make_deps,
        make_graph,
        make_fake_llm([ASK, WRITER]),
        resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}),
        rx_facts=FakeRxFacts(error=RuntimeError("down")),
    )
    assert out["status"] == "answered" and out["meta"]["facts_unavailable"] is True


@respx.mock
async def test_rxnav_down_under_the_real_facts_source_marks_the_answer_and_counts_errors(
    make_fake_llm, make_deps, make_graph, corpus
):
    """The real client and facts source against a 503. The source swallows the outage and returns
    no lines, so the only signal is its log line: this is the guard for that coupling."""
    respx.get(url__regex=RX_URL).mock(return_value=Response(503))
    client = rx_client()
    facts = RxNavFactsSource(client, InMemoryCache(), 5)
    metrics = Metrics()
    handler = attach_rxnav_error_counter(metrics)
    try:
        deps = make_deps(
            make_fake_llm([ASK, WRITER]),
            resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}),
            rx_facts=facts,
        )
        out = await run_ask(make_graph(deps), deps, question=QUESTION)
    finally:
        detach_rxnav_error_counter(handler)
        await client.aclose()
    assert out["status"] == "answered"  # NFR-4: the answer is still given, from label text
    assert out["meta"]["facts_unavailable"] is True
    assert metrics.registry.get_sample_value("pharmarag_rxnav_errors_total") >= 1.0


@respx.mock
async def test_a_tripped_circuit_breaker_still_marks_facts_unavailable(
    make_fake_llm, make_deps, make_graph, corpus
):
    """Breaker open means the client raises without logging "rxnav unavailable"; the facts
    source's own line covers that case."""
    respx.get(url__regex=RX_URL).mock(return_value=Response(503))
    client = rx_client(breaker_threshold=1)
    with pytest.raises(RxNavUnavailable):
        await client.find_rxcuis("x")  # one failed call trips the breaker
    assert client.breaker_open
    handler = attach_rxnav_error_counter(Metrics())
    try:
        deps = make_deps(
            make_fake_llm([ASK, WRITER]),
            resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}),
            rx_facts=RxNavFactsSource(client, InMemoryCache(), 5),
        )
        out = await run_ask(make_graph(deps), deps, question=QUESTION)
    finally:
        detach_rxnav_error_counter(handler)
        await client.aclose()
    assert out["meta"]["facts_unavailable"] is True


async def test_concurrent_requests_do_not_share_an_outage_flag(
    make_fake_llm, make_deps, make_graph, corpus
):
    """One request sees an RxNav outage while another, in flight at the same time, does not."""
    import asyncio

    class SlowFlaky:
        async def facts(self, rxcuis):
            await asyncio.sleep(0.05)
            raise RuntimeError("down")

    class SlowOk:
        async def facts(self, rxcuis):
            await asyncio.sleep(0.05)
            return []

    async def run(rx):
        deps = make_deps(
            make_fake_llm([ASK, WRITER]),
            resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}),
            rx_facts=rx,
        )
        return await run_ask(make_graph(deps), deps, question=QUESTION)

    bad, good = await asyncio.gather(run(SlowFlaky()), run(SlowOk()))
    assert (bad["meta"]["facts_unavailable"], good["meta"]["facts_unavailable"]) == (True, False)


async def test_an_outage_answer_is_not_cached_and_the_recovered_request_is_a_miss(
    make_fake_llm, make_deps, make_graph, corpus, settings
):
    """The reviewer's scenario end to end: real graph, real finalize, real answer-cache path."""
    import httpx

    from pharmarag.app.main import create_app
    from pharmarag.app.runtime import Runtime

    facts = FakeRxFacts(error=RuntimeError("RxNav down"))
    llm = make_fake_llm([ASK, WRITER] * 3)
    deps = make_deps(llm, resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}), rx_facts=facts)
    deps.settings = settings.model_copy(update={"cache_backend": "memory", "answer_cache": True})
    cache = FakeAnswerCache()
    app = create_app(deps.settings)
    app.state.runtime = Runtime(deps=deps, graph=make_graph(deps), answer_cache=cache)
    transport = httpx.ASGITransport(app=app)

    async def post(client):
        return (await client.post("/v1/ask", json={"question": QUESTION})).json()

    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        outage = await post(client)
        assert cache.puts == []  # not stored while RxNav is down
        facts.error = None  # RxNav recovers
        recovered = await post(client)
        again = await post(client)

    assert outage["status"] == "answered" and outage["meta"]["facts_unavailable"] is True
    assert (
        recovered["meta"]["cache_hit"] is False and recovered["meta"]["facts_unavailable"] is False
    )
    assert again["meta"]["cache_hit"] is True
    assert len(cache.puts) == 1
