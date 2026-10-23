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
from pharmarag.graph.run import run_ask
from pharmarag.resolver.rxnav_client import RxNavClient, RxNavUnavailable
from tests.graph.fakes import (
    DOSE_TEXT,
    FakeQueryEmbeddingCache,
    analysis,
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
