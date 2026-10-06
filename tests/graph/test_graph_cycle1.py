"""End-to-end graph runs with FakeLLM and FakeEmbedder over real PostgreSQL.

These started in cycle 1 as the baseline pipeline. With no resolver in `Deps` the graph still
runs it (the AT-1 baseline); the full-graph routes are in test_graph_routes.py.
"""

import json

import pytest

from pharmarag.graph.run import run_ask
from pharmarag.graph.state import Flags, Resolution
from pharmarag.llm.errors import TRUNCATION_MARKER, LLMRefusalError
from pharmarag.retrieval.scope import FilterRequiredError
from tests.graph.fakes import FakeResolver

pytestmark = pytest.mark.pg

QUESTION = "What is the hydroxyzine dosage for anxiety in adults?"


def analysis(**kw) -> str:
    base = {
        "intent": "label_qa",
        "mentions": [],
        "sections": [],
        "search_query": "hydroxyzine dosage for anxiety in adults",
    }
    return json.dumps({**base, **kw})


async def test_happy_path_runs_end_to_end(make_fake_llm, make_deps, make_graph, corpus):
    llm = make_fake_llm(
        [
            analysis(),
            "Hydroxyzine dosage for anxiety in adults is 50 to 100 mg four times daily. [c1]",
        ]
    )
    deps = make_deps(llm)
    out = await run_ask(make_graph(deps), deps, question=QUESTION, debug=True)

    assert out["status"] == "answered"
    assert out["notice"] == "From FDA drug labels via DailyMed. Not medical advice."
    assert [s["text"] for s in out["answer"]] == [
        "Hydroxyzine dosage for anxiety in adults is 50 to 100 mg four times daily."
    ]
    assert out["answer"][0]["citations"] == ["c1"]
    (src,) = out["sources"]
    assert src["id"] == "c1"
    assert src["setid"] == str(corpus["hydroxyzine"])
    assert src["url"] == (
        f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={corpus['hydroxyzine']}"
    )
    assert src["spl_version"] == 1 and src["effective_date"] == "2024-05-01"
    assert src["section"] == "Dosage and administration"
    meta = out["meta"]
    assert meta["corpus_version"] == "corpus-abc123"
    assert meta["retrieval_attempts"] == 1 and meta["generation_attempts"] == 1
    assert meta["removed_sentences"] == 0 and meta["latency_ms"] >= 0
    # Deps has neither a resolver nor a verifier, and the response says so.
    assert meta["verified"] is False and meta["resolver"] is False
    # Every node that ran recorded its time (NFR-5), including the three parallel branches.
    assert set(out["debug"]["timings_ms"]) == {
        "analyze",
        "resolve",
        "retrieve_dense",
        "retrieve_lexical",
        "fetch_rx_facts",
        "fuse_grade",
        "answer",
        "finalize",
    }
    # The writer got the delimited evidence blocks.
    writer_call = llm.calls[1]
    assert "<evidence>" in writer_call["messages"][0]["content"]
    assert "[c1]" in writer_call["messages"][0]["content"]


async def test_declined_path_skips_retrieval_and_writer(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm([analysis(intent="patient_advice")])
    deps = make_deps(llm)
    out = await run_ask(
        make_graph(deps), deps, question="Should I stop taking my pills?", debug=True
    )
    assert out["status"] == "declined"
    assert out["answer"] == [] and out["sources"] == []
    assert out["notice"].startswith("From FDA drug labels")
    assert len(llm.calls) == 1  # analyze only; the writer was never called
    assert set(out["debug"]["timings_ms"]) == {"analyze", "finalize"}  # no retrieval, no grading
    assert out["meta"]["retrieval_attempts"] == 0 and out["meta"]["generation_attempts"] == 0


async def test_weak_evidence_on_both_attempts_never_calls_the_writer(
    make_fake_llm, make_deps, make_graph, pg_conn
):
    # Empty corpus: nothing retrieved, evidence_ok is False, the one rewrite finds nothing too.
    llm = make_fake_llm([analysis(), json.dumps({"search_query": "dose of the drug"})])
    deps = make_deps(llm)
    out = await run_ask(make_graph(deps), deps, question=QUESTION, debug=True)
    assert out["status"] == "insufficient_evidence"
    assert out["answer"] == []
    assert len(llm.calls) == 2  # analyze and rewrite_query; the writer was never called
    assert "answer" not in out["debug"]["timings_ms"]
    assert out["meta"]["corpus_version"] == "unknown"
    assert out["meta"]["retrieval_attempts"] == 2


async def test_writer_insufficient_becomes_insufficient_evidence(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm([analysis(), "INSUFFICIENT"])
    deps = make_deps(llm)
    out = await run_ask(make_graph(deps), deps, question=QUESTION)
    assert out["status"] == "insufficient_evidence"
    assert out["answer"] == [] and out["sources"] == []
    assert out["meta"]["generation_attempts"] == 1


async def test_uncited_sentence_is_removed_and_counted(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm(
        [analysis(), "Hydroxyzine is taken four times daily. [c1]\nIt is a popular medicine."]
    )
    deps = make_deps(llm)
    out = await run_ask(make_graph(deps), deps, question=QUESTION, debug=True)
    assert out["status"] == "answered"
    assert [s["text"] for s in out["answer"]] == ["Hydroxyzine is taken four times daily."]
    assert out["meta"]["removed_sentences"] == 1
    (removed,) = out["debug"]["removed_sentences"]
    assert removed["text"] == "It is a popular medicine." and removed["reason"] == "no_citation"


async def test_without_a_resolver_the_flag_is_forced_off_and_mentions_are_never_extracted(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm([analysis(mentions=["hydroxyzine"]), "Taken four times daily. [c1]"])
    deps = make_deps(llm)  # resolver=None
    out = await run_ask(
        make_graph(deps), deps, question=QUESTION, flags=Flags(resolver=True), debug=True
    )
    assert out["debug"]["flags"]["resolver"] is False
    assert out["debug"]["mentions"] == []


async def test_resolver_that_returns_no_key_for_a_mention_never_reaches_unfiltered_ranking(
    make_fake_llm, make_deps, make_graph, corpus
):
    # FR-3 fail-closed: a resolver bug that marks a mention resolved without an ingredient_key
    # must stop retrieval, not widen it to the whole corpus.
    buggy = FakeResolver(
        {"hydroxyzine": Resolution(mention="hydroxyzine", status="resolved", name="hydroxyzine")}
    )
    llm = make_fake_llm([analysis(mentions=["hydroxyzine"])])
    deps = make_deps(llm, resolver=buggy)
    with pytest.raises(FilterRequiredError):
        await run_ask(make_graph(deps), deps, question=QUESTION)


async def test_hits_from_two_branches_are_fused_not_duplicated(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm([analysis(), "Taken four times daily. [c1]"])
    deps = make_deps(llm)
    await run_ask(make_graph(deps), deps, question=QUESTION)
    prompt = llm.calls[1]["messages"][0]["content"]
    ids = [line.split("]")[0] for line in prompt.splitlines() if line.startswith("[c")]
    assert len(ids) == len(set(ids)) >= 2  # each chunk once, even if both branches found it


def _refuse(messages, system):
    raise LLMRefusalError("declined by provider")


async def test_refusal_during_analyze_ends_as_declined_without_retrieval(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm([_refuse])
    deps = make_deps(llm)
    out = await run_ask(make_graph(deps), deps, question=QUESTION, debug=True)
    assert out["status"] == "declined"
    assert out["answer"] == []
    assert set(out["debug"]["timings_ms"]) == {"analyze", "finalize"}


async def test_refusal_during_writing_ends_as_insufficient_evidence(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm([analysis(), _refuse])
    deps = make_deps(llm)
    out = await run_ask(make_graph(deps), deps, question=QUESTION)
    assert out["status"] == "insufficient_evidence"
    assert out["answer"] == [] and out["sources"] == []


async def test_truncated_writer_reply_drops_the_cut_off_line_end_to_end(
    make_fake_llm, make_deps, make_graph, corpus
):
    reply = "Hydroxyzine is taken four times daily. [c1]\nThe maximum is 5" + TRUNCATION_MARKER
    llm = make_fake_llm([analysis(), reply])
    deps = make_deps(llm)
    out = await run_ask(make_graph(deps), deps, question=QUESTION)
    assert [s["text"] for s in out["answer"]] == ["Hydroxyzine is taken four times daily."]
