"""End-to-end graph runs with FakeLLM and FakeEmbedder over real PostgreSQL."""

import json

import pytest

from pharmarag.graph.run import run_ask
from pharmarag.graph.state import Flags
from pharmarag.retrieval.scope import FilterRequiredError

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
    # Every node recorded its time (NFR-5), including the two parallel branches.
    assert set(out["debug"]["timings_ms"]) == {
        "analyze",
        "retrieve_dense",
        "retrieve_lexical",
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


async def test_weak_evidence_goes_to_finalize_without_calling_the_writer(
    make_fake_llm, make_deps, make_graph, pg_conn
):
    # Empty corpus: nothing retrieved, evidence_ok is False.
    llm = make_fake_llm([analysis()])
    deps = make_deps(llm)
    out = await run_ask(make_graph(deps), deps, question=QUESTION, debug=True)
    assert out["status"] == "insufficient_evidence"
    assert out["answer"] == []
    assert len(llm.calls) == 1
    assert "answer" not in out["debug"]["timings_ms"]
    assert out["meta"]["corpus_version"] == "unknown"
    assert out["meta"]["retrieval_attempts"] == 1


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


async def test_cycle1_forces_resolver_flag_off_so_mentions_are_never_extracted(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm([analysis(mentions=["hydroxyzine"]), "Taken four times daily. [c1]"])
    deps = make_deps(llm)  # resolver_available defaults to False
    out = await run_ask(
        make_graph(deps), deps, question=QUESTION, flags=Flags(resolver=True), debug=True
    )
    assert out["debug"]["flags"]["resolver"] is False
    assert out["debug"]["mentions"] == []


async def test_mention_without_resolver_never_reaches_unfiltered_ranking(
    make_fake_llm, make_deps, make_graph, corpus
):
    # FR-3 fail-closed: if someone enables the resolver flag without wiring a `resolve` node,
    # retrieval refuses to rank rather than searching the whole corpus.
    llm = make_fake_llm([analysis(mentions=["hydroxyzine"])])
    deps = make_deps(llm, resolver_available=True)
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
