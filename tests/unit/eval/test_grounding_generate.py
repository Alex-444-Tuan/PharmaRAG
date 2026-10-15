"""Claim generation (spec section 11.2, "Claim set"): no network, a scripted graph."""

from datetime import date

import pytest

from pharmarag.eval import grounding_generate as gg
from pharmarag.eval.grounding_data import Question
from pharmarag.graph.state import Chunk, Fact, Sentence, Span
from pharmarag.llm.base import FakeLLM

SETID = "00000000-0000-0000-0000-000000000001"


def ev(cid="c1", text="Take 10 mg."):
    return Chunk(
        id=cid,
        chunk_id=1,
        setid=SETID,
        spl_version=1,
        ingredient_key="1",
        section_loinc="34068-7",
        section_title="Dosage",
        text=text,
        spans=[Span(i=0, start=0, end=len(text))],
        effective_date=date(2026, 1, 1),
        fused_score=0.1,
    )


def question(i=1, answerable=True):
    return Question(id=f"q{i:02d}", question=f"Q{i}?", answerable=answerable, drug="d")


def state(sentences, attempt=1, facts=()):
    return {
        "sentences": sentences,
        "evidence": [ev()],
        "rx_facts": list(facts),
        "retrieval_attempt": attempt,
        "response": {"status": "answered"},
        "evidence_ok": True,
    }


def test_claims_carry_question_sentence_citations_and_evidence_snapshot():
    final = state([Sentence(text="Take 10 mg.", citations=["c1"])])
    (c,) = gg.claims_from_state(question(), final, 10, "claude-sonnet-5-5", 1)
    assert c.question == "Q1?" and c.sentence == "Take 10 mg." and c.citations == ["c1"]
    assert c.evidence[0].text == "Take 10 mg."
    assert c.meta["llm_model_id"] == "claude-sonnet-5-5"
    assert c.meta["question_id"] == "q01" and c.label is None


def test_uncited_sentences_are_not_claims():
    final = state(
        [
            Sentence(text="Cited.", citations=["c1"]),
            Sentence(text="Not cited.", citations=[], verdict="unsupported", reason="no_citation"),
        ]
    )
    assert [c.sentence for c in gg.claims_from_state(question(), final, 10, "m", 1)] == ["Cited."]


def test_facts_are_those_of_the_final_attempt_and_capped_like_the_writer_saw_them():
    facts = [
        Fact(id="r1", rxcui="1", text="old", retrieval_attempt=1),
        Fact(id="r1", rxcui="1", text="new1", retrieval_attempt=2),
        Fact(id="r2", rxcui="1", text="new2", retrieval_attempt=2),
        Fact(id="r3", rxcui="1", text="new3", retrieval_attempt=2),
    ]
    final = state([Sentence(text="S.", citations=["r1"])], attempt=2, facts=facts)
    (c,) = gg.claims_from_state(question(), final, 2, "m", 1)
    assert [f.text for f in c.facts] == ["new1", "new2"]


def make_claims(per_question: dict[str, int]):
    out = []
    for qid, n in per_question.items():
        for i in range(n):
            q = Question(id=qid, question=qid, answerable=True, drug="d")
            final = state([Sentence(text=f"{qid} sentence {i}.", citations=["c1"])])
            out += gg.claims_from_state(q, final, 10, "m", 1)
    return out


def test_selection_is_round_robin_over_questions_up_to_the_target():
    cands = make_claims({"q01": 10, "q02": 10, "q03": 2})
    chosen = gg.select_claims(cands, target=9)
    counts = {q: sum(c.meta["question_id"] == q for c in chosen) for q in ("q01", "q02", "q03")}
    assert counts == {"q01": 4, "q02": 3, "q03": 2}  # q03 runs dry, the others share the rest
    assert len(chosen) == 9


def test_selection_keeps_each_questions_sentence_order():
    chosen = gg.select_claims(make_claims({"q01": 3}), target=3)
    assert [c.sentence for c in chosen] == [f"q01 sentence {i}." for i in range(3)]


def test_selection_drops_duplicates_and_numbers_ids():
    cands = make_claims({"q01": 2}) + make_claims({"q01": 2})
    chosen = gg.select_claims(cands, target=40)
    assert len(chosen) == 2
    assert [c.id for c in chosen] == ["g01", "g02"]


def test_ids_have_room_for_the_target_width():
    chosen = gg.select_claims(make_claims({"q01": 120}), target=120)
    assert chosen[0].id == "g001" and chosen[-1].id == "g120"


class ScriptedGraph:
    """Stands in for the compiled graph: returns one scripted final state per call."""

    def __init__(self, finals):
        self.finals = list(finals)
        self.invocations = []

    async def ainvoke(self, state, config=None):
        self.invocations.append((state, config))
        return self.finals.pop(0)


class FakeDeps:
    def __init__(self, verifier_available=False, eval_mode=True):
        from types import SimpleNamespace

        self.settings = SimpleNamespace(eval_mode=eval_mode, verify_mode="stream")
        self.thresholds = SimpleNamespace(
            graph=SimpleNamespace(recursion_limit=12),
            retrieval=SimpleNamespace(max_facts=10),
        )
        self.resolver_available = True
        self.verifier_available = verifier_available


async def test_generate_asks_every_question_with_the_verifier_off():
    finals = [state([Sentence(text=f"A{i}.", citations=["c1"])]) for i in range(3)]
    graph = ScriptedGraph(finals)
    llm = gg.CountingLLM(FakeLLM(model_id="claude-sonnet-5-5"))
    claims, log = await gg.generate(
        graph, FakeDeps(), [question(1), question(2), question(3)], llm=llm, target=3
    )
    assert len(claims) == 3 and len(log) == 3
    flags = [s["flags"] for s, _ in graph.invocations]
    assert all(f.verifier is False for f in flags)
    assert all(cfg["recursion_limit"] == 12 for _, cfg in graph.invocations)


async def test_generate_repeats_passes_until_enough_claims():
    one = lambda t: state([Sentence(text=t, citations=["c1"])])  # noqa: E731
    graph = ScriptedGraph([one("A."), one("B."), one("C.")])
    claims, log = await gg.generate(
        graph, FakeDeps(), [question(1)], llm=gg.CountingLLM(FakeLLM()), target=3, max_passes=5
    )
    assert [c.sentence for c in claims] == ["A.", "B.", "C."]
    assert [r["pass"] for r in log] == [1, 2, 3]


async def test_generate_stops_after_max_passes_with_what_it_has():
    one = lambda t: state([Sentence(text=t, citations=["c1"])])  # noqa: E731
    graph = ScriptedGraph([one("A."), one("B.")])
    claims, _ = await gg.generate(
        graph, FakeDeps(), [question(1)], llm=gg.CountingLLM(FakeLLM()), target=40, max_passes=2
    )
    assert len(claims) == 2


async def test_eval_mode_honours_verifier_off_even_when_a_verifier_exists():
    graph = ScriptedGraph([state([])])
    await gg.ask_with_evidence(graph, FakeDeps(verifier_available=True, eval_mode=True), "q")
    assert graph.invocations[0][0]["flags"].verifier is False


async def test_generation_cannot_run_outside_eval_mode():
    # Outside EVAL_MODE `effective_flags` forces the verifier on (when one exists); generation
    # must then fail rather than silently produce verified sentences for an unverified baseline.
    with pytest.raises(RuntimeError, match="verifier off"):
        await gg.ask_with_evidence(
            ScriptedGraph([]), FakeDeps(verifier_available=True, eval_mode=False), "q"
        )


async def test_counting_llm_counts_calls_and_sizes():
    llm = gg.CountingLLM(FakeLLM(["hello world", "again"], model_id="m"))
    assert llm.model_id == "m"
    await llm.complete([{"role": "user", "content": "x" * 400}], system="y" * 400)
    await llm.complete([{"role": "user", "content": "z"}])
    u = llm.usage()
    assert u["calls"] == 2
    assert u["approx_input_tokens"] == (801) // 4
    assert u["approx_output_tokens"] == len("hello world" + "again") // 4


async def test_counting_llm_counts_structured_and_stream_calls():
    from pydantic import BaseModel

    class Out(BaseModel):
        a: int

    llm = gg.CountingLLM(FakeLLM(['{"a": 1}', "one two"]))
    assert (await llm.structured([{"role": "user", "content": "q"}], Out)).a == 1
    pieces = [p async for p in llm.stream([{"role": "user", "content": "q"}])]
    assert "".join(pieces) == "one two"
    assert llm.calls == 2
