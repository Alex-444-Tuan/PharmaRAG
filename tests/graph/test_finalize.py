import datetime
import uuid

import pytest

from pharmarag.graph.nodes.finalize import NOTICE, finalize, split_sentences
from pharmarag.graph.state import Chunk, Fact, Flags, Resolution, Sentence

SETID = uuid.UUID(int=7)


def chunk(i: int) -> Chunk:
    return Chunk(
        id=f"c{i}",
        chunk_id=i,
        setid=SETID,
        spl_version=3,
        ingredient_key="1",
        section_loinc="34068-7",
        section_title="Dosage and administration",
        text="Take 10 mg daily.",
        spans=[],
        effective_date=datetime.date(2025, 1, 2),
        fused_score=0.03,
    )


def S(text, verdict="unverified", citations=("c1",), **kw) -> Sentence:
    return Sentence(text=text, citations=list(citations), verdict=verdict, **kw)


def test_split_sentences_verifier_off_keeps_unverified_but_never_unsupported():
    sents = [S("a"), S("b", "unsupported", (), reason="no_citation"), S("c", "contradicted")]
    shown, removed = split_sentences(sents, verifier_on=False)
    assert [s.text for s in shown] == ["a"]
    assert [s.text for s in removed] == ["b", "c"]


def test_split_sentences_verifier_on_is_fail_closed():
    sents = [S("a"), S("b", "supported"), S("c", "unsupported"), S("d", "contradicted")]
    shown, removed = split_sentences(sents, verifier_on=True)
    assert [s.text for s in shown] == ["b"]  # "unverified" is not good enough when verifying
    assert len(removed) == 3


@pytest.mark.pg
async def test_answered_response_shape(make_deps, make_fake_llm, pg_conn):
    from tests.graph.seed import set_corpus_version

    await set_corpus_version(pg_conn, "v-xyz")
    await pg_conn.commit()
    state = {
        "flags": Flags(verifier=True),
        "intent": "label_qa",
        "evidence_ok": True,
        "evidence": [chunk(1), chunk(2)],
        "rx_facts": [Fact(id="r1", rxcui="5", text="Brand: Atarax", retrieval_attempt=1)],
        "sentences": [
            S("Take 10 mg daily.", "supported", ("c1", "r1"), score=0.93, span="c1:0"),
            S("Rejected one.", "unsupported", ("c2",)),
        ],
        "resolutions": [
            Resolution(
                mention="hydroxyzin",
                status="resolved",
                match="approximate",
                ingredient_key="1",
                name="hydroxyzine",
            ),
            Resolution(mention="zzz", status="unknown"),
        ],
        "retrieval_attempt": 1,
        "generation_attempt": 1,
        "timings": {"analyze": 5.0},
    }
    out = await finalize(state, make_deps(make_fake_llm([])))
    r = out["response"]
    assert out["status"] == r["status"] == "answered"
    assert r["drugs"] == [
        {"mention": "hydroxyzin", "name": "hydroxyzine", "rxcui": "1", "match": "approximate"}
    ]
    assert r["answer"] == [
        {
            "text": "Take 10 mg daily.",
            "citations": ["c1", "r1"],
            "support": {"span": "c1:0", "score": 0.93},
        }
    ]
    # Sources list only what shown sentences cite, with the label version and DailyMed URL.
    assert [s["id"] for s in r["sources"]] == ["c1", "r1"]
    assert r["sources"][0] == {
        "id": "c1",
        "setid": str(SETID),
        "spl_version": 3,
        "section": "Dosage and administration",
        "effective_date": "2025-01-02",
        "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={SETID}",
    }
    assert r["notice"] == NOTICE
    assert r["meta"]["corpus_version"] == "v-xyz"
    assert r["meta"]["removed_sentences"] == 1
    assert len(out["removed_sentences"]) == 1


@pytest.mark.pg
async def test_status_derivation(make_deps, make_fake_llm):
    deps = make_deps(make_fake_llm([]))
    base = {"flags": Flags(verifier=False), "retrieval_attempt": 1}
    ok = {
        **base,
        "intent": "label_qa",
        "evidence_ok": True,
        "evidence": [chunk(1)],
        "sentences": [S("a")],
    }
    assert (await finalize(ok, deps))["status"] == "answered"
    weak = {**ok, "evidence_ok": False}
    assert (await finalize(weak, deps))["status"] == "insufficient_evidence"
    empty = {**ok, "sentences": []}
    assert (await finalize(empty, deps))["status"] == "insufficient_evidence"
    declined = {**base, "intent": "out_of_scope"}
    out = await finalize(declined, deps)
    assert out["status"] == "declined" and out["response"]["answer"] == []
    preset = {**ok, "status": "not_covered"}
    out = await finalize(preset, deps)
    assert out["status"] == "not_covered" and out["response"]["answer"] == []
