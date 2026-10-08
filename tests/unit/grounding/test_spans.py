"""Span records and candidate selection (spec section 9, "Verifier, per sentence", step 1)."""

import numpy as np
import pytest

from pharmarag.graph.state import Chunk, Fact, Span
from pharmarag.grounding.spans import SpanRecord, build_spans, select_candidates

SETID = "00000000-0000-0000-0000-000000000001"


def chunk(cid: str, sentences: list[str], **kw) -> Chunk:
    text, spans = "", []
    for i, s in enumerate(sentences):
        start = len(text) + (1 if text else 0)
        text = f"{text} {s}" if text else s
        spans.append(Span(i=i, start=start, end=start + len(s)))
    return Chunk(
        id=cid,
        chunk_id=int(cid[1:]),
        setid=SETID,
        spl_version=1,
        ingredient_key="1",
        section_loinc="34068-7",
        section_title=kw.get("section_title", "2 DOSAGE"),
        subsection_title=kw.get("subsection_title"),
        text=text,
        spans=spans,
        fused_score=0.5,
    )


def fact(fid: str, text: str) -> Fact:
    return Fact(id=fid, rxcui="1", text=text, retrieval_attempt=1)


# --- build_spans ---


def test_spans_are_cut_by_offset_and_ids_use_the_stored_span_index():
    c = chunk("c2", ["Take 5 mg daily.", "Avoid alcohol.", "Store dry."])
    recs = build_spans([c], [])
    assert [r.id for r in recs] == ["c2:0", "c2:1", "c2:2"]
    assert [r.text for r in recs] == ["Take 5 mg daily.", "Avoid alcohol.", "Store dry."]
    assert all(r.source == "c2" for r in recs)


def test_span_index_is_the_stored_one_not_the_position():
    c = chunk("c1", ["One.", "Two.", "Three."])
    c = c.model_copy(update={"spans": [c.spans[0], c.spans[2]]})  # span 1 missing
    assert [r.id for r in build_spans([c], [])] == ["c1:0", "c1:2"]


def test_context_is_previous_sentence_or_next_for_the_first():
    c = chunk("c1", ["First.", "Second.", "Third."])
    ctx = {r.id: r.context for r in build_spans([c], [])}
    assert ctx["c1:0"] == "First. Second."  # no previous sentence: use the next
    assert ctx["c1:1"] == "First. Second."
    assert ctx["c1:2"] == "Second. Third."


def test_single_span_chunk_has_no_neighbour():
    (r,) = build_spans([chunk("c1", ["Only one."])], [])
    assert r.context == r.text == "Only one."


def test_facts_are_spans_with_their_own_id():
    recs = build_spans([chunk("c1", ["A."])], [fact("r1", "Brand: Zestril"), fact("r2", "  ")])
    assert [r.id for r in recs] == ["c1:0", "r1"]  # blank fact dropped
    assert recs[1].source == "r1" and recs[1].context == "Brand: Zestril"


def test_chunk_without_stored_spans_becomes_one_span():
    c = chunk("c1", ["Whole text here."]).model_copy(update={"spans": []})
    (r,) = build_spans([c], [])
    assert r.id == "c1:0" and r.text == "Whole text here."


def test_headings_are_kept_for_the_premise():
    c = chunk(
        "c1",
        ["A."],
        section_title="8 USE IN SPECIFIC POPULATIONS",
        subsection_title="8.1 Pregnancy",
    )
    (r,) = build_spans([c], [])
    assert (r.section, r.subsection) == ("8 USE IN SPECIFIC POPULATIONS", "8.1 Pregnancy")


# --- select_candidates ---


def records(sources: list[str]) -> list[SpanRecord]:
    return [
        SpanRecord(id=f"{s}:{i}", source=s, text="t", context="t") for i, s in enumerate(sources)
    ]


def unit_matrix(sims: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Span vectors whose dot product with the sentence vector is exactly `sims`."""
    sentence = np.array([1.0, 0.0], dtype=np.float32)
    rows = [[s, float(np.sqrt(max(0.0, 1 - s * s)))] for s in sims]
    return sentence, np.array(rows, dtype=np.float32)


def pick(sims, sources, citations, nearest=3, k=5):
    sentence, matrix = unit_matrix(sims)
    recs = records(sources)
    return select_candidates(
        sentence, matrix, recs, citations, nearest_spans=nearest, candidate_spans=k
    )


def test_cited_spans_and_the_nearest_three_are_candidates():
    # c1 cited (indexes 0,1); c2 not cited. Nearest overall are indexes 4, 3, 2.
    sims = [0.10, 0.20, 0.50, 0.90, 0.95]
    idx = pick(sims, ["c1", "c1", "c2", "c2", "c2"], ["c1"], nearest=3, k=10)
    assert sorted(idx) == [0, 1, 2, 3, 4]


def test_result_is_ordered_by_cosine_descending():
    sims = [0.1, 0.9, 0.5, 0.7]
    assert pick(sims, ["c1"] * 4, ["c1"], k=10) == [1, 3, 2, 0]


def test_only_top_k_kept_and_nearest_survive_the_cut():
    # 8 cited spans, k=5, plus the 3 nearest overall (all uncited, highest cosine).
    sims = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.99, 0.98, 0.97]
    sources = ["c1"] * 8 + ["c2"] * 3
    idx = pick(sims, sources, ["c1"], nearest=3, k=5)
    assert len(idx) == 5
    assert idx[:3] == [8, 9, 10]  # the 3 nearest are never trimmed away
    assert set(idx[3:]) == {7, 6}  # then the best cited spans


def test_uncited_evidence_can_supply_support_when_the_citation_is_wrong():
    sims = [0.2, 0.2, 0.96]  # the matching span is in c2, but the sentence cites c1
    idx = pick(sims, ["c1", "c1", "c2"], ["c1"], nearest=1, k=5)
    assert 2 in idx


def test_ties_keep_evidence_order():
    assert pick([0.5, 0.5, 0.5], ["c1"] * 3, ["c1"], nearest=1, k=2) == [0, 1]


def test_no_citations_still_gets_the_nearest():
    assert pick([0.1, 0.9, 0.2], ["c1"] * 3, [], nearest=2, k=5) == [1, 2]


def test_empty_evidence_gives_no_candidates():
    assert (
        select_candidates(
            np.zeros(2, dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            [],
            ["c1"],
            nearest_spans=3,
            candidate_spans=5,
        )
        == []
    )


def test_fewer_spans_than_k():
    assert len(pick([0.1, 0.2], ["c1", "c1"], ["c1"], k=5)) == 2


@pytest.mark.parametrize("nearest", [0, 1, 3])
def test_nearest_zero_means_cited_only(nearest):
    sims = [0.1, 0.2, 0.99]
    idx = pick(sims, ["c1", "c1", "c2"], ["c1"], nearest=nearest, k=5)
    assert (2 in idx) == (nearest > 0)
