"""Outcome classification and the metric (spec section 11.1), pure functions."""

import pytest

from pharmarag.eval import lookalike as la
from pharmarag.graph.state import Candidate, Resolution

GOLD = "5553"
ROW = la.Row(
    question="What is the dose of hydroxizine?",
    mention="hydroxizine",
    gold_ingredient_key=GOLD,
    gold_name="hydroxyzine",
    partner_name="hydralazine",
    kind="misspelling",
)


def res(status, key=None, candidates=()):
    return Resolution(
        mention="m",
        status=status,
        match="approximate" if key else None,
        ingredient_key=key,
        candidates=[Candidate(ingredient_key=k, name=n, score=0.9) for k, n in candidates],
    )


# --- resolver-only: Resolution ---


def test_resolved_to_gold_is_correct():
    assert la.classify_resolution(res("resolved", GOLD), GOLD) == la.CORRECT


def test_resolved_to_another_drug_is_wrong_drug():
    assert la.classify_resolution(res("resolved", "5470"), GOLD) == la.WRONG_DRUG


def test_not_covered_for_another_drug_is_still_acting_on_the_wrong_drug():
    assert la.classify_resolution(res("not_covered", "11289"), GOLD) == la.WRONG_DRUG


def test_not_covered_for_gold_is_not_covered():
    # Cannot happen with a gold drug that has a label, but the category must exist.
    assert la.classify_resolution(res("not_covered", GOLD), GOLD) == la.NOT_COVERED


def test_ambiguous_with_gold_among_the_candidates_is_clarified():
    r = res("ambiguous", None, [("5470", "hydralazine"), (GOLD, "hydroxyzine")])
    assert la.classify_resolution(r, GOLD) == la.CLARIFIED


def test_ambiguous_without_gold_is_not_a_clarification():
    r = res("ambiguous", None, [("5470", "hydralazine"), ("1", "x")])
    assert la.classify_resolution(r, GOLD) == la.CLARIFY_WITHOUT_GOLD


def test_unknown_for_a_drug_in_the_corpus_is_not_recognised():
    assert la.classify_resolution(res("unknown"), GOLD) == la.NOT_RECOGNISED


def test_misinterpreted_is_exactly_the_four_failure_outcomes():
    assert la.MISINTERPRETED == {
        la.WRONG_DRUG,
        la.NOT_RECOGNISED,
        la.NOT_COVERED,
        la.CLARIFY_WITHOUT_GOLD,
    }
    # A clarification is counted on its own, never as a misinterpretation (spec 11.1).
    assert la.CLARIFIED not in la.MISINTERPRETED
    assert la.OTHER not in la.MISINTERPRETED


# --- full mode: response ---

KEYS = {"s-gold": GOLD, "s-other": "5470"}


def answered(*sources):
    return {"status": "answered", "sources": list(sources)}


def test_an_answer_citing_only_gold_chunks_is_correct():
    out, keys = la.classify_response(
        answered({"id": "c1", "setid": "s-gold"}, {"id": "c2", "setid": "s-gold"}), ROW, KEYS
    )
    assert (out, keys) == (la.CORRECT, [GOLD, GOLD])


def test_an_answer_citing_any_other_drugs_chunk_is_wrong_drug():
    out, keys = la.classify_response(
        answered({"id": "c1", "setid": "s-gold"}, {"id": "c2", "setid": "s-other"}), ROW, KEYS
    )
    assert out == la.WRONG_DRUG
    assert keys == [GOLD, "5470"]


def test_an_rxnorm_fact_of_another_ingredient_counts_too():
    out, _ = la.classify_response(answered({"id": "r1", "rxcui": "5470", "text": "x"}), ROW, KEYS)
    assert out == la.WRONG_DRUG
    out, _ = la.classify_response(answered({"id": "r1", "rxcui": GOLD, "text": "x"}), ROW, KEYS)
    assert out == la.CORRECT


def test_a_source_of_unknown_origin_is_not_assumed_to_be_gold():
    out, keys = la.classify_response(answered({"id": "c1", "setid": "unknown"}), ROW, KEYS)
    assert out == la.WRONG_DRUG
    assert keys == ["?"]


def test_an_answer_with_no_sources_is_other_not_correct():
    assert la.classify_response(answered(), ROW, KEYS)[0] == la.OTHER


@pytest.mark.parametrize(
    "status, expected",
    [
        ("drug_not_recognised", la.NOT_RECOGNISED),
        ("not_covered", la.NOT_COVERED),
        ("insufficient_evidence", la.OTHER),
        ("declined", la.OTHER),
    ],
)
def test_other_statuses(status, expected):
    assert la.classify_response({"status": status}, ROW, KEYS)[0] == expected


@pytest.mark.parametrize(
    "candidates, expected",
    [
        ([{"ingredient_key": GOLD, "name": "hydroxyzine"}], la.CLARIFIED),
        ([{"rxcui": GOLD}, {"rxcui": "5470"}], la.CLARIFIED),
        ([{"name": "Hydroxyzine"}], la.CLARIFIED),
        ([GOLD], la.CLARIFIED),
        ([{"ingredient_key": "5470", "name": "hydralazine"}], la.CLARIFY_WITHOUT_GOLD),
        ([], la.CLARIFY_WITHOUT_GOLD),
        ([None, 5], la.CLARIFY_WITHOUT_GOLD),
    ],
)
def test_needs_clarification_counts_as_clarified_only_with_gold_among_the_candidates(
    candidates, expected
):
    response = {"status": "needs_clarification", "candidates": candidates}
    assert la.classify_response(response, ROW, KEYS)[0] == expected


def test_needs_clarification_without_a_candidates_field_is_not_a_clarification():
    assert la.classify_response({"status": "needs_clarification"}, ROW, KEYS)[0] == (
        la.CLARIFY_WITHOUT_GOLD
    )


# --- counts and the metric ---


def test_counts_sum_misinterpreted_and_keep_every_outcome():
    c = la.counts([la.CORRECT, la.CLARIFIED, la.WRONG_DRUG, la.NOT_RECOGNISED, la.OTHER])
    assert c["misinterpreted"] == 2
    assert c["total"] == 5
    assert c[la.CLARIFIED] == 1
    assert all(o in c for o in la.OUTCOMES)


def test_reduction_is_b_minus_s_over_b():
    assert la.reduction(13, 2) == pytest.approx(11 / 13)  # the spec's 85% example
    assert round(la.reduction(13, 2) * 100) == 85
    assert la.reduction(10, 10) == 0
    assert la.reduction(4, 0) == 1.0


def test_reduction_is_none_when_the_baseline_has_no_errors():
    assert la.reduction(0, 0) is None
    assert la.reduction(0, 3) is None


def test_reduction_can_be_negative_when_the_system_is_worse():
    assert la.reduction(2, 3) == pytest.approx(-0.5)
