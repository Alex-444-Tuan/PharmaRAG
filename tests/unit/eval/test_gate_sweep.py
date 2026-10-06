import pytest

from pharmarag.eval import gate_sweep as gs
from pharmarag.resolver.cascade import Cascaded, Scored


def test_edit_variants_are_single_edits_and_never_the_word_itself():
    v = gs.edit_variants("prednisone")
    assert "prednisone" not in v
    assert "prednisne" in v  # a deleted letter
    assert "prendisone" in v  # swapped neighbours
    assert "prednisona" in v  # a changed vowel
    assert "predniisone" in v  # a doubled letter
    assert v == sorted(v)
    assert all(len(x) >= gs.MIN_LENGTH for x in v)


def test_edit_variants_skip_words_that_become_too_short():
    # Deleting a letter from a five-letter word leaves four letters: not a query.
    assert not any(len(x) < gs.MIN_LENGTH for x in gs.edit_variants("abcde"))


def scored(*pairs):
    return Cascaded("m", False, [Scored(k, k, s) for k, s in pairs])


def test_tally_applies_the_given_gate_setting(thresholds):
    cases = [
        ("1", scored(("1", 0.97), ("2", 0.70))),  # clear and right
        ("1", scored(("2", 0.96), ("1", 0.94))),  # a wrong top, saved only by the margin
        ("1", scored(("1", 0.50))),  # nothing close enough
    ]
    cur = (0.90, 0.05, 0.80)
    assert gs.tally(cases, thresholds.resolver, cur) == {
        "correct": 1,
        "clarified": 1,
        "clarify_no_gold": 0,
        "not_recognised": 1,
        "wrong": 0,
    }
    # With the margin gone the wrong top is acted on: the sweep must show that as WRONG.
    assert gs.tally(cases, thresholds.resolver, (0.90, 0.01, 0.80))["wrong"] == 1


def test_the_grid_contains_the_current_setting(thresholds):
    r = thresholds.resolver
    assert (r.resolved_min, r.margin, r.ambiguous_min) in gs.grid()


@pytest.mark.corpus
async def test_no_single_edit_misspelling_of_a_lookalike_resolves_to_another_drug(
    corpus_url, thresholds
):
    """The dev evidence behind the frozen thresholds, as a regression guard."""
    report = await gs.run(corpus_url)
    assert report["n_queries"] > 1500
    cur = tuple(report["current"])
    row = next(
        g for g in report["grid"] if (g["resolved_min"], g["margin"], g["ambiguous_min"]) == cur
    )
    assert row["wrong"] == 0
    assert row["clarify_no_gold"] == 0
    assert row["correct"] > 0.8 * report["n_queries"]
    # The check can fail: with the margin at 0.02 the same variants include wrong resolutions.
    loose = next(g for g in report["grid"] if g["margin"] == 0.02 and g["resolved_min"] == 0.90)
    assert loose["wrong"] > 0
