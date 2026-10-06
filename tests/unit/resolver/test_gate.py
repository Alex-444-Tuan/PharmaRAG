"""The decision gate at every edge of its thresholds (spec section 7).

With the frozen thresholds: resolved_min 0.90, margin 0.05, ambiguous_min 0.80, 3 candidates.
"""

import pytest

from pharmarag.app.settings import ResolverThresholds
from pharmarag.resolver.gate import KeyScore, decide, rank


@pytest.fixture
def th(thresholds) -> ResolverThresholds:
    assert (
        thresholds.resolver.resolved_min,
        thresholds.resolver.margin,
        thresholds.resolver.ambiguous_min,
    ) == (0.90, 0.05, 0.80), "these edge tests are written for the proposed thresholds"
    return thresholds.resolver


def ks(*pairs: tuple[str, float]) -> list[KeyScore]:
    return [KeyScore(k, s) for k, s in pairs]


# --- resolved_min edge -------------------------------------------------------------------------


def test_top_of_exactly_090_resolves(th):
    v = decide(ks(("1", 0.90), ("2", 0.60)), th)
    assert v.outcome == "resolved"
    assert v.top.key == "1"


def test_top_just_below_090_is_ambiguous_not_resolved(th):
    v = decide(ks(("1", 0.8999), ("2", 0.60)), th)
    assert v.outcome == "ambiguous"
    assert [c.key for c in v.candidates] == ["1"]


def test_a_single_candidate_with_a_high_score_resolves(th):
    assert decide(ks(("1", 0.97)), th).outcome == "resolved"


def test_float_noise_does_not_move_a_score_across_090(th):
    # 0.7 + 0.2 is 0.8999999999999999 in floating point. It is 0.90 for the gate.
    assert 0.7 + 0.2 < 0.90
    assert decide(ks(("1", 0.7 + 0.2)), th).outcome == "resolved"


# --- margin edge ---------------------------------------------------------------------------------


def test_margin_of_exactly_005_resolves(th):
    # 0.95 - 0.90 is 0.04999999999999993 in floating point; the rule says "at least 0.05".
    assert 0.95 - 0.90 < 0.05
    v = decide(ks(("1", 0.95), ("2", 0.90)), th)
    assert v.outcome == "resolved"
    assert v.top.key == "1"


def test_margin_just_under_005_is_ambiguous(th):
    v = decide(ks(("1", 0.95), ("2", 0.9001)), th)
    assert v.outcome == "ambiguous"
    assert [c.key for c in v.candidates] == ["1", "2"]


def test_two_distinct_ingredients_within_the_margin_are_ambiguous(th):
    v = decide(ks(("5470", 0.97), ("5553", 0.95)), th)
    assert v.outcome == "ambiguous"
    assert [c.key for c in v.candidates] == ["5470", "5553"]


def test_a_tie_is_ambiguous(th):
    assert decide(ks(("1", 0.96), ("2", 0.96)), th).outcome == "ambiguous"


def test_the_margin_is_measured_to_the_next_distinct_ingredient_not_to_the_same_one(th):
    # Two names of the same ingredient must not make it "ambiguous with itself".
    v = decide(ks(("1", 0.97), ("1", 0.96), ("2", 0.70)), th)
    assert v.outcome == "resolved"
    assert v.top.key == "1"


# --- ambiguous_min edge --------------------------------------------------------------------------


def test_top_of_exactly_080_is_ambiguous(th):
    v = decide(ks(("1", 0.80)), th)
    assert v.outcome == "ambiguous"
    assert [c.key for c in v.candidates] == ["1"]


def test_top_just_below_080_is_unknown(th):
    v = decide(ks(("1", 0.7999), ("2", 0.78)), th)
    assert v.outcome == "unknown"
    assert v.candidates == ()


def test_unknown_wins_over_a_close_second_below_the_floor(th):
    assert decide(ks(("1", 0.79), ("2", 0.79)), th).outcome == "unknown"


def test_no_candidates_is_unknown(th):
    v = decide([], th)
    assert v.outcome == "unknown"
    assert v.top is None


# --- candidates ----------------------------------------------------------------------------------


def test_ambiguous_returns_at_most_max_candidates(th):
    v = decide(ks(("1", 0.93), ("2", 0.92), ("3", 0.91), ("4", 0.90)), th)
    assert v.outcome == "ambiguous"
    assert [c.key for c in v.candidates] == ["1", "2", "3"]


def test_candidates_below_the_floor_are_not_offered(th):
    v = decide(ks(("1", 0.88), ("2", 0.85), ("3", 0.55)), th)
    assert v.outcome == "ambiguous"
    assert [c.key for c in v.candidates] == ["1", "2"]


def test_max_candidates_comes_from_the_thresholds(th):
    one = th.model_copy(update={"max_candidates": 1})
    v = decide(ks(("1", 0.93), ("2", 0.92)), one)
    assert [c.key for c in v.candidates] == ["1"]


def test_ranking_is_best_first_with_ties_broken_by_key(th):
    assert [s.key for s in rank(ks(("9", 0.9), ("2", 0.95), ("1", 0.9)))] == ["2", "1", "9"]


def test_thresholds_are_read_from_the_config_not_hard_coded(th):
    strict = th.model_copy(update={"resolved_min": 0.95})
    assert decide(ks(("1", 0.93)), th).outcome == "resolved"
    assert decide(ks(("1", 0.93)), strict).outcome == "ambiguous"
    wide = th.model_copy(update={"margin": 0.20})
    assert decide(ks(("1", 0.97), ("2", 0.85)), th).outcome == "resolved"
    assert decide(ks(("1", 0.97), ("2", 0.85)), wide).outcome == "ambiguous"


# --- exact match ---------------------------------------------------------------------------------


def test_one_exact_ingredient_resolves(th):
    v = decide(ks(("1", 1.0)), th, exact=True)
    assert v.outcome == "resolved"


def test_exact_resolves_whatever_the_score(th):
    assert decide(ks(("1", 0.5)), th, exact=True).outcome == "resolved"


def test_two_distinct_exact_ingredients_are_ambiguous(th):
    # A name that two ingredients share: the user has to say which.
    v = decide(ks(("1", 1.0), ("2", 1.0)), th, exact=True)
    assert v.outcome == "ambiguous"
    assert [c.key for c in v.candidates] == ["1", "2"]
