"""Label selection ordering, ingredient keys, and corpus_version."""

from datetime import date

import pytest

from pharmarag.ingest.load import compute_corpus_version
from pharmarag.ingest.select import (
    Candidate,
    group_by_key,
    ingredient_key,
    is_nda_or_bla,
    rank,
    select_labels,
)


def cand(setid: str, category: str | None, d: date | None, key: str = "6809", version: int = 1):
    return Candidate(setid, version, category, d, key)


# --- ingredient_key -----------------------------------------------------------------------


def test_single_ingredient_key():
    assert ingredient_key(["6809"]) == "6809"


def test_combination_key_is_sorted_numerically_not_lexically():
    # the spec's own example: 6809+593411 (lexically "593411" would sort first)
    assert ingredient_key(["593411", "6809"]) == "6809+593411"
    assert ingredient_key(["29046", "5487", "310798"]) == "5487+29046+310798"


def test_key_is_order_independent_and_deduplicated():
    assert ingredient_key(["8", "3", "8"]) == ingredient_key(["3", "8"]) == "3+8"


def test_empty_key():
    assert ingredient_key([]) == ""


# --- ordering -----------------------------------------------------------------------------


def test_nda_beats_anda_even_when_the_anda_label_is_newer():
    nda = cand("b", "NDA", date(2020, 1, 1))
    anda = cand("a", "ANDA", date(2026, 1, 1))
    assert rank([anda, nda]) == [nda, anda]


def test_bla_counts_as_brand_tier():
    bla = cand("b", "BLA", date(2019, 1, 1))
    anda = cand("a", "ANDA", date(2026, 1, 1))
    assert rank([anda, bla])[0] == bla


def test_authorized_generic_and_unapproved_are_not_brand_tier():
    assert not is_nda_or_bla("NDA AUTHORIZED GENERIC")
    assert not is_nda_or_bla("UNAPPROVED DRUG OTHER")
    assert not is_nda_or_bla(None)
    assert is_nda_or_bla("nda") and is_nda_or_bla(" BLA ")


def test_latest_date_wins_within_a_tier():
    old = cand("a", "NDA", date(2020, 5, 1))
    new = cand("z", "NDA", date(2024, 5, 1))
    assert rank([old, new]) == [new, old]


def test_lowest_setid_breaks_a_full_tie():
    d = date(2024, 1, 1)
    x = cand("bbbbbbbb-0000-0000-0000-000000000000", "NDA", d)
    y = cand("aaaaaaaa-0000-0000-0000-000000000000", "NDA", d)
    assert rank([x, y]) == [y, x]


def test_missing_date_sorts_after_any_dated_label_in_the_same_tier():
    dated = cand("z", "NDA", date(1999, 1, 1))
    undated = cand("a", "NDA", None)
    assert rank([undated, dated]) == [dated, undated]


def test_ranking_is_independent_of_input_order():
    items = [
        cand("c", "ANDA", date(2025, 1, 1)),
        cand("b", "NDA", date(2021, 1, 1)),
        cand("a", "NDA", date(2023, 1, 1)),
        cand("d", None, None),
    ]
    assert rank(items) == rank(list(reversed(items)))
    assert [c.setid for c in rank(items)] == ["a", "b", "c", "d"]


# --- one label per ingredient set ---------------------------------------------------------


def test_one_label_per_ingredient_set_combination_gets_its_own_label():
    single_nda = cand("s1", "NDA", date(2022, 1, 1), key="6809")
    single_anda = cand("s2", "ANDA", date(2026, 1, 1), key="6809")
    combo_anda = cand("c1", "ANDA", date(2024, 1, 1), key="6809+593411")
    combo_nda = cand("c2", "NDA", date(2020, 1, 1), key="6809+593411")
    picked = select_labels([single_anda, combo_anda, single_nda, combo_nda])
    assert picked == [single_nda, combo_nda]


def test_labels_per_ingredient_raises_the_limit():
    items = [cand(f"s{i}", "NDA", date(2020 + i, 1, 1)) for i in range(4)]
    assert [c.setid for c in select_labels(items, per_key=3)] == ["s3", "s2", "s1"]
    assert len(select_labels(items, per_key=10)) == 4


def test_group_by_key_ranks_each_group():
    a = cand("a", "ANDA", date(2024, 1, 1), key="1")
    b = cand("b", "NDA", date(2020, 1, 1), key="1")
    c = cand("c", "NDA", date(2020, 1, 1), key="2")
    groups = group_by_key([a, b, c])
    assert groups == {"1": [b, a], "2": [c]}


@pytest.mark.parametrize("per_key", [1, 2])
def test_selection_is_deterministic(per_key):
    items = [cand(f"s{i}", "ANDA", date(2024, 1, 1)) for i in range(5)]
    assert select_labels(items, per_key) == select_labels(list(reversed(items)), per_key)


# --- corpus_version -----------------------------------------------------------------------

PAIRS = [("aaaaaaaa-0000-0000-0000-000000000001", 3), ("bbbbbbbb-0000-0000-0000-000000000002", 1)]
MODEL = "BAAI/bge-small-en-v1.5"


def test_corpus_version_is_deterministic_and_order_independent():
    v = compute_corpus_version(PAIRS, MODEL)
    assert v == compute_corpus_version(PAIRS, MODEL)
    assert v == compute_corpus_version(list(reversed(PAIRS)), MODEL)
    assert len(v) == 16 and int(v, 16) >= 0


def test_corpus_version_changes_when_an_spl_version_changes():
    bumped = [(PAIRS[0][0], 4), PAIRS[1]]
    assert compute_corpus_version(bumped, MODEL) != compute_corpus_version(PAIRS, MODEL)


def test_corpus_version_changes_when_the_model_name_changes():
    assert compute_corpus_version(PAIRS, "other-model") != compute_corpus_version(PAIRS, MODEL)


def test_corpus_version_changes_when_a_label_is_added_or_removed():
    base = compute_corpus_version(PAIRS, MODEL)
    assert compute_corpus_version(PAIRS[:1], MODEL) != base
    assert (
        compute_corpus_version([*PAIRS, ("cccccccc-0000-0000-0000-000000000003", 1)], MODEL) != base
    )


def test_corpus_version_setid_case_does_not_matter():
    upper = [(PAIRS[0][0].upper(), PAIRS[0][1]), PAIRS[1]]
    assert compute_corpus_version(upper, MODEL) == compute_corpus_version(PAIRS, MODEL)


def test_corpus_version_of_empty_corpus_still_depends_on_model():
    assert compute_corpus_version([], "a") != compute_corpus_version([], "b")
