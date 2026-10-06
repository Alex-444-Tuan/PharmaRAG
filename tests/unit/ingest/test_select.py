"""Label selection ordering, ingredient keys, and corpus_version."""

from datetime import date

import pytest

from pharmarag.ingest.load import compute_corpus_version
from pharmarag.ingest.select import (
    Candidate,
    group_by_key,
    ingredient_key,
    is_base_form,
    is_nda_or_bla,
    is_repackager,
    labeler_from_title,
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


# --- repackager and form tie-breaks -------------------------------------------------------


def titled(setid, title, category="NDA", d=date(2024, 1, 1)):
    return Candidate(setid, 1, category, d, "6809", title)


def test_labeler_is_the_last_bracketed_text_of_the_title():
    assert (
        labeler_from_title("NORVASC (AMLODIPINE) TABLET [VIATRIS SPECIALTY LLC]")
        == "VIATRIS SPECIALTY LLC"
    )
    assert labeler_from_title("X [A] TABLET [B, INC.]") == "B, INC."
    assert labeler_from_title("NO LABELER") == ""


@pytest.mark.parametrize(
    "labeler",
    [
        "REMEDYREPACK INC.",
        "Bryant Ranch Prepack",
        "A-S Medication Solutions",
        "Proficient Rx LP",
        "NuCare Pharmaceuticals, Inc.",
        "Quality Care Products, LLC",
        "Direct_Rx",
        "PD-Rx Pharmaceuticals, Inc.",
        "Preferred Pharmaceuticals Inc.",
        "Physicians Total Care, Inc.",
        "Coupler LLC",
        "ST MARY'S MEDICAL PARK PHARMACY",
        "Golden State Medical Supply, Inc.",
    ],
)
def test_known_repackagers_are_recognised(labeler):
    assert is_repackager(labeler)


@pytest.mark.parametrize(
    "labeler", ["Pfizer Laboratories Div Pfizer Inc", "Viatris Specialty LLC", "Teva", ""]
)
def test_manufacturers_are_not_repackagers(labeler):
    assert not is_repackager(labeler)


def test_repackager_loses_to_the_manufacturer_even_when_newer_but_nda_tier_still_wins():
    maker = titled("b", "NORVASC (AMLODIPINE) TABLET [PFIZER]", d=date(2020, 1, 1))
    repack = titled("a", "NORVASC (AMLODIPINE) TABLET [REMEDYREPACK INC.]", d=date(2026, 1, 1))
    generic = titled("c", "AMLODIPINE TABLET [TEVA]", category="ANDA", d=date(2026, 6, 1))
    assert rank([repack, generic, maker]) == [maker, repack, generic]


@pytest.mark.parametrize(
    "title",
    [
        "GLYBURIDE TABLET [TEVA]",
        "NORVASC (AMLODIPINE BESYLATE) TABLET [PFIZER]",
        "LIPITOR (ATORVASTATIN CALCIUM) TABLET, FILM COATED [VIATRIS]",
        "HYDROXYZINE PAMOATE CAPSULE [X]",
        "BUSPIRONE CAPSULE, GELATIN COATED [X]",
        "HEPARIN SODIUM INJECTION, SOLUTION [PFIZER]",
        "VINCRISTINE SULFATE INJECTION [HOSPIRA]",
        "TRILEPTAL (OXCARBAZEPINE) TABLET, FILM COATED "
        "TRILEPTAL (OXCARBAZEPINE) SUSPENSION [NOVARTIS]",
        "TEGRETOL (CARBAMAZEPINE) SUSPENSION TEGRETOL (CARBAMAZEPINE) TABLET "
        "TEGRETOL XR (CARBAMAZEPINE) TABLET, EXTENDED RELEASE [NOVARTIS]",
    ],
)
def test_plain_tablets_capsules_and_injections_are_base_forms(title):
    assert is_base_form(title)


@pytest.mark.parametrize(
    "title",
    [
        "OXTELLAR XR (OXCARBAZEPINE) TABLET [SUPERNUS]",
        "WELLBUTRIN SR (BUPROPION HYDROCHLORIDE) TABLET, FILM COATED [GLAXOSMITHKLINE LLC]",
        "AUVELITY (DEXTROMETHORPHAN, BUPROPION) TABLET, MULTILAYER, EXTENDED RELEASE [AXSOME]",
        "GLUCOTROLXL (GLIPIZIDE) TABLET, EXTENDED RELEASE [ROERIG]",
        "CARBATROL (CARBAMAZEPINE) CAPSULE, EXTENDED RELEASE [TAKEDA]",
        "BUPROPION HCL ER (XL) (BUPROPION HCL ER) TABLET, EXTENDED RELEASE [LIFESTAR]",
        "QLONILIK (CLONIDINE HYDROCHLORIDE) SOLUTION [CMP PHARMA, INC.]",
        "ORAL SUSPENSION OF X (X) SUSPENSION [Y]",
        "METFORMIN TABLET, ORALLY DISINTEGRATING [Y]",
        "OMEPRAZOLE CAPSULE, DELAYED RELEASE [Y]",
        "INVEGA SUSTENNA (PALIPERIDONE) INJECTION, SUSPENSION, EXTENDED RELEASE [JANSSEN]",
    ],
)
def test_extended_release_and_liquid_forms_are_niche(title):
    assert not is_base_form(title)


def test_base_form_beats_niche_form_inside_a_tier_even_if_older():
    base = titled("b", "GLUCOTROL (GLIPIZIDE) TABLET [PFIZER]", d=date(2019, 1, 1))
    niche = titled(
        "a", "GLUCOTROLXL (GLIPIZIDE) TABLET, EXTENDED RELEASE [PFIZER]", d=date(2026, 1, 1)
    )
    assert rank([niche, base]) == [base, niche]


def test_repackager_rule_outranks_form_rule_and_form_rule_outranks_date():
    repack_base = titled("1", "X TABLET [BRYANT RANCH PREPACK]", d=date(2026, 1, 1))
    maker_niche = titled("2", "X TABLET, EXTENDED RELEASE [TEVA]", d=date(2026, 1, 1))
    maker_base_old = titled("3", "X TABLET [TEVA]", d=date(2010, 1, 1))
    ranked = rank([repack_base, maker_niche, maker_base_old])
    assert [c.setid for c in ranked] == ["3", "2", "1"]


def test_an_ingredient_that_only_has_niche_forms_still_gets_a_label():
    only = [
        titled("1", "X TABLET, EXTENDED RELEASE [A]", d=date(2020, 1, 1)),
        titled("2", "X SOLUTION [B]", d=date(2022, 1, 1)),
    ]
    assert [c.setid for c in select_labels(only)] == ["2"]


def test_candidates_without_a_title_are_not_penalised():
    assert (
        rank([cand("a", "NDA", date(2020, 1, 1)), cand("b", "NDA", date(2021, 1, 1))])[0].setid
        == "b"
    )


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
