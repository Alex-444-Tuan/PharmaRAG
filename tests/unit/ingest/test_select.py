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
        "STATE OF FLORIDA DOH CENTRAL PHARMACY",
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


def test_base_form_beats_niche_form_even_if_older():
    base = titled("b", "GLUCOTROL (GLIPIZIDE) TABLET [PFIZER]", d=date(2019, 1, 1))
    niche = titled(
        "a", "GLUCOTROLXL (GLIPIZIDE) TABLET, EXTENDED RELEASE [PFIZER]", d=date(2026, 1, 1)
    )
    assert rank([niche, base]) == [base, niche]


def test_precedence_is_form_then_tier_then_repackager_then_date_then_setid():
    ranked = rank(
        [
            titled("1", "X TABLET [BRYANT RANCH PREPACK]", "NDA", date(2026, 1, 1)),
            titled("2", "X TABLET, EXTENDED RELEASE [TEVA]", "NDA", date(2026, 1, 1)),
            titled("3", "X TABLET [TEVA]", "ANDA", date(2010, 1, 1)),
            titled("4", "X TABLET [TEVA]", "NDA", date(2010, 1, 1)),
            titled("5", "X TABLET [TEVA]", "NDA", date(2020, 1, 1)),
            titled("6", "X TABLET [TEVA]", "NDA", date(2020, 1, 1)),
        ]
    )
    # base forms first: NDA non-repackager (newest, then lowest setid), then NDA repackager,
    # then ANDA; the extended-release label is last even though it is an NDA from a maker
    assert [c.setid for c in ranked] == ["5", "6", "4", "1", "3", "2"]


def test_anda_immediate_release_tablet_beats_nda_er_only_and_nda_epidural_injection():
    """The clonidine, bupropion and glipizide cases from the first smoke run."""
    anda_tablet = titled("a", "CLONIDINE HYDROCHLORIDE TABLET [TEVA]", "ANDA", date(2020, 1, 1))
    nda_er = titled("b", "KAPVAY (CLONIDINE) TABLET, EXTENDED RELEASE [CONCORDIA]", "NDA")
    nda_epidural = titled("c", "DURACLON (CLONIDINE) INJECTION, SOLUTION [MYLAN]", "NDA")
    nda_solution = titled("d", "QLONILIK (CLONIDINE) SOLUTION [CMP PHARMA]", "NDA")
    ranked = rank([nda_epidural, nda_er, nda_solution, anda_tablet])
    assert ranked[0] == anda_tablet
    assert select_labels([nda_epidural, nda_er, nda_solution, anda_tablet]) == [anda_tablet]


def test_nda_immediate_release_tablet_still_beats_an_anda_tablet():
    nda = titled("n", "CATAPRES (CLONIDINE) TABLET [BOEHRINGER]", "NDA", date(2010, 1, 1))
    anda = titled("a", "CLONIDINE TABLET [TEVA]", "ANDA", date(2026, 1, 1))
    assert rank([anda, nda])[0] == nda


def test_injection_is_base_only_when_the_ingredient_has_no_oral_label():
    injection = titled("i", "HEPARIN SODIUM INJECTION, SOLUTION [PFIZER]", "NDA")
    er = titled("e", "HEPARIN X TABLET, EXTENDED RELEASE [PFIZER]", "NDA", date(2026, 1, 1))
    liquid = titled("l", "HEPARIN X SOLUTION [PFIZER]", "NDA", date(2026, 6, 1))
    # injection-only ingredient: the injection is base and beats the newer niche oral forms
    assert rank([er, liquid, injection])[0] == injection
    # an oral tablet exists: it wins, and the injection falls back among the others
    tablet = titled("t", "HEPARIN X TABLET [TEVA]", "ANDA", date(2000, 1, 1))
    ranked = rank([injection, er, liquid, tablet])
    assert ranked[0] == tablet
    # the injection is no longer base, so it ties with the niche forms and loses on date
    assert ranked[-1] == injection


def test_has_oral_base_is_judged_within_each_ingredient_group():
    inj_single = Candidate("i", 1, "NDA", None, "1", "X INJECTION [PFIZER]")
    er_single = Candidate("e", 1, "NDA", date(2026, 1, 1), "1", "X TABLET, EXTENDED RELEASE [A]")
    oral_combo = Candidate("c", 1, "ANDA", None, "1+2", "X AND Y TABLET [B]")
    groups = group_by_key([er_single, oral_combo, inj_single])
    assert groups["1"] == [inj_single, er_single]  # the oral combination does not demote it


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
