"""Number-and-unit check (spec section 9, step 2). The check must catch a wrong dose that NLI
models accept, and must not reject a sentence for a spelling difference."""

from decimal import Decimal

import pytest

from pharmarag.grounding.numeric import (
    extract_quantities,
    missing_quantities,
    parse_number,
)


def keys(text: str) -> set[tuple[Decimal, str]]:
    return {q.key for q in extract_quantities(text)}


def q(value: str, unit: str) -> tuple[Decimal, str]:
    return (Decimal(value), unit)


# --- number parsing ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,000", "1000"),
        ("0.5", "0.5"),
        (".5", "0.5"),
        ("0.50", "0.5"),
        ("10", "10"),
        ("1,000.25", "1000.25"),
    ],
)
def test_parse_number_normalises(raw, expected):
    assert parse_number(raw) == Decimal(expected)


def test_parse_number_rejects_garbage():
    assert parse_number("abc") is None


# --- extraction ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Take 10 mg daily.", {q("10", "mg")}),
        ("Take 10mg daily.", {q("10", "mg")}),
        ("A 10-mg tablet.", {q("10", "mg")}),
        ("Up to 1,000 mg per day.", {q("1000", "mg")}),
        ("Start at 0.5 mg.", {q("0.5", "mg")}),
        ("Start at .5 mg.", {q("0.5", "mg")}),
        ("Start at 0.50 mg.", {q("0.5", "mg")}),
        ("Give 250 mcg.", {q("0.25", "mg")}),
        ("Give 250 µg.", {q("0.25", "mg")}),  # micro sign
        ("Give 250 μg.", {q("0.25", "mg")}),  # greek mu
        ("Give 250 micrograms.", {q("0.25", "mg")}),
        ("Give 2 g.", {q("2000", "mg")}),
        ("Give 5 mL.", {q("5", "ml")}),
        ("Give 5 milliliters.", {q("5", "ml")}),
        ("Incidence was 12%.", {q("12", "%")}),
        ("Incidence was 12 %.", {q("12", "%")}),
        ("Incidence was 12 percent.", {q("12", "%")}),
        ("Every 8 hours.", {q("8", "hour")}),
        ("Every 8 hr.", {q("8", "hour")}),
        ("Every 8 h.", {q("8", "hour")}),
        ("A 12-hour interval.", {q("12", "hour")}),
        ("For 7 days.", {q("168", "hour")}),
        ("For 2 weeks.", {q("336", "hour")}),
        ("In patients over 65 years.", {q("569400", "hour")}),
        ("5 mg/kg once.", {q("5", "mg/kg")}),
        ("5 mg per kg once.", {q("5", "mg/kg")}),
        ("25 mg/m2 weekly.", {q("25", "mg/m2")}),
        ("25 mg/m² weekly.", {q("25", "mg/m2")}),
        ("Use 100 units.", {q("100", "units")}),
        ("Use 100 IU.", {q("100", "iu")}),
    ],
)
def test_extract_quantities(text, expected):
    assert keys(text) == expected


def test_day_denominator_is_dropped():
    # "10 mg/day" and "10 mg once daily" must compare equal (SPEC note in numeric.py)
    assert keys("Maximum 10 mg/day.") == {q("10", "mg")}
    assert keys("5 mg/kg/day") == {q("5", "mg/kg")}


def test_other_denominators_are_kept():
    assert q("5", "mg/kg") in keys("5 mg/kg")
    assert q("5", "mg/kg") not in keys("5 mg")
    assert keys("5 mg/mL") == {q("5", "mg/ml")}


@pytest.mark.parametrize(
    "text",
    [
        "Patients had 3 events.",
        "CYP2D6 and CYP3A4 are involved.",
        "See Table 2 for details.",
        "Type 2 diabetes mellitus.",
        "Take 2 tablets twice daily.",
        "The 5-HT3 receptor.",
        "About 10 hepatic events.",  # "h" must not match the start of "hepatic"
    ],
)
def test_numbers_without_units_are_ignored(text):
    assert extract_quantities(text) == []


def test_unit_word_boundary():
    assert extract_quantities("10 gastric ulcers") == []
    assert extract_quantities("3 minutes") == []  # minutes are not in the unit list
    assert extract_quantities("5 mgs") == []


# --- ranges and lists ---


@pytest.mark.parametrize(
    "text",
    ["10-20 mg", "10 to 20 mg", "10 mg to 20 mg", "10–20 mg", "10 mg - 20 mg", "10 or 20 mg"],
)
def test_ranges_give_both_endpoints(text):
    assert keys(text) == {q("10", "mg"), q("20", "mg")}


def test_list_shares_the_trailing_unit():
    assert keys("Doses of 5, 10, or 20 mg.") == {q("5", "mg"), q("10", "mg"), q("20", "mg")}


def test_range_in_percent_and_years():
    assert keys("1 to 5%") == {q("1", "%"), q("5", "%")}
    assert keys("12 to 17 years") == {q(str(12 * 8760), "hour"), q(str(17 * 8760), "hour")}


def test_reference_numbers_are_not_pulled_into_a_list():
    # "Table 2, 10 mg": the 2 is a table number, not a 2 mg dose
    assert keys("In Table 2, 10 mg was used.") == {q("10", "mg")}
    assert keys("Study 1 and 10 mg") == {q("10", "mg")}


def test_a_number_with_its_own_unit_is_not_rebound():
    assert keys("5 mg or 10 mcg") == {q("5", "mg"), q("0.01", "mg")}


# --- the check ---


def test_equal_spellings_pass():
    assert missing_quantities("Take 10mg daily.", ["The dose is 10 mg once daily."]) == []
    assert missing_quantities("Up to 1000 mg.", ["Maximum 1,000 mg"]) == []
    assert missing_quantities("Start at .5 mg.", ["Start at 0.5 mg"]) == []
    assert missing_quantities("Start at 0.5 mg.", ["Start at .5 mg"]) == []


def test_unit_conversion_passes():
    assert missing_quantities("Give 0.5 g.", ["Give 500 mg"]) == []
    assert missing_quantities("Give 500 mcg.", ["Give 0.5 mg"]) == []
    assert missing_quantities("For 7 days.", ["For 1 week"]) == []
    assert missing_quantities("Every 24 hours.", ["Once every 1 day"]) == []


def test_range_in_sentence_matches_range_in_evidence():
    assert missing_quantities("The dose is 10-20 mg.", ["Doses range from 10 to 20 mg."]) == []
    assert missing_quantities("The dose is 10 to 20 mg.", ["Either 10 mg or 20 mg."]) == []


def test_wrong_dose_is_caught():
    miss = missing_quantities("The dose is 20 mg.", ["The dose is 10 mg."])
    assert [m.key for m in miss] == [q("20", "mg")]


def test_wrong_unit_is_caught():
    assert missing_quantities("Take 10 mg.", ["Take 10 mcg."])
    assert missing_quantities("Take 10 mg.", ["Take 10 mL."])
    assert missing_quantities("Take 10 hours.", ["Take 10 days."])


def test_per_kg_dose_is_not_a_flat_dose():
    assert missing_quantities("Give 5 mg.", ["Give 5 mg/kg."])
    assert missing_quantities("Give 5 mg/kg.", ["Give 5 mg."])
    assert missing_quantities("Give 5 mg/kg.", ["Give 5 mg/kg."]) == []


def test_one_bad_number_among_good_ones_is_caught():
    miss = missing_quantities(
        "Start at 5 mg and increase to 40 mg over 2 weeks.",
        ["Start at 5 mg and increase to 20 mg over 2 weeks."],
    )
    assert [m.key for m in miss] == [q("40", "mg")]


def test_number_may_come_from_any_candidate_span():
    texts = ["Start at 5 mg.", "Max is 20 mg."]
    assert missing_quantities("Start at 5 mg, maximum 20 mg.", texts) == []


def test_no_numbers_passes():
    assert missing_quantities("Avoid in pregnancy.", ["Avoid in pregnancy."]) == []
    assert missing_quantities("Avoid in pregnancy.", []) == []


def test_no_evidence_fails_when_sentence_has_a_number():
    assert missing_quantities("Take 10 mg.", [])


def test_duplicates_reported_once():
    assert len(missing_quantities("10 mg then 10 mg again", ["x"])) == 1


def test_evidence_number_without_unit_does_not_satisfy():
    assert missing_quantities("Take 10 mg.", ["There were 10 patients."])


def test_percent_mismatch_caught():
    assert missing_quantities("Seen in 25% of patients.", ["Seen in 2.5% of patients."])
    assert missing_quantities("Seen in 2.5% of patients.", ["Seen in 2.5% of patients."]) == []
