import pytest

from pharmarag.resolver.normalize import collapse, normalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Hydroxyzine", "hydroxyzine"),
        ("  HYDROXYZINE   ", "hydroxyzine"),
        ("hydroxyzine 25 mg", "hydroxyzine"),
        ("hydroxyzine 25mg tablets", "hydroxyzine"),
        ("Metformin ER 500 mg", "metformin"),
        ("Metformin HCl ER 500mg tab", "metformin hydrochloride"),
        ("wellbutrin XL 150 mg", "wellbutrin"),
        ("Zoloft (sertraline) 50mg", "zoloft sertraline"),
        ("insulin glargine 100 units/mL", "insulin glargine"),
        ("amoxicillin 0.5 g capsule.", "amoxicillin"),
        ("Toprol-XL", "toprol-xl"),  # a hyphen is part of a brand name, not a separator
        ("penicillin V", "penicillin v"),
        ("hydroxyzine\t25 mg\n tablet", "hydroxyzine"),
        ("ER", ""),
        ("25 mg tablet", ""),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_salt_words_are_kept_so_a_salt_match_stays_possible():
    assert normalize("hydroxyzine pamoate 25 mg") == "hydroxyzine pamoate"
    assert normalize("amlodipine besylate") == "amlodipine besylate"


def test_a_spelled_out_salt_is_kept_and_hcl_is_expanded_to_it():
    assert normalize("hydroxyzine hydrochloride") == "hydroxyzine hydrochloride"
    assert normalize("hydroxyzine HCl") == normalize("hydroxyzine hydrochloride")


def test_collapse_keeps_form_words():
    # Some brands carry a form word; the alias table is tried with this form too.
    assert collapse("Glucotrol  XL") == "glucotrol xl"
    assert collapse("Glucotrol XL 10 mg tablets") == "glucotrol xl"
    assert collapse("Zoloft (sertraline)") == "zoloft sertraline"
    assert normalize("Glucotrol XL 10 mg tablets") == "glucotrol"
