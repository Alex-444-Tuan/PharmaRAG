"""Perturbations and rewrites: a hallucination must really be absent from the evidence, and a
rewrite must keep the meaning (spec section 11.2, dev set and top-up rule)."""

import random

import pytest

from pharmarag.eval import perturb
from pharmarag.grounding.numeric import missing_quantities


def rng(seed=1):
    return random.Random(seed)


# --- hallucination perturbations ---


def test_change_number_produces_a_quantity_the_evidence_does_not_state():
    s = "The usual dose is 10 mg once daily for 7 days."
    ev = [s]
    for seed in range(20):
        out = perturb.change_number(s, ev, rng(seed))
        assert out and out != s
        assert missing_quantities(out, ev), out  # the new number is absent from the evidence


def test_change_number_refuses_when_no_unit_number():
    assert perturb.change_number("Take it with food.", ["Take it with food."], rng()) is None
    assert perturb.change_number("Seen in 3 patients.", ["Seen in 3 patients."], rng()) is None


def test_change_number_avoids_a_value_that_appears_elsewhere_in_the_evidence():
    s = "The dose is 10 mg."
    # 20 mg is stated elsewhere in the evidence, so it is not a hallucination: choose another
    for seed in range(30):
        out = perturb.change_number(s, [s, "A dose of 20 mg is also used."], rng(seed))
        assert out and "20 mg" not in out


def test_change_number_handles_decimals_and_formatting():
    out = perturb.change_number("Start at 0.5 mg daily.", ["Start at 0.5 mg daily."], rng(3))
    assert out and "0.5 mg" not in out and " mg daily" in out


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Zestril is contraindicated in pregnancy.", "is not contraindicated"),
        ("Tablets should not be crushed.", "should be crushed"),
        ("The drug can cause dizziness.", "does not cause"),
        ("Avoid grapefruit juice.", "Use grapefruit juice"),
    ],
)
def test_negate_flips_polarity(sentence, expected):
    out = perturb.negate(sentence, [sentence])
    assert out and expected in out


def test_negate_skips_a_sentence_that_would_get_a_double_negative():
    # "no ... were observed" -> "no ... were not observed" would not be a clear contradiction
    assert perturb.negate("No differences were observed between groups.", ["x"]) is None


def test_negate_can_still_remove_an_existing_negation():
    out = perturb.negate("Do not crush the tablets. They should not be split.", ["x"])
    assert out and out != "Do not crush the tablets. They should not be split."


def test_negate_refuses_a_flip_the_evidence_already_contains():
    s = "The drug is not recommended in children."
    # the evidence also says "is recommended" somewhere, so flipping would not be a hallucination
    assert perturb.negate(s, [s, "In adults the drug is recommended."]) is None


def test_swap_frequency_and_population():
    f = perturb.swap_frequency("Take 5 mg once daily.", ["Take 5 mg once daily."])
    assert f and "three times daily" in f
    p = perturb.swap_population(
        "Safety in pediatric patients is established.", ["Safety in pediatric patients."]
    )
    assert p and "adult patients" in p


def test_swap_refuses_when_the_new_phrase_is_in_the_evidence():
    assert (
        perturb.swap_frequency("Take it daily.", ["Take it daily.", "Applied weekly in trials."])
        is None
    )


def test_overclaim_strengthens_the_claim():
    out = perturb.overclaim("Nitrates may aggravate angina.", ["Nitrates may aggravate angina."])
    assert out == "Nitrates will aggravate angina."


def test_rules_return_none_when_nothing_applies():
    plain = "The tablet is white and round."
    for fn in (perturb.negate, perturb.swap_frequency, perturb.swap_population, perturb.overclaim):
        assert fn(plain, [plain]) is None


# --- supported rewrites ---


def test_clean_sentence_drops_section_references():
    assert (
        perturb.clean_sentence("Avoid use (see 5.2) in pregnancy (5.1, 5.3).")
        == "Avoid use in pregnancy."
    )


def test_lower_first_keeps_acronyms_and_names():
    assert perturb.lower_first("Patients may bleed.") == "patients may bleed."
    assert perturb.lower_first("NSAIDs may bleed.") == "NSAIDs may bleed."
    assert perturb.lower_first("CYP3A4 is involved.") == "CYP3A4 is involved."


def test_join_sentences_keeps_both_facts():
    out = perturb.join_sentences("Take with food.", "Avoid alcohol.")
    assert out == "Take with food; avoid alcohol."


def test_rewrite_light_changes_the_text_but_not_the_numbers():
    s = "Take 10 mg twice daily in patients with renal impairment."
    seen = set()
    for seed in range(40):
        out = perturb.rewrite_light(s, rng(seed))
        assert out and out != s
        assert not missing_quantities(out, [s]), out  # no number was invented or lost
        seen.add(out)
    assert len(seen) >= 2  # more than one rule is used


def test_rewrite_light_none_when_no_rule_applies():
    assert perturb.rewrite_light("The tablet is white.", rng()) is None


def test_rewrite_rules_individually():
    assert perturb._paraphrase("Take it twice daily.") == "Take it two times a day."
    assert perturb._first_clause("Take the tablet with food each day; avoid alcohol.") == (
        "Take the tablet with food each day."
    )
    assert perturb._first_clause("Take with food; avoid alcohol.") is None  # head too short
    assert perturb._drop_discourse("However, rates were lower.") == "Rates were lower."
    assert perturb._drop_parenthetical("Rates were lower (p<0.05) overall today.") == (
        "Rates were lower overall today."
    )
    assert perturb._spacing("Give 10 mg now.") == "Give 10mg now."
