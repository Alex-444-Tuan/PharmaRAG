"""Number check: table header units, rates, count-before-comma, and the "/day" trade-off
(spec review log round 5.4)."""

from decimal import Decimal

from pharmarag.grounding.numeric import extract_quantities, missing_quantities


def keys(text: str) -> set[tuple[Decimal, str]]:
    return {q.key for q in extract_quantities(text)}


def q(value: str, unit: str) -> tuple[Decimal, str]:
    return (Decimal(value), unit)


GABAPENTIN_ROW = (
    "Renal Function Creatinine Clearance (mL/min): >30 to 59; "
    "Total Daily Dose Range (mg/day): 400 to 1400; "
    "Dose Regimen (mg): 200 BID; Dose Regimen (mg): 300 BID"
)


def test_header_unit_is_carried_to_the_bare_numbers_of_its_cell():
    assert missing_quantities("The total daily dose is 400 to 1400 mg/day.", [GABAPENTIN_ROW]) == []
    assert missing_quantities("Give 200 mg BID.", [GABAPENTIN_ROW]) == []
    assert missing_quantities("Give 300 mg twice daily.", [GABAPENTIN_ROW]) == []


def test_header_unit_does_not_make_a_wrong_number_pass():
    assert missing_quantities("The total daily dose is 400 to 1500 mg/day.", [GABAPENTIN_ROW])
    assert missing_quantities("Give 250 mg BID.", [GABAPENTIN_ROW])
    assert missing_quantities("The dose is 400 to 1400 mcg/day.", [GABAPENTIN_ROW])


def test_header_unit_covers_a_clearance_cell_as_a_rate():
    assert missing_quantities("For creatinine clearance of 59 mL/min.", [GABAPENTIN_ROW]) == []


def test_percent_column_headers():
    row = (
        "Adverse Reaction: Arthralgia; % Placebo N=7,311: 6.5; % 10 mg N=3,908: 8.9; "
        "% 20 mg N=188: 11.7"
    )
    assert missing_quantities("Arthralgia occurred in 8.9% of patients on 10 mg.", [row]) == []
    assert missing_quantities("Arthralgia occurred in 11.7% of patients.", [row]) == []
    assert missing_quantities("Arthralgia occurred in 9.8% of patients.", [row])


def test_a_cell_that_already_has_its_unit_is_not_changed():
    row = "Dose (mg): 5 mg; Interval (h): 8 h"
    assert keys(row) == {q("5", "mg"), q("8", "hour")}


def test_text_that_is_not_a_table_is_untouched():
    assert keys("Dose: 5; Other: 6") == set()  # a header with no unit gives nothing
    assert keys("Study (placebo-controlled): 12") == set()  # parentheses that are not a unit
    assert keys("Note: take 5 mg; see label: 10 mg") == {q("5", "mg"), q("10", "mg")}


def test_ml_per_min_is_a_rate_not_a_volume():
    assert q("30", "ml/min") in keys("CrCl 30 mL/min")
    assert missing_quantities("Volume of 30 mL.", ["Creatinine clearance 30 mL/min."])
    assert missing_quantities("Clearance below 30 mL/min.", ["Give 30 mL."])
    assert missing_quantities("Clearance below 30 mL/min.", ["Clearance below 30 mL/min."]) == []


def test_a_count_before_a_comma_is_not_a_dose():
    assert keys("Take 2, 500 mg tablets.") == {q("500", "mg")}
    assert keys("Take two or 2, 500 mg tablets.") == {q("500", "mg")}


def test_real_lists_still_work_after_the_comma_rule():
    assert keys("Doses of 5, 10, or 20 mg.") == {q("5", "mg"), q("10", "mg"), q("20", "mg")}
    assert keys("Doses of 5, 10 and 20 mg.") == {q("5", "mg"), q("10", "mg"), q("20", "mg")}
    assert keys("5 or 10 mg") == {q("5", "mg"), q("10", "mg")}


def test_per_day_denominator_trade_off_is_a_known_false_pass():
    """ "/day" is dropped so that "10 mg once daily" matches a label's "10 mg/day". The price: a
    sentence "10 mg/day" also matches a label that says "10 mg twice daily (20 mg/day)", where
    10 mg is the single dose. The number check passes it and NLI has to catch the frequency.
    This test pins the behaviour so changing it is a decision, not an accident."""
    label = "Give 10 mg twice daily (20 mg/day)."
    assert missing_quantities("The total daily dose is 10 mg/day.", [label]) == []
    assert missing_quantities("The total daily dose is 30 mg/day.", [label])  # nowhere in the label
    assert missing_quantities("The dose is 10 mg/kg/day.", [label])  # per-kg is not a flat dose
