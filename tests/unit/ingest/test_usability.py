"""Which parsed labels may be the reference label: content in Indications and Dosage."""

from pathlib import Path

import pytest

from pharmarag.ingest.pipeline import usability_problem
from pharmarag.ingest.spl_parser import parse_spl

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "spl"


def problem(name: str) -> str | None:
    return usability_problem(parse_spl((FIXTURES / name).read_bytes()))


@pytest.mark.parametrize(
    "name",
    [
        "norvasc_plr.xml",
        "lipitor_plr.xml",
        "zestoretic_old_combo.xml",
        "clonidine_old.xml",
        "hydroxyzine_old.xml",
    ],
)
def test_complete_labels_are_usable(name):
    assert problem(name) is None


def test_label_whose_dosage_section_is_coded_unclassified_is_usable_after_title_mapping():
    # QLONILIK: the 34068-7 section is Highlights only; the real text is coded 42229-5.
    assert problem("qlonilik_plr_title_coded.xml") is None


def test_label_without_a_dosage_and_administration_section_is_rejected():
    # RemedyRepack hydroxyzine pamoate: dosing text sits inside OVERDOSAGE. It has 8 other
    # sections, so the old "at least 3 sections" rule accepted it.
    label = parse_spl((FIXTURES / "hydroxyzine_pamoate_no_dosage.xml").read_bytes())
    assert len({s.loinc for s in label.sections}) >= 8
    assert "34068-7" not in {s.loinc for s in label.sections}
    assert problem("hydroxyzine_pamoate_no_dosage.xml") == "no text in Dosage and administration"


def _label(sections: str) -> bytes:
    return (
        '<document xmlns="urn:hl7-org:v3"><code code="34391-3"/><title>t</title>'
        '<setId root="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"/><versionNumber value="1"/>'
        f"<component><structuredBody>{sections}</structuredBody></component></document>"
    ).encode()


def sec(code: str, text: str = "<paragraph>Some text.</paragraph>") -> str:
    return f'<component><section><code code="{code}"/><text>{text}</text></section></component>'


def test_sections_without_text_do_not_count():
    # Indications and Dosage exist but are empty shells; only 3 sections have text overall
    xml = _label(
        sec("34067-9", "") + sec("34068-7", "") + sec("34070-3") + sec("34088-5") + sec("34069-5")
    )
    assert (
        usability_problem(parse_spl(xml))
        == "no text in Indications and usage and Dosage and administration"
    )


def test_both_core_sections_and_three_sections_with_text_are_required():
    only_core = _label(sec("34067-9") + sec("34068-7"))
    assert usability_problem(parse_spl(only_core)) == "only 2 sections with text"
    ok = _label(sec("34067-9") + sec("34068-7") + sec("34070-3"))
    assert usability_problem(parse_spl(ok)) is None
    no_indications = _label(sec("34068-7") + sec("34070-3") + sec("34088-5"))
    assert usability_problem(parse_spl(no_indications)) == "no text in Indications and usage"
