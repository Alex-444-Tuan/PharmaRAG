"""SPL parser: five real labels (PLR and older format) plus synthetic edge cases."""

import logging
from datetime import date
from pathlib import Path

import pytest

from pharmarag.ingest.spl_parser import ParsedLabel, SplParseError, code_for_title, parse_spl
from pharmarag.sections import PRESCRIPTION_LABEL_LOINC, SECTIONS

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "spl"
PLR = ["norvasc_plr.xml", "lipitor_plr.xml"]
OLD = ["zestoretic_old_combo.xml", "clonidine_old.xml", "hydroxyzine_old.xml"]


def load(name: str) -> ParsedLabel:
    return parse_spl((FIXTURES / name).read_bytes())


def codes(label: ParsedLabel) -> list[str]:
    return [s.loinc for s in label.sections]


def all_blocks(label: ParsedLabel, loinc: str):
    return [
        b for s in label.sections if s.loinc == loinc for sub in s.subsections for b in sub.blocks
    ]


# --- real fixtures ------------------------------------------------------------------------


def test_fixture_labels_exist():
    assert {p.name for p in FIXTURES.glob("*.xml")} >= set(PLR + OLD)


def test_metadata_norvasc():
    label = load("norvasc_plr.xml")
    assert label.setid == "7367289c-b0b0-466a-83e2-558e2985c29f"
    assert label.version == 10
    assert label.doc_type_loinc == PRESCRIPTION_LABEL_LOINC
    assert label.labeler == "Viatris Specialty LLC"
    assert label.effective_date == date(2023, 2, 15)
    assert label.marketing_category == "NDA"
    assert label.title == "NORVASC (AMLODIPINE BESYLATE) TABLET [VIATRIS SPECIALTY LLC]"
    assert label.active_moieties == ["AMLODIPINE"]


def test_metadata_old_format_generic():
    label = load("clonidine_old.xml")
    assert label.marketing_category == "ANDA"
    assert label.title == "CLONIDINE HYDROCHLORIDE TABLET [ALEMBIC PHARMACEUTICALS LIMITED]"
    assert label.effective_date == date(2026, 9, 30)


def test_combination_product_lists_both_active_moieties():
    label = load("zestoretic_old_combo.xml")
    assert label.active_moieties == ["LISINOPRIL ANHYDROUS", "HYDROCHLOROTHIAZIDE"]


@pytest.mark.parametrize("name", PLR)
def test_plr_labels_use_warnings_and_precautions_code(name):
    found = codes(load(name))
    assert "43685-7" in found
    assert "34071-1" not in found and "42232-9" not in found


@pytest.mark.parametrize("name", OLD)
def test_old_labels_use_warnings_and_precautions_codes(name):
    found = codes(load(name))
    assert "34071-1" in found
    assert "42232-9" in found
    assert "43685-7" not in found


@pytest.mark.parametrize("name", PLR + OLD)
def test_only_known_codes_kept_and_every_section_has_text(name):
    label = load(name)
    assert label.sections
    for section in label.sections:
        assert section.loinc in SECTIONS
        assert section.title
        assert section.subsections
        for sub in section.subsections:
            assert sub.blocks
            assert all(b.text and b.text == b.text.strip() for b in sub.blocks)
            assert all("\n" not in b.text and "  " not in b.text for b in sub.blocks)


@pytest.mark.parametrize("name", PLR + OLD)
def test_unknown_codes_are_skipped_and_counted(name):
    label = load(name)
    assert label.skipped_codes["51945-4"] >= 1  # package label panel
    assert label.skipped_codes["48780-1"] == 1  # product data elements
    assert not set(label.skipped_codes) & set(SECTIONS)


def test_unknown_code_skip_is_logged(caplog):
    with caplog.at_level(logging.DEBUG, logger="pharmarag.ingest.spl_parser"):
        load("hydroxyzine_old.xml")
    assert any("skipping section with code 51945-4" in r.message for r in caplog.records)


def test_boxed_warning_is_kept_for_zestoretic():
    label = load("zestoretic_old_combo.xml")
    boxed = next(s for s in label.sections if s.loinc == "34066-1")
    assert boxed.title == "WARNING: FETAL TOXICITY"
    assert any("fetal" in b.text.lower() for sub in boxed.subsections for b in sub.blocks)


def test_subsection_titles_come_from_nested_sections():
    label = load("norvasc_plr.xml")
    warnings = next(s for s in label.sections if s.loinc == "43685-7")
    assert [sub.title for sub in warnings.subsections] == [
        "5.1 Hypotension",
        "5.2 Increased Angina or Myocardial Infarction",
        "5.3 Patients with Hepatic Failure",
    ]


def test_deeper_nesting_joins_titles():
    label = load("norvasc_plr.xml")
    interactions = next(s for s in label.sections if s.loinc == "34073-7")
    titles = [sub.title for sub in interactions.subsections]
    assert any(t and t.startswith("7.1 Impact of Other Drugs on Amlodipine > ") for t in titles)


def test_nested_known_code_becomes_its_own_section_not_duplicated_in_parent():
    label = load("zestoretic_old_combo.xml")
    order = codes(label)
    assert order.count("34073-7") == 1
    assert order.index("42232-9") < order.index("34073-7")
    precautions = next(s for s in label.sections if s.loinc == "42232-9")
    interactions = next(s for s in label.sections if s.loinc == "34073-7")
    precaution_text = " ".join(b.text for sub in precautions.subsections for b in sub.blocks)
    first_interaction = interactions.subsections[0].blocks[0].text
    assert first_interaction not in precaution_text


def test_missing_section_title_falls_back_to_canonical_name():
    label = load("hydroxyzine_old.xml")
    supplied = next(s for s in label.sections if s.loinc == "34069-5")
    assert supplied.title == SECTIONS["34069-5"]


def test_table_rows_become_header_value_lines_in_real_label():
    label = load("norvasc_plr.xml")
    rows = [b.text for b in all_blocks(label, "34084-4") if b.kind == "row"]
    edema = "Edema; 2.5 mg N=275: 1.8; Amlodipine 5 mg N=296: 3.0; 10 mg N=268: 10.8; "
    assert edema + "Placebo N=520: 0.6" in rows
    assert "Fatigue; NORVASC (%) (N=1730): 4.5; Placebo (%) (N=1250): 2.8" in rows
    # header-only and blank rows are not emitted as data lines
    assert not any(r.startswith(("N=", "2.5 mg;")) for r in rows)


def test_empty_paragraphs_are_dropped():
    label = load("hydroxyzine_old.xml")
    assert all(b.text for b in all_blocks(label, "34069-5"))


# --- synthetic documents ------------------------------------------------------------------


def spl(
    body: str, *, setid: str = "11111111-2222-3333-4444-555555555555", version: str = "3"
) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3">
  <code code="34391-3"/>
  <title>Example</title>
  <effectiveTime value="20240102"/>
  <setId root="{setid}"/>
  <versionNumber value="{version}"/>
  <component><structuredBody>{body}</structuredBody></component>
</document>""".encode()


def section(code: str, title: str, inner: str = "", children: str = "") -> str:
    return (
        f'<component><section><code code="{code}"/><title>{title}</title>'
        f"<text>{inner}</text>{children}</section></component>"
    )


def test_table_with_thead_pairs_every_cell_with_its_header():
    table = """<table><thead><tr><th>Reaction</th><th>Drug (%)</th><th>Placebo (%)</th></tr></thead>
      <tbody><tr><td>Nausea</td><td>5</td><td>2</td></tr><tr><td>Rash</td><td>1</td><td>1</td></tr></tbody></table>"""
    label = parse_spl(spl(section("34084-4", "ADVERSE REACTIONS", table)))
    rows = [b for b in label.sections[0].subsections[0].blocks]
    assert [b.text for b in rows] == [
        "Reaction: Nausea; Drug (%): 5; Placebo (%): 2",
        "Reaction: Rash; Drug (%): 1; Placebo (%): 1",
    ]
    assert all(b.kind == "row" for b in rows)


def test_table_without_thead_uses_first_row_as_header():
    table = """<table><tbody><tr><td>Dose</td><td>Frequency</td></tr>
      <tr><td>5 mg</td><td>daily</td></tr></tbody></table>"""
    label = parse_spl(spl(section("34068-7", "DOSAGE", table)))
    assert [b.text for b in label.sections[0].subsections[0].blocks] == [
        "Dose: 5 mg; Frequency: daily"
    ]


def test_table_caption_and_full_width_label_rows():
    table = """<table><caption>Table 1. Adverse reactions</caption><tbody>
      <tr><th>Event</th><th>Rate</th></tr>
      <tr><td colspan="2">Cardiovascular</td></tr>
      <tr><td>Edema</td><td>10%</td></tr></tbody></table>"""
    label = parse_spl(spl(section("34084-4", "AR", table)))
    blocks = label.sections[0].subsections[0].blocks
    assert [(b.kind, b.text) for b in blocks] == [
        ("text", "Table 1. Adverse reactions"),
        ("row", "Cardiovascular"),
        ("row", "Event: Edema; Rate: 10%"),
    ]


def test_table_colspan_header_spreads_over_columns():
    table = """<table><tbody>
      <tr><td/><td colspan="2">Drug</td></tr>
      <tr><td/><td>5 mg</td><td>10 mg</td></tr>
      <tr><td>Edema</td><td>3</td><td>10</td></tr></tbody></table>"""
    label = parse_spl(spl(section("34084-4", "AR", table)))
    assert [b.text for b in label.sections[0].subsections[0].blocks] == [
        "Edema; Drug 5 mg: 3; Drug 10 mg: 10"
    ]


def test_full_width_title_row_is_not_mistaken_for_the_header():
    table = """<table><tbody>
      <tr><td colspan="2">Adverse reactions by dose</td></tr>
      <tr><td>Event</td><td>Rate</td></tr>
      <tr><td>Edema</td><td>10%</td></tr></tbody></table>"""
    label = parse_spl(spl(section("34084-4", "AR", table)))
    assert [(b.kind, b.text) for b in label.sections[0].subsections[0].blocks] == [
        ("text", "Adverse reactions by dose"),
        ("row", "Event: Edema; Rate: 10%"),
    ]


def test_lists_flatten_to_one_block_per_item_including_nested():
    inner = """<paragraph>Avoid with:</paragraph>
      <list listType="unordered"><item>Strong inhibitors</item>
        <item>Grapefruit juice<list><item>in large amounts</item></list></item></list>"""
    label = parse_spl(spl(section("34073-7", "DI", inner)))
    assert [b.text for b in label.sections[0].subsections[0].blocks] == [
        "Avoid with:",
        "Strong inhibitors",
        "Grapefruit juice",
        "in large amounts",
    ]


def test_inline_markup_and_breaks_are_flattened():
    inner = (
        '<paragraph>Dose is 10 mg/m<sup>2</sup> <content styleCode="bold">daily</content>.'
        "<br/>Next line</paragraph>"
        '<renderMultiMedia referencedObject="x"/>'
    )
    label = parse_spl(spl(section("34068-7", "DOSAGE", inner)))
    assert [b.text for b in label.sections[0].subsections[0].blocks] == [
        "Dose is 10 mg/m2 daily. Next line"
    ]


def test_br_separated_text_without_paragraph_elements_becomes_blocks():
    inner = "First point.<br/><br/>Second point."
    label = parse_spl(spl(section("34070-3", "CONTRAINDICATIONS", inner)))
    assert [b.text for b in label.sections[0].subsections[0].blocks] == [
        "First point.",
        "Second point.",
    ]


def test_unknown_top_level_section_is_skipped_but_its_known_children_are_found():
    wrapper = section(
        "42229-5",
        "Wrapper",
        "<paragraph>ignored</paragraph>",
        children='<component><section><code code="34088-5"/><title>OVERDOSAGE</title>'
        "<text><paragraph>Hemodialysis.</paragraph></text></section></component>",
    )
    label = parse_spl(spl(wrapper))
    assert codes(label) == ["34088-5"]
    assert label.skipped_codes == {"42229-5": 1}


def test_unclassified_nested_section_without_title_inherits_parent_path():
    child = (
        '<component><section><code code="42229-5"/><title>Geriatric Use</title>'
        "<text><paragraph>Be careful.</paragraph></text>"
        '<component><section><code code="42229-5"/>'
        "<text><paragraph>Deeper.</paragraph></text></section></component></section></component>"
    )
    label = parse_spl(spl(section("43684-0", "8 USE IN SPECIFIC POPULATIONS", "", children=child)))
    assert [(s.title, s.blocks[0].text) for s in label.sections[0].subsections] == [
        ("Geriatric Use", "Be careful."),
        ("Geriatric Use", "Deeper."),
    ]


def test_missing_set_id_raises():
    with pytest.raises(SplParseError, match="setId"):
        parse_spl(b'<document xmlns="urn:hl7-org:v3"><title>x</title></document>')


def test_invalid_xml_raises():
    with pytest.raises(SplParseError, match="invalid XML"):
        parse_spl(b"<document><unclosed></document>")


# --- title-coded sections and Highlights excerpts -----------------------------------------


def test_dosage_section_coded_unclassified_is_found_by_its_title(caplog):
    """QLONILIK codes the real D&A section 42229-5; the 34068-7 section is Highlights only."""
    with caplog.at_level(logging.INFO, logger="pharmarag.ingest.spl_parser"):
        label = load("qlonilik_plr_title_coded.xml")
    dosing = [s for s in label.sections if s.loinc == "34068-7"]
    assert len(dosing) == 1  # the empty Highlights shell is not a second section
    assert dosing[0].title == "2 DOSAGE AND ADMINISTRATION"
    assert [sub.title for sub in dosing[0].subsections] == [
        "2.1 Recommended Dosage",
        "2.2 Administration Instructions",
    ]
    assert all(b.text for sub in dosing[0].subsections for b in sub.blocks)
    assert any(
        "mapped to 34068-7" in r.getMessage() and "2 DOSAGE AND ADMINISTRATION" in r.getMessage()
        for r in caplog.records
    )


def test_highlights_excerpt_text_is_never_read():
    label = load("qlonilik_plr_title_coded.xml")
    every_text = " ".join(
        b.text for s in label.sections for sub in s.subsections for b in sub.blocks
    )
    assert "See full prescribing information" not in every_text
    assert all(s.subsections for s in label.sections)  # no empty shells survive


def test_excerpt_only_section_without_full_text_is_dropped():
    body = (
        '<component><section><code code="34068-7"/><text/>'
        "<excerpt><highlight><text><paragraph>Take 5 mg daily.</paragraph></text></highlight>"
        "</excerpt></section></component>"
    )
    assert parse_spl(spl(body)).sections == []


@pytest.mark.parametrize(
    ("title", "code"),
    [
        ("2 DOSAGE AND ADMINISTRATION", "34068-7"),
        ("DOSAGE & ADMINISTRATION", "34068-7"),
        ("Dosage and Administration:", "34068-7"),
        ("1 INDICATIONS AND USAGE", "34067-9"),
        ("INDICATIONS", "34067-9"),
        ("3 DOSAGE FORMS AND STRENGTHS", "43678-2"),
        ("4 CONTRAINDICATIONS", "34070-3"),
        ("5 WARNINGS AND PRECAUTIONS", "43685-7"),
        ("WARNINGS", "34071-1"),
        ("PRECAUTIONS", "42232-9"),
        ("6 ADVERSE REACTIONS", "34084-4"),
        ("7 DRUG INTERACTIONS", "34073-7"),
        ("8 USE IN SPECIFIC POPULATIONS", "43684-0"),
        ("10 OVERDOSAGE", "34088-5"),
        ("12 CLINICAL PHARMACOLOGY", "34090-1"),
        ("16 HOW SUPPLIED/STORAGE AND HANDLING", "34069-5"),
        ("HOW SUPPLIED", "34069-5"),
        ("WARNING: FETAL TOXICITY", "34066-1"),
        ("BOXED WARNING", "34066-1"),
    ],
)
def test_code_for_title_maps_every_kept_heading(title, code):
    assert code_for_title(title) == code


@pytest.mark.parametrize(
    "title",
    [
        "",
        "5.1 Hypotension",
        "11 DESCRIPTION",
        "14 CLINICAL STUDIES",
        "DOSAGE FORMS AND DOSES",
        "Overdosage in children",
    ],
)
def test_code_for_title_rejects_other_headings(title):
    assert code_for_title(title) is None


def test_every_kept_code_is_reachable_by_title():
    from pharmarag.ingest.spl_parser import _TITLE_RULES

    assert {code for code, _ in _TITLE_RULES} == set(SECTIONS)
