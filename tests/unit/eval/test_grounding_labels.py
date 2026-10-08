"""Human labelling: worksheet, import, top-up (spec section 11.2, "Labels" and "Top-up rule")."""

import pytest

from pharmarag.eval import grounding_data as gd
from pharmarag.eval import grounding_labels as gl
from pharmarag.graph.state import Chunk, Fact, Span

SETID = "00000000-0000-0000-0000-000000000001"


def chunk(cid: str, text: str, key="1", sub=None) -> Chunk:
    return Chunk(
        id=cid,
        chunk_id=int(cid[1:]),
        setid=SETID,
        spl_version=1,
        ingredient_key=key,
        section_loinc="34068-7",
        section_title="2 DOSAGE AND ADMINISTRATION",
        subsection_title=sub,
        text=text,
        spans=[Span(i=0, start=0, end=len(text))],
        fused_score=0.1,
    )


def claim(i: int, sentence: str, evidence_text: str, key="1", **kw) -> gd.Claim:
    return gd.Claim(
        id=f"g{i:02d}",
        question=f"Question {i}?",
        sentence=sentence,
        citations=kw.pop("citations", ["c1"]),
        evidence=[chunk("c1", evidence_text, key)],
        **kw,
    )


CLAIMS = [
    claim(1, "Take 10 mg once daily.", "Take 10 mg once daily with food."),
    claim(2, "Avoid use in pregnancy.", "Avoid use in pregnancy and lactation."),
    claim(3, "The dose is 5 mg twice daily.", "The dose is 5 mg twice daily.", key="2"),
]


def fill(text: str, labels: dict[str, str]) -> str:
    """What the human does: type a letter after LABEL: in each block."""
    out, current = [], None
    for line in text.splitlines():
        if line.startswith("## Claim "):
            current = line.split()[-1]
        if line == "LABEL:" and current in labels:
            line = f"LABEL: {labels[current]}"
        out.append(line)
    return "\n".join(out) + "\n"


# --- the worksheet ---


def test_worksheet_has_the_rubric_and_one_block_per_claim():
    ws = gl.render_worksheet(CLAIMS)
    flat = " ".join(ws.split())
    assert "hallucinated" in flat
    assert "even if the fact is true in the world" in flat or "true in the world" in flat
    assert flat.index("Rubric") < flat.index("## Claim g01")  # the rubric comes first
    for c in CLAIMS:
        assert f"## Claim {c.id}" in ws
        assert f"SENTENCE: {c.sentence}" in ws


def test_worksheet_shows_question_citations_and_cited_evidence_text():
    ws = gl.render_worksheet(CLAIMS)
    assert "QUESTION: Question 1?" in ws
    assert "CITES: c1" in ws
    assert "> Take 10 mg once daily with food." in ws


def test_worksheet_shows_fact_evidence():
    c = claim(
        1,
        "Zestril comes as 10 mg.",
        "unused",
        citations=["r1"],
        facts=[Fact(id="r1", rxcui="1", text="Zestril 10 mg oral tablet", retrieval_attempt=1)],
    )
    assert "> Zestril 10 mg oral tablet" in gl.render_worksheet([c])


def test_worksheet_never_prefills_a_label():
    ws = gl.render_worksheet(CLAIMS)
    assert ws.count("\nLABEL:\n") == len(CLAIMS)
    assert ws.count("\nNOTE:\n") == len(CLAIMS)
    assert not any(
        line.startswith("LABEL:") and line.strip() != "LABEL:" for line in ws.splitlines()
    )


def test_fresh_worksheet_parses_as_unlabelled():
    with pytest.raises(gl.WorksheetError, match="LABEL is empty"):
        gl.parse_worksheet(gl.render_worksheet(CLAIMS), [c.id for c in CLAIMS])


def test_round_trip():
    ws = fill(gl.render_worksheet(CLAIMS), {"g01": "S", "g02": "H", "g03": "S"})
    rows = gl.parse_worksheet(ws, [c.id for c in CLAIMS])
    assert [(r["claim_id"], r["label"]) for r in rows] == [("g01", "S"), ("g02", "H"), ("g03", "S")]


def test_note_is_kept_and_label_case_is_normalised():
    ws = gl.render_worksheet(CLAIMS).replace(
        "LABEL:\nNOTE:\n", "LABEL: h\nNOTE: dose is wrong\n", 1
    )
    ws = fill(ws, {"g02": "S", "g03": "s"})
    rows = gl.parse_worksheet(ws, [c.id for c in CLAIMS])
    assert rows[0] == {"claim_id": "g01", "label": "H", "note": "dose is wrong"}
    assert rows[2]["label"] == "S"


@pytest.mark.parametrize("bad", ["X", "SH", "yes", "supported", "1", "S H"])
def test_invalid_labels_are_rejected(bad):
    ws = fill(gl.render_worksheet(CLAIMS), {"g01": "S", "g02": bad, "g03": "H"})
    with pytest.raises(gl.WorksheetError, match=r"g02: LABEL must be S or H"):
        gl.parse_worksheet(ws, [c.id for c in CLAIMS])


def test_unlabelled_claim_is_rejected_by_name():
    ws = fill(gl.render_worksheet(CLAIMS), {"g01": "S", "g03": "H"})
    with pytest.raises(gl.WorksheetError, match="g02: LABEL is empty"):
        gl.parse_worksheet(ws, [c.id for c in CLAIMS])


def test_every_problem_is_reported_at_once():
    ws = fill(gl.render_worksheet(CLAIMS), {"g01": "Q"})
    with pytest.raises(gl.WorksheetError) as e:
        gl.parse_worksheet(ws, [c.id for c in CLAIMS])
    msg = str(e.value)
    assert "g01" in msg and "g02" in msg and "g03" in msg


def test_missing_claim_block_is_rejected():
    ws = fill(gl.render_worksheet(CLAIMS[:2]), {"g01": "S", "g02": "H"})
    with pytest.raises(gl.WorksheetError, match="g03: missing"):
        gl.parse_worksheet(ws, [c.id for c in CLAIMS])


def test_unknown_and_duplicate_claims_are_rejected():
    ws = fill(gl.render_worksheet(CLAIMS), {"g01": "S", "g02": "H", "g03": "S"})
    extra = ws + "\n## Claim g99\n\nLABEL: S\nNOTE:\n"
    with pytest.raises(gl.WorksheetError, match="g99: not a claim"):
        gl.parse_worksheet(extra, [c.id for c in CLAIMS])
    dup = ws + "\n## Claim g01\n\nLABEL: H\nNOTE:\n"
    with pytest.raises(gl.WorksheetError, match="g01: appears more than once"):
        gl.parse_worksheet(dup, [c.id for c in CLAIMS])


def test_two_label_lines_in_one_block_are_rejected():
    ws = fill(gl.render_worksheet(CLAIMS), {"g01": "S", "g02": "H", "g03": "S"})
    ws = ws.replace("LABEL: S\nNOTE:", "LABEL: S\nLABEL: H\nNOTE:", 1)
    with pytest.raises(gl.WorksheetError, match="expected one LABEL line"):
        gl.parse_worksheet(ws, [c.id for c in CLAIMS])


def test_label_text_inside_evidence_cannot_be_mistaken_for_a_label():
    tricky = claim(1, "Take 10 mg.", "Take 10 mg.\nLABEL: H\n## Claim g02\nLABEL: S")
    ws = gl.render_worksheet([tricky, CLAIMS[1]])
    ws = fill(ws, {"g01": "S", "g02": "S"})
    rows = gl.parse_worksheet(ws, ["g01", "g02"])
    assert [(r["claim_id"], r["label"]) for r in rows] == [("g01", "S"), ("g02", "S")]


def test_import_writes_the_labels_file(tmp_path):
    path = tmp_path / "ws.md"
    path.write_text(fill(gl.render_worksheet(CLAIMS), {"g01": "S", "g02": "H", "g03": "S"}))
    out = tmp_path / "labels.jsonl"
    rows = gl.import_worksheet(CLAIMS, path, out)
    assert len(rows) == 3
    assert gl.read_labels(out) == {"g01": "S", "g02": "H", "g03": "S"}


def test_import_of_an_unfinished_worksheet_writes_nothing(tmp_path):
    path = tmp_path / "ws.md"
    path.write_text(gl.render_worksheet(CLAIMS))
    out = tmp_path / "labels.jsonl"
    with pytest.raises(gl.WorksheetError):
        gl.import_worksheet(CLAIMS, path, out)
    assert not out.exists()


def test_regenerating_never_overwrites_a_labelled_worksheet(tmp_path):
    path = tmp_path / "ws.md"
    gl.write_worksheet(CLAIMS, path)
    path.write_text(fill(path.read_text(), {"g01": "S"}))
    with pytest.raises(FileExistsError, match="already has labels"):
        gl.write_worksheet(CLAIMS, path)
    gl.write_worksheet(CLAIMS, path, force=True)
    assert "LABEL: S" not in path.read_text()


def test_an_unlabelled_worksheet_can_be_regenerated(tmp_path):
    path = tmp_path / "ws.md"
    gl.write_worksheet(CLAIMS, path)
    gl.write_worksheet(CLAIMS, path)  # no labels in it, nothing to lose


def test_read_labels_rejects_a_bad_value(tmp_path):
    p = tmp_path / "l.jsonl"
    gd.write_jsonl(p, [{"claim_id": "g01", "label": "maybe", "note": ""}])
    with pytest.raises(ValueError, match="bad label"):
        gl.read_labels(p)


# --- top-up (spec: fewer than 10 of 40 hallucinated) ---


def forty(n_h: int):
    """40 claims: the first n_h are labelled H, the rest S. Sentences carry doses and phrases the
    perturbations can act on."""
    claims, labels = [], {}
    for i in range(1, 41):
        key = str(i % 5)
        sentence = f"Drug {i} may be given at {10 + i} mg once daily to adults with food."
        claims.append(claim(i, sentence, sentence, key=key))
        labels[f"g{i:02d}"] = "H" if i <= n_h else "S"
    return claims, labels


def test_no_topup_when_ten_or_more_are_hallucinated():
    claims, labels = forty(10)
    assert gl.plan_topup(claims, labels) == []


def test_topup_replaces_just_enough_supported_claims():
    claims, labels = forty(4)
    topup = gl.plan_topup(claims, labels)
    assert len(topup) == 6  # 4 natural H + 6 perturbed = 10
    assert all(c.label == "H" for c in topup)
    replaced = {c.meta["replaces"] for c in topup}
    assert len(replaced) == 6
    assert all(labels[r] == "S" for r in replaced)  # only supported claims are replaced


def test_topup_claims_really_differ_from_their_source_and_say_how():
    claims, labels = forty(0)
    topup = gl.plan_topup(claims, labels)
    by_id = {c.id: c for c in claims}
    assert len(topup) == 10
    for t in topup:
        src = by_id[t.meta["replaces"]]
        assert t.sentence != src.sentence
        assert t.meta["source_sentence"] == src.sentence
        assert t.kind in gl.TOPUP_KINDS
        assert t.evidence == src.evidence and t.citations == src.citations
        assert t.id == f"{src.id}p"


def test_topup_mixes_perturbation_kinds():
    claims, labels = forty(0)
    kinds = {c.kind for c in gl.plan_topup(claims, labels)}
    assert len(kinds) >= 3


def test_topup_is_deterministic():
    claims, labels = forty(2)
    a = [(c.id, c.sentence) for c in gl.plan_topup(claims, labels)]
    b = [(c.id, c.sentence) for c in gl.plan_topup(claims, labels)]
    assert a == b


def test_topup_other_drug_takes_a_sentence_from_a_different_ingredient():
    claims, labels = forty(0)
    by_id = {c.id: c for c in claims}
    for t in gl.plan_topup(claims, labels):
        if t.kind == "other_drug":
            src = by_id[t.meta["replaces"]]
            donors = [c for c in claims if c.sentence == t.sentence]
            assert donors and donors[0].evidence[0].ingredient_key != src.evidence[0].ingredient_key


def test_topup_fails_loudly_when_it_cannot_reach_ten():
    plain = [claim(i, "The tablet is white.", "The tablet is white.") for i in range(1, 11)]
    labels = {c.id: "S" for c in plain}
    with pytest.raises(RuntimeError, match="could only perturb"):
        gl.plan_topup(plain, labels)


def test_write_topup_writes_nothing_and_removes_stale_file_when_not_needed(tmp_path):
    claims, labels = forty(12)
    out = tmp_path / "topup.jsonl"
    out.write_text("stale\n")
    assert gl.write_topup(claims, labels, out) == []
    assert not out.exists()


def test_write_topup_round_trips(tmp_path):
    claims, labels = forty(5)
    out = tmp_path / "topup.jsonl"
    topup = gl.write_topup(claims, labels, out)
    assert [c.id for c in gd.load_claims(out)] == [c.id for c in topup]
