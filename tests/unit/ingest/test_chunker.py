"""Chunker: size limit, overlap, section boundaries, and exact span offsets."""

from pathlib import Path

import pytest

from pharmarag.ingest.chunker import (
    MAX_TOKENS,
    OVERLAP_TOKENS,
    ChunkRecord,
    chunk_blocks,
    chunk_label,
    embed_text,
    split_sentences,
)
from pharmarag.ingest.spl_parser import Block, Subsection, parse_spl
from pharmarag.models.embedder import FakeEmbedder

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "spl"
count = FakeEmbedder().count_tokens  # whitespace-separated words


def sentence(i: int, words: int = 10) -> str:
    """A sentence of exactly `words` words, unique per i."""
    return " ".join([f"S{i}", *["w"] * (words - 2), "end."])


def span_texts(text: str, spans) -> list[str]:
    return [text[s.start : s.end] for s in spans]


# --- sentence splitting -------------------------------------------------------------------


def test_split_basic():
    assert split_sentences("One thing. Another thing! Is it? Yes.") == [
        "One thing.",
        "Another thing!",
        "Is it?",
        "Yes.",
    ]


def test_split_keeps_abbreviations_decimals_and_references_together():
    text = "Use e.g. 1.5 mg daily (see Fig. 2). Take with food. See Table 3. Results vary."
    assert split_sentences(text) == [
        "Use e.g. 1.5 mg daily (see Fig. 2).",
        "Take with food.",
        "See Table 3. Results vary.",
    ]


def test_split_requires_capital_or_digit_after_boundary():
    assert split_sentences("Dose is 5 mg. approximately daily.") == [
        "Dose is 5 mg. approximately daily."
    ]


def test_split_after_closing_paren_and_quote():
    assert split_sentences('It may occur (rarely). Stop the drug. "Seek help." Call.') == [
        "It may occur (rarely).",
        "Stop the drug.",
        '"Seek help."',
        "Call.",
    ]


def test_split_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# --- size, overlap, spans -----------------------------------------------------------------


def long_blocks(n_sentences: int = 120, words: int = 10) -> list[Block]:
    # Four sentences per paragraph so both separators (" " and "\n") are exercised.
    blocks = []
    for p in range(0, n_sentences, 4):
        blocks.append(Block(" ".join(sentence(i, words) for i in range(p, p + 4))))
    return blocks


def test_every_chunk_is_within_max_tokens():
    chunks = chunk_blocks(long_blocks(), count, max_tokens=35, overlap_tokens=12)
    assert len(chunks) > 5
    assert all(count(text) <= 35 for text, _ in chunks)


def test_default_limits_are_350_and_50():
    assert (MAX_TOKENS, OVERLAP_TOKENS) == (350, 50)
    blocks = [Block(" ".join(sentence(i, 17) for i in range(j, j + 6))) for j in range(0, 600, 6)]
    chunks = chunk_blocks(blocks, count)
    assert len(chunks) > 1
    assert max(count(t) for t, _ in chunks) <= 350
    assert max(count(t) for t, _ in chunks) > 300  # chunks are actually filled


def test_consecutive_chunks_overlap_by_whole_units_up_to_the_limit():
    chunks = chunk_blocks(long_blocks(), count, max_tokens=35, overlap_tokens=12)
    for (prev_text, prev_spans), (text, spans) in zip(chunks, chunks[1:], strict=False):
        prev_units = span_texts(prev_text, prev_spans)
        units = span_texts(text, spans)
        shared = [u for u in units if u in prev_units]
        assert shared, "chunks must overlap"
        # the shared units are exactly the tail of the previous chunk and the head of this one
        assert prev_units[-len(shared) :] == shared == units[: len(shared)]
        assert sum(count(u) for u in shared) <= 12
        assert len(units) > len(shared), "each chunk adds at least one new unit"


def test_zero_overlap_means_no_shared_units():
    chunks = chunk_blocks(long_blocks(), count, max_tokens=35, overlap_tokens=0)
    seen: set[str] = set()
    for text, spans in chunks:
        units = set(span_texts(text, spans))
        assert not units & seen
        seen |= units


def test_overlap_is_skipped_when_the_last_unit_is_larger_than_the_overlap():
    blocks = [Block(sentence(i, 30)) for i in range(6)]  # 30-token sentences
    chunks = chunk_blocks(blocks, count, max_tokens=70, overlap_tokens=20)
    firsts = [span_texts(t, s) for t, s in chunks]
    assert firsts == [
        [sentence(0, 30), sentence(1, 30)],
        [sentence(2, 30), sentence(3, 30)],
        [sentence(4, 30), sentence(5, 30)],
    ]


def test_all_units_are_covered_in_order_and_loop_terminates():
    blocks = long_blocks(80)
    chunks = chunk_blocks(blocks, count, max_tokens=35, overlap_tokens=12)
    expected = [s for b in blocks for s in split_sentences(b.text)]
    seen: list[str] = []
    for text, spans in chunks:
        for u in span_texts(text, spans):
            if u not in seen:
                seen.append(u)
    assert seen == expected


def test_spans_slice_exactly_each_sentence_and_row():
    blocks = [
        Block("First sentence here. Second one follows. Third."),
        Block("Edema; 5 mg: 3. 10 mg: 10", kind="row"),  # contains ". " but is one row
        Block("Fourth sentence after the table."),
    ]
    [(text, spans)] = chunk_blocks(blocks, count)
    assert span_texts(text, spans) == [
        "First sentence here.",
        "Second one follows.",
        "Third.",
        "Edema; 5 mg: 3. 10 mg: 10",
        "Fourth sentence after the table.",
    ]
    assert [s.i for s in spans] == [0, 1, 2, 3, 4]
    # sentences of one paragraph are joined by a space, blocks and rows by a newline
    assert text == (
        "First sentence here. Second one follows. Third.\n"
        "Edema; 5 mg: 3. 10 mg: 10\n"
        "Fourth sentence after the table."
    )


def test_spans_are_exact_in_every_chunk_after_overlap():
    blocks = long_blocks(60) + [Block("Row a; b: 1", kind="row")] * 5
    for text, spans in chunk_blocks(blocks, count, max_tokens=35, overlap_tokens=12):
        assert [s.i for s in spans] == list(range(len(spans)))
        for s in spans:
            piece = text[s.start : s.end]
            assert piece and piece == piece.strip()
        assert spans[0].start == 0 and spans[-1].end == len(text)


def test_oversize_unit_is_cut_at_word_boundaries_into_fragments():
    big = " ".join(f"word{i}" for i in range(100)) + "."
    chunks = chunk_blocks([Block(big)], count, max_tokens=30, overlap_tokens=0)
    fragments = [u for t, s in chunks for u in span_texts(t, s)]
    assert all(count(t) <= 30 for t, _ in chunks)
    assert " ".join(fragments) == big


def test_empty_blocks_make_no_chunks():
    assert chunk_blocks([], count) == []


def test_chunking_is_deterministic():
    blocks = long_blocks(50)
    assert chunk_blocks(blocks, count, 35, 12) == chunk_blocks(blocks, count, 35, 12)


# --- whole labels -------------------------------------------------------------------------


def units_of(sub: Subsection) -> set[str]:
    return {
        b.text if b.kind == "row" else s
        for b in sub.blocks
        for s in ([b.text] if b.kind == "row" else split_sentences(b.text))
    }


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
def test_real_labels_never_cross_a_section_or_subsection_and_respect_limits(name):
    label = parse_spl((FIXTURES / name).read_bytes())
    chunks = chunk_label(label, count)
    assert chunks
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    sources: dict[tuple[str, str, str | None], set[str]] = {}
    for section in label.sections:
        for sub in section.subsections:
            sources.setdefault((section.loinc, section.title, sub.title), set()).update(
                units_of(sub)
            )
    for c in chunks:
        assert count(c.text) <= MAX_TOKENS
        own = sources[(c.section_loinc, c.section_title, c.subsection_title)]
        pieces = span_texts(c.text, c.spans)
        # every unit in the chunk belongs to the chunk's own subsection, i.e. nothing was
        # merged across a section or subsection boundary
        assert set(pieces) <= own, (c.section_title, c.subsection_title)
        assert [s.i for s in c.spans] == list(range(len(c.spans)))


def test_chunks_from_two_sections_are_never_merged_even_when_both_are_tiny():
    label = parse_spl(
        b"""<document xmlns="urn:hl7-org:v3"><code code="34391-3"/><title>t</title>
        <setId root="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"/><versionNumber value="1"/>
        <component><structuredBody>
        <component><section><code code="34070-3"/><title>CONTRAINDICATIONS</title>
          <text><paragraph>None.</paragraph></text></section></component>
        <component><section><code code="34088-5"/><title>OVERDOSAGE</title>
          <text><paragraph>Call poison control.</paragraph></text></section></component>
        </structuredBody></component></document>"""
    )
    chunks = chunk_label(label, count)
    assert [(c.section_loinc, c.text) for c in chunks] == [
        ("34070-3", "None."),
        ("34088-5", "Call poison control."),
    ]


def test_chunk_label_without_overlap_across_boundary():
    # The overlap tail of one subsection must not appear at the head of the next.
    label = parse_spl(
        b"""<document xmlns="urn:hl7-org:v3"><code code="34391-3"/><title>t</title>
        <setId root="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"/><versionNumber value="1"/>
        <component><structuredBody><component><section><code code="34084-4"/><title>AR</title>
        <text><paragraph>Alpha one. Alpha two. Alpha three.</paragraph></text>
        <component><section><code code="42229-5"/><title>Sub</title>
        <text><paragraph>Beta one. Beta two.</paragraph></text></section></component>
        </section></component></structuredBody></component></document>"""
    )
    chunks = chunk_label(label, count, max_tokens=4, overlap_tokens=2)
    for c in chunks:
        words = c.text.split()
        assert not ({"Alpha", "Beta"} <= set(words))
    assert {c.subsection_title for c in chunks} == {None, "Sub"}


# --- embed text ---------------------------------------------------------------------------


def test_embed_text_format_with_and_without_subsection():
    base = dict(ordinal=0, section_loinc="34084-4", section_title="6 ADVERSE REACTIONS", spans=[])
    with_sub = ChunkRecord(subsection_title="6.1 Clinical Trials", text="Edema is common.", **base)
    no_sub = ChunkRecord(subsection_title=None, text="Edema is common.", **base)
    assert embed_text("amlodipine", with_sub) == (
        "amlodipine | 6 ADVERSE REACTIONS | 6.1 Clinical Trials\nEdema is common."
    )
    assert (
        embed_text("amlodipine", no_sub) == "amlodipine | 6 ADVERSE REACTIONS | \nEdema is common."
    )
