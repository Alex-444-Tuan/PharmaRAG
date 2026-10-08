"""LineSplitter: the one place that decides where the writer's lines end, for both paths."""

import pytest

from pharmarag.graph.state import Chunk
from pharmarag.grounding.writer import LineSplitter, parse_line, parse_writer_output
from pharmarag.llm.errors import TRUNCATION_MARKER
from tests.unit.grounding.test_writer import EVIDENCE, FACTS


def feed_all(pieces: list[str]) -> tuple[list[list[str]], list[str], LineSplitter]:
    """(lines released after each piece, lines returned by finish, the splitter)."""
    splitter = LineSplitter()
    per_piece = [splitter.feed(p) for p in pieces]
    return per_piece, splitter.finish(), splitter


def test_a_line_is_held_until_the_next_line_starts():
    per_piece, rest, _ = feed_all(["A is true. [c1]", "\n", "B is", " true. [c1]"])
    assert per_piece == [[], [], ["A is true. [c1]"], []]
    assert rest == ["B is true. [c1]"]


def test_a_line_followed_only_by_blank_text_is_not_released_early():
    per_piece, rest, _ = feed_all(["A. [c1]\n", "\n", "  \n", "B. [c1]"])
    assert per_piece[:3] == [[], [], []]
    assert per_piece[3] == ["A. [c1]"] and rest == ["B. [c1]"]


def test_pieces_may_carry_several_newlines_and_split_mid_word():
    per_piece, rest, _ = feed_all(["A. [c1]\nB. [c", "1]\nC. [c1]"])
    assert per_piece == [["A. [c1]"], ["B. [c1]"]]  # B is released once C has started
    assert rest == ["C. [c1]"]


def test_truncation_marker_drops_the_cut_off_line_and_nothing_before_it():
    per_piece, rest, sp = feed_all(["A. [c1]\nB. [c1]\nC. is cut of", TRUNCATION_MARKER])
    released = [line for lines in per_piece for line in lines]
    assert released == ["A. [c1]", "B. [c1]"]
    assert rest == [] and sp.truncated


def test_a_line_before_the_marker_is_never_released_as_complete():
    # The marker starts with a newline, so "B. [c1]" looks finished when its newline arrives.
    # It must not be handed out until the marker is ruled out.
    splitter = LineSplitter()
    assert splitter.feed("A. [c1]\nB. [c1]") == ["A. [c1]"]
    assert splitter.feed("\n") == []  # B waits
    assert splitter.feed("<<TRUNCATED_AT") == []  # looks like the start of the marker: still waits
    assert splitter.feed("_MAX_TOKENS>>") == []  # marker complete: B is the cut-off line, dropped
    assert splitter.truncated and splitter.finish() == []


def test_a_partial_marker_that_turns_out_to_be_text_releases_the_line():
    splitter = LineSplitter()
    assert splitter.feed("A. [c1]\n<<not the marker") == ["A. [c1]"]


def test_text_after_the_marker_is_ignored():
    splitter = LineSplitter()
    splitter.feed(f"A. [c1]\nB.{TRUNCATION_MARKER}")
    assert splitter.feed("late text\nmore") == []


@pytest.mark.parametrize(
    "text",
    [
        "A. [c1]\nB. [c2]\nC. [r1]",
        "A. [c1]\n\n\nB. [c2]\n",
        "INSUFFICIENT",
        "- A. [c1]\n- B. [c2]",
        "A. [c1] B. [c2]\nno citation here",
        f"A. [c1]\nB. [c2]\nC. cut{TRUNCATION_MARKER}",
        f"A. [c1]\n{TRUNCATION_MARKER}",
        f"A. [c1]\n\n{TRUNCATION_MARKER}",
        "",
    ],
)
@pytest.mark.parametrize("step", [1, 3, 7, 1000])
def test_any_chunking_of_the_stream_gives_the_batch_parsers_sentences(text, step):
    """Streaming sentences = batch sentences, whatever the token boundaries (mode equivalence)."""
    valid = {c.id for c in EVIDENCE} | {f.id for f in FACTS}
    splitter = LineSplitter()
    lines: list[str] = []
    for i in range(0, len(text), step):
        lines += splitter.feed(text[i : i + step])
    lines += splitter.finish()
    streamed = [s for line in lines for s in parse_line(line, valid)]
    batch = parse_writer_output(text, EVIDENCE, FACTS).sentences
    assert streamed == batch


def test_parse_line_drops_the_insufficient_line_and_checks_citations():
    valid = {"c1"}
    assert parse_line("INSUFFICIENT.", valid) == []
    only = parse_line("Take it. [c1][c9]", valid)
    assert [s.citations for s in only] == [["c1"]]
    assert parse_line("Take it.", valid)[0].reason == "no_citation"


def test_evidence_fixture_is_what_the_equivalence_test_assumes():
    assert all(isinstance(c, Chunk) for c in EVIDENCE) and FACTS[0].id == "r1"
