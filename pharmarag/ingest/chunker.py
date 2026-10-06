"""Section-aware chunker with sentence and table-row spans (spec section 4, steps 4-6).

A chunk is at most `max_tokens` tokens and starts with up to `overlap_tokens` tokens of
whole units (sentences or table rows) repeated from the end of the previous chunk. A chunk
never crosses a section or subsection boundary: each (section, subsection) pair is chunked on
its own, so the overlap never spans a boundary either.

A *unit* is one sentence or one table row. `Span(i, start, end)` marks a unit by character
offsets into the chunk text, so `text[start:end]` is exactly that sentence or row. A unit
longer than `max_tokens` is cut at word boundaries and each fragment is its own unit.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from pharmarag.graph.state import Span
from pharmarag.ingest.spl_parser import Block, ParsedLabel

MAX_TOKENS = 350
OVERLAP_TOKENS = 50

_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+")
_OPENERS = re.compile(r"[A-Z0-9\"'(\[]")
_ABBREVIATIONS = frozenset(
    "dr mr mrs ms vs no nos fig figs approx ca cf inc co corp ltd st mt vol al p pp conc "
    "e.g i.e u.s u.k".split()
)
_INITIALISM = re.compile(r"^(?:[a-z]\.)*[a-z]$", re.IGNORECASE)
_REFERENCE = re.compile(r"(?:table|figure|section|chapter|study|trial|appendix)\s+[\dA-Z]+$", re.I)


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by whitespace and a capital, digit, quote, or bracket.

    Does not split after common abbreviations ("Fig.", "e.g.", "et al."), single-letter
    initials, or references such as "Table 3.". Decimal numbers never split because no
    whitespace follows their period.
    """
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        nxt = m.end()
        if nxt >= len(text) or not _OPENERS.match(text[nxt]):
            continue
        if text[m.start()] == ".":
            before = text[start : m.start()]
            word = re.search(r"(\S+)$", before)
            token = word.group(1).lower().lstrip("([\"'") if word else ""
            if token in _ABBREVIATIONS or _INITIALISM.match(token) or _REFERENCE.search(before):
                continue
        sentence = text[start:nxt].strip()
        if sentence:
            sentences.append(sentence)
        start = nxt
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@dataclass
class _Unit:
    text: str
    sep: str  # separator placed before this unit when it is not first in a chunk
    tokens: int


@dataclass
class ChunkRecord:
    ordinal: int
    section_loinc: str
    section_title: str
    subsection_title: str | None
    text: str
    spans: list[Span]


def embed_text(drug_name: str, chunk: ChunkRecord) -> str:
    """Text that is embedded: `{drug} | {section} | {subsection}\\n{chunk text}`."""
    return f"{drug_name} | {chunk.section_title} | {chunk.subsection_title or ''}\n{chunk.text}"


def _hard_split(text: str, count: Callable[[str], int], max_tokens: int) -> list[str]:
    """Cut an over-long unit at word boundaries into fragments of at most max_tokens."""
    words = text.split(" ")
    parts: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join([*cur, w])
        if cur and count(trial) > max_tokens:
            parts.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        parts.append(" ".join(cur))
    return parts


def _units(blocks: list[Block], count: Callable[[str], int], max_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    for block in blocks:
        pieces = [block.text] if block.kind == "row" else split_sentences(block.text)
        for j, piece in enumerate(pieces):
            sep = "\n" if j == 0 else " "
            n = count(piece)
            if n <= max_tokens:
                units.append(_Unit(piece, sep, n))
                continue
            for k, frag in enumerate(_hard_split(piece, count, max_tokens)):
                units.append(_Unit(frag, sep if k == 0 else " ", count(frag)))
    return units


def _render(units: list[_Unit]) -> tuple[str, list[Span]]:
    text = ""
    spans: list[Span] = []
    for i, u in enumerate(units):
        if i:
            text += u.sep
        start = len(text)
        text += u.text
        spans.append(Span(i=i, start=start, end=len(text)))
    return text, spans


def chunk_blocks(
    blocks: list[Block],
    count_tokens: Callable[[str], int],
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[tuple[str, list[Span]]]:
    """Chunk one subsection's blocks. Returns (text, spans) per chunk.

    Every chunk after the first starts with the trailing units of the previous chunk that
    fit in `overlap_tokens`, but always leaves room for at least one new unit, so each chunk
    ends strictly after the previous one and the loop terminates.
    """
    units = _units(blocks, count_tokens, max_tokens)
    out: list[tuple[str, list[Span]]] = []
    n = len(units)
    start = prev_end = 0
    while start < n:
        # Unit token counts add up exactly for wordpiece tokenizers (whitespace is free).
        end, total = start, 0
        while end < n and (end == start or total + units[end].tokens <= max_tokens):
            total += units[end].tokens
            end += 1
        end = max(end, prev_end + 1)
        text, spans = _render(units[start:end])
        # Verify against the real tokenizer and shrink if it disagrees with the sum:
        # first drop new trailing units (keeping one new unit), then carried units.
        while count_tokens(text) > max_tokens and end > prev_end + 1 and end - start > 1:
            end -= 1
            text, spans = _render(units[start:end])
        while count_tokens(text) > max_tokens and start < prev_end:
            start += 1
            text, spans = _render(units[start:end])
        out.append((text, spans))
        if end >= n:
            break
        prev_end = end
        nxt, carried = end, 0
        while (
            nxt - 1 > start
            and carried + units[nxt - 1].tokens <= overlap_tokens
            and carried + units[nxt - 1].tokens + units[end].tokens <= max_tokens
        ):
            nxt -= 1
            carried += units[nxt].tokens
        start = nxt
    return out


def chunk_label(
    label: ParsedLabel,
    count_tokens: Callable[[str], int],
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[ChunkRecord]:
    """Chunk every kept section of a parsed label. `ordinal` runs over the whole label."""
    records: list[ChunkRecord] = []
    for section in label.sections:
        for sub in section.subsections:
            for text, spans in chunk_blocks(sub.blocks, count_tokens, max_tokens, overlap_tokens):
                records.append(
                    ChunkRecord(
                        ordinal=len(records),
                        section_loinc=section.loinc,
                        section_title=section.title,
                        subsection_title=sub.title,
                        text=text,
                        spans=spans,
                    )
                )
    return records
