"""SPL (Structured Product Labeling) XML parser (spec section 4, "Parsing and chunking").

`parse_spl` reads one label and returns its metadata plus the kept sections as plain text.
Only LOINC codes in `pharmarag.sections.SECTIONS` are kept. Unknown top-level codes are
counted in `ParsedLabel.skipped_codes` and logged at debug level.

Text rules:
- Paragraphs and list items become one `Block` each.
- A table row becomes one `Block(kind="row")`: a line of `header: value` pairs joined by "; ".
- A section's own text is the subsection whose title is None. Nested sections with a code
  that is not in SECTIONS (typically 42229-5, "unclassified") become subsections of the
  nearest kept ancestor. Their title is the chain of titles below that ancestor joined
  with " > ".
- A section coded 42229-5 (or any code outside SECTIONS) whose title is a standard heading
  ("2 DOSAGE AND ADMINISTRATION") is mapped to the matching kept code by `code_for_title`.
- <excerpt> content (Highlights duplicates) is ignored; sections left with no text are dropped.
- A nested section whose code is in SECTIONS but differs from its parent's code (for
  example Drug Interactions inside an old-format Precautions section) is emitted as its
  own section, so its text is not duplicated in the parent.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from lxml import etree

from pharmarag.sections import SECTIONS

log = logging.getLogger(__name__)

NS = "urn:hl7-org:v3"
_WS = re.compile(r"\s+")


class SplParseError(ValueError):
    """The document is not a parseable SPL label."""


@dataclass
class Block:
    """One paragraph, list item, caption, or table row."""

    text: str
    kind: Literal["text", "row"] = "text"


@dataclass
class Subsection:
    title: str | None  # None for the section's own text
    blocks: list[Block]


@dataclass
class Section:
    loinc: str
    title: str
    subsections: list[Subsection] = field(default_factory=list)


@dataclass
class ParsedLabel:
    setid: str
    version: int
    title: str
    labeler: str | None
    effective_date: date | None
    marketing_category: str | None  # displayName of the approval code, e.g. "NDA", "ANDA"
    doc_type_loinc: str
    product_name: str | None
    generic_name: str | None
    active_moieties: list[str]
    sections: list[Section]
    skipped_codes: Counter[str] = field(default_factory=Counter)


def _q(tag: str) -> str:
    return f"{{{NS}}}{tag}"


def _local(el: etree._Element) -> str:
    tag = el.tag
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _norm(text: str) -> str:
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


def _inline_text(el: etree._Element) -> str:
    """All text under `el`, with <br/> as a space and media elements ignored."""
    parts: list[str] = []

    def rec(e: etree._Element) -> None:
        if e.text:
            parts.append(e.text)
        for ch in e:
            name = _local(ch)
            if name == "br":
                parts.append(" ")
            elif name != "renderMultiMedia":
                rec(ch)
            if ch.tail:
                parts.append(ch.tail)

    rec(el)
    return _norm("".join(parts))


# --- tables -----------------------------------------------------------------------------


@dataclass
class _Cell:
    text: str
    is_th: bool


def _table_rows(table: etree._Element) -> tuple[list[list[_Cell]], int]:
    """Rows with colspans expanded (the cell text is repeated across the span).

    Returns (rows, number_of_thead_rows). thead rows come first in the list.
    """
    thead_rows: list[list[_Cell]] = []
    body_rows: list[list[_Cell]] = []
    for tr in table.iter(_q("tr")):
        cells: list[_Cell] = []
        for c in tr:
            name = _local(c)
            if name not in ("td", "th"):
                continue
            text = _inline_text(c)
            try:
                span = max(1, int(c.get("colspan", "1")))
            except ValueError:
                span = 1
            cells.extend(_Cell(text, name == "th") for _ in range(span))
        in_head = any(_local(a) == "thead" for a in tr.iterancestors())
        (thead_rows if in_head else body_rows).append(cells)
    return thead_rows + body_rows, len(thead_rows)


def _is_empty(row: list[_Cell]) -> bool:
    return not any(c.text for c in row)


def _merge_headers(header_rows: list[list[_Cell]], width: int) -> list[str]:
    headers: list[str] = []
    for i in range(width):
        parts: list[str] = []
        for row in header_rows:
            t = row[i].text if i < len(row) else ""
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        headers.append(" ".join(parts))
    return headers


def table_to_blocks(table: etree._Element) -> list[Block]:
    """Caption as a text block, then one `row` block per non-empty data row."""
    out: list[Block] = []
    caption = table.find(_q("caption"))
    if caption is not None:
        cap = _inline_text(caption)
        if cap:
            out.append(Block(cap))

    rows, n_thead = _table_rows(table)
    rows = [r for r in rows]
    # Drop leading empty rows (they carry no header information).
    start = 0
    while start < len(rows) and _is_empty(rows[start]):
        start += 1
    if start >= len(rows):
        return out
    n_thead = max(0, n_thead - start)
    rows = rows[start:]

    def distinct(row: list[_Cell]) -> set[str]:
        return {c.text for c in row if c.text}

    # A first row with one text repeated across every column is a title, not a header.
    if (
        len(rows) > 1
        and len(rows[0]) > 1
        and all(c.text for c in rows[0])
        and len(distinct(rows[0])) == 1
    ):
        out.append(Block(next(iter(distinct(rows[0])))))
        rows = rows[1:]
        n_thead = max(0, n_thead - 1)
        while rows and _is_empty(rows[0]):
            rows = rows[1:]
            n_thead = max(0, n_thead - 1)
        if not rows:
            return out

    width = max(len(r) for r in rows)
    if n_thead > 0:
        n_header = n_thead
    elif all(c.is_th for c in rows[0] if c.text):
        n_header = 1
        while (
            n_header < len(rows)
            and not _is_empty(rows[n_header])
            and all(c.is_th for c in rows[n_header] if c.text)
        ):
            n_header += 1
    else:
        n_header = 1
        # A blank stub cell in the first column means the header continues on the next rows.
        if not rows[0][0].text:
            while (
                n_header < len(rows)
                and not _is_empty(rows[n_header])
                and not rows[n_header][0].text
            ):
                n_header += 1
    if len(rows) == 1:
        n_header = 0  # a one-row table has nothing to pair against

    headers = _merge_headers(rows[:n_header], width)
    for row in rows[n_header:]:
        if _is_empty(row):
            continue
        uniq = distinct(row)
        if len(uniq) == 1:  # a full-width label row such as "Cardiovascular"
            out.append(Block(next(iter(uniq)), "row"))
            continue
        pairs: list[str] = []
        for i, cell in enumerate(row):
            if not cell.text:
                continue
            h = headers[i] if i < len(headers) else ""
            pairs.append(f"{h}: {cell.text}" if h and h != cell.text else cell.text)
        if pairs:
            out.append(Block("; ".join(pairs), "row"))
    return out


# --- text -------------------------------------------------------------------------------


def text_blocks(container: etree._Element) -> list[Block]:
    """Flatten a <text> (or <item>, <td>) element into blocks, in document order."""
    out: list[Block] = []
    buf: list[str] = []

    def flush() -> None:
        t = _norm("".join(buf))
        buf.clear()
        if t:
            out.append(Block(t))

    def has_block_child(e: etree._Element) -> bool:
        return any(_local(d) in ("paragraph", "list", "table") for d in e.iterdescendants())

    def visit(parent: etree._Element) -> None:
        if parent.text:
            buf.append(parent.text)
        for ch in parent:
            name = _local(ch)
            if name == "paragraph":
                flush()
                t = _inline_text(ch)
                if t:
                    out.append(Block(t))
            elif name == "list":
                flush()
                for sub in ch:
                    sname = _local(sub)
                    if sname == "caption":
                        t = _inline_text(sub)
                        if t:
                            out.append(Block(t))
                    elif sname == "item":
                        out.extend(text_blocks(sub))
            elif name == "table":
                flush()
                out.extend(table_to_blocks(ch))
            elif name == "br":
                flush()
            elif name == "renderMultiMedia":
                pass
            elif has_block_child(ch):
                flush()
                visit(ch)
                flush()
            else:
                buf.append(_inline_text(ch))
            if ch.tail:
                buf.append(ch.tail)

    visit(container)
    flush()
    return out


# --- sections ---------------------------------------------------------------------------


def _section_code(sec: etree._Element) -> str | None:
    code = sec.find(_q("code"))
    return code.get("code") if code is not None else None


def _section_title(sec: etree._Element) -> str:
    title = sec.find(_q("title"))
    return _inline_text(title) if title is not None else ""


def _child_sections(sec: etree._Element) -> list[etree._Element]:
    return [s for c in sec.iterchildren(_q("component")) for s in c.iterchildren(_q("section"))]


def _own_blocks(sec: etree._Element) -> list[Block]:
    text = sec.find(_q("text"))
    return text_blocks(text) if text is not None else []


_TITLE_RULES: list[tuple[str, re.Pattern[str]]] = [
    (code, re.compile(pattern))
    for code, pattern in [
        ("34066-1", r"(BOXED WARNING|WARNING:.*)"),
        ("34067-9", r"INDICATIONS?( AND USAGE)?"),
        ("34068-7", r"DOSAGE( ?(AND|/) ?ADMINISTRATION)?"),
        ("43678-2", r"DOSAGE FORMS( AND STRENGTHS)?"),
        ("34070-3", r"CONTRAINDICATIONS?"),
        ("43685-7", r"WARNINGS AND PRECAUTIONS"),
        ("34071-1", r"WARNINGS"),
        ("42232-9", r"PRECAUTIONS"),
        ("34084-4", r"ADVERSE (REACTIONS|EVENTS)"),
        ("34073-7", r"DRUG INTERACTIONS"),
        ("43684-0", r"USE IN SPECIFIC POPULATIONS"),
        ("34088-5", r"OVERDOSAGE"),
        ("34090-1", r"CLINICAL PHARMACOLOGY"),
        ("34069-5", r"HOW SUPPLIED.*"),
    ]
]
_LEADING_NUMBER = re.compile(r"^\s*\d+(?:\s*\.\s*\d+)*\s*\.?\s*")


def code_for_title(title: str) -> str | None:
    """Map a section title to a kept LOINC code, or None.

    Some labels code a real section 42229-5 ("unclassified") and give the heading only in the
    title, for example "2 DOSAGE AND ADMINISTRATION". The title is upper-cased, a leading
    section number is dropped ("2", "5.1", "16."), "&" becomes "AND", and trailing
    punctuation is stripped. The whole title must match one rule, so "DOSAGE FORMS AND
    STRENGTHS" never lands in Dosage and administration.
    """
    t = _LEADING_NUMBER.sub("", title.upper().replace("&", " AND "))
    t = " ".join(t.split()).rstrip(" :.-")
    for code, pattern in _TITLE_RULES:
        if pattern.fullmatch(t):
            return code
    return None


def _walk(sec: etree._Element, out: list[Section], skipped: Counter[str]) -> None:
    """Add `sec` (and, for sections that are not kept, their descendants) to `out`.

    Only the section's own <text> is read. <excerpt> blocks (the Highlights duplicate of a
    section) are never read, so a Highlights-only section has no content and is dropped by
    `parse_spl`.
    """
    code = _section_code(sec)
    if code not in SECTIONS:
        mapped = code_for_title(_section_title(sec))
        if mapped is None:
            skipped[code or "none"] += 1
            log.debug("skipping section with code %s (%s)", code, _section_title(sec)[:60])
            for child in _child_sections(sec):
                _walk(child, out, skipped)
            return
        log.info(
            "section coded %s titled %r mapped to %s (%s) by title",
            code,
            _section_title(sec)[:60],
            mapped,
            SECTIONS[mapped],
        )
        code = mapped

    section = Section(loinc=code, title=_section_title(sec) or SECTIONS[code])
    out.append(section)
    blocks = _own_blocks(sec)
    if blocks:
        section.subsections.append(Subsection(None, blocks))

    def gather(parent: etree._Element, path: list[str]) -> None:
        for child in _child_sections(parent):
            ccode = _section_code(child)
            if ccode in SECTIONS and ccode != code:
                _walk(child, out, skipped)
                continue
            title = _section_title(child)
            child_path = path + [title] if title else path
            blocks = _own_blocks(child)
            if blocks:
                section.subsections.append(Subsection(" > ".join(child_path) or None, blocks))
            gather(child, child_path)

    gather(sec, [])


# --- document ---------------------------------------------------------------------------


def _text_of(root: etree._Element, path: str) -> str | None:
    el = root.find(path, {"h": NS})
    if el is None:
        return None
    t = _inline_text(el)
    return t or None


def _parse_date(value: str | None) -> date | None:
    if not value or len(value) < 8 or not value[:8].isdigit():
        return None
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except ValueError:
        return None


def _dailymed_style_title(
    product: str | None, generic: str | None, form: str | None, labeler: str | None
) -> str | None:
    """Rebuild the title DailyMed shows in lists: NAME (GENERIC) FORM [LABELER]."""
    name = product or generic
    if not name:
        return None
    if product and generic and product.lower() != generic.lower():
        name = f"{product} ({generic})"
    parts = [name]
    if form:
        parts.append(form)
    if labeler:
        parts.append(f"[{labeler}]")
    return " ".join(parts).upper()


def parse_spl(xml: bytes | str) -> ParsedLabel:
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    try:
        root = etree.fromstring(xml, etree.XMLParser(recover=False, huge_tree=True))
    except etree.XMLSyntaxError as e:
        raise SplParseError(f"invalid XML: {e}") from e

    ns = {"h": NS}
    setid_el = root.find("h:setId", ns)
    if setid_el is None or not setid_el.get("root"):
        raise SplParseError("no setId")
    version_el = root.find("h:versionNumber", ns)
    try:
        version = int(version_el.get("value")) if version_el is not None else 1
    except (TypeError, ValueError) as e:
        raise SplParseError("bad versionNumber") from e
    doc_code = root.find("h:code", ns)
    eff = root.find("h:effectiveTime", ns)

    labeler = _text_of(root, "h:author/h:assignedEntity/h:representedOrganization/h:name")
    approval = root.find(".//h:approval/h:code", ns)
    marketing = approval.get("displayName") if approval is not None else None

    product = generic = form = None
    moieties: list[str] = []
    for mp in root.iterfind(".//h:manufacturedProduct", ns):
        pname = mp.find("h:name", ns)
        if pname is None:
            continue
        product = _inline_text(pname) or None
        gname = mp.find("h:asEntityWithGeneric/h:genericMedicine/h:name", ns)
        generic = _inline_text(gname) if gname is not None else None
        fc = mp.find("h:formCode", ns)
        form = fc.get("displayName") if fc is not None else None
        for ing in mp.iterfind("h:ingredient", ns):
            if ing.get("classCode") not in ("ACTIB", "ACTIM", "ACTIR"):
                continue
            nm = ing.find("h:ingredientSubstance/h:activeMoiety/h:activeMoiety/h:name", ns)
            if nm is not None and _inline_text(nm):
                moieties.append(_inline_text(nm))
        break

    title = _dailymed_style_title(product, generic, form, labeler) or _text_of(root, "h:title")
    if not title:
        raise SplParseError("no title or product name")

    sections: list[Section] = []
    skipped: Counter[str] = Counter()
    body = root.find("h:component/h:structuredBody", ns)
    if body is not None:
        for comp in body.iterchildren(_q("component")):
            for sec in comp.iterchildren(_q("section")):
                _walk(sec, sections, skipped)

    return ParsedLabel(
        setid=setid_el.get("root").lower(),
        version=version,
        title=title,
        labeler=labeler,
        effective_date=_parse_date(eff.get("value") if eff is not None else None),
        marketing_category=marketing,
        doc_type_loinc=doc_code.get("code") if doc_code is not None else "",
        product_name=product,
        generic_name=generic,
        active_moieties=list(dict.fromkeys(moieties)),
        sections=[s for s in sections if s.subsections],  # drop Highlights-only shells
        skipped_codes=skipped,
    )
