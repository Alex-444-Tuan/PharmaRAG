"""Deterministic number-and-unit check (spec section 9, "Verifier, per sentence", step 2).

General NLI models often accept a sentence whose only error is a dose, so every number that
carries a unit in the sentence must appear, with the same unit, in the candidate evidence.

Normalisation, so that the check compares quantities and not spellings:

- "1,000" is 1000, ".5" is 0.5, "0.50" is 0.5, "10mg", "10 mg" and "10-mg" are the same.
- Mass is compared in mg (1 g = 1000 mg, 1 mcg = 0.001 mg), time in hours (1 day = 24 hours,
  1 week = 168 hours, 1 year = 8760 hours), volume in mL.
- A range gives one quantity per endpoint: "10-20 mg", "10 to 20 mg", "10 mg to 20 mg" all give
  10 mg and 20 mg. A list sharing one unit ("5, 10, or 20 mg") gives one quantity per number.
- Rates keep their denominator: "5 mg/kg" and "5 mg" differ (a per-kg dose is not a flat dose).

# SPEC: the spec's unit list is mg, mcg, mL, %, hours, days, years, mg/kg. This also reads
# g, weeks, mg/m2, units, IU, and long forms (milligram, percent, ...), as the lane brief says.
# SPEC: a "/day" denominator is dropped ("10 mg/day" equals "10 mg"), because writers say
# "10 mg once daily" for a label that says "10 mg/day". Other denominators are kept.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_D = Decimal

# unit spelling (lower case) -> (canonical unit, factor to the canonical unit)
_NUMERATORS: dict[str, tuple[str, Decimal]] = {
    "mg": ("mg", _D(1)),
    "milligram": ("mg", _D(1)),
    "milligrams": ("mg", _D(1)),
    "mcg": ("mg", _D("0.001")),
    "µg": ("mg", _D("0.001")),  # micro sign U+00B5
    "μg": ("mg", _D("0.001")),  # greek mu U+03BC
    "microgram": ("mg", _D("0.001")),
    "micrograms": ("mg", _D("0.001")),
    "g": ("mg", _D(1000)),
    "gram": ("mg", _D(1000)),
    "grams": ("mg", _D(1000)),
    "ml": ("ml", _D(1)),
    "milliliter": ("ml", _D(1)),
    "milliliters": ("ml", _D(1)),
    "millilitre": ("ml", _D(1)),
    "millilitres": ("ml", _D(1)),
    "%": ("%", _D(1)),
    "percent": ("%", _D(1)),
    "h": ("hour", _D(1)),
    "hr": ("hour", _D(1)),
    "hrs": ("hour", _D(1)),
    "hour": ("hour", _D(1)),
    "hours": ("hour", _D(1)),
    "day": ("hour", _D(24)),
    "days": ("hour", _D(24)),
    "week": ("hour", _D(168)),
    "weeks": ("hour", _D(168)),
    "year": ("hour", _D(8760)),
    "years": ("hour", _D(8760)),
    "yr": ("hour", _D(8760)),
    "yrs": ("hour", _D(8760)),
    "unit": ("units", _D(1)),
    "units": ("units", _D(1)),
    "iu": ("iu", _D(1)),
}

_DENOMINATORS: dict[str, str | None] = {
    "kg": "kg",
    "m2": "m2",
    "m²": "m2",
    "m^2": "m2",
    "ml": "ml",
    "dose": "dose",
    "day": None,  # dropped, see the module docstring
    "d": None,
    "hr": "hour",
    "h": "hour",
    "hour": "hour",
    "min": "min",  # mL/min is a rate (creatinine clearance), not a volume
    "minute": "min",
}

_NUM = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
_NUM_RE = re.compile(_NUM)
_UNIT_WORD = "|".join(
    sorted((re.escape(k) for k in _NUMERATORS if k != "%"), key=len, reverse=True)
)
_DEN_WORD = "|".join(sorted((re.escape(k) for k in _DENOMINATORS), key=len, reverse=True))
# A number, an optional space or hyphen, a unit, and up to two "/den" or " per den" parts.
_QUANTITY = re.compile(
    rf"(?<![\w.,])(?P<num>{_NUM})[  ]?-?[  ]?"
    rf"(?:(?P<pct>%)|(?P<unit>{_UNIT_WORD})(?![A-Za-z0-9µμ])"
    rf"(?P<dens>(?:\s*(?:/|\bper\b)\s*(?:{_DEN_WORD})(?![A-Za-z0-9µμ])){{0,2}}))",
    re.IGNORECASE,
)
_DEN_PART = re.compile(rf"(?:/|\bper\b)\s*({_DEN_WORD})", re.IGNORECASE)

# Bare numbers that share the unit of the quantity after them: "10 to 20 mg", "5, 10, or 20 mg".
_LINK = r"(?:\s*(?:-|–|—|,\s*or|,\s*and|,|\bto\b|\bor\b|\band\b)\s*)"
_PREV_NUM = re.compile(rf"(?<![\w.,])({_NUM})({_LINK})$", re.IGNORECASE)
# A number after one of these words is a label (Table 2, Study 3), not a dose.
_REFERENCE_WORD = re.compile(
    r"\b(?:table|figure|study|studies|trial|trials|week|day|days|section|visit|cohort|part"
    r"|group|arm|example|no\.|number|page|step|phase)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Quantity:
    """A number with a unit, normalised. `raw` is the text it came from (for reports)."""

    value: Decimal
    unit: str
    raw: str = ""

    @property
    def key(self) -> tuple[Decimal, str]:
        return (self.value, self.unit)


def parse_number(text: str) -> Decimal | None:
    """ "1,000" -> 1000, ".5" -> 0.5, "0.50" -> 0.5. None when it is not a number."""
    try:
        return _D(text.replace(",", "")).normalize()
    except InvalidOperation:
        return None


def _canonical(value: Decimal, unit_word: str, dens: str) -> tuple[Decimal, str]:
    unit, factor = _NUMERATORS[unit_word.lower()]
    parts = [unit]
    for m in _DEN_PART.finditer(dens or ""):
        den = _DENOMINATORS[m.group(1).lower()]
        if den is not None:
            parts.append(den)
    return (value * factor).normalize(), "/".join(parts)


_UNIT_ONLY = re.compile(
    rf"(?:%|(?:{_UNIT_WORD})(?:\s*(?:/|\bper\b)\s*(?:{_DEN_WORD})){{0,2}})", re.IGNORECASE
)
_LEADING_VALUE = re.compile(
    rf"^(\s*(?:>=|<=|[<>≥≤=~])?\s*{_NUM}(?:\s*(?:to|-|–|—|or)\s*{_NUM})*)(?=\s|$)",
    re.IGNORECASE,
)
_HEADER_UNIT = re.compile(r"\(([^()]+)\)\s*$")


def header_unit(header: str) -> str | None:
    """The unit a table column header declares: "Dose (mg/day)" -> "mg/day", "% 10 mg" -> "%"."""
    h = header.strip()
    m = _HEADER_UNIT.search(h)
    if m and _UNIT_ONLY.fullmatch(m.group(1).strip()):
        return m.group(1).strip()
    if h.startswith("%"):
        return "%"
    return None


def carry_header_units(text: str) -> str:
    """Give the bare numbers of a table cell the unit of its column header.

    The chunker writes a table row as "Header (unit): value; Header (unit): value". A reader
    sees "900 to 3600" under "Total Daily Dose Range (mg/day)" as 900 to 3600 mg/day, so the
    number check must too. "Dose Regimen (mg): 300 TID" becomes "Dose Regimen (mg): 300 mg TID".
    A cell whose value already carries a unit is left alone.
    """
    if ": " not in text:
        return text
    cells = []
    for cell in text.split("; "):
        header, sep, value = cell.partition(": ")
        unit = header_unit(header) if sep else None
        m = _LEADING_VALUE.match(value) if unit else None
        if m and not _QUANTITY.match(value.lstrip("<>≥≤=~ ")):
            value = f"{m.group(1)} {unit}{value[m.end() :]}"
        cells.append(f"{header}{sep}{value}")
    return "; ".join(cells)


def extract_quantities(text: str) -> list[Quantity]:
    """Every number-with-unit in `text`, ranges and unit-sharing lists expanded."""
    text = carry_header_units(text)
    matches = list(_QUANTITY.finditer(text))
    with_unit_starts = {m.start("num") for m in matches}
    out: list[Quantity] = []
    for m in matches:
        value = parse_number(m.group("num"))
        if value is None:
            continue
        unit_word = "%" if m.group("pct") else m.group("unit")
        dens = m.group("dens") or ""
        v, unit = _canonical(value, unit_word, dens)
        out.append(Quantity(v, unit, m.group(0)))
        # Walk back over bare numbers joined by "to", "-", "or", ",": they share this unit.
        # The link right before the unit-bearing number may not be a bare comma: in "Take 2,
        # 500 mg tablets" the 2 counts tablets. A list ("5, 10, or 20 mg") ends in or/and.
        pos = m.start("num")
        first = True
        while True:
            prev = _PREV_NUM.search(text[:pos])
            if prev is None or prev.start(1) in with_unit_starts:
                break
            if first and prev.group(2).strip() == ",":
                break
            first = False
            if _REFERENCE_WORD.search(text[: prev.start(1)]):
                break
            bare = parse_number(prev.group(1))
            if bare is None:
                break
            v2, unit2 = _canonical(bare, unit_word, dens)
            out.append(Quantity(v2, unit2, prev.group(1)))
            pos = prev.start(1)
    return out


def missing_quantities(sentence: str, evidence_texts: Iterable[str]) -> list[Quantity]:
    """Quantities in `sentence` that appear in none of `evidence_texts` (same value and unit).

    An empty result means the check passes. A sentence with no unit-bearing number always
    passes: the spec checks numbers with units, and leaves the rest to the NLI model.
    """
    wanted = extract_quantities(sentence)
    if not wanted:
        return []
    have: set[tuple[Decimal, str]] = set()
    for t in evidence_texts:
        have.update(q.key for q in extract_quantities(t))
    seen: set[tuple[Decimal, str]] = set()
    missing: list[Quantity] = []
    for q in wanted:
        if q.key not in have and q.key not in seen:
            seen.add(q.key)
            missing.append(q)
    return missing
