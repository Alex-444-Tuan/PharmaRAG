"""Label selection rules (spec section 4, "Corpus selection").

Only human prescription labels are eligible (checked by the caller against
`PRESCRIPTION_LABEL_LOINC`). Within one ingredient set the preference order is:

1. NDA or BLA marketing category,
2. latest effective date,
3. lowest SETID, so the choice is deterministic.

`labels_per_ingredient` (settings) is the number of labels kept per ingredient set.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

BRAND_CATEGORIES = frozenset({"NDA", "BLA"})


def ingredient_key(rxcuis: Iterable[str]) -> str:
    """Sorted, de-duplicated ingredient RxCUIs joined by '+'. Sorted numerically.

    A single ingredient yields "6809"; a combination yields "6809+593411". The spec's own
    example orders 6809 before 593411, so the sort is numeric, not lexical. The Cycle 2
    resolver must build keys with this function.
    """
    return "+".join(sorted(set(rxcuis), key=int))


@dataclass(frozen=True)
class Candidate:
    setid: str
    spl_version: int
    marketing_category: str | None  # "NDA", "BLA", "ANDA", ...
    effective_date: date | None
    ingredient_key: str


def is_nda_or_bla(category: str | None) -> bool:
    return (category or "").strip().upper() in BRAND_CATEGORIES


def preference(c: Candidate) -> tuple[int, int, str]:
    """Sort key: smaller is better. A missing date sorts after every real date."""
    tier = 0 if is_nda_or_bla(c.marketing_category) else 1
    ordinal = c.effective_date.toordinal() if c.effective_date else 0
    return (tier, -ordinal, c.setid)


def rank(candidates: Iterable[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=preference)


def group_by_key(candidates: Iterable[Candidate]) -> dict[str, list[Candidate]]:
    """Candidates per ingredient key, each list ranked best first."""
    groups: dict[str, list[Candidate]] = {}
    for c in rank(candidates):
        groups.setdefault(c.ingredient_key, []).append(c)
    return groups


def select_labels(candidates: Iterable[Candidate], per_key: int = 1) -> list[Candidate]:
    """Best `per_key` candidates for every ingredient set, ordered by key then rank."""
    out: list[Candidate] = []
    groups = group_by_key(candidates)
    for key in sorted(groups, key=lambda k: [int(p) for p in k.split("+")]):
        out.extend(groups[key][:per_key])
    return out
