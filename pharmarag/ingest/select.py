"""Label selection rules (spec section 4, "Corpus selection").

Only human prescription labels are eligible (checked by the caller against
`PRESCRIPTION_LABEL_LOINC`). Within one ingredient set the preference order is:

1. NDA or BLA marketing category (the spec's rule),
2. not a repackager label,
3. a base immediate-release oral or injectable form,
4. latest effective date (the spec's rule),
5. lowest SETID, so the choice is deterministic.

Rules 2 and 3 are tie-breaks inside the spec's tier. They exist because "latest date" alone
picks whichever repackager republished last, and the pick would change every week.

`labels_per_ingredient` (settings) is the number of labels kept per ingredient set.
"""

from __future__ import annotations

import re
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


# SPEC: repackager demotion. The spec ranks NDA/BLA, then latest date, so a repackager that
# republishes a brand's label every few weeks always wins and the chosen setid (and with it
# corpus_version) changes weekly. A labeler is a repackager if its normalised name contains
# one of these. Chosen over "labeler differs from the application holder" because DailyMed's
# list API does not return the application holder, so that rule would cost an XML download per
# candidate. The list is a heuristic: a repackager missing from it is simply not demoted.
REPACKAGERS = (
    "REMEDYREPACK",
    "BRYANT RANCH",
    "A S MEDICATION",
    "PROFICIENT RX",
    "NUCARE",
    "QUALITY CARE",
    "DIRECT RX",
    "PD RX",
    "PREFERRED PHARMACEUTICALS",
    "PHYSICIANS TOTAL CARE",
    "REBEL DISTRIBUTORS",
    "AMERICAN HEALTH PACKAGING",
    "MEDSOURCE",
    "LAKE ERIE MEDICAL",
    "ST MARY S MEDICAL PARK",
    "UNIT DOSE SERVICES",
    "NORTHWIND",
    "DENTON PHARMA",
    "HENRY SCHEIN",
    "CARDINAL HEALTH",
    "MCKESSON",
    "NCS HEALTHCARE",
    "VANGARD",
    "GSMS",
    "MEDVANTX",
    "BLENHEIM",
    "CLINICAL SOLUTIONS WHOLESALE",
    "ADVANCED RX",
    "RXCHANGE",
    "QPHARMA",
    "COUNTY LINE",
    "SAFECOR",
    "MAJOR PHARMACEUTICALS",
    "AVKARE",
    "AVPAK",
    "HJ HARKINS",
    "COUPLER",
    "GOLDEN STATE MEDICAL SUPPLY",
    "AIDAREX",
    "PACK PHARMACEUTICALS",
    "READYMEDS",
    "RPK",
    "NEW HORIZON RX",
    "MEDICAL PURCHASING SOLUTIONS",
    "REPACKAG",
    "RELABEL",
)

# SPEC: base-form preference. Extended/delayed-release products and oral liquids are niche
# formulations of an ingredient; a reference label for the ingredient should describe the
# base product (QLONILIK oral solution, Oxtellar XR and Glucotrol XL were picked for
# clonidine, oxcarbazepine and glipizide). The title carries the dose form, so no XML is
# needed. An ingredient with only niche forms still gets one, because this is only a sort key.
_ER_WORDS = re.compile(
    r"\b(EXTENDED|DELAYED|SUSTAINED|CONTROLLED|MODIFIED|LIPOSOMAL)[ -]?RELEASE\b"
    r"|\b(ER|XR|XL|SR|CR|DR|ODT)\b|\bLIPOSOM\w*"
)
_LIQUID_ETC = re.compile(
    r"\b(SOLUTION|SUSPENSION|SYRUP|ELIXIR|CONCENTRATE|POWDER|LOZENGE|GRANULE|PATCH|TOPICAL|"
    r"CREAM|OINTMENT|GEL|LOTION|SPRAY|OPHTHALMIC|OTIC|INHALATION|SUPPOSITORY|KIT|EMULSION)\b"
)
# One dose-form phrase: a solid or injectable noun plus its comma-separated descriptors.
_FORM_PHRASE = re.compile(
    r"\b(?P<noun>TABLET|CAPSULE|CAPLET|INJECTION)"
    r"(?P<desc>(?:\s*,\s*(?:FILM COATED|COATED|GELATIN COATED|SUGAR COATED|ENTERIC COATED|"
    r"LIQUID FILLED|HARD|SOFT|CHEWABLE|ORALLY DISINTEGRATING|EFFERVESCENT|SUBLINGUAL|"
    r"MULTILAYER|FOR SUSPENSION|FOR SOLUTION|POWDER|SOLUTION|SUSPENSION|EMULSION|"
    r"LYOPHILIZED|CONCENTRATE|EXTENDED RELEASE|DELAYED RELEASE|[A-Z]+ RELEASE))*)"
)
_NICHE_DESCRIPTORS = re.compile(
    r"CHEWABLE|ORALLY DISINTEGRATING|EFFERVESCENT|SUBLINGUAL|MULTILAYER|FOR SUSPENSION|"
    r"RELEASE|LIPOSOM"
)


def labeler_from_title(title: str) -> str:
    """The text of the last [..] in a DailyMed list title: the labeler."""
    m = re.search(r"\[([^\[\]]*)\]\s*$", title.strip())
    return m.group(1).strip() if m else ""


def is_repackager(labeler: str) -> bool:
    name = " ".join(re.sub(r"[^A-Z0-9]+", " ", labeler.upper()).split())
    return any(re.search(r"\b" + re.escape(r), name) for r in REPACKAGERS)


def is_base_form(title: str) -> bool:
    """True if the label covers a plain tablet, capsule or injection, judged from its title.

    A DailyMed title lists every product: "NAME (GENERIC) FORM NAME2 (GENERIC) FORM2 [LABELER]".
    Each dose-form phrase is checked on its own, together with the brand text in front of it
    (so "OXTELLAR XR (...) TABLET" is extended release although its form says only TABLET). A
    label with at least one plain product counts as base: Trileptal lists tablets and a
    suspension, Tegretol lists tablets, XR tablets and a suspension. A title with no solid or
    injectable phrase is base unless it names a liquid, patch, extended release, and so on.
    """
    head = re.sub(r"\[[^\]]*\]\s*$", "", title.upper())
    found = False
    previous_end = 0
    for m in _FORM_PHRASE.finditer(head):
        found = True
        brand = head[previous_end : m.start()]
        previous_end = m.end()
        if _ER_WORDS.search(brand) or _NICHE_DESCRIPTORS.search(m.group("desc")):
            continue
        return True
    if found:
        return False
    return not (_ER_WORDS.search(head) or _LIQUID_ETC.search(head))


@dataclass(frozen=True)
class Candidate:
    setid: str
    spl_version: int
    marketing_category: str | None  # "NDA", "BLA", "ANDA", ...
    effective_date: date | None
    ingredient_key: str
    title: str = ""  # DailyMed list title, "NAME (GENERIC) FORM [LABELER]"


def is_nda_or_bla(category: str | None) -> bool:
    return (category or "").strip().upper() in BRAND_CATEGORIES


def preference(c: Candidate) -> tuple[int, int, int, int, str]:
    """Sort key: smaller is better. A missing date sorts after every real date."""
    tier = 0 if is_nda_or_bla(c.marketing_category) else 1
    repackager = 1 if is_repackager(labeler_from_title(c.title)) else 0
    niche = 0 if is_base_form(c.title) or not c.title else 1
    ordinal = c.effective_date.toordinal() if c.effective_date else 0
    return (tier, repackager, niche, -ordinal, c.setid)


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
