"""Label selection rules (spec section 4, "Corpus selection").

Only human prescription labels are eligible (checked by the caller against
`PRESCRIPTION_LABEL_LOINC`). Within one ingredient set the preference order is:

1. base form: an immediate-release oral tablet or capsule; an injection counts as base only
   when the ingredient has no such oral label (user decision, deviates from spec section 4),
2. not a repackager label (orchestrator decision),
3. NDA or BLA marketing category (the spec's first rule),
4. latest effective date (the spec's second rule),
5. lowest SETID, so the choice is deterministic.

Rules 1 and 2 are not in the spec and both outrank its NDA/BLA tier. Rule 2 exists because
"latest date" alone picks whichever repackager republished last, and the pick would change
every week; a manufacturer's label is also the better reference even when a repackager's copy
carries the NDA category (a state pharmacy's clonidine tablet label was filed as NDA). Rule 1
outranks the tier because the reference label should represent typical use, and an NDA niche
form misrepresents it: for clonidine the only NDA labels were an epidural injection and an oral
solution, and bupropion and glipizide had only extended-release NDA labels, while ordinary
ANDA tablets exist.

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
    "STATE OF FLORIDA",
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


def form_class(title: str) -> str:
    """ "oral", "injectable" or "niche", judged from the DailyMed list title.

    A title lists every product: "NAME (GENERIC) FORM NAME2 (GENERIC) FORM2 [LABELER]". Each
    dose-form phrase is checked on its own, together with the brand text in front of it (so
    "OXTELLAR XR (...) TABLET" is extended release although its form says only TABLET). A label
    with at least one plain tablet or capsule is "oral": Trileptal lists tablets and a
    suspension, Tegretol lists tablets, XR tablets and a suspension. Otherwise a plain
    injection makes it "injectable". A title with no tablet, capsule or injection phrase is
    "oral" unless it names a liquid, patch, extended release and so on, which is "niche".
    """
    head = re.sub(r"\[[^\]]*\]\s*$", "", title.upper())
    found = injectable = False
    previous_end = 0
    for m in _FORM_PHRASE.finditer(head):
        found = True
        brand = head[previous_end : m.start()]
        previous_end = m.end()
        if _ER_WORDS.search(brand) or _NICHE_DESCRIPTORS.search(m.group("desc")):
            continue
        if m.group("noun") == "INJECTION":
            injectable = True
            continue
        return "oral"
    if injectable:
        return "injectable"
    if found:
        return "niche"
    return "niche" if (_ER_WORDS.search(head) or _LIQUID_ETC.search(head)) else "oral"


def is_base_form(title: str) -> bool:
    """True for a plain oral solid or an injection (see `form_class`)."""
    return form_class(title) != "niche"


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


def has_oral_base(candidates: Iterable[Candidate]) -> bool:
    """True if any candidate is a plain oral tablet or capsule label.

    A candidate with no title counts as oral. The pool may include combination products, which
    can make an injection-only ingredient look like it has an oral form. The cost is only a
    slightly different order among its injection and niche labels, because candidates with
    the wrong ingredient set are skipped later anyway.
    """
    return any(form_class(c.title) == "oral" for c in candidates)


def preference(c: Candidate, injection_is_base: bool = False) -> tuple[int, int, int, int, str]:
    """Sort key: smaller is better. A missing date sorts after every real date.

    SPEC: deviates from spec section 4 by user decision. The form rule, then the repackager
    rule, come before the NDA/BLA tier. See the module docstring for why.
    """
    cls = form_class(c.title) if c.title else "oral"
    base = cls == "oral" or (cls == "injectable" and injection_is_base)
    tier = 0 if is_nda_or_bla(c.marketing_category) else 1
    repackager = 1 if is_repackager(labeler_from_title(c.title)) else 0
    ordinal = c.effective_date.toordinal() if c.effective_date else 0
    return (0 if base else 1, repackager, tier, -ordinal, c.setid)


def rank(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Best first. Injection is base only if no candidate in the pool is an oral label."""
    pool = list(candidates)
    injection_is_base = not has_oral_base(pool)
    return sorted(pool, key=lambda c: preference(c, injection_is_base))


def group_by_key(candidates: Iterable[Candidate]) -> dict[str, list[Candidate]]:
    """Candidates per ingredient key, each list ranked best first (judged within the group)."""
    raw: dict[str, list[Candidate]] = {}
    for c in candidates:
        raw.setdefault(c.ingredient_key, []).append(c)
    return {key: rank(group) for key, group in raw.items()}


def select_labels(candidates: Iterable[Candidate], per_key: int = 1) -> list[Candidate]:
    """Best `per_key` candidates for every ingredient set, ordered by key then rank."""
    out: list[Candidate] = []
    groups = group_by_key(candidates)
    for key in sorted(groups, key=lambda k: [int(p) for p in k.split("+")]):
        out.extend(groups[key][:per_key])
    return out
