"""Audit the chosen reference labels for special-indication or special-population picks.

    DATABASE_URL=.../pharmarag_dev uv run python -m pharmarag.ingest.audit

The ranking in `select.py` prefers a base form and a manufacturer, but it cannot tell that
BRISDELLE (paroxetine 7.5 mg for hot flashes) or GRALISE (once-daily gabapentin for one
indication) is not the label a user means by "paroxetine" or "gabapentin". This script lists the
picks that look like that, so a person can re-pin them in `data/label_pins.csv`. It reports,
it does not decide. A flag is a prompt for judgement, not a verdict.

Flags: a form that is not an ordinary oral solid or injection (`select.form_class`); a
repackager labeler; an effective date before 2016 (a label nobody maintains); a product name
from SPECIAL_PRODUCTS (low-dose, single-indication or restricted-population brands of a
common drug); a title whose dose form is an implant, ring, enema, film, insert, spray or patch.
"""

from __future__ import annotations

import asyncio
import re
import sys
from datetime import date

from pharmarag.app.settings import get_settings
from pharmarag.db import connect
from pharmarag.ingest.select import form_class, is_repackager, labeler_from_title

OLD_BEFORE = date(2016, 1, 1)

# Brands whose label covers a narrower use than the drug: low dose, one indication, one
# population, or a restricted program. Extend it when an audit finds another.
SPECIAL_PRODUCTS = frozenset(
    {
        "ABSTRAL", "ASPRUZYO", "BELBUCA", "BRISDELLE", "CORTENEMA", "DEPAKOTE SPRINKLES",
        "DEPAKOTESPRINKLES", "DEXTENZA", "DISKETS", "ENDOMETRIN", "ESTRING", "GRALISE",
        "HEMADY", "LODOCO",
        "MITIGARE", "ONUREG", "OZURDEX", "PEXEVA", "SARAFEM", "SAXENDA", "SILENOR", "SOLTAMOX",
        "THALITONE", "VIVLODEX", "VYBRIQUE", "ZIPSOR", "ZORVOLEX", "ZYBAN", "PROZAC WEEKLY",
    }
)  # fmt: skip
# "FILM COATED" tablets and "LIQUID FILLED" capsules are ordinary oral solids.
_SPECIAL_FORM = re.compile(
    r"\b(IMPLANT|RING|ENEMA|FILM(?! COATED)|INSERT|SPRAY|PATCH|SYSTEM|LIQUID(?! FILLED)|GRANULE)\b"
)


def flags(title: str, effective_date: date | None) -> list[str]:
    """Why a chosen label deserves a second look. Empty when it looks like a standard pick."""
    out: list[str] = []
    head = re.sub(r"\[[^\]]*\]\s*$", "", title.upper())
    name = head.split("(")[0].strip()
    if any(name.startswith(p) for p in SPECIAL_PRODUCTS):
        out.append("special-indication product")
    if form_class(title) == "niche":
        out.append("niche dose form")
    elif _SPECIAL_FORM.search(head):
        out.append("non-oral-solid form")
    if is_repackager(labeler_from_title(title)):
        out.append("repackager")
    if effective_date is not None and effective_date < OLD_BEFORE:
        out.append(f"old label ({effective_date.year})")
    return out


async def audit(database_url: str) -> list[tuple[str, str, list[str]]]:
    conn = await connect(database_url)
    try:
        cur = await conn.execute(
            "SELECT coalesce(i.name, l.ingredient_key), l.title, l.effective_date "
            "FROM label l LEFT JOIN ingredient i ON i.rxcui = l.ingredient_key ORDER BY 1"
        )
        rows = await cur.fetchall()
    finally:
        await conn.close()
    return [(n, t, f) for n, t, d in rows if (f := flags(t, d))]


def main() -> int:
    for name, title, why in asyncio.run(audit(get_settings().database_url)):
        print(f"{name:24} {'; '.join(why):40} {title[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
