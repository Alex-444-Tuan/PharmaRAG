"""Shared helpers for the dense and lexical branches (spec section 8)."""

from typing import Any, Literal

from pharmarag.graph.state import Hit, Span

# Columns every branch selects from `scoped` / `chunk`, plus `effective_date` from `label`.
CHUNK_COLUMNS = (
    "id, setid, spl_version, ingredient_key, section_loinc, section_title, "
    "subsection_title, text, spans"
)


def section_param(sections: list[str] | None) -> list[str] | None:
    """An empty section list means "no section filter"; SQL receives NULL for it."""
    return list(sections) if sections else None


def rows_to_hits(
    rows: list[dict[str, Any]], *, source: Literal["dense", "lexical"], attempt: int
) -> list[Hit]:
    return [
        Hit(
            chunk_id=r["id"],
            setid=r["setid"],
            spl_version=r["spl_version"],
            ingredient_key=r["ingredient_key"],
            section_loinc=r["section_loinc"],
            section_title=r["section_title"],
            subsection_title=r["subsection_title"],
            text=r["text"],
            spans=[Span(**s) for s in r["spans"]],
            effective_date=r["effective_date"],
            source=source,
            rank=i,
            score=float(r["score"]),
            retrieval_attempt=attempt,
        )
        for i, r in enumerate(rows, start=1)
    ]
