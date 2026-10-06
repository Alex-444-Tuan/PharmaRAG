"""Write ingredients, aliases, labels and chunks to PostgreSQL (spec sections 4 and 5).

Loading is idempotent on `(setid, spl_version)`. A new version of a setid replaces the old
version's chunks in the same transaction, so a reader never sees a label with no chunks.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from pharmarag.ingest.chunker import ChunkRecord
from pharmarag.ingest.rxnav import Alias, Ingredient


@dataclass(frozen=True)
class LabelRow:
    setid: str
    spl_version: int
    title: str
    labeler: str | None
    marketing_category: str | None
    effective_date: date | None
    ingredient_key: str
    ingredient_rxcuis: list[str]
    product_rxcuis: list[str]


LoadStatus = Literal["inserted", "unchanged", "replaced", "stale"]


def compute_corpus_version(pairs: Iterable[tuple[str, int]], embedding_model: str) -> str:
    """Hash of the sorted (setid, spl_version) list plus the embedding model name.

    Order of `pairs` does not matter. Changing any version, adding or removing a label, or
    changing the model name changes the result. 16 hex characters of SHA-256.
    """
    lines = [f"{setid.lower()}:{version}" for setid, version in sorted(set(pairs))]
    payload = "\n".join([*lines, f"model={embedding_model}"])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


async def upsert_ingredients(
    conn: psycopg.AsyncConnection, ingredients: Iterable[Ingredient]
) -> None:
    rows = [(i.rxcui, i.name) for i in ingredients]
    if not rows:
        return
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO ingredient (rxcui, name) VALUES (%s, %s) "
            "ON CONFLICT (rxcui) DO UPDATE SET name = EXCLUDED.name",
            rows,
        )
    await conn.commit()


async def upsert_aliases(conn: psycopg.AsyncConnection, aliases: Iterable[Alias]) -> None:
    rows = [(a.alias, a.tty, a.source_rxcui, a.ingredient_key) for a in aliases]
    if not rows:
        return
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO drug_alias (alias, alias_tty, source_rxcui, ingredient_key) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (alias, source_rxcui) DO UPDATE "
            "SET alias_tty = EXCLUDED.alias_tty, ingredient_key = EXCLUDED.ingredient_key",
            rows,
        )
    await conn.commit()


async def label_loaded(conn: psycopg.AsyncConnection, setid: str, spl_version: int) -> bool:
    cur = await conn.execute(
        "SELECT 1 FROM label WHERE setid = %s AND spl_version = %s", (uuid.UUID(setid), spl_version)
    )
    found = await cur.fetchone() is not None
    await conn.commit()
    return found


async def load_label(
    conn: psycopg.AsyncConnection,
    label: LabelRow,
    chunks: Sequence[ChunkRecord],
    embeddings: np.ndarray,
) -> LoadStatus:
    """Insert one label version with its chunks, atomically.

    Returns "unchanged" if this exact version is already loaded, "stale" if a newer version
    of the setid is loaded (nothing is written), "replaced" if older versions were swapped
    out, and "inserted" for a setid not seen before.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    status = await _load_in_transaction(conn, label, chunks, embeddings)
    await conn.commit()  # also ends the implicit transaction when a savepoint was used
    return status


async def _load_in_transaction(
    conn: psycopg.AsyncConnection,
    label: LabelRow,
    chunks: Sequence[ChunkRecord],
    embeddings: np.ndarray,
) -> LoadStatus:
    setid = uuid.UUID(label.setid)
    async with conn.transaction():
        cur = await conn.execute("SELECT spl_version FROM label WHERE setid = %s", (setid,))
        existing = [r[0] for r in await cur.fetchall()]
        if label.spl_version in existing:
            return "unchanged"
        if existing and max(existing) > label.spl_version:
            return "stale"
        if existing:
            # ON DELETE CASCADE removes the old versions' chunks.
            await conn.execute("DELETE FROM label WHERE setid = %s", (setid,))
        await conn.execute(
            "INSERT INTO label (setid, spl_version, title, labeler, marketing_category, "
            "effective_date, ingredient_key, ingredient_rxcuis, product_rxcuis) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                setid,
                label.spl_version,
                label.title,
                label.labeler,
                label.marketing_category,
                label.effective_date,
                label.ingredient_key,
                label.ingredient_rxcuis,
                label.product_rxcuis,
            ),
        )
        async with conn.cursor() as cur2:
            await cur2.executemany(
                "INSERT INTO chunk (setid, spl_version, ingredient_key, section_loinc, "
                "section_title, subsection_title, ordinal, text, spans, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        setid,
                        label.spl_version,
                        label.ingredient_key,
                        c.section_loinc,
                        c.section_title,
                        c.subsection_title,
                        c.ordinal,
                        c.text,
                        Jsonb([s.model_dump() for s in c.spans]),
                        np.asarray(emb, dtype=np.float32),
                    )
                    for c, emb in zip(chunks, embeddings, strict=True)
                ],
            )
    return "replaced" if existing else "inserted"


async def prune_other_labels(
    conn: psycopg.AsyncConnection, ingredient_key: str, keep_setids: Iterable[str]
) -> int:
    """Delete labels of this ingredient set that are no longer selected. Returns the count.

    Keeps the "one label per ingredient set" rule true when a re-run picks a different
    setid (for example a newer NDA label appeared).
    """
    keep = [uuid.UUID(s) for s in keep_setids]
    cur = await conn.execute(
        "DELETE FROM label WHERE ingredient_key = %s AND NOT (setid = ANY(%s))",
        (ingredient_key, keep),
    )
    await conn.commit()
    return cur.rowcount


async def refresh_corpus_version(conn: psycopg.AsyncConnection, embedding_model: str) -> str:
    """Recompute `corpus_version` from the label table and upsert it into `corpus_meta`."""
    cur = await conn.execute("SELECT setid, spl_version FROM label")
    pairs = [(str(r[0]), r[1]) for r in await cur.fetchall()]
    version = compute_corpus_version(pairs, embedding_model)
    await conn.execute(
        "INSERT INTO corpus_meta (id, corpus_version, updated_at) VALUES (1, %s, now()) "
        "ON CONFLICT (id) DO UPDATE SET corpus_version = EXCLUDED.corpus_version, "
        "updated_at = now()",
        (version,),
    )
    await conn.commit()
    return version
