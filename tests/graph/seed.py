"""Insert fixture labels and chunks directly in SQL (no dependency on the ingest lane).

Vectors come from FakeEmbedder, so a query that shares words with a chunk is close to it.
"""

import json
import uuid
from datetime import date

from psycopg import AsyncConnection

from pharmarag.models.embedder import Embedder

SECTION_TITLES = {
    "34067-9": "Indications and usage",
    "34068-7": "Dosage and administration",
    "34070-3": "Contraindications",
    "34073-7": "Drug interactions",
    "34084-4": "Adverse reactions",
}


async def seed_label(
    conn: AsyncConnection,
    embedder: Embedder,
    *,
    key: str,
    name: str,
    chunks: list[tuple[str, str]],
    setid: uuid.UUID | None = None,
    spl_version: int = 1,
    effective_date: date | None = date(2024, 5, 1),
    product_rxcuis: list[str] | None = None,
) -> uuid.UUID:
    """Insert one label and its chunks. `chunks` is a list of (section_loinc, text)."""
    setid = setid or uuid.uuid4()
    await conn.execute(
        "INSERT INTO label (setid, spl_version, title, ingredient_key, ingredient_rxcuis, "
        "product_rxcuis, effective_date) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (
            setid,
            spl_version,
            f"{name} label",
            key,
            key.split("+"),
            product_rxcuis or [],
            effective_date,
        ),
    )
    vectors = embedder.embed_documents([text for _, text in chunks])
    for ordinal, ((loinc, text), vec) in enumerate(zip(chunks, vectors, strict=True)):
        spans = [{"i": 0, "start": 0, "end": len(text)}]
        await conn.execute(
            "INSERT INTO chunk (setid, spl_version, ingredient_key, section_loinc, section_title, "
            "ordinal, text, spans, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
            (
                setid,
                spl_version,
                key,
                loinc,
                SECTION_TITLES.get(loinc, loinc),
                ordinal,
                text,
                json.dumps(spans),
                vec,
            ),
        )
    return setid


async def set_corpus_version(conn: AsyncConnection, version: str) -> None:
    await conn.execute(
        "INSERT INTO corpus_meta (id, corpus_version) VALUES (1, %s) "
        "ON CONFLICT (id) DO UPDATE SET corpus_version = EXCLUDED.corpus_version",
        (version,),
    )


# Two look-alike drugs. Key "1" is hydroxyzine, key "2" is hydralazine.
HYDROXYZINE = [
    ("34068-7", "Hydroxyzine dosage for anxiety in adults is 50 to 100 mg four times daily."),
    ("34070-3", "Hydroxyzine is contraindicated in early pregnancy and in QT prolongation."),
    ("34073-7", "Hydroxyzine with CYP3A4 inhibitors may raise hydroxyzine plasma levels."),
]
HYDRALAZINE = [
    ("34068-7", "Hydralazine dosage for hypertension in adults is 10 mg four times daily."),
    (
        "34070-3",
        "Hydralazine is contraindicated in coronary artery disease and in "
        "mitral valve rheumatic heart disease.",
    ),
]
