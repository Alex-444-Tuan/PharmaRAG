"""Dense branch (spec section 8).

Two paths, and the choice between them is the FR-3 guarantee:

* Filtered (an `ingredient_key` is given): a materialised CTE applies the hard filter first,
  then PostgreSQL sorts the few dozen surviving rows exactly. The HNSW index is never used,
  so the filter cannot under-return the way index-scan-then-WHERE does.
* Unfiltered (`ingredient_key is None`): the no-mention path. Uses the HNSW index, with
  iterative scan when a section filter applies so the WHERE clause cannot starve the result.
"""

import numpy as np
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from pharmarag.graph.state import Hit
from pharmarag.retrieval.common import CHUNK_COLUMNS, rows_to_hits, section_param

DENSE_FILTERED_SQL = f"""
WITH scoped AS MATERIALIZED (
  SELECT {CHUNK_COLUMNS}, embedding
  FROM chunk
  WHERE ingredient_key = %(key)s
    AND (%(sections)s::text[] IS NULL OR section_loinc = ANY(%(sections)s::text[]))
)
SELECT s.id, s.setid, s.spl_version, s.ingredient_key, s.section_loinc, s.section_title,
       s.subsection_title, s.text, s.spans, l.effective_date,
       1 - (s.embedding <=> %(vec)s::vector) AS score
FROM scoped s
JOIN label l ON l.setid = s.setid AND l.spl_version = s.spl_version
ORDER BY s.embedding <=> %(vec)s::vector, s.id
LIMIT %(k)s
"""

DENSE_UNFILTERED_SQL = f"""
SELECT c.id, c.setid, c.spl_version, c.ingredient_key, c.section_loinc, c.section_title,
       c.subsection_title, c.text, c.spans, l.effective_date,
       1 - (c.embedding <=> %(vec)s::vector) AS score
FROM (
  SELECT {CHUNK_COLUMNS}, embedding
  FROM chunk
  WHERE (%(sections)s::text[] IS NULL OR section_loinc = ANY(%(sections)s::text[]))
  ORDER BY embedding <=> %(vec)s::vector
  LIMIT %(k)s
) c
JOIN label l ON l.setid = c.setid AND l.spl_version = c.spl_version
ORDER BY c.embedding <=> %(vec)s::vector, c.id
"""


async def dense_search(
    conn: AsyncConnection,
    query_vec: np.ndarray,
    *,
    ingredient_key: str | None,
    sections: list[str] | None,
    top_k: int,
    attempt: int,
) -> list[Hit]:
    """Top `top_k` chunks by cosine similarity. `rank` is 1-based within this call."""
    params = {
        "key": ingredient_key,
        "sections": section_param(sections),
        "vec": query_vec,
        "k": top_k,
    }
    async with conn.cursor(row_factory=dict_row) as cur:
        if ingredient_key is not None:
            await cur.execute(DENSE_FILTERED_SQL, params)
        elif params["sections"] is not None:
            # SET LOCAL needs a transaction; it reverts at its end, so the pooled connection
            # is left clean.
            async with conn.transaction():
                await cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                await cur.execute(DENSE_UNFILTERED_SQL, params)
        else:
            await cur.execute(DENSE_UNFILTERED_SQL, params)
        rows = await cur.fetchall()
    return rows_to_hits(rows, source="dense", attempt=attempt)
