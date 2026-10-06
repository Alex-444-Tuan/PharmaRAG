"""Lexical branch (spec section 8): full-text search that catches exact tokens such as
enzyme names, lab values, and dose strings that embeddings blur.

Same two paths as the dense branch. The filtered path reuses the materialised `scoped` CTE.
"""

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from pharmarag.graph.state import Hit
from pharmarag.retrieval.common import CHUNK_COLUMNS, rows_to_hits, section_param

# SPEC: section 8 ranks the scoped rows by ts_rank_cd only. We also require `tsv @@ query`,
# otherwise rows that match no query term (rank 0) would fill the list and inflate the
# "chunk appears in both lists" count used by the evidence grader.
LEXICAL_FILTERED_SQL = f"""
WITH scoped AS MATERIALIZED (
  SELECT {CHUNK_COLUMNS}, tsv
  FROM chunk
  WHERE ingredient_key = %(key)s
    AND (%(sections)s::text[] IS NULL OR section_loinc = ANY(%(sections)s::text[]))
), q AS (
  SELECT websearch_to_tsquery('english', %(q)s) AS tsq
)
SELECT s.id, s.setid, s.spl_version, s.ingredient_key, s.section_loinc, s.section_title,
       s.subsection_title, s.text, s.spans, l.effective_date,
       ts_rank_cd(s.tsv, q.tsq) AS score
FROM scoped s
CROSS JOIN q
JOIN label l ON l.setid = s.setid AND l.spl_version = s.spl_version
WHERE s.tsv @@ q.tsq
ORDER BY score DESC, s.id
LIMIT %(k)s
"""

LEXICAL_UNFILTERED_SQL = f"""
WITH q AS (
  SELECT websearch_to_tsquery('english', %(q)s) AS tsq
)
SELECT c.id, c.setid, c.spl_version, c.ingredient_key, c.section_loinc, c.section_title,
       c.subsection_title, c.text, c.spans, l.effective_date,
       ts_rank_cd(c.tsv, q.tsq) AS score
FROM (SELECT {CHUNK_COLUMNS}, tsv FROM chunk) c
CROSS JOIN q
JOIN label l ON l.setid = c.setid AND l.spl_version = c.spl_version
WHERE c.tsv @@ q.tsq
  AND (%(sections)s::text[] IS NULL OR c.section_loinc = ANY(%(sections)s::text[]))
ORDER BY score DESC, c.id
LIMIT %(k)s
"""


async def lexical_search(
    conn: AsyncConnection,
    query: str,
    *,
    ingredient_key: str | None,
    sections: list[str] | None,
    top_k: int,
    attempt: int,
) -> list[Hit]:
    """Top `top_k` chunks by ts_rank_cd. `rank` is 1-based within this call."""
    params = {"key": ingredient_key, "sections": section_param(sections), "q": query, "k": top_k}
    sql = LEXICAL_FILTERED_SQL if ingredient_key is not None else LEXICAL_UNFILTERED_SQL
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(sql, params)
        rows = await cur.fetchall()
    return rows_to_hits(rows, source="lexical", attempt=attempt)
