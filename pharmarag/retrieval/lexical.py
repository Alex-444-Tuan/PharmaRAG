"""Lexical branch (spec section 8): full-text search that catches exact tokens such as
enzyme names, lab values, and dose strings that embeddings blur.

Same two paths as the dense branch. The filtered path reuses the materialised `scoped` CTE.
"""

import re

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from pharmarag.graph.state import Hit
from pharmarag.retrieval.common import CHUNK_COLUMNS, rows_to_hits, section_param

# SPEC: section 8 names `websearch_to_tsquery`, which ANDs every term. On real labels a natural
# language question ("hydroxyzine dosage anxiety adults") then matches one chunk where an OR
# query matches 153, so the lexical list was almost empty and the "2 chunks in both lists"
# rule of the grader could never fire. We build an OR query from the question's word tokens
# (see `or_query`) and rank with ts_rank_cd, which scores chunks matching more terms higher.
# We also keep `tsv @@ query`, otherwise rank-0 rows would fill the list.
LEXICAL_FILTERED_SQL = f"""
WITH scoped AS MATERIALIZED (
  SELECT {CHUNK_COLUMNS}, tsv
  FROM chunk
  WHERE ingredient_key = %(key)s
    AND (%(sections)s::text[] IS NULL OR section_loinc = ANY(%(sections)s::text[]))
), q AS (
  SELECT to_tsquery('english', %(q)s) AS tsq
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
  SELECT to_tsquery('english', %(q)s) AS tsq
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

_TOKEN = re.compile(r"[A-Za-z0-9]+")
MAX_QUERY_TERMS = 32


def or_query(text: str) -> str:
    """`to_tsquery` input that ORs the word tokens of `text`, or "" if there are none.

    Only [A-Za-z0-9] runs survive, so user text cannot inject tsquery operators. The result
    is passed to PostgreSQL as a bound parameter, never formatted into the SQL string.
    `to_tsquery('english', ...)` then stems each term and drops stop words.
    """
    terms = list(dict.fromkeys(t.lower() for t in _TOKEN.findall(text)))
    return " | ".join(terms[:MAX_QUERY_TERMS])


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
    q = or_query(query)
    if not q:
        return []
    params = {"key": ingredient_key, "sections": section_param(sections), "q": q, "k": top_k}
    sql = LEXICAL_FILTERED_SQL if ingredient_key is not None else LEXICAL_UNFILTERED_SQL
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(sql, params)
        rows = await cur.fetchall()
    return rows_to_hits(rows, source="lexical", attempt=attempt)
