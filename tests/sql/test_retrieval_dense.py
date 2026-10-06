"""Dense retrieval against real PostgreSQL + pgvector (spec section 8, FR-3)."""

import datetime

import pytest

from pharmarag.retrieval.dense import DENSE_FILTERED_SQL, DENSE_UNFILTERED_SQL, dense_search
from tests.graph.seed import HYDRALAZINE, HYDROXYZINE, seed_label

pytestmark = pytest.mark.pg

# Closer to the hydralazine chunks than to any hydroxyzine chunk.
QUERY = "hydralazine dosage for hypertension in adults four times daily"


@pytest.fixture
async def two_drugs(pg_conn, fake_embedder):
    s1 = await seed_label(pg_conn, fake_embedder, key="1", name="hydroxyzine", chunks=HYDROXYZINE)
    s2 = await seed_label(pg_conn, fake_embedder, key="2", name="hydralazine", chunks=HYDRALAZINE)
    return s1, s2


async def test_filtered_returns_only_requested_ingredient_even_when_other_drug_is_closer(
    pg_conn, fake_embedder, two_drugs
):
    vec = fake_embedder.embed_query(QUERY)
    unfiltered = await dense_search(
        pg_conn, vec, ingredient_key=None, sections=None, top_k=20, attempt=1
    )
    assert unfiltered[0].ingredient_key == "2"  # the premise: the wrong drug ranks first

    hits = await dense_search(pg_conn, vec, ingredient_key="1", sections=None, top_k=20, attempt=1)
    assert {h.ingredient_key for h in hits} == {"1"}
    assert len(hits) == len(HYDROXYZINE)  # exact scan: nothing under-returned
    assert hits[0].text.startswith("Hydroxyzine dosage")


async def test_filtered_hits_carry_rank_score_attempt_and_label_date(
    pg_conn, fake_embedder, two_drugs
):
    vec = fake_embedder.embed_query(QUERY)
    hits = await dense_search(pg_conn, vec, ingredient_key="2", sections=None, top_k=20, attempt=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].score >= hits[1].score
    assert all(h.source == "dense" and h.retrieval_attempt == 2 for h in hits)
    assert all(h.effective_date == datetime.date(2024, 5, 1) for h in hits)
    assert hits[0].spans[0].start == 0 and hits[0].spans[0].end == len(hits[0].text)
    assert hits[0].section_title == "Dosage and administration"
    expected = float(vec @ fake_embedder.embed_documents([hits[0].text])[0])
    assert hits[0].score == pytest.approx(expected, abs=1e-5)


async def test_filtered_respects_top_k_and_section_filter(pg_conn, fake_embedder, two_drugs):
    vec = fake_embedder.embed_query(QUERY)
    hits = await dense_search(pg_conn, vec, ingredient_key="1", sections=None, top_k=2, attempt=1)
    assert len(hits) == 2
    hits = await dense_search(
        pg_conn, vec, ingredient_key="1", sections=["34070-3"], top_k=20, attempt=1
    )
    assert [h.section_loinc for h in hits] == ["34070-3"]
    # An empty list means "no section filter", not "no sections".
    hits = await dense_search(pg_conn, vec, ingredient_key="1", sections=[], top_k=20, attempt=1)
    assert len(hits) == len(HYDROXYZINE)


async def test_combination_key_is_not_matched_by_single_ingredient_key(pg_conn, fake_embedder):
    await seed_label(pg_conn, fake_embedder, key="1+2", name="combo", chunks=HYDROXYZINE[:1])
    vec = fake_embedder.embed_query("hydroxyzine dosage")
    assert (
        await dense_search(pg_conn, vec, ingredient_key="1", sections=None, top_k=20, attempt=1)
        == []
    )
    hits = await dense_search(
        pg_conn, vec, ingredient_key="1+2", sections=None, top_k=20, attempt=1
    )
    assert len(hits) == 1


async def _plan(conn, sql, params) -> str:
    await conn.execute(
        "SET LOCAL enable_seqscan = off"
    )  # make an index plan as attractive as possible
    cur = await conn.execute("EXPLAIN " + sql, params)
    return "\n".join(row[0] for row in await cur.fetchall())


async def test_explain_filtered_dense_query_never_uses_the_hnsw_index(
    pg_conn, fake_embedder, two_drugs
):
    # Leave the planner no btree to prefer, so only the MATERIALIZED fence keeps HNSW out of
    # the plan (without it the planner inlines the CTE and picks the vector index).
    await pg_conn.execute("DROP INDEX chunk_key_section")
    vec = fake_embedder.embed_query(QUERY)
    params = {"key": "1", "sections": None, "vec": vec, "k": 20}
    assert "chunk_hnsw" not in await _plan(pg_conn, DENSE_FILTERED_SQL, params)
    params["sections"] = ["34068-7"]
    assert "chunk_hnsw" not in await _plan(pg_conn, DENSE_FILTERED_SQL, params)


async def test_explain_control_unfiltered_query_does_use_the_hnsw_index(
    pg_conn, fake_embedder, two_drugs
):
    """Guards the guard: the plan check above can only fail if the index is reachable."""
    vec = fake_embedder.embed_query(QUERY)
    plan = await _plan(pg_conn, DENSE_UNFILTERED_SQL, {"sections": None, "vec": vec, "k": 20})
    assert "chunk_hnsw" in plan


async def test_unfiltered_with_sections_uses_iterative_scan_so_rare_section_is_not_starved(
    pg_conn, fake_embedder
):
    # 150 dosage chunks that all look like the query, plus 3 interaction chunks that do not.
    filler = [
        ("34068-7", f"hydralazine dosage hypertension adults four times daily note {i}")
        for i in range(150)
    ]
    rare = [("34073-7", f"unrelated text about grapefruit juice number {i}") for i in range(3)]
    await seed_label(pg_conn, fake_embedder, key="2", name="hydralazine", chunks=filler + rare)
    # Force the HNSW path on a tiny table: no seq scan, and no btree that the planner could
    # scan instead (the DDL is rolled back with the test transaction).
    await pg_conn.execute("SET LOCAL enable_seqscan = off")
    await pg_conn.execute("DROP INDEX chunk_key_section")
    vec = fake_embedder.embed_query(QUERY)
    hits = await dense_search(
        pg_conn, vec, ingredient_key=None, sections=["34073-7"], top_k=20, attempt=1
    )
    # hnsw.ef_search defaults to 40, so a plain index scan followed by the WHERE clause would
    # return nothing here. relaxed_order iterative scan keeps scanning until it has the rows.
    assert len(hits) == 3
    assert {h.section_loinc for h in hits} == {"34073-7"}
