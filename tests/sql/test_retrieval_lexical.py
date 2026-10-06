"""Lexical retrieval against real PostgreSQL full-text search (spec section 8)."""

import pytest

from pharmarag.retrieval.lexical import lexical_search
from tests.graph.seed import HYDRALAZINE, HYDROXYZINE, seed_label

pytestmark = pytest.mark.pg


@pytest.fixture
async def two_drugs(pg_conn, fake_embedder):
    await seed_label(pg_conn, fake_embedder, key="1", name="hydroxyzine", chunks=HYDROXYZINE)
    await seed_label(pg_conn, fake_embedder, key="2", name="hydralazine", chunks=HYDRALAZINE)


async def test_finds_exact_token_like_cyp3a4(pg_conn, two_drugs):
    hits = await lexical_search(
        pg_conn, "CYP3A4", ingredient_key=None, sections=None, top_k=20, attempt=1
    )
    assert len(hits) == 1
    assert "CYP3A4" in hits[0].text
    assert hits[0].source == "lexical" and hits[0].rank == 1 and hits[0].score > 0

    filtered = await lexical_search(
        pg_conn, "CYP3A4", ingredient_key="1", sections=None, top_k=20, attempt=1
    )
    assert [h.chunk_id for h in filtered] == [hits[0].chunk_id]


async def test_filtered_lexical_never_returns_other_drug(pg_conn, two_drugs):
    # "dosage" appears in both drugs' chunks.
    both = await lexical_search(
        pg_conn, "dosage", ingredient_key=None, sections=None, top_k=20, attempt=1
    )
    assert {h.ingredient_key for h in both} == {"1", "2"}
    only_2 = await lexical_search(
        pg_conn, "dosage", ingredient_key="2", sections=None, top_k=20, attempt=1
    )
    assert {h.ingredient_key for h in only_2} == {"2"}


async def test_nonmatching_chunks_are_not_returned(pg_conn, two_drugs):
    hits = await lexical_search(
        pg_conn, "CYP3A4", ingredient_key="2", sections=None, top_k=20, attempt=1
    )
    assert hits == []


async def test_ranking_rank_attempt_label_date_and_top_k(pg_conn, fake_embedder):
    await seed_label(
        pg_conn,
        fake_embedder,
        key="1",
        name="x",
        chunks=[
            ("34068-7", "warfarin warfarin warfarin bleeding"),
            ("34070-3", "warfarin bleeding"),
            ("34073-7", "nothing relevant"),
        ],
    )
    hits = await lexical_search(
        pg_conn, "warfarin", ingredient_key="1", sections=None, top_k=20, attempt=3
    )
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].score >= hits[1].score
    assert hits[0].text.startswith("warfarin warfarin")
    assert all(h.retrieval_attempt == 3 and h.effective_date is not None for h in hits)
    one = await lexical_search(
        pg_conn, "warfarin", ingredient_key="1", sections=None, top_k=1, attempt=1
    )
    assert len(one) == 1


async def test_section_filter_on_both_paths(pg_conn, two_drugs):
    for key in ("1", None):
        hits = await lexical_search(
            pg_conn, "hydroxyzine", ingredient_key=key, sections=["34070-3"], top_k=20, attempt=1
        )
        assert [h.section_loinc for h in hits] == ["34070-3"]


async def test_websearch_syntax_is_safe_for_arbitrary_user_text(pg_conn, two_drugs):
    # websearch_to_tsquery never raises on punctuation or stray operators.
    hits = await lexical_search(
        pg_conn,
        'what is the "dosage" -- & | ( ! hydroxyzine ::',
        ingredient_key="1",
        sections=None,
        top_k=20,
        attempt=1,
    )
    assert hits
