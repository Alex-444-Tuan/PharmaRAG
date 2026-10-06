"""The resolver's SQL against a throwaway Postgres: alias lookups, trigram, coverage."""

import pytest

from pharmarag.db import create_pool
from pharmarag.resolver.local import PgLocalStore
from tests.unit.resolver.fakes import seed

pytestmark = pytest.mark.pg


@pytest.fixture
async def store(pg_url, pg_conn):
    await seed(pg_conn)
    pool = await create_pool(pg_url, min_size=1, max_size=1)  # one connection: leaks would show
    yield PgLocalStore(pool)
    await pool.close()


async def test_exact_matches_any_of_the_forms_and_ignores_empty_ones(store):
    rows = await store.exact({"vistaril", "glucotrol xl", ""})
    assert {(r.alias, r.tty, r.ingredient_key) for r in rows} == {
        ("vistaril", "BN", "5553"),
        ("glucotrol xl", "BN", "5454"),
    }
    assert await store.exact(set()) == []
    assert await store.exact({"vistari"}) == []  # exact means exact


async def test_exact_ignores_punctuation_in_the_stored_alias(store):
    # The normaliser turns "insulin aspart, human" into "insulin aspart human".
    rows = await store.exact({"insulin aspart human"})
    assert [(r.alias, r.ingredient_key) for r in rows] == [("insulin aspart, human", "51428")]
    # A hyphen is not punctuation here: brand and suffix names keep it.
    assert await store.exact({"toprol xl"}) == []


async def test_trigram_returns_the_nearest_aliases_first(store):
    rows = await store.trigram("klonipin", min_similarity=0.4, k=5)
    assert rows and rows[0].alias == "klonopin"


async def test_trigram_applies_the_similarity_floor(store):
    assert await store.trigram("klonipin", min_similarity=0.99, k=5) == []
    assert await store.trigram("qqqqqqq", min_similarity=0.4, k=5) == []


async def test_trigram_returns_at_most_k_aliases(store):
    rows = await store.trigram("hydroxyzine", min_similarity=0.2, k=2)
    assert len(rows) == 2
    assert rows[0].alias == "hydroxyzine"


async def test_the_trigram_threshold_does_not_leak_to_the_next_query(store):
    await store.trigram("klonipin", min_similarity=0.9, k=5)
    # The pool has one connection, so the next query would see a leaked SET.
    async with store._pool.connection() as conn:
        cur = await conn.execute("SHOW pg_trgm.similarity_threshold")
        assert float((await cur.fetchone())[0]) == pytest.approx(0.3)


async def test_by_source_maps_concepts_to_their_alias_row_and_prefers_the_ingredient(store):
    got = await store.by_source({"5553", "282519", "999"})
    assert set(got) == {"5553", "282519"}
    assert (got["5553"].tty, got["5553"].ingredient_key) == ("IN", "5553")
    assert (got["282519"].tty, got["282519"].ingredient_key) == ("BN", "5553")
    # 4815 is both the IN "glyburide" and the SY "glibenclamide": the IN row wins.
    assert (await store.by_source({"4815"}))["4815"].tty == "IN"
    assert await store.by_source(set()) == {}


async def test_aliases_of_lists_every_name_of_each_key(store):
    got = await store.aliases_of({"5553", "6809+857974", "nope"})
    assert got["5553"] == [
        "hydroxyzine",
        "hydroxyzine hydrochloride",
        "hydroxyzine pamoate",
        "vistaril",
    ]
    assert got["6809+857974"] == ["kombiglyze"]
    assert "nope" not in got


async def test_ingredient_names(store):
    assert await store.ingredient_names({"5553", "1"}) == {"5553": "hydroxyzine"}


async def test_covered_means_a_label_exists_for_the_exact_key(store):
    # zolpidem has aliases but no label; the combination key has aliases but no label.
    assert await store.covered({"5553", "39993", "6809+857974", "6809"}) == {"5553", "6809"}
    assert await store.covered(set()) == set()
