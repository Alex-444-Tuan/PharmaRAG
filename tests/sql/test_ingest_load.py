"""Loader against real PostgreSQL + pgvector: idempotency, version replacement, aliases."""

import uuid
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
import pytest

from pharmarag.ingest.chunker import chunk_label, embed_text
from pharmarag.ingest.load import (
    LabelRow,
    compute_corpus_version,
    label_loaded,
    load_label,
    prune_other_labels,
    refresh_corpus_version,
    upsert_aliases,
    upsert_ingredients,
)
from pharmarag.ingest.rxnav import Alias, Ingredient
from pharmarag.ingest.spl_parser import parse_spl
from pharmarag.models.embedder import FakeEmbedder

pytestmark = pytest.mark.pg

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "spl" / "norvasc_plr.xml"
SETID = "7367289c-b0b0-466a-83e2-558e2985c29f"
MODEL = "fake-embedder"


@pytest.fixture(scope="module")
def parsed():
    return parse_spl(FIXTURE.read_bytes())


@pytest.fixture
def prepared(parsed):
    emb = FakeEmbedder()
    chunks = chunk_label(parsed, emb.count_tokens)
    vectors = emb.embed_documents([embed_text("amlodipine", c) for c in chunks])
    return chunks, vectors


def row(version: int = 10, setid: str = SETID, key: str = "17767") -> LabelRow:
    return LabelRow(
        setid=setid,
        spl_version=version,
        title="NORVASC (AMLODIPINE BESYLATE) TABLET [VIATRIS SPECIALTY LLC]",
        labeler="Viatris Specialty LLC",
        marketing_category="NDA",
        effective_date=date(2023, 2, 15),
        ingredient_key=key,
        ingredient_rxcuis=key.split("+"),
        product_rxcuis=["197361", "308135"],
    )


async def scalar(conn, sql, *args):
    cur = await conn.execute(sql, args)
    return (await cur.fetchone())[0]


async def test_load_writes_label_and_chunks_with_filter_key(pg_conn, prepared):
    chunks, vectors = prepared
    assert await load_label(pg_conn, row(), chunks, vectors) == "inserted"

    lab = await (
        await pg_conn.execute(
            "SELECT title, labeler, marketing_category, effective_date, ingredient_key, "
            "ingredient_rxcuis, product_rxcuis FROM label WHERE setid = %s",
            (uuid.UUID(SETID),),
        )
    ).fetchone()
    assert lab == (
        "NORVASC (AMLODIPINE BESYLATE) TABLET [VIATRIS SPECIALTY LLC]",
        "Viatris Specialty LLC",
        "NDA",
        date(2023, 2, 15),
        "17767",
        ["17767"],
        ["197361", "308135"],
    )
    assert await scalar(pg_conn, "SELECT count(*) FROM chunk") == len(chunks)
    # the denormalised filter column is set on every chunk
    assert await scalar(pg_conn, "SELECT count(*) FROM chunk WHERE ingredient_key <> '17767'") == 0
    # tsv is generated and the embedding round-trips
    assert await scalar(pg_conn, "SELECT count(*) FROM chunk WHERE tsv IS NULL") == 0
    assert await scalar(pg_conn, "SELECT vector_dims(embedding) FROM chunk LIMIT 1") == 384
    ordinals = [
        r[0]
        for r in await (await pg_conn.execute("SELECT ordinal FROM chunk ORDER BY id")).fetchall()
    ]
    assert ordinals == list(range(len(chunks)))


async def test_stored_spans_slice_the_stored_text_exactly(pg_conn, prepared):
    chunks, vectors = prepared
    await load_label(pg_conn, row(), chunks, vectors)
    rows = await (
        await pg_conn.execute("SELECT text, spans FROM chunk ORDER BY ordinal")
    ).fetchall()
    assert len(rows) == len(chunks)
    for (text, spans), original in zip(rows, chunks, strict=True):
        assert text == original.text
        assert [s["i"] for s in spans] == list(range(len(spans)))
        for s, expected in zip(spans, original.spans, strict=True):
            assert text[s["start"] : s["end"]] == original.text[expected.start : expected.end]
            assert text[s["start"] : s["end"]].strip() == text[s["start"] : s["end"]]


async def test_loading_the_same_version_twice_is_idempotent(pg_conn, prepared):
    chunks, vectors = prepared
    assert await load_label(pg_conn, row(), chunks, vectors) == "inserted"
    ids_before = [
        r[0] for r in await (await pg_conn.execute("SELECT id FROM chunk ORDER BY id")).fetchall()
    ]
    assert await label_loaded(pg_conn, SETID, 10)

    assert await load_label(pg_conn, row(), chunks, vectors) == "unchanged"

    assert await scalar(pg_conn, "SELECT count(*) FROM label") == 1
    ids_after = [
        r[0] for r in await (await pg_conn.execute("SELECT id FROM chunk ORDER BY id")).fetchall()
    ]
    assert ids_after == ids_before  # nothing was deleted and re-inserted


async def test_new_spl_version_replaces_the_old_chunks(pg_conn, prepared):
    chunks, vectors = prepared
    await load_label(pg_conn, row(version=10), chunks, vectors)
    old_ids = {r[0] for r in await (await pg_conn.execute("SELECT id FROM chunk")).fetchall()}

    new_chunks = chunks[:5]
    status = await load_label(pg_conn, row(version=11), new_chunks, vectors[:5])

    assert status == "replaced"
    assert await (await pg_conn.execute("SELECT spl_version FROM label")).fetchall() == [(11,)]
    new = await (await pg_conn.execute("SELECT id, spl_version FROM chunk")).fetchall()
    assert len(new) == 5
    assert {v for _, v in new} == {11}
    assert not old_ids & {i for i, _ in new}


async def test_failed_replacement_rolls_back_and_keeps_the_old_version(pg_conn, prepared):
    chunks, vectors = prepared
    await load_label(pg_conn, row(version=10), chunks, vectors)
    bad_vectors = np.zeros((len(chunks), 3), dtype=np.float32)  # wrong dimension

    with pytest.raises(psycopg.Error):
        await load_label(pg_conn, row(version=11), chunks, bad_vectors)
    await pg_conn.rollback()

    assert await (await pg_conn.execute("SELECT spl_version FROM label")).fetchall() == [(10,)]
    assert await scalar(pg_conn, "SELECT count(*) FROM chunk WHERE spl_version = 10") == len(chunks)
    assert await scalar(pg_conn, "SELECT count(*) FROM chunk WHERE spl_version = 11") == 0


async def test_older_version_does_not_replace_a_newer_one(pg_conn, prepared):
    chunks, vectors = prepared
    await load_label(pg_conn, row(version=10), chunks, vectors)
    assert await load_label(pg_conn, row(version=9), chunks[:2], vectors[:2]) == "stale"
    assert await (await pg_conn.execute("SELECT spl_version FROM label")).fetchall() == [(10,)]
    assert await scalar(pg_conn, "SELECT count(*) FROM chunk") == len(chunks)


async def test_mismatched_chunk_and_embedding_counts_are_rejected(pg_conn, prepared):
    chunks, vectors = prepared
    with pytest.raises(ValueError, match="embeddings"):
        await load_label(pg_conn, row(), chunks, vectors[:-1])
    assert await scalar(pg_conn, "SELECT count(*) FROM label") == 0


async def test_alias_rows_are_present_and_upsert_is_idempotent(pg_conn):
    aliases = [
        Alias("amlodipine", "IN", "17767", "17767"),
        Alias("amlodipine besylate", "PIN", "104416", "17767"),
        Alias("norvasc", "BN", "1", "17767"),
        Alias("tribenzor", "BN", "2", "5487+17767+73494"),
        Alias("amlodipine", "SY", "999", "17767"),  # same text, different source concept
    ]
    await upsert_aliases(pg_conn, aliases)
    await upsert_aliases(pg_conn, aliases)  # idempotent

    rows = await (
        await pg_conn.execute(
            "SELECT alias, alias_tty, source_rxcui, ingredient_key FROM drug_alias ORDER BY 1, 3"
        )
    ).fetchall()
    assert rows == [
        ("amlodipine", "IN", "17767", "17767"),
        ("amlodipine", "SY", "999", "17767"),
        ("amlodipine besylate", "PIN", "104416", "17767"),
        ("norvasc", "BN", "1", "17767"),
        ("tribenzor", "BN", "2", "5487+17767+73494"),
    ]
    assert all(r[0] == r[0].lower() for r in rows)


async def test_alias_upsert_updates_the_key_of_an_existing_row(pg_conn):
    await upsert_aliases(pg_conn, [Alias("norvasc", "BN", "1", "17767")])
    await upsert_aliases(pg_conn, [Alias("norvasc", "BN", "1", "17767+5487")])
    rows = await (await pg_conn.execute("SELECT ingredient_key FROM drug_alias")).fetchall()
    assert rows == [("17767+5487",)]


async def test_ingredient_upsert_updates_the_name(pg_conn):
    await upsert_ingredients(pg_conn, [Ingredient("17767", "amlodipine")])
    await upsert_ingredients(
        pg_conn, [Ingredient("17767", "amlodipine"), Ingredient("5487", "hctz")]
    )
    rows = await (
        await pg_conn.execute("SELECT rxcui, name FROM ingredient ORDER BY rxcui")
    ).fetchall()
    assert rows == [("17767", "amlodipine"), ("5487", "hctz")]


async def test_prune_removes_other_labels_of_the_same_key_only(pg_conn, prepared):
    chunks, vectors = prepared
    other = "aaaaaaaa-0000-0000-0000-000000000001"
    third = "bbbbbbbb-0000-0000-0000-000000000002"
    await load_label(pg_conn, row(setid=SETID), chunks[:3], vectors[:3])
    await load_label(pg_conn, row(setid=other), chunks[:3], vectors[:3])
    await load_label(pg_conn, row(setid=third, key="5487"), chunks[:3], vectors[:3])

    removed = await prune_other_labels(pg_conn, "17767", [SETID])

    assert removed == 1
    left = {str(r[0]) for r in await (await pg_conn.execute("SELECT setid FROM label")).fetchall()}
    assert left == {SETID, third}
    assert (
        await scalar(pg_conn, "SELECT count(*) FROM chunk WHERE setid = %s", uuid.UUID(other)) == 0
    )


async def test_refresh_corpus_version_writes_corpus_meta_and_tracks_changes(pg_conn, prepared):
    chunks, vectors = prepared
    assert await refresh_corpus_version(pg_conn, MODEL) == compute_corpus_version([], MODEL)

    await load_label(pg_conn, row(version=10), chunks[:3], vectors[:3])
    v10 = await refresh_corpus_version(pg_conn, MODEL)
    assert v10 == compute_corpus_version([(SETID, 10)], MODEL)
    assert await scalar(pg_conn, "SELECT corpus_version FROM corpus_meta") == v10

    await load_label(pg_conn, row(version=11), chunks[:3], vectors[:3])
    v11 = await refresh_corpus_version(pg_conn, MODEL)
    assert v11 != v10
    assert await (
        await pg_conn.execute("SELECT id, corpus_version FROM corpus_meta")
    ).fetchall() == [(1, v11)]

    assert await refresh_corpus_version(pg_conn, "another-model") != v11
