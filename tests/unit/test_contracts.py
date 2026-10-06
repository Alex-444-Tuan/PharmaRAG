"""Smoke tests for the cycle 0 contract files."""

import numpy as np
import pytest
from pydantic import BaseModel

from pharmarag.graph.state import merge_dicts


def test_thresholds_load(thresholds):
    assert thresholds.graph.max_retrieval_attempts == 2
    assert thresholds.graph.recursion_limit == 12


def test_settings_defaults(settings):
    assert settings.embedding_dim == 384
    assert settings.verify_mode == "stream"


def test_merge_dicts():
    assert merge_dicts({"a": 1.0}, {"b": 2.0}) == {"a": 1.0, "b": 2.0}


def test_fake_embedder_normalised_and_similar(fake_embedder):
    docs = fake_embedder.embed_documents(["hydroxyzine dosage anxiety", "warfarin bleeding risk"])
    q = fake_embedder.embed_query("hydroxyzine anxiety dose")
    assert docs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert docs[0] @ q > docs[1] @ q


async def test_fake_llm_scripted(make_fake_llm):
    class Out(BaseModel):
        x: int

    llm = make_fake_llm(["hello world", '{"x": 3}', "a b a"])
    assert await llm.complete([{"role": "user", "content": "hi"}]) == "hello world"
    assert (await llm.structured([{"role": "user", "content": "hi"}], Out)).x == 3
    assert "".join([t async for t in llm.stream([{"role": "user", "content": "hi"}])]) == "a b a"
    with pytest.raises(AssertionError):
        await llm.complete([{"role": "user", "content": "hi"}])


@pytest.mark.pg
async def test_schema_migrated(pg_conn):
    cur = await pg_conn.execute(
        "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm') ORDER BY 1"
    )
    assert [r[0] for r in await cur.fetchall()] == ["pg_trgm", "vector"]
    cur = await pg_conn.execute("SELECT count(*) FROM chunk")
    assert (await cur.fetchone())[0] == 0
