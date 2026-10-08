"""Grounding pieces against the real ingested corpus (pharmarag_dev). Skipped without it."""

import hashlib
import random

import pytest

from pharmarag.eval import grounding_data as gd
from pharmarag.eval import grounding_dev
from pharmarag.graph.state import Chunk, Span
from pharmarag.grounding.spans import build_spans
from pharmarag.grounding.verifier import GroundingVerifier
from pharmarag.models.embedder import FakeEmbedder
from tests.unit.grounding.test_verifier import TH, FakeNLI

pytestmark = pytest.mark.corpus

DEV_CORPUS_VERSION = "24bdf02ae716c922"  # the corpus the frozen dev set was built from


async def real_chunks(conn, n=200, seed=1) -> list[Chunk]:
    cur = await conn.execute(
        "SELECT id, setid, spl_version, ingredient_key, section_loinc, section_title, "
        "subsection_title, text, spans FROM chunk ORDER BY id"
    )
    rows = await cur.fetchall()
    random.Random(seed).shuffle(rows)
    return [
        Chunk(
            id=f"c{i + 1}",
            chunk_id=r[0],
            setid=r[1],
            spl_version=r[2],
            ingredient_key=r[3],
            section_loinc=r[4],
            section_title=r[5],
            subsection_title=r[6],
            text=r[7],
            spans=[Span(**s) for s in r[8]],
            fused_score=0.0,
        )
        for i, r in enumerate(rows[:n])
    ]


async def test_real_chunk_spans_slice_to_nonempty_unique_records(corpus_conn):
    for chunk in await real_chunks(corpus_conn):
        recs = build_spans([chunk], [])
        assert recs, chunk.chunk_id
        ids = [r.id for r in recs]
        assert len(ids) == len(set(ids))
        assert all(r.text in r.context for r in recs)
        assert all(r.section == chunk.section_title for r in recs)


async def test_the_three_nearest_spans_are_always_candidates_on_real_evidence(corpus_conn):
    import numpy as np

    from pharmarag.graph.state import Sentence

    chunks = await real_chunks(corpus_conn, n=8)
    emb = FakeEmbedder()
    v = GroundingVerifier(emb, FakeNLI(lambda p, h: (0.9, 0.0)), TH)
    prepared = await v.prepare(chunks, [])
    sentence = Sentence(text="Take with food and avoid alcohol.", citations=["c3"])
    analysis = await v.analyse_sentence(sentence, prepared)
    sims = prepared.matrix @ emb.embed_documents([sentence.text])[0]
    nearest = [prepared.records[i].id for i in np.argsort(-sims, kind="stable")[:3]]
    assert set(nearest) <= set(analysis.candidates)
    assert len(analysis.candidates) == TH.candidate_spans


async def test_the_frozen_dev_set_is_reproduced_byte_for_byte_by_the_script(corpus_conn, tmp_path):
    cur = await corpus_conn.execute("SELECT corpus_version FROM corpus_meta WHERE id = 1")
    if (await cur.fetchone())[0] != DEV_CORPUS_VERSION:
        pytest.skip("the corpus changed since the dev set was built; rebuild with --build-dev")
    claims = await grounding_dev.build_dev(corpus_conn)
    out = tmp_path / "dev.jsonl"
    gd.write_jsonl(out, claims)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == gd.sha256_file(gd.DEV_CLAIMS)


async def test_dev_evidence_is_the_real_corpus_text(corpus_conn):
    claims = gd.load_claims(gd.DEV_CLAIMS)[:15]
    for c in claims:
        for e in c.evidence:
            cur = await corpus_conn.execute("SELECT text FROM chunk WHERE id = %s", (e.chunk_id,))
            row = await cur.fetchone()
            if row is not None:  # a re-ingest may have renumbered chunks; then there is no id
                assert row[0] == e.text, (c.id, e.id)
