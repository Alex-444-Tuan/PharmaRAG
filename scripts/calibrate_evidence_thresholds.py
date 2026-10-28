"""Calibrate dense_min / overlap_min on a small dev set over the real corpus.

Usage: uv run python scripts/calibrate_evidence_thresholds.py  (needs pharmarag_dev)
Writes reports/calibration_evidence_rows.json. Numbers in config/thresholds.yaml.

Answerable: label topics. Unanswerable: questions a drug label cannot answer.
Retrieval is filtered by the drug's ingredient_key, as in the system path.
"""

import asyncio
import json
import statistics
from pathlib import Path

from pharmarag.app.settings import get_settings, load_thresholds
from pharmarag.db import connect
from pharmarag.models.embedder import SentenceTransformerEmbedder
from pharmarag.retrieval.dense import dense_search
from pharmarag.retrieval.fusion import fuse
from pharmarag.retrieval.lexical import lexical_search

URL = "postgresql://postgres:postgres@localhost:5432/pharmarag_dev"
DRUGS = [
    "hydroxyzine",
    "clonidine",
    "metformin",
    "lamotrigine",
    "amlodipine",
    "sertraline",
    "warfarin",
    "atorvastatin",
    "lisinopril",
    "prednisone",
    "tramadol",
    "glipizide",
    "risperidone",
    "levothyroxine",
    "gabapentin",
]
ANSWERABLE = [
    "Can {d} be used during pregnancy?",
    "What drugs interact with {d}?",
    "What happens if someone takes too much {d}?",
    "How should {d} be stored?",
    "Who should not take {d}?",
    "Is dose adjustment of {d} needed in kidney impairment?",
]
UNANSWERABLE = [
    "How much does {d} cost at Walmart without insurance?",
    "Which celebrities have publicly said they take {d}?",
    "What is the stock price of the company that makes {d}?",
    "Can I bring {d} on a plane to Japan?",
]


async def main() -> None:
    th = load_thresholds()
    emb = SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5", revision=get_settings().embedding_model_revision or None
    )
    conn = await connect(URL)
    rows = []
    for d in DRUGS:
        cur = await conn.execute(
            "SELECT DISTINCT ingredient_key FROM drug_alias WHERE alias = %s AND alias_tty = 'IN'",
            (d,),
        )
        keys = [r[0] for r in await cur.fetchall()]
        cur = await conn.execute("SELECT 1 FROM label WHERE ingredient_key = ANY(%s)", (keys,))
        if not keys or not await cur.fetchone():
            print(f"skip {d}: not in corpus")
            continue
        key = keys[0]
        for label, templates in (("answerable", ANSWERABLE), ("unanswerable", UNANSWERABLE)):
            for t in templates:
                q = t.format(d=d)
                vec = await asyncio.to_thread(emb.embed_query, q)
                kw = dict(
                    ingredient_key=key,
                    sections=None,
                    top_k=th.retrieval.top_k_per_branch,
                    attempt=1,
                )
                hits = await dense_search(conn, vec, **kw) + await lexical_search(conn, q, **kw)
                evidence, ok = fuse(hits, attempt=1, ingredient_keys=[key], thresholds=th.retrieval)
                dense_ids = {h.chunk_id for h in hits if h.source == "dense"}
                lex_ids = {h.chunk_id for h in hits if h.source == "lexical"}
                lex_nodrug = await lexical_search(conn, q.replace(d, ""), **kw)
                lex_nd_ids = {h.chunk_id for h in lex_nodrug}
                rows.append(
                    {
                        "drug": d,
                        "kind": label,
                        "q": q,
                        "best_dense": max(
                            (h.score for h in hits if h.source == "dense"), default=0
                        ),
                        "overlap": sum(
                            1 for c in evidence if c.chunk_id in dense_ids and c.chunk_id in lex_ids
                        ),
                        "overlap_without_drug_term": sum(
                            1
                            for c in evidence
                            if c.chunk_id in dense_ids and c.chunk_id in lex_nd_ids
                        ),
                        "evidence_ok_current": ok,
                    }
                )
    await conn.close()
    for kind in ("answerable", "unanswerable"):
        rs = [r for r in rows if r["kind"] == kind]
        bd = sorted(r["best_dense"] for r in rs)
        print(f"\n{kind} (n={len(rs)})")
        print(
            f"  best_dense min/median/max: {bd[0]:.3f} / {statistics.median(bd):.3f} / {bd[-1]:.3f}"
        )
        print(f"  overlap (with drug term) median: {statistics.median(r['overlap'] for r in rs)}")
        no_drug = statistics.median(r["overlap_without_drug_term"] for r in rs)
        print(f"  overlap (drug term removed) median: {no_drug}")
        for t in (0.70, 0.72, 0.74, 0.76):
            print(f"  ok at dense_min={t}: {sum(r['best_dense'] >= t for r in rs)}/{len(rs)}")
        lowest = sorted(rs, key=lambda r: r["best_dense"])[:3]
        print(f"  lowest 3: {[(r['q'], round(r['best_dense'], 3)) for r in lowest]}")
        ok_now = sum(r["evidence_ok_current"] for r in rs)
        print(f"  evidence_ok under current thresholds: {ok_now}/{len(rs)}")
    return rows


if __name__ == "__main__":
    out = asyncio.run(main())
    Path("reports").mkdir(exist_ok=True)
    Path("reports/calibration_evidence_rows.json").write_text(json.dumps(out, indent=1))
