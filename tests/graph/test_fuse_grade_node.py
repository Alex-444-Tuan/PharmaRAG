import json
import uuid

import pytest

from pharmarag.graph.nodes.fuse_grade import fuse_grade
from pharmarag.graph.state import Hit, Resolution
from tests.graph.seed import seed_label

pytestmark = pytest.mark.pg


def hit(chunk_id, setid, source, rank, attempt=1) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        setid=setid,
        spl_version=1,
        ingredient_key="1",
        section_loinc="34068-7",
        section_title="Dosage and administration",
        text="t",
        spans=[],
        source=source,
        rank=rank,
        score=0.1,
        retrieval_attempt=attempt,
    )


async def test_product_hint_label_wins_a_tie_via_label_product_rxcuis(
    make_deps, make_fake_llm, pg_conn, fake_embedder
):
    plain = await seed_label(pg_conn, fake_embedder, key="1", name="x", chunks=[("34068-7", "a")])
    branded = await seed_label(
        pg_conn, fake_embedder, key="1", name="x", chunks=[("34068-7", "b")], product_rxcuis=["999"]
    )
    await pg_conn.commit()
    # Exact RRF tie: dense rank 1 vs lexical rank 1, different labels.
    hits = [hit(1, plain, "dense", 1), hit(2, branded, "lexical", 1)]
    resolutions = [
        Resolution(mention="brand", status="resolved", ingredient_key="1", product_hint="999")
    ]
    deps = make_deps(make_fake_llm([]))
    out = await fuse_grade(
        {"hits": hits, "resolutions": resolutions, "ingredient_keys": ["1"]}, deps
    )
    assert [c.setid for c in out["evidence"]] == [branded, plain]
    out = await fuse_grade({"hits": hits, "ingredient_keys": ["1"]}, deps)
    assert [c.setid for c in out["evidence"]] == [plain, branded]


async def test_only_current_attempt_is_fused(make_deps, make_fake_llm):
    sid = uuid.uuid4()
    hits = [hit(1, sid, "dense", 1, attempt=1), hit(2, sid, "dense", 1, attempt=2)]
    deps = make_deps(make_fake_llm([]))
    out = await fuse_grade({"hits": hits, "retrieval_attempt": 2}, deps)
    assert [c.chunk_id for c in out["evidence"]] == [2]


async def test_two_retrieval_branches_join_before_fuse_grade(
    make_deps, make_fake_llm, make_graph, corpus
):
    llm = make_fake_llm(
        [
            json.dumps(
                {
                    "intent": "label_qa",
                    "mentions": [],
                    "sections": [],
                    "search_query": "hydroxyzine dosage anxiety adults",
                }
            ),
            "Four times daily. [c1]",
        ]
    )
    graph = make_graph(make_deps(llm))
    state = {
        "question": "hydroxyzine dosage anxiety adults",
        "retrieval_attempt": 1,
        "generation_attempt": 1,
        "hits": [],
        "rx_facts": [],
        "timings": {},
    }
    order = []
    async for update in graph.astream(state, stream_mode="updates"):
        order += list(update)
    assert order.count("fuse_grade") == 1
    assert order.index("fuse_grade") > max(
        order.index("retrieve_dense"), order.index("retrieve_lexical")
    )
    assert order[0] == "analyze" and order[-1] == "finalize"
