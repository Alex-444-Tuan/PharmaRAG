"""Graph-level filter-leak test (FR-3, spec section 13) over the real ingested corpus.

`tests/sql/test_filter_leak.py` calls the SQL functions directly. That proves the filtered
queries cannot leak, but not that the graph always reaches them filtered. This test drives the
real compiled graph: analyze (scripted mention) -> the real CascadeResolver in local-only mode ->
the real retrieval nodes -> fusion. For every request:

* if it reached retrieval, every hit of every attempt and every evidence chunk belongs to one of
  the resolved `ingredient_keys` (not just the final evidence: the first attempt's hits too);
* if the resolver stopped it (clarification, not recognised, not covered), nothing was retrieved.

Requests: every seed name, two misspellings of each look-alike pair member, and one brand name of
each. The LLM is scripted, the embedder is FakeEmbedder, RxNav is stubbed down, so the test is
deterministic and offline. The corpus is opened read-only.
"""

import json
from collections import Counter

import pytest

from pharmarag.cache import InMemoryCache
from pharmarag.db import create_pool
from pharmarag.graph.build import build_graph
from pharmarag.graph.deps import Deps
from pharmarag.graph.run import effective_flags, initial_state
from pharmarag.graph.state import Flags
from pharmarag.resolver.cascade import CascadeResolver
from pharmarag.resolver.local import PgLocalStore
from tests.sql import test_filter_leak as sql_leak
from tests.unit.resolver.fakes import FakeRx

pytestmark = pytest.mark.corpus


def scripted_llm_responder(messages, system):
    """analyze: the mention is whatever sits between the markers; sections come from the question
    template. rewrite_query and the writer get a fixed reply."""
    if system and "You analyse" in system:
        question = messages[0]["content"].split("<question>\n", 1)[1].rsplit("\n</question>", 1)[0]
        mention, _, rest = question.partition(" || ")
        sections = json.loads(rest.split(" || ")[0]) if rest else []
        text = question.split(" || ")[-1]
        return json.dumps(
            {
                "intent": "label_qa",
                "mentions": [mention],
                "sections": sections,
                "search_query": text,
            }
        )
    if system and "second try" in system:
        return json.dumps({"search_query": "dosage warnings adverse reactions"})
    return "The label states this. [c1]"


def build_request(mention: str, i: int) -> tuple[str, str]:
    """(question text sent to the graph, mention). The question contains the mention verbatim."""
    template, sections = sql_leak.QUESTIONS[i % len(sql_leak.QUESTIONS)]
    return template.format(m=mention), json.dumps(sections or [])


@pytest.fixture
async def pool(corpus_url):
    p = await create_pool(corpus_url + "?options=-c%20default_transaction_read_only%3Don")
    yield p
    await p.close()


@pytest.fixture
def deps(pool, settings, thresholds, fake_embedder, make_fake_llm):
    down = FakeRx()
    down.down = True  # RxNav unavailable: alias table and trigram only (NFR-4)
    resolver = CascadeResolver(PgLocalStore(pool), down, thresholds.resolver, InMemoryCache())
    return Deps(
        llm=make_fake_llm([scripted_llm_responder] * 5000),
        embedder=fake_embedder,
        pool=pool,
        settings=settings,
        thresholds=thresholds,
        resolver=resolver,
    )


async def run_graph(graph, deps, question: str, flags: Flags | None = None) -> dict:
    state = initial_state(question, flags=effective_flags(flags or Flags(), deps))
    return await graph.ainvoke(
        state, config={"recursion_limit": deps.thresholds.graph.recursion_limit}
    )


def question_for(mention: str, i: int) -> str:
    """The scripted analyzer reads 'mention || sections || text' from the question."""
    text, sections = build_request(mention, i)
    return f"{mention} || {sections} || {text}"


async def all_mentions(pool) -> list[str]:
    names = sql_leak.seed_names()
    mentions = list(names)
    for name in sql_leak.seed_pairs():
        mentions += sql_leak.MISSPELLINGS[name]
    # One brand name per look-alike member, as the SQL leak test does. Their keys come from the
    # same alias table the resolver reads.
    key_of = {}
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT lower(name), rxcui FROM ingredient")
        key_of = {n: k for n, k in await cur.fetchall()}
    for name in sql_leak.seed_pairs():
        key = key_of.get(name.lower()) or sql_leak.RENAMED_SEEDS.get(name)
        if key:
            mentions += await sql_leak.brands_of(pool, key)
    return mentions


async def test_the_compiled_graph_never_ranks_a_chunk_outside_the_resolved_keys(deps, pool):
    graph = build_graph(deps)
    mentions = await all_mentions(pool)

    outcomes: Counter[str] = Counter()
    reached = chunks = hits_seen = leaks = retried = 0
    for i, mention in enumerate(mentions):
        state = await run_graph(graph, deps, question_for(mention, i))
        assert state["mentions"] == [mention], "the scripted analyzer must hand the mention over"
        status = state["response"]["status"]
        outcomes[status] += 1
        keys = set(state.get("ingredient_keys", []))
        if "evidence" not in state:
            # Stopped before retrieval (or declined): nothing may have been ranked at all.
            assert not state.get("hits"), f"{mention!r}: hits without reaching fusion"
            assert status in {"needs_clarification", "drug_not_recognised", "not_covered"}
            continue
        reached += 1
        retried += state["retrieval_attempt"] > 1
        assert keys, f"{mention!r}: retrieval ran with no ingredient key"
        stray = [h for h in state["hits"] if h.ingredient_key not in keys]
        stray += [c for c in state["evidence"] if c.ingredient_key not in keys]
        hits_seen += len(state["hits"])
        chunks += len(state["evidence"])
        leaks += len(stray)
        assert not stray, f"{mention!r} -> {keys}, but ranked {stray[0].ingredient_key}"

    print(
        f"\ngraph-level filter leak: requests={len(mentions)} reached_retrieval={reached} "
        f"(with a rewrite: {retried}) hits={hits_seen} evidence_chunks={chunks} leaks={leaks} "
        f"outcomes={dict(outcomes)}"
    )
    assert len(mentions) >= 280
    assert reached >= 200 and chunks > 0 and hits_seen > chunks
    assert leaks == 0


async def test_negative_control_the_same_graph_with_the_resolver_off_does_leak(deps):
    """If the baseline did not return several drugs, the zero above would prove nothing."""
    graph = build_graph(deps)
    foreign = 0
    for i, mention in enumerate(["hydroxyzine", "warfarin", "metformin"]):
        state = await run_graph(graph, deps, question_for(mention, i), flags=Flags(resolver=False))
        assert state["mentions"] == [] and state["ingredient_keys"] == []
        foreign += len({h.ingredient_key for h in state["hits"]})
    assert foreign > 3
