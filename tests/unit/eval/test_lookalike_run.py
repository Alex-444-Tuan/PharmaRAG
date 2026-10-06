"""The look-alike eval end to end over the real corpus, with FakeLLM and no network.

Full mode runs `run_ask` through the real compiled graph (`pharmarag.graph.build.build_graph`)
with a local-only resolver, a scripted FakeLLM (analyze JSON, writer lines citing chunk ids) and
FakeEmbedder vectors. Nothing here calls a real LLM or RxNav.
"""

import json
import re

import pytest

from pharmarag.cache import InMemoryCache
from pharmarag.db import create_pool
from pharmarag.eval import lookalike as la
from pharmarag.eval import lookalike_run as lr
from pharmarag.eval.report import write_report
from pharmarag.eval.run import parse_args
from pharmarag.graph.build import build_graph
from pharmarag.graph.deps import Deps
from pharmarag.graph.run import run_ask
from pharmarag.llm.base import FakeLLM
from pharmarag.resolver.cascade import CascadeResolver
from pharmarag.resolver.local import PgLocalStore
from tests.unit.resolver.fakes import FakeRx

pytestmark = pytest.mark.corpus


@pytest.fixture
async def pool(corpus_url):
    p = await create_pool(corpus_url + "?options=-c%20default_transaction_read_only%3Don")
    yield p
    await p.close()


def local_resolver(pool, thresholds) -> CascadeResolver:
    down = FakeRx()
    down.down = True  # RxNav unavailable: the resolver runs on the alias table alone
    return CascadeResolver(PgLocalStore(pool), down, thresholds.resolver, InMemoryCache())


def scripted_llm(rows: list[la.Row]) -> FakeLLM:
    """Analyze JSON for each dataset question, and a writer line citing the first chunk."""
    mention_of = {r.question: r.mention for r in rows}

    def respond(messages, system):
        content = messages[0]["content"]
        if system.startswith("You analyse"):
            question = re.search(r"<question>\n(.*?)\n</question>", content, re.S).group(1)
            return json.dumps(
                {
                    "intent": "label_qa",
                    "mentions": [mention_of[question]],
                    "sections": [],
                    "search_query": question,
                }
            )
        assert "[c1]" in content  # the writer is only called with evidence
        return "The label describes this use. [c1]"

    return FakeLLM([respond] * 1000, model_id="fake-llm-eval")


@pytest.fixture
def full_deps(pool, settings, thresholds, fake_embedder):
    rows = la.load_dataset("dev")
    # FakeEmbedder vectors against the corpus's real bge vectors give cosines near zero or
    # negative, so the grader's dense_min would reject every retrieval. Only that threshold is
    # relaxed, for this test.
    relaxed = thresholds.model_copy(
        update={"retrieval": thresholds.retrieval.model_copy(update={"dense_min": -1.0})}
    )
    return Deps(
        llm=scripted_llm(rows),
        embedder=fake_embedder,
        pool=pool,
        settings=settings.model_copy(update={"eval_mode": True}),
        thresholds=relaxed,
        resolver=local_resolver(pool, thresholds),
    )


@pytest.fixture
def ask(full_deps):
    graph = build_graph(full_deps)
    responses: list[tuple[str, bool, dict]] = []

    async def _ask(question, flags):
        out = await run_ask(graph, full_deps, question=question, flags=flags)
        responses.append((question, flags.resolver, out))
        return out

    _ask.responses = responses
    return _ask


async def test_full_mode_baseline_vs_system_on_dev(ask, full_deps, corpus_conn, thresholds):
    report = await lr.run_full(
        split="dev",
        mode="both",
        runs=2,
        ask=ask,
        conn=corpus_conn,
        model_id=full_deps.llm.model_id,
        thresholds=thresholds,
    )

    # Provenance (spec 11 "Rules for all three").
    cur = await corpus_conn.execute("SELECT corpus_version FROM corpus_meta WHERE id = 1")
    assert report["corpus_version"] == (await cur.fetchone())[0]
    assert report["model_id"] == "fake-llm-eval"
    assert report["runs"] == 2
    assert report["n_queries"] == 20
    assert report["thresholds"]["resolver"]["margin"] == 0.05
    assert {"thresholds_commit", "thresholds_dirty", "timestamp"} <= set(report)

    base, system = report["baseline"], report["system"]
    assert len(base["runs"]) == len(system["runs"]) == 2
    assert all(r["counts"]["total"] == 20 for r in base["runs"] + system["runs"])

    # Unfiltered retrieval over 200 labels does cite other drugs, and the filter stops that.
    b = base["misinterpreted"]["values"]
    s = system["misinterpreted"]["values"]
    assert min(b) >= 10, b
    assert max(s) == 0, s
    # The resolver clarifies some misspellings instead of guessing: counted on its own.
    assert system["runs"][0]["counts"][la.CLARIFIED] >= 1
    assert system["runs"][0]["counts"][la.CORRECT] + system["runs"][0]["counts"][la.CLARIFIED] >= 18

    m = report["metric"]
    assert m["baseline_misinterpreted"] == b and m["system_misinterpreted"] == s
    assert m["per_run"] == [pytest.approx((bv - sv) / bv) for bv, sv in zip(b, s, strict=True)]


async def test_the_classification_agrees_with_an_independent_look_at_every_response(
    ask, corpus_conn, thresholds, full_deps
):
    """Re-derive each outcome from the raw response with fresh SQL, not with the eval's code."""
    rows = la.load_dataset("dev")
    report = await lr.run_full(
        split="dev",
        mode="both",
        runs=1,
        ask=ask,
        conn=corpus_conn,
        model_id="fake",
        thresholds=thresholds,
    )
    gold = {r.question: r.gold_ingredient_key for r in rows}
    reported = {
        "baseline": {r["question"]: r["outcome"] for r in report["baseline"]["runs"][0]["rows"]},
        "system": {r["question"]: r["outcome"] for r in report["system"]["runs"][0]["rows"]},
    }
    assert len(ask.responses) == 40
    for question, resolver_on, response in ask.responses:
        arm = "system" if resolver_on else "baseline"
        cited_setids = [s["setid"] for s in response["sources"]]
        cur = await corpus_conn.execute(
            "SELECT DISTINCT ingredient_key FROM label WHERE setid::text = ANY(%s)", (cited_setids,)
        )
        keys = {r[0] for r in await cur.fetchall()}
        status = response["status"]
        if status == "answered":
            expected = la.CORRECT if keys == {gold[question]} else la.WRONG_DRUG
        elif status == "needs_clarification":
            names = {c["ingredient_key"] for c in response["candidates"]}
            expected = la.CLARIFIED if gold[question] in names else la.CLARIFY_WITHOUT_GOLD
        elif status in ("drug_not_recognised", "not_covered"):
            expected = la.NOT_RECOGNISED if status == "drug_not_recognised" else la.NOT_COVERED
        else:
            expected = la.OTHER
        assert reported[arm][question] == expected, (arm, question, status)


async def test_the_baseline_arm_switches_the_resolver_off_and_the_system_arm_leaves_it_on(
    ask, corpus_conn, thresholds
):
    await lr.run_full(
        split="dev",
        mode="both",
        runs=1,
        ask=ask,
        conn=corpus_conn,
        model_id="fake",
        thresholds=thresholds,
    )
    flags = {(q, on) for q, on, _ in ask.responses}
    rows = la.load_dataset("dev")
    assert flags == {(r.question, on) for r in rows for on in (False, True)}
    base_meta = [resp["meta"] for _, on, resp in ask.responses if not on]
    sys_meta = [resp["meta"] for _, on, resp in ask.responses if on]
    assert not any(m["resolver"] for m in base_meta)
    assert all(m["resolver"] for m in sys_meta)


async def test_a_system_arm_without_a_resolver_is_an_error_not_a_second_baseline(
    ask, corpus_conn, thresholds
):
    with pytest.raises(ValueError, match="resolver"):
        await lr.run_full(
            split="dev",
            mode="both",
            runs=1,
            ask=ask,
            conn=corpus_conn,
            model_id="fake",
            thresholds=thresholds,
            resolver_available=False,
        )
    # The baseline alone needs none.
    report = await lr.run_full(
        split="dev",
        mode="baseline",
        runs=1,
        ask=ask,
        conn=corpus_conn,
        model_id="fake",
        thresholds=thresholds,
        resolver_available=False,
    )
    assert "system" not in report and "metric" not in report


# --- resolver-only mode ---


async def test_resolver_only_mode_reports_system_counts_and_a_labelled_proxy(
    pool, corpus_conn, thresholds, fake_embedder, tmp_path
):
    report = await lr.run_resolver_only(
        split="dev",
        mode="both",
        runs=2,
        resolver_factory=lambda: local_resolver(pool, thresholds),
        conn=corpus_conn,
        embedder=fake_embedder,
        thresholds=thresholds,
        rxnav_mode="offline",
    )
    assert report["resolver_only"] and report["model_id"] is None
    system = report["system"]
    assert len(system["runs"]) == 2
    counts = system["runs"][0]["counts"]
    assert counts["total"] == 20
    assert counts["misinterpreted"] == 0
    assert counts[la.CORRECT] + counts[la.CLARIFIED] == 20
    assert counts[la.WRONG_DRUG] == counts[la.NOT_RECOGNISED] == 0

    proxy = report["baseline_proxy"]
    assert "not the AT-1 number" in proxy["note"]
    assert proxy["counts"]["total"] == 20
    assert proxy["counts"]["any_top8_wrong"] >= proxy["counts"]["top1_wrong"]
    assert proxy["counts"]["any_top8_wrong"] > 0  # unfiltered search mixes drugs
    assert proxy["embedding_model"] == fake_embedder.model_name
    # No percentage is computed from a proxy.
    assert "metric" not in report

    md = lr.render_markdown(report)
    assert "PROXY, not the AT-1 number" in md
    assert "resolver-only" in md
    assert report["corpus_version"] in md

    json_path, md_path = write_report(report, tmp_path / "reports", md)
    assert re.fullmatch(r"lookalike_dev_\d{8}T\d{6}Z\.json", json_path.name)
    assert md_path.name == json_path.name.replace(".json", ".md")
    assert json.loads(json_path.read_text())["corpus_version"] == report["corpus_version"]
    assert md_path.read_text() == md


async def test_resolver_only_degraded_rows_are_counted(
    pool, corpus_conn, thresholds, fake_embedder
):
    report = await lr.run_resolver_only(
        split="dev",
        mode="system",
        runs=1,
        resolver_factory=lambda: local_resolver(pool, thresholds),
        conn=corpus_conn,
        embedder=None,
        thresholds=thresholds,
        rxnav_mode="offline",
    )
    # Misspellings and nothing else fall through to the (down) RxNav step.
    degraded = report["system"]["degraded_rows"]["values"][0]
    n_misspelled = sum(r.kind == "misspelling" for r in la.load_dataset("dev"))
    assert 0 < degraded <= n_misspelled
    assert "baseline_proxy" not in report


async def test_the_proxy_needs_an_embedder(corpus_conn, thresholds, pool):
    with pytest.raises(ValueError, match="embedder"):
        await lr.run_resolver_only(
            split="dev",
            mode="baseline",
            runs=1,
            resolver_factory=lambda: local_resolver(pool, thresholds),
            conn=corpus_conn,
            embedder=None,
            thresholds=thresholds,
            rxnav_mode="n/a",
        )


# --- CLI ---


def test_cli_defaults_and_flags():
    a = parse_args(["lookalike", "--split", "dev"])
    assert (a.split, a.mode, a.runs, a.resolver_only, a.offline) == ("dev", "both", 1, False, False)
    b = parse_args(
        ["lookalike", "--split", "test", "--mode", "system", "--runs", "3", "--resolver-only"]
    )
    assert (b.split, b.mode, b.runs, b.resolver_only) == ("test", "system", 3, True)


@pytest.mark.parametrize(
    "argv",
    [
        ["lookalike"],  # split is required
        ["lookalike", "--split", "val"],
        ["lookalike", "--split", "dev", "--mode", "neither"],
        ["lookalike", "--split", "dev", "--runs", "0"],
        ["lookalike", "--split", "dev", "--offline"],  # only with --resolver-only
    ],
)
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
