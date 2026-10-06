"""Shape of the compiled graph: supersteps, step budget, node retry policies (section 6, NFR-6).

Supersteps are read from LangGraph's own step numbers (`stream_mode="debug"`), never from timing.
"""

from collections import defaultdict

import pytest
from langgraph.errors import GraphRecursionError

from pharmarag.graph.build import LLM_NODES, build_graph, llm_retry_policy
from pharmarag.graph.run import effective_flags, initial_state, run_ask
from pharmarag.graph.state import Flags
from pharmarag.llm.errors import LLMUnavailableError
from tests.graph.fakes import (
    DOSE_TEXT,
    GIBBERISH,
    HYDROXYZIN_Q,
    FakeResolver,
    FakeVerifier,
    analysis,
    resolved,
    rewrite,
)

pytestmark = pytest.mark.pg

BRANCHES = {"retrieve_dense", "retrieve_lexical", "fetch_rx_facts"}
HYDROXYZINE = resolved("hydroxyzin", "1", "hydroxyzine", match="approximate")


async def trace(
    graph, deps, question=HYDROXYZIN_Q, *, recursion_limit=None
) -> dict[int, list[str]]:
    """{superstep number: [node names that ran in it]}."""
    state = initial_state(question, flags=effective_flags(Flags(), deps))
    limit = recursion_limit or deps.thresholds.graph.recursion_limit
    steps: dict[int, list[str]] = defaultdict(list)
    async for ev in graph.astream(state, config={"recursion_limit": limit}, stream_mode="debug"):
        if ev["type"] == "task":
            steps[ev["step"]].append(ev["payload"]["name"])
    return dict(steps)


def longest_path_llm(make_fake_llm):
    """Weak evidence on attempt 1, then low support on generation 1: both retries used."""
    return make_fake_llm(
        [
            analysis(mentions=["hydroxyzin"], search_query=GIBBERISH),
            rewrite(DOSE_TEXT),
            "Sentence A is true. [c1]\nSentence B is invented. [c1]",
            "Sentence C is true. [c1]\nSentence D is true. [c1]",
        ]
    )


def longest_path_deps(make_fake_llm, make_deps):
    verifier = FakeVerifier([["supported", "unsupported"], ["supported", "supported"]])
    return make_deps(
        longest_path_llm(make_fake_llm),
        resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}),
        verifier=verifier,
    )


async def test_the_three_branches_run_in_the_same_superstep(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), f"{DOSE_TEXT} [c1]"])
    deps = make_deps(llm, resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}))
    steps = await trace(make_graph(deps), deps)

    branch_steps = [s for s, names in steps.items() if BRANCHES & set(names)]
    assert len(branch_steps) == 1, "the branches must not be spread over several supersteps"
    assert set(steps[branch_steps[0]]) == BRANCHES
    # fuse_grade is its own, later superstep: it ran once, after all three.
    (fuse_step,) = [s for s, names in steps.items() if "fuse_grade" in names]
    assert fuse_step > branch_steps[0] and steps[fuse_step] == ["fuse_grade"]


async def test_the_retry_reruns_all_three_branches_together_again(
    make_fake_llm, make_deps, make_graph, corpus
):
    deps = longest_path_deps(make_fake_llm, make_deps)
    steps = await trace(make_graph(deps), deps)
    branch_steps = sorted(s for s, names in steps.items() if BRANCHES & set(names))
    assert len(branch_steps) == 2
    assert all(set(steps[s]) == BRANCHES for s in branch_steps)
    assert [names for _, names in sorted(steps.items()) if "rewrite_query" in names] == [
        ["rewrite_query"]
    ]


async def test_longest_path_fits_the_recursion_limit(make_fake_llm, make_deps, make_graph, corpus):
    """Retrieval retry plus regeneration, every node on the way: 10 supersteps, limit 12."""
    deps = longest_path_deps(make_fake_llm, make_deps)
    steps = await trace(make_graph(deps), deps)
    expected = [
        ["analyze"],
        ["resolve"],
        sorted(BRANCHES),
        ["fuse_grade"],
        ["rewrite_query"],
        sorted(BRANCHES),
        ["fuse_grade"],
        ["answer"],
        ["answer"],
        ["finalize"],
    ]
    assert [sorted(names) for _, names in sorted(steps.items())] == expected
    assert len(steps) == 10
    assert max(steps) <= deps.thresholds.graph.recursion_limit  # NFR-6 backstop is never hit


async def test_step_count_boundary_is_exact(make_fake_llm, make_deps, make_graph, corpus):
    # Pins what the limit means to LangGraph: the longest path is 10 supersteps and needs a
    # recursion_limit of exactly 11 (the limit counts the closing step too). The configured 12
    # leaves one spare, so a wiring change that adds a hop to this path is caught here.
    deps = longest_path_deps(make_fake_llm, make_deps)
    graph = make_graph(deps)
    assert len(await trace(graph, deps, recursion_limit=11)) == 10
    deps = longest_path_deps(make_fake_llm, make_deps)
    with pytest.raises(GraphRecursionError):
        await trace(make_graph(deps), deps, recursion_limit=10)


# --- retry policies ----------------------------------------------------------------------------


def test_production_retry_policy_is_three_attempts_with_exponential_backoff_on_llm_errors():
    policy = llm_retry_policy()
    assert policy.max_attempts == 3
    assert policy.backoff_factor == 2.0 and policy.initial_interval > 0
    assert policy.retry_on is LLMUnavailableError  # not Exception: a refusal or 4xx is not retried


async def test_only_the_llm_nodes_carry_a_retry_policy(make_fake_llm, make_deps, pool):
    deps = make_deps(make_fake_llm([]), resolver=FakeResolver({}))
    graph = build_graph(deps)  # the production policy
    with_policy = {name for name, node in graph.nodes.items() if node.retry_policy}
    assert with_policy == set(LLM_NODES) == {"analyze", "rewrite_query", "answer"}
    # `resolve` has none: Lane A's RxNav client retries internally (3 x 3 would be too many).
    assert not graph.nodes["resolve"].retry_policy
    assert not graph.nodes["fetch_rx_facts"].retry_policy


def _flaky(n: int, then: str):
    """Responders: raise LLMUnavailableError n times, then return `then`."""

    def fail(messages, system):
        raise LLMUnavailableError("429")

    return [fail] * n + [then]


async def test_analyze_retries_transient_llm_errors_then_succeeds(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm([*_flaky(2, analysis(intent="out_of_scope"))])
    deps = make_deps(llm)
    out = await run_ask(make_graph(deps), deps, question="q")
    assert out["status"] == "declined" and len(llm.calls) == 3  # 2 failures + 1 success


async def test_llm_node_gives_up_after_three_attempts(make_fake_llm, make_deps, make_graph, corpus):
    llm = make_fake_llm([*_flaky(3, analysis())])
    deps = make_deps(llm)
    with pytest.raises(LLMUnavailableError):
        await run_ask(make_graph(deps), deps, question="q")
    assert len(llm.calls) == 3


async def test_non_transient_errors_are_not_retried(make_fake_llm, make_deps, make_graph, corpus):
    def bad(messages, system):
        raise ValueError("a 400 from the provider")

    llm = make_fake_llm([bad, analysis()])
    deps = make_deps(llm)
    with pytest.raises(ValueError):
        await run_ask(make_graph(deps), deps, question="q")
    assert len(llm.calls) == 1


async def test_rewrite_and_answer_nodes_retry_too(make_fake_llm, make_deps, make_graph, corpus):
    llm = make_fake_llm(
        [
            analysis(mentions=["hydroxyzin"], search_query=GIBBERISH),
            *_flaky(1, rewrite(DOSE_TEXT)),
            *_flaky(2, f"{DOSE_TEXT} [c1]"),
        ]
    )
    deps = make_deps(llm, resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}))
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q)
    assert out["status"] == "answered" and len(llm.calls) == 1 + 2 + 3


async def test_resolve_is_not_retried(make_fake_llm, make_deps, make_graph, corpus):
    class Flaky:
        calls = 0

        async def resolve(self, mentions, *, pinned_rxcuis=None):
            Flaky.calls += 1
            raise LLMUnavailableError("would be retried if resolve had the LLM policy")

    llm = make_fake_llm([analysis(mentions=["hydroxyzin"])])
    deps = make_deps(llm, resolver=Flaky())
    with pytest.raises(LLMUnavailableError):
        await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q)
    assert Flaky.calls == 1
