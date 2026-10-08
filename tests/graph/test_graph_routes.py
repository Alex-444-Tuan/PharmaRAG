"""One test per route of the section 6 graph, with fake LLM, resolver, facts source and
verifier over real PostgreSQL (spec section 13, "Graph")."""

import pytest

from pharmarag.graph.run import effective_flags, initial_state, run_ask
from pharmarag.graph.state import Flags
from pharmarag.grounding.writer import STRICT_ADDENDUM
from pharmarag.llm.errors import LLMRefusalError
from tests.graph.fakes import (
    DOSE_TEXT,
    GIBBERISH,
    HYDRALAZINE_DOSE_TEXT,
    HYDROXYZIN_Q,
    FakeResolver,
    FakeRxFacts,
    FakeVerifier,
    ambiguous,
    analysis,
    not_covered,
    resolved,
    rewrite,
    unknown,
)

pytestmark = pytest.mark.pg

HYDROXYZINE = resolved("hydroxyzin", "1", "hydroxyzine", match="approximate")


def hydroxyzin_resolver() -> FakeResolver:
    return FakeResolver({"hydroxyzin": HYDROXYZINE})


async def final_state(graph, deps, question, *, flags=None, pinned=None) -> dict:
    """The full final graph state, for assertions that the response does not expose."""
    flags = effective_flags(flags or Flags(), deps)
    return await graph.ainvoke(
        initial_state(question, flags=flags, pinned_rxcuis=pinned),
        config={"recursion_limit": deps.thresholds.graph.recursion_limit},
    )


def writer_prompts(llm) -> list[str]:
    return [c["messages"][0]["content"] for c in llm.calls if c["kind"] == "stream"]


# --- declined ----------------------------------------------------------------------------------


async def test_declined_never_reaches_the_resolver_or_retrieval(
    make_fake_llm, make_deps, make_graph, corpus
):
    resolver, facts = hydroxyzin_resolver(), FakeRxFacts()
    llm = make_fake_llm([analysis(intent="patient_advice", mentions=["hydroxyzin"])])
    deps = make_deps(llm, resolver=resolver, rx_facts=facts)
    out = await run_ask(
        make_graph(deps), deps, question="Should I take hydroxyzin tonight?", debug=True
    )
    assert out["status"] == "declined" and out["answer"] == []
    assert resolver.calls == [] and facts.calls == []
    assert set(out["debug"]["timings_ms"]) == {"analyze", "finalize"}


# --- clarification, not recognised, not covered -----------------------------------------------


async def test_ambiguous_name_asks_for_clarification_with_three_candidates(
    make_fake_llm, make_deps, make_graph, corpus
):
    candidates = [
        ("1", "hydroxyzine", 0.88),
        ("2", "hydralazine", 0.86),
        ("3", "hydroxyurea", 0.82),
        ("4", "hydrocodone", 0.71),
    ]
    resolver = FakeResolver({"hydroxyzin": ambiguous("hydroxyzin", candidates)})
    facts = FakeRxFacts({"1": ["Brand: Atarax"]})
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"])])
    deps = make_deps(llm, resolver=resolver, rx_facts=facts)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q, debug=True)

    assert out["status"] == "needs_clarification"
    assert out["candidates"] == [
        {"ingredient_key": "1", "name": "hydroxyzine", "score": 0.88},
        {"ingredient_key": "2", "name": "hydralazine", "score": 0.86},
        {"ingredient_key": "3", "name": "hydroxyurea", "score": 0.82},
    ]
    assert out["ambiguous_mention"] == "hydroxyzin"
    assert out["answer"] == [] and out["sources"] == []
    # FR-8: nothing was searched, not even unfiltered, and the writer never ran.
    assert len(llm.calls) == 1 and facts.calls == []
    assert set(out["debug"]["timings_ms"]) == {"analyze", "resolve", "finalize"}
    assert out["meta"]["retrieval_attempts"] == 0 and out["meta"]["generation_attempts"] == 0


async def test_unknown_name_is_drug_not_recognised(make_fake_llm, make_deps, make_graph, corpus):
    resolver = FakeResolver({})  # every mention is unknown
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"])])
    deps = make_deps(llm, resolver=resolver)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q, debug=True)
    assert out["status"] == "drug_not_recognised"
    assert out["answer"] == [] and out["drugs"] == [] and "candidates" not in out
    assert set(out["debug"]["timings_ms"]) == {"analyze", "resolve", "finalize"}
    assert len(llm.calls) == 1


async def test_resolved_drug_without_a_label_is_not_covered_and_named(
    make_fake_llm, make_deps, make_graph, corpus
):
    resolver = FakeResolver({"hydroxyzin": not_covered("hydroxyzin", "999", "rarexamab")})
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"])])
    deps = make_deps(llm, resolver=resolver)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q, debug=True)
    assert out["status"] == "not_covered" and out["answer"] == []
    # The client can say "rarexamab is not in the corpus" by name.
    assert [d["name"] for d in out["drugs"]] == ["rarexamab"]
    assert set(out["debug"]["timings_ms"]) == {"analyze", "resolve", "finalize"}


async def test_several_failures_report_clarification_before_unknown_before_not_covered(
    make_fake_llm, make_deps, make_graph, corpus
):
    table = {
        "hydroxyzin": ambiguous("hydroxyzin", [("1", "hydroxyzine", 0.88), ("2", "x", 0.86)]),
        "zzzdrug": unknown("zzzdrug"),
        "rarexamab": not_covered("rarexamab"),
    }
    q = "hydroxyzin zzzdrug rarexamab"

    async def status_for(*mentions: str) -> str:
        deps = make_deps(
            make_fake_llm([analysis(mentions=list(mentions))]), resolver=FakeResolver(table)
        )
        return (await run_ask(make_graph(deps), deps, question=q))["status"]

    assert await status_for("hydroxyzin", "zzzdrug", "rarexamab") == "needs_clarification"
    assert await status_for("zzzdrug", "rarexamab") == "drug_not_recognised"
    assert await status_for("rarexamab") == "not_covered"


async def test_one_unresolved_mention_stops_the_whole_request(
    make_fake_llm, make_deps, make_graph, corpus
):
    # An interaction question where only one drug resolved must not be answered for one drug.
    resolver = FakeResolver({"hydroxyzin": HYDROXYZINE})  # "zzzdrug" is unknown
    llm = make_fake_llm([analysis(intent="interaction", mentions=["hydroxyzin", "zzzdrug"])])
    deps = make_deps(llm, resolver=resolver)
    out = await run_ask(make_graph(deps), deps, question="hydroxyzin with zzzdrug?")
    assert out["status"] == "drug_not_recognised" and out["answer"] == []
    assert len(llm.calls) == 1


# --- no mention --------------------------------------------------------------------------------


async def test_class_question_without_a_drug_runs_the_unfiltered_branches(
    make_fake_llm, make_deps, make_graph, corpus
):
    resolver, facts = hydroxyzin_resolver(), FakeRxFacts({"1": ["Brand: Atarax"]})
    query = "Hydroxyzine is contraindicated in early pregnancy and in QT prolongation."
    llm = make_fake_llm([analysis(intent="class_question", search_query=query), f"{query} [c1]"])
    deps = make_deps(llm, resolver=resolver, rx_facts=facts)
    q = "Which labels mention QT prolongation?"
    out = await run_ask(make_graph(deps), deps, question=q, debug=True)
    assert out["status"] == "answered"
    assert resolver.calls == [] and facts.calls == []  # nothing to resolve, no keys, no facts
    assert out["debug"]["ingredient_keys"] == []
    assert {"retrieve_dense", "retrieve_lexical", "fetch_rx_facts"} <= set(
        out["debug"]["timings_ms"]
    )


@pytest.mark.parametrize("intent", ["label_qa", "interaction"])
async def test_no_mention_on_a_drug_question_never_runs_unfiltered(
    make_fake_llm, make_deps, make_graph, corpus, intent
):
    resolver = hydroxyzin_resolver()
    llm = make_fake_llm([analysis(intent=intent, mentions=[])])
    deps = make_deps(llm, resolver=resolver)
    out = await run_ask(make_graph(deps), deps, question="What is the max daily dose?", debug=True)
    assert out["status"] == "drug_not_recognised" and out["answer"] == []
    assert resolver.calls == []
    assert set(out["debug"]["timings_ms"]) == {"analyze", "resolve", "finalize"}
    assert len(llm.calls) == 1


# --- happy path --------------------------------------------------------------------------------


async def test_happy_path_filters_to_the_resolved_drug_and_shows_how_it_matched(
    make_fake_llm, make_deps, make_graph, corpus
):
    resolver = hydroxyzin_resolver()
    facts = FakeRxFacts(
        {"1": ["Brand: Atarax", "Strength: 25 mg tablet"], "2": ["Brand: Apresoline"]}
    )
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), f"{DOSE_TEXT} [c1][r1]"])
    deps = make_deps(llm, resolver=resolver, rx_facts=facts)
    graph = make_graph(deps)
    state = await final_state(graph, deps, HYDROXYZIN_Q)

    assert resolver.calls == [{"mentions": ["hydroxyzin"], "pinned": []}]
    assert facts.calls == [["1"]]
    assert state["ingredient_keys"] == ["1"]
    # FR-3: every ranked hit belongs to the resolved drug, though hydralazine is in the corpus.
    assert state["hits"] and {h.ingredient_key for h in state["hits"]} == {"1"}
    assert {c.ingredient_key for c in state["evidence"]} == {"1"}
    assert [f.id for f in state["rx_facts"]] == ["r1", "r2"]
    assert {f.retrieval_attempt for f in state["rx_facts"]} == {1}
    assert "Hydralazine" not in writer_prompts(llm)[0]
    assert "[r1] Brand: Atarax" in writer_prompts(llm)[0]

    out = state["response"]
    assert out["status"] == "answered"
    assert out["drugs"] == [
        {
            "mention": "hydroxyzin",
            "name": "hydroxyzine",
            "rxcui": "1",
            "ingredient_key": "1",
            "match": "approximate",
        }
    ]
    assert out["answer"][0]["citations"] == ["c1", "r1"]
    assert [s["id"] for s in out["sources"]] == ["c1", "r1"]
    assert out["sources"][1] == {"id": "r1", "rxcui": "1", "text": "Brand: Atarax"}
    assert out["meta"]["resolver"] is True and out["meta"]["verified"] is False


# --- weak evidence, then success; retry exhausted ---------------------------------------------


async def test_weak_evidence_then_rewrite_drops_sections_keeps_keys_and_succeeds(
    make_fake_llm, make_deps, make_graph, corpus
):
    # Attempt 1 filters to adverse reactions, a section hydroxyzine's label has no chunk for
    # (the "dosing under a messy section code" case): nothing is retrieved.
    llm = make_fake_llm(
        [
            analysis(mentions=["hydroxyzin"], sections=["34084-4"]),
            rewrite("dosage for anxiety in adults"),
            f"{DOSE_TEXT} [c1]",
        ]
    )
    facts = FakeRxFacts({"1": ["Brand: Atarax"]})
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), rx_facts=facts)
    state = await final_state(make_graph(deps), deps, HYDROXYZIN_Q)

    assert state["response"]["status"] == "answered"
    assert state["response"]["meta"]["retrieval_attempts"] == 2
    assert [c["kind"] for c in llm.calls] == ["structured", "structured", "stream"]
    # The rewrite saw the first query and the question as delimited data.
    assert "<previous_query>" in llm.calls[1]["messages"][0]["content"]
    # It dropped the section filter and changed the query, and nothing else.
    assert state["sections"] == []
    assert state["search_query"] == "dosage for anxiety in adults"
    assert state["ingredient_keys"] == ["1"] and state["mentions"] == ["hydroxyzin"]
    assert state["retrieval_attempt"] == 2
    assert {h.retrieval_attempt for h in state["hits"]} == {2}  # attempt 1 found nothing
    assert {h.ingredient_key for h in state["hits"]} == {"1"}  # still filtered to the drug
    # The facts branch ran on both attempts, with ids restarting each time.
    assert facts.calls == [["1"], ["1"]]
    assert [(f.id, f.retrieval_attempt) for f in state["rx_facts"]] == [("r1", 1), ("r1", 2)]


async def test_retry_fuses_only_attempt_two_hits(make_fake_llm, make_deps, make_graph, corpus):
    # Attempt 1 retrieves three weak dense hits (a gibberish query). The state keeps them
    # (the reducer only appends), and attempt 2 retrieves the same chunks again. Fusion must
    # score attempt-2 ranks only, or every chunk's fused score would be inflated by the stale
    # ranks from attempt 1.
    llm = make_fake_llm(
        [
            analysis(mentions=["hydroxyzin"], search_query=GIBBERISH),
            rewrite(DOSE_TEXT),
            f"{DOSE_TEXT} [c1]",
        ]
    )
    deps = make_deps(llm, resolver=hydroxyzin_resolver())
    state = await final_state(make_graph(deps), deps, HYDROXYZIN_Q)

    assert {h.retrieval_attempt for h in state["hits"]} == {1, 2}
    assert len([h for h in state["hits"] if h.retrieval_attempt == 1]) == 3
    k = deps.thresholds.retrieval.rrf_k
    by_id = {c.id: c.fused_score for c in state["evidence"]}
    assert by_id["c1"] == pytest.approx(2 / (k + 1))  # dense rank 1 + lexical rank 1, attempt 2
    assert by_id["c2"] == pytest.approx(1 / (k + 2))  # dense rank 2 only: no stale rank added
    assert by_id["c3"] == pytest.approx(1 / (k + 3))


async def test_weak_evidence_after_one_retry_is_insufficient_evidence(
    make_fake_llm, make_deps, make_graph, corpus
):
    llm = make_fake_llm(
        [analysis(mentions=["hydroxyzin"], search_query=GIBBERISH), rewrite(GIBBERISH)]
    )
    facts = FakeRxFacts()
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), rx_facts=facts)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q, debug=True)

    assert out["status"] == "insufficient_evidence" and out["answer"] == []
    assert out["meta"]["retrieval_attempts"] == 2 and out["meta"]["generation_attempts"] == 0
    assert len(llm.calls) == 2  # analyze, one rewrite; no second rewrite and no writer
    assert "answer" not in out["debug"]["timings_ms"]
    assert facts.calls == [["1"], ["1"]]  # the facts branch ran on both attempts
    # The user still learns which drug the request was about.
    assert [d["name"] for d in out["drugs"]] == ["hydroxyzine"]


# --- regeneration ------------------------------------------------------------------------------

TWO_SENTENCES = "Sentence A is true. [c1]\nSentence B is invented. [c1]"
TWO_MORE = "Sentence C is true. [c1]\nSentence D is true. [c1]"


async def test_low_support_regenerates_once_with_a_stricter_prompt(
    make_fake_llm, make_deps, make_graph, corpus
):
    verifier = FakeVerifier([["supported", "unsupported"], ["supported", "supported"]])
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), TWO_SENTENCES, TWO_MORE])
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), verifier=verifier)
    graph = make_graph(deps)
    state = await final_state(graph, deps, HYDROXYZIN_Q)
    out = state["response"]

    assert out["status"] == "answered"
    assert [s["text"] for s in out["answer"]] == ["Sentence C is true.", "Sentence D is true."]
    assert out["meta"]["generation_attempts"] == 2 and state["generation_attempt"] == 2
    assert out["meta"]["verified"] is True
    assert len(verifier.calls) == 2 and state["support_ratio"] == 1.0
    # The first attempt's rejected sentence is counted and kept for the debug payload.
    assert out["meta"]["removed_sentences"] == 1
    assert [s.text for s in state["removed_sentences"]] == ["Sentence B is invented."]

    first, second = writer_prompts(llm)
    assert "<do_not_repeat>" not in first
    assert "<do_not_repeat>\n- Sentence B is invented.\n</do_not_repeat>" in second
    assert "Sentence A" not in second.split("<do_not_repeat>")[1]  # supported ones are not listed
    assert STRICT_ADDENDUM not in llm.calls[1]["system"]
    assert STRICT_ADDENDUM in llm.calls[2]["system"]


async def test_support_at_the_threshold_does_not_regenerate(
    make_fake_llm, make_deps, make_graph, corpus
):
    # 4 of 5 is exactly the 0.8 minimum: "at least 0.8" passes.
    reply = "\n".join(f"Sentence {i} is true. [c1]" for i in range(5))
    verifier = FakeVerifier([["supported"] * 4 + ["unsupported"]])
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), reply])
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), verifier=verifier)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q)
    assert out["meta"]["generation_attempts"] == 1 and len(verifier.calls) == 1
    assert len(out["answer"]) == 4 and out["meta"]["removed_sentences"] == 1


async def test_regeneration_exhausted_keeps_only_supported_sentences_of_the_last_attempt(
    make_fake_llm, make_deps, make_graph, corpus
):
    verifier = FakeVerifier([["supported", "unsupported"], ["unsupported", "supported"]])
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), TWO_SENTENCES, TWO_MORE])
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), verifier=verifier)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q, debug=True)

    # Still below 0.8 after attempt 2, and there is no third attempt (NFR-6).
    assert len(llm.calls) == 3 and len(verifier.calls) == 2
    assert out["meta"]["generation_attempts"] == 2
    assert out["status"] == "answered"
    assert [s["text"] for s in out["answer"]] == ["Sentence D is true."]  # FR-6: supported only
    assert out["meta"]["removed_sentences"] == 2
    assert [s["text"] for s in out["debug"]["removed_sentences"]] == [
        "Sentence B is invented.",
        "Sentence C is true.",
    ]
    assert out["debug"]["support_ratio"] == 0.5


async def test_regeneration_exhausted_with_nothing_supported_is_insufficient_evidence(
    make_fake_llm, make_deps, make_graph, corpus
):
    verifier = FakeVerifier([["unsupported", "contradicted"]])
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), TWO_SENTENCES, TWO_MORE])
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), verifier=verifier)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q)
    assert out["status"] == "insufficient_evidence"
    assert out["answer"] == [] and out["sources"] == []
    assert out["meta"]["removed_sentences"] == 4 and out["meta"]["generation_attempts"] == 2


async def test_uncited_sentences_count_against_support_and_trigger_regeneration(
    make_fake_llm, make_deps, make_graph, corpus
):
    verifier = FakeVerifier([lambda ss: ["supported"] * len(ss)])
    llm = make_fake_llm(
        [
            analysis(mentions=["hydroxyzin"]),
            "No citation here.\nCited sentence. [c1]",
            "Another cited sentence. [c1]",
        ]
    )
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), verifier=verifier)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q)
    assert out["meta"]["generation_attempts"] == 2
    # 1 of 2 sentences supported (the uncited one counts in the denominator): 0.5 < 0.8.
    assert [s["text"] for s in out["answer"]] == ["Another cited sentence."]
    assert out["meta"]["removed_sentences"] == 1


@pytest.mark.parametrize("second_reply", ["INSUFFICIENT", "refusal"])
async def test_second_attempt_that_writes_nothing_keeps_the_first_attempts_supported_sentences(
    make_fake_llm, make_deps, make_graph, corpus, second_reply
):
    # One rule for both: a failed retry does not make verified sentences false (FR-6 holds,
    # they were verified), and the first attempt's rejected sentence is still counted once.
    def refuse(messages, system):
        raise LLMRefusalError("declined")

    reply = refuse if second_reply == "refusal" else "INSUFFICIENT"
    verifier = FakeVerifier([["supported", "unsupported"]])
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), TWO_SENTENCES, reply])
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), verifier=verifier)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q, debug=True)
    assert out["status"] == "answered"
    assert [s["text"] for s in out["answer"]] == ["Sentence A is true."]
    assert out["meta"]["generation_attempts"] == 2 and out["meta"]["removed_sentences"] == 1
    assert len(verifier.calls) == 1  # nothing new to verify


async def test_debug_payload_says_when_a_drug_was_resolved_from_the_local_table_only(
    make_fake_llm, make_deps, make_graph, corpus
):
    local_only = HYDROXYZINE.model_copy(update={"degraded": True})  # RxNav was down (NFR-4)
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), f"{DOSE_TEXT} [c1]"])
    deps = make_deps(llm, resolver=FakeResolver({"hydroxyzin": local_only}))
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q, debug=True)
    assert out["debug"]["degraded"] is True
    assert out["debug"]["resolutions"][0]["degraded"] is True

    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), f"{DOSE_TEXT} [c1]"])
    deps = make_deps(llm, resolver=hydroxyzin_resolver())
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q, debug=True)
    assert out["debug"]["degraded"] is False


async def test_writer_saying_insufficient_is_not_regenerated(
    make_fake_llm, make_deps, make_graph, corpus
):
    verifier = FakeVerifier([[]])
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), "INSUFFICIENT"])
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), verifier=verifier)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q)
    assert out["status"] == "insufficient_evidence"
    assert len(llm.calls) == 2 and verifier.calls == []
    assert out["meta"]["generation_attempts"] == 1


async def test_verifier_flag_off_returns_every_sentence_and_never_regenerates(
    make_fake_llm, make_deps, make_graph, corpus
):
    # AT-2 baseline: retrieval, prompt and model unchanged, verifier switched off.
    verifier = FakeVerifier([["unsupported", "unsupported"]])
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), TWO_SENTENCES])
    deps = make_deps(llm, resolver=hydroxyzin_resolver(), verifier=verifier)
    out = await run_ask(make_graph(deps), deps, question=HYDROXYZIN_Q, flags=Flags(verifier=False))
    assert verifier.calls == [] and len(llm.calls) == 2
    assert len(out["answer"]) == 2 and out["meta"]["verified"] is False
    assert out["meta"]["generation_attempts"] == 1


async def test_verifier_requested_but_missing_fails_closed(
    make_fake_llm, make_deps, make_graph, corpus
):
    # run_ask would force the flag off. Invoking the graph directly with the flag on and no
    # verifier must not leak unchecked sentences (FR-6).
    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), TWO_SENTENCES])
    deps = make_deps(llm, resolver=hydroxyzin_resolver())
    graph = make_graph(deps)
    state = initial_state(HYDROXYZIN_Q, flags=Flags(verifier=True))
    final = await graph.ainvoke(state, config={"recursion_limit": 12})
    assert final["response"]["status"] == "insufficient_evidence"
    assert final["response"]["answer"] == []


# --- resolver off: the AT-1 baseline -----------------------------------------------------------

BASELINE_QUERY = "dosage adults four times daily"


@pytest.mark.parametrize("build", ["no_resolver_in_build", "flag_off_in_eval_mode"])
async def test_resolver_off_baseline_extracts_no_mentions_and_searches_every_label(
    make_fake_llm, make_deps, make_graph, corpus, build
):
    resolver = hydroxyzin_resolver()
    llm = make_fake_llm(
        [
            analysis(mentions=["hydroxyzin"], search_query=BASELINE_QUERY),
            "Both labels give a dosage. [c1][c2]",
        ]
    )
    if build == "no_resolver_in_build":
        deps, flags = make_deps(llm), Flags()
    else:
        deps, flags = make_deps(llm, resolver=resolver), Flags(resolver=False)
    state = await final_state(make_graph(deps), deps, HYDROXYZIN_Q, flags=flags)

    assert resolver.calls == []
    assert state["mentions"] == [] and state["ingredient_keys"] == []
    # Unfiltered over the whole corpus: the look-alike drug's text is in the evidence too.
    assert {c.ingredient_key for c in state["evidence"]} == {"1", "2"}
    prompt = writer_prompts(llm)[0]
    assert DOSE_TEXT in prompt and HYDRALAZINE_DOSE_TEXT in prompt
    out = state["response"]
    assert out["status"] == "answered" and out["drugs"] == []
    assert out["meta"]["resolver"] is False


async def test_resolver_off_runs_unfiltered_whatever_the_intent(
    make_fake_llm, make_deps, make_graph, corpus
):
    # A label_qa question with no mention is `drug_not_recognised` with the resolver on, and
    # an unfiltered search with it off: that is the baseline the 85% claim is measured against.
    llm = make_fake_llm([analysis(search_query=BASELINE_QUERY), "Dosage is given. [c1]"])
    deps = make_deps(llm, resolver=hydroxyzin_resolver())
    out = await run_ask(make_graph(deps), deps, question="max dose?", flags=Flags(resolver=False))
    assert out["status"] == "answered"


# --- two drugs ---------------------------------------------------------------------------------


async def test_interaction_question_retrieves_evidence_for_both_drugs(
    make_fake_llm, make_deps, make_graph, corpus
):
    resolver = FakeResolver(
        {
            "hydroxyzin": HYDROXYZINE,
            "hydralazine": resolved("hydralazine", "2", "hydralazine"),
        }
    )
    facts = FakeRxFacts({"1": ["Brand: Atarax"], "2": ["Brand: Apresoline"]})
    llm = make_fake_llm(
        [
            analysis(
                intent="interaction",
                mentions=["hydroxyzin", "hydralazine"],
                search_query=BASELINE_QUERY,
            ),
            "Each label gives a dosage. [c1][c2][r1][r2]",
        ]
    )
    deps = make_deps(llm, resolver=resolver, rx_facts=facts)
    graph = make_graph(deps)
    state = await final_state(graph, deps, "Does hydroxyzin interact with hydralazine?")

    assert state["ingredient_keys"] == ["1", "2"]
    assert facts.calls == [["1", "2"]]
    # One query per drug in each branch: both drugs have ranked hits in both lists.
    assert {(h.ingredient_key, h.source) for h in state["hits"]} == {
        ("1", "dense"),
        ("2", "dense"),
        ("1", "lexical"),
        ("2", "lexical"),
    }
    assert {c.ingredient_key for c in state["evidence"]} == {"1", "2"}
    out = state["response"]
    assert out["status"] == "answered"
    assert [d["name"] for d in out["drugs"]] == ["hydroxyzine", "hydralazine"]
    assert len(resolver.calls) == 1 and resolver.calls[0]["mentions"] == [
        "hydroxyzin",
        "hydralazine",
    ]


async def test_two_mentions_of_one_drug_search_it_once(
    make_fake_llm, make_deps, make_graph, corpus
):
    resolver = FakeResolver(
        {
            "atarax": resolved("atarax", "1", "hydroxyzine", match="brand", product_hint="77"),
            "hydroxyzine": resolved("hydroxyzine", "1", "hydroxyzine", match="exact"),
        }
    )
    llm = make_fake_llm([analysis(mentions=["Atarax", "hydroxyzine"]), f"{DOSE_TEXT} [c1]"])
    deps = make_deps(llm, resolver=resolver)
    state = await final_state(make_graph(deps), deps, "Is Atarax the same as hydroxyzine?")
    assert state["ingredient_keys"] == ["1"]
    assert len([h for h in state["hits"] if h.source == "dense"]) == 3  # one query, not two
    assert [d["match"] for d in state["response"]["drugs"]] == ["brand", "exact"]


# --- clarification round trip ------------------------------------------------------------------


async def test_pinned_rxcui_resend_after_a_clarification_resolves_and_answers(
    make_fake_llm, make_deps, make_graph, corpus
):
    resolver = FakeResolver(
        {
            "hydroxyzin": ambiguous(
                "hydroxyzin", [("1", "hydroxyzine", 0.88), ("2", "hydralazine", 0.86)]
            )
        }
    )
    llm = make_fake_llm(
        [
            analysis(mentions=["hydroxyzin"]),
            analysis(mentions=["hydroxyzin"]),
            f"{DOSE_TEXT} [c1]",
        ]
    )
    deps = make_deps(llm, resolver=resolver)
    graph = make_graph(deps)

    first = await run_ask(graph, deps, question=HYDROXYZIN_Q)
    assert first["status"] == "needs_clarification"
    pick = first["candidates"][0]["ingredient_key"]

    second = await run_ask(graph, deps, question=HYDROXYZIN_Q, pinned_rxcuis=[pick])
    assert resolver.calls[0]["pinned"] == [] and resolver.calls[1]["pinned"] == ["1"]
    assert second["status"] == "answered"
    assert second["drugs"][0]["match"] == "pinned" and second["drugs"][0]["name"] == "hydroxyzine"
    assert "Hydralazine" not in writer_prompts(llm)[0]
