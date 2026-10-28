"""A plain-language `message` on every response that is not an answer (JSON and SSE `done`)."""

import httpx
import pytest

from pharmarag.app.main import create_app
from pharmarag.app.runtime import Runtime
from pharmarag.graph.messages import (
    NO_DRUG_NAMED,
    OUT_OF_SCOPE,
    PATIENT_ADVICE,
    status_message,
)
from pharmarag.graph.run import run_ask
from tests.graph.fakes import (
    DOSE_TEXT,
    FakeResolver,
    ambiguous,
    analysis,
    not_covered,
    resolved,
    unknown,
)
from tests.graph.test_sse import parse_sse

pytestmark = pytest.mark.pg

TYLENOL_Q = "can i take tylenol when i have high fever?"
DOSE_Q = "What is the max daily dose of hydroxyzin?"
TYLENOL_OTC = not_covered("tylenol", key="161", name="acetaminophen")
HYDROXYZINE = resolved("hydroxyzin", "1", "hydroxyzine", match="approximate")

PATIENT_ADVICE_TEXT = (
    "I can't give personal medical advice — please ask a pharmacist or doctor. "
    "I can tell you what a drug's FDA label says, e.g. "
    "'What does the acetaminophen label say about fever?'"
)


async def ask(make_fake_llm, make_deps, make_graph, replies, question, **deps_kw):
    deps = make_deps(make_fake_llm(replies), **deps_kw)
    out = await run_ask(make_graph(deps), deps, question=question, debug=True)
    return out, deps


# --- declined ------------------------------------------------------------------------------------


async def test_patient_advice_is_declined_with_the_required_message_and_names_the_drug(
    make_fake_llm, make_deps, make_graph, corpus
):
    resolver = FakeResolver({"tylenol": TYLENOL_OTC})
    out, _ = await ask(
        make_fake_llm,
        make_deps,
        make_graph,
        [analysis(intent="patient_advice", mentions=["tylenol"])],
        TYLENOL_Q,
        resolver=resolver,
    )
    assert out["status"] == "declined" and out["answer"] == [] and out["sources"] == []
    assert out["message"] == PATIENT_ADVICE_TEXT == PATIENT_ADVICE
    assert out["notice"] == "From FDA drug labels via DailyMed. Not medical advice."
    # SPEC (finalize.declined_drug_entries): the typed name is echoed, unresolved.
    assert out["drugs"] == [
        {"mention": "tylenol", "name": None, "rxcui": None, "ingredient_key": None, "match": None}
    ]


async def test_a_declined_request_never_resolves_or_retrieves(
    make_fake_llm, make_deps, make_graph, corpus
):
    resolver = FakeResolver({"tylenol": TYLENOL_OTC})
    llm_replies = [analysis(intent="patient_advice", mentions=["tylenol"])]
    out, deps = await ask(
        make_fake_llm, make_deps, make_graph, llm_replies, TYLENOL_Q, resolver=resolver
    )
    assert resolver.calls == []  # no RxNav, no alias lookup
    assert set(out["debug"]["timings_ms"]) == {"analyze", "finalize"}  # no retrieval node ran
    assert len(deps.llm.calls) == 1  # analyze only: no writer
    assert out["meta"]["retrieval_attempts"] == 0 and out["meta"]["generation_attempts"] == 0


async def test_out_of_scope_is_declined_with_a_message_that_steers_back_to_labels(
    make_fake_llm, make_deps, make_graph, corpus
):
    out, _ = await ask(
        make_fake_llm,
        make_deps,
        make_graph,
        [analysis(intent="out_of_scope")],
        "What is the weather like?",
    )
    assert out["status"] == "declined" and out["message"] == OUT_OF_SCOPE
    assert "label" in out["message"] and "FDA" in out["message"]
    assert out["message"] != PATIENT_ADVICE
    assert out["drugs"] == []


async def test_a_declined_request_with_no_drug_has_no_drugs(
    make_fake_llm, make_deps, make_graph, corpus
):
    out, _ = await ask(
        make_fake_llm,
        make_deps,
        make_graph,
        [analysis(intent="patient_advice")],
        "should i stop my pills?",
    )
    assert out["status"] == "declined" and out["drugs"] == []


async def test_a_provider_refusal_to_analyse_is_declined_with_the_out_of_scope_message(
    make_fake_llm, make_deps, make_graph, corpus
):
    from pharmarag.llm.errors import LLMRefusalError

    class Refuses:
        model_id = "fake"
        calls: list = []

        async def structured(self, *a, **k):
            raise LLMRefusalError("refused")

    deps = make_deps(Refuses())
    out = await run_ask(make_graph(deps), deps, question="something blocked")
    assert out["status"] == "declined" and out["message"] == OUT_OF_SCOPE


# --- the other statuses --------------------------------------------------------------------------


async def test_not_covered_names_the_resolved_drug_and_says_why(
    make_fake_llm, make_deps, make_graph, corpus
):
    out, _ = await ask(
        make_fake_llm,
        make_deps,
        make_graph,
        [analysis(mentions=["tylenol"])],
        "what does the tylenol label say about fever?",
        resolver=FakeResolver({"tylenol": TYLENOL_OTC}),
    )
    assert out["status"] == "not_covered"
    assert out["message"] == (
        "Acetaminophen resolved, but its label isn't in this service's corpus "
        "(prescription labels only; over-the-counter drugs are not covered)."
    )
    # The drug is still listed, so a client can show "acetaminophen" next to the message.
    assert [d["name"] for d in out["drugs"]] == ["acetaminophen"]


async def test_needs_clarification_lists_the_candidates_and_says_to_pin(
    make_fake_llm, make_deps, make_graph, corpus
):
    amb = ambiguous("hydroxazine", [("1", "hydroxyzine", 0.9), ("2", "hydralazine", 0.88)])
    out, _ = await ask(
        make_fake_llm,
        make_deps,
        make_graph,
        [analysis(mentions=["hydroxazine"])],
        "dose of hydroxazine?",
        resolver=FakeResolver({"hydroxazine": amb}),
    )
    assert out["status"] == "needs_clarification"
    msg = out["message"]
    assert "'hydroxazine'" in msg and "hydroxyzine (1)" in msg and "hydralazine (2)" in msg
    assert "pinned_rxcuis" in msg
    assert [c["ingredient_key"] for c in out["candidates"]] == ["1", "2"]


async def test_drug_not_recognised_asks_to_check_the_spelling_and_names_the_mention(
    make_fake_llm, make_deps, make_graph, corpus
):
    out, _ = await ask(
        make_fake_llm,
        make_deps,
        make_graph,
        [analysis(mentions=["zzxqv"])],
        "dose of zzxqv?",
        resolver=FakeResolver({"zzxqv": unknown("zzxqv")}),
    )
    assert out["status"] == "drug_not_recognised"
    assert out["message"] == "I didn't recognise 'zzxqv'. Please check the spelling and try again."


async def test_drug_not_recognised_when_no_drug_was_named(
    make_fake_llm, make_deps, make_graph, corpus
):
    out, _ = await ask(
        make_fake_llm,
        make_deps,
        make_graph,
        [analysis()],
        "What is the max dose?",
        resolver=FakeResolver({}),
    )
    assert out["status"] == "drug_not_recognised" and out["message"] == NO_DRUG_NAMED
    assert "spelling" in out["message"]


async def test_insufficient_evidence_says_the_label_does_not_answer(
    make_fake_llm, make_deps, make_graph, corpus
):
    out, _ = await ask(
        make_fake_llm, make_deps, make_graph, [analysis(), "INSUFFICIENT"], "What is the dose?"
    )
    assert out["status"] == "insufficient_evidence"
    assert "label doesn't answer this question" in out["message"]


async def test_an_answered_response_has_no_message(make_fake_llm, make_deps, make_graph, corpus):
    out, _ = await ask(
        make_fake_llm,
        make_deps,
        make_graph,
        [analysis(), f"{DOSE_TEXT} [c1]"],
        "What is the hydroxyzine dosage for anxiety in adults?",
    )
    assert out["status"] == "answered" and "message" not in out


# --- the pure function ---------------------------------------------------------------------------


def test_not_covered_with_several_drugs_uses_the_plural():
    msg = status_message(
        "not_covered",
        intent="label_qa",
        resolutions=[
            not_covered("tylenol", name="acetaminophen"),
            not_covered("advil", key="5640", name="ibuprofen"),
        ],
    )
    assert msg.startswith("Acetaminophen and ibuprofen resolved, but their labels aren't")


def test_not_covered_falls_back_to_the_mention_when_the_resolution_has_no_name():
    r = not_covered("tylenol").model_copy(update={"name": None})
    assert status_message("not_covered", intent=None, resolutions=[r]).startswith(
        "Tylenol resolved"
    )


def test_every_non_answered_status_has_a_message_and_answered_has_none():
    for status in (
        "declined",
        "not_covered",
        "needs_clarification",
        "drug_not_recognised",
        "insufficient_evidence",
    ):
        assert status_message(status, intent=None, resolutions=[])
    assert status_message("answered", intent=None, resolutions=[]) is None


# --- JSON and SSE over HTTP ----------------------------------------------------------------------


@pytest.fixture
def api(make_deps, make_graph, settings, make_fake_llm):
    def _make(replies, **deps_kw):
        deps = make_deps(make_fake_llm(replies), **deps_kw)
        deps.settings = settings.model_copy(update={"cache_backend": "memory"})
        app = create_app(deps.settings)
        app.state.runtime = Runtime(deps=deps, graph=make_graph(deps))
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")

    return _make


async def test_the_json_endpoint_returns_the_message(api, corpus):
    client = api(
        [analysis(intent="patient_advice", mentions=["tylenol"])],
        resolver=FakeResolver({"tylenol": TYLENOL_OTC}),
    )
    async with client:
        body = (await client.post("/v1/ask", json={"question": TYLENOL_Q})).json()
    assert body["status"] == "declined" and body["message"] == PATIENT_ADVICE_TEXT
    assert body["drugs"][0]["mention"] == "tylenol" and body["drugs"][0]["name"] is None


async def test_sse_done_carries_the_message_for_a_declined_request(api, corpus):
    client = api(
        [analysis(intent="patient_advice", mentions=["tylenol"])],
        resolver=FakeResolver({"tylenol": TYLENOL_OTC}),
    )
    async with client:
        r = await client.post("/v1/ask", json={"question": TYLENOL_Q, "stream": True})
    events = parse_sse(r.text)
    assert [n for n, _ in events] == ["done"]  # a declined request resolves nothing: no drugs event
    done = events[-1][1]
    assert done["status"] == "declined" and done["message"] == PATIENT_ADVICE_TEXT
    assert done["drugs"][0]["mention"] == "tylenol"


async def test_sse_done_carries_the_message_for_not_covered(api, corpus):
    client = api([analysis(mentions=["tylenol"])], resolver=FakeResolver({"tylenol": TYLENOL_OTC}))
    async with client:
        r = await client.post(
            "/v1/ask", json={"question": "what does the tylenol label say?", "stream": True}
        )
    events = parse_sse(r.text)
    assert [n for n, _ in events] == ["drugs", "done"]
    done = events[-1][1]
    assert done["status"] == "not_covered" and done["message"].startswith("Acetaminophen resolved")


async def test_sse_done_has_no_message_when_answered(api, corpus):
    client = api(
        [analysis(mentions=["hydroxyzin"]), f"{DOSE_TEXT} [c1]"],
        resolver=FakeResolver({"hydroxyzin": HYDROXYZINE}),
    )
    async with client:
        r = await client.post("/v1/ask", json={"question": DOSE_Q, "stream": True})
    done = parse_sse(r.text)[-1][1]
    assert done["status"] == "answered" and "message" not in done
