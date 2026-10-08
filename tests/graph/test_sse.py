"""POST /v1/ask with stream=true over the real graph and PostgreSQL (spec section 10, FR-6).

The writer, the resolver and the verifier are fakes. What is under test is the wiring: event
order, that a sentence that failed verification can never reach the client, that `done` is the
authoritative answer, early exits, and what happens when the LLM fails mid-answer.
"""

import json

import httpx
import pytest

from pharmarag.app.main import create_app, get_runtime
from pharmarag.app.runtime import Runtime
from pharmarag.llm.errors import LLMUnavailableError
from tests.graph.fakes import (
    HYDROXYZIN_Q,
    FakeResolver,
    FakeVerifier,
    ambiguous,
    analysis,
    not_covered,
    resolved,
)

pytestmark = pytest.mark.pg

HYDROXYZINE = resolved("hydroxyzin", "1", "hydroxyzine", match="approximate")
THREE = "Sentence A is true. [c1]\nSentence B is invented. [c1]\nSentence C is true. [c1]"
ASK = analysis(mentions=["hydroxyzin"])


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        name, data = "", ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        events.append((name, json.loads(data)))
    return events


@pytest.fixture
def client_for(make_deps, make_graph, settings, corpus):
    """client_for(llm, verifier, mode=...) -> (httpx client, deps). One app per call."""

    def _make(llm, verifier=None, *, mode="stream", resolver=None, eval_mode=True):
        deps = make_deps(
            llm,
            resolver=resolver or FakeResolver({"hydroxyzin": HYDROXYZINE}),
            verifier=verifier,
        )
        deps.settings = settings.model_copy(update={"verify_mode": mode, "eval_mode": eval_mode})
        app = create_app(deps.settings)
        app.dependency_overrides[get_runtime] = lambda: Runtime(deps=deps, graph=make_graph(deps))
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        return client, deps

    return _make


async def ask(client, *, stream: bool, question=HYDROXYZIN_Q, headers=None, **extra):
    async with client:
        return await client.post(
            "/v1/ask", json={"question": question, "stream": stream, **extra}, headers=headers
        )


def reject_b(sentences):
    return ["unsupported" if "invented" in s.text else "supported" for s in sentences]


# --- order and shape ----------------------------------------------------------------------------


async def test_events_come_in_order_drugs_then_sentences_then_done(client_for, make_fake_llm):
    llm = make_fake_llm([ASK, "Sentence A is true. [c1]\nSentence C is true. [c1]"])
    client, _ = client_for(llm, FakeVerifier([lambda ss: ["supported"] * len(ss)]))
    r = await ask(client, stream=True)

    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache" and r.headers["x-request-id"]
    events = parse_sse(r.text)
    assert [n for n, _ in events] == ["drugs", "sentence", "sentence", "done"]
    assert events[0][1] == [
        {
            "mention": "hydroxyzin",
            "name": "hydroxyzine",
            "rxcui": "1",
            "ingredient_key": "1",
            "match": "approximate",
        }
    ]
    first = events[1][1]
    assert first == {
        "index": 0,
        "text": "Sentence A is true.",
        "citations": ["c1"],
        "support": {"span": "c1:0", "score": 0.93},
    }
    assert events[2][1]["index"] == 1
    done = events[3][1]
    assert done["status"] == "answered" and done["notice"]
    assert done["sources"][0]["id"] == "c1" and done["sources"][0]["url"].startswith("https://")
    assert set(done["meta"]) >= {"corpus_version", "latency_ms", "generation_attempts"}


# --- FR-6 over SSE ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["stream", "sequential"])
async def test_a_rejected_sentence_is_never_emitted_and_released_equals_done(
    client_for, make_fake_llm, mode
):
    # B is rejected, so the support ratio is 2/3 = 0.67 < 0.8. In stream mode that is final for
    # SSE (A and C are already out); in sequential mode a regeneration follows (below).
    replies = [ASK, THREE] + (["Sentence D is true. [c1]\nSentence E is true. [c1]"] * 2)
    client, _ = client_for(
        make_fake_llm(replies),
        FakeVerifier([reject_b, lambda ss: ["supported"] * len(ss)]),
        mode=mode,
    )
    r = await ask(client, stream=True)
    events = parse_sse(r.text)
    sent = [d["text"] for n, d in events if n == "sentence"]
    done = events[-1][1]
    assert events[-1][0] == "done"
    assert "Sentence B is invented." not in sent
    assert "Sentence B is invented." not in r.text.split("event: done")[0]
    assert "Sentence B is invented." not in [a["text"] for a in done["answer"]]
    # Everything the client saw is in the authoritative payload, in the same order.
    assert sent == [a["text"] for a in done["answer"]]


async def test_contradicted_and_uncited_sentences_are_not_emitted_either(client_for, make_fake_llm):
    llm = make_fake_llm([ASK, "Contradicted one. [c1]\nNo citation at all.\nGood one. [c1]"] * 2)
    client, _ = client_for(llm, FakeVerifier([["contradicted", "supported", "supported"]]))
    events = parse_sse((await ask(client, stream=True)).text)
    assert [d["text"] for n, d in events if n == "sentence"] == ["Good one."]


async def test_a_sentence_is_emitted_before_the_llm_finished_writing(client_for):
    """Stream mode releases a verified sentence while the writer is still mid-answer."""
    seen_before_end: list[int] = []

    class WatchingLLM:
        model_id = "watch"
        calls: list = []

        async def structured(self, messages, schema, **kw):
            return schema.model_validate_json(ASK)

        async def stream(self, messages, **kw):
            yield "Sentence A is true. [c1]\n"
            yield "Sentence C is true. [c1]\n"  # A is released once this has started
            for _ in range(50):  # give the verifier and the event loop time to run
                await __import__("asyncio").sleep(0.01)
                if verifier.calls:
                    break
            seen_before_end.append(len(verifier.calls))
            yield "Sentence D is true. [c1]"

    verifier = FakeVerifier([lambda ss: ["supported"] * len(ss)])
    client, _ = client_for(WatchingLLM(), verifier)
    events = parse_sse((await ask(client, stream=True)).text)
    assert seen_before_end == [1]  # the verifier had already been called for sentence A
    assert [n for n, _ in events].count("sentence") == 3


# --- done matches the non-stream response --------------------------------------------------------


CLEAN = "Sentence A is true. [c1]\nSentence C is true. [c3]"
DRAFT_WITH_ONE_BAD = (
    "Sentence A is true. [c1]\nSentence B is invented. [c2]\nSentence C is true. [c3]"
)
REDRAFT = "Sentence D is true. [c1]\nSentence E is true. [c2]"


def verdicts_by_text(sentences):
    return ["unsupported" if "invented" in s.text else "supported" for s in sentences]


@pytest.mark.parametrize(
    ("mode", "replies"),
    [
        ("off", [CLEAN]),
        ("sequential", [CLEAN]),
        ("stream", [CLEAN]),
        # A low-support draft that gets regenerated: sequential SSE buffers the first draft, so
        # its `done` is the non-stream response. (Stream mode differs by design: next test.)
        ("sequential", [DRAFT_WITH_ONE_BAD, REDRAFT]),
        ("off", [DRAFT_WITH_ONE_BAD]),
    ],
)
async def test_done_equals_the_non_stream_response(client_for, make_fake_llm, mode, replies):
    def build():
        llm = make_fake_llm([ASK, *replies])
        return client_for(
            llm, FakeVerifier([verdicts_by_text, lambda ss: ["supported"] * len(ss)]), mode=mode
        )

    plain_client, _ = build()
    stream_client, _ = build()
    plain = (await ask(plain_client, stream=False, debug=True)).json()
    events = parse_sse((await ask(stream_client, stream=True, debug=True)).text)
    done = events[-1][1]

    for body in (plain, done):
        body["meta"].pop("latency_ms")
        body["debug"].pop("timings_ms")
        body["debug"].pop("request_id")
    assert events[-1][0] == "done"
    assert done == plain
    assert [d["text"] for n, d in events if n == "sentence"] == [a["text"] for a in plain["answer"]]


async def test_stream_sse_done_differs_from_the_non_stream_response_only_by_not_regenerating(
    client_for, make_fake_llm
):
    """The one documented divergence (SPEC, spec sections 6 and 10). Same input, stream mode.
    Non-stream replaces the low-support draft with a second one. SSE has already shown A and C,
    so its `done` keeps them and does not regenerate."""
    script = [ASK, DRAFT_WITH_ONE_BAD, REDRAFT]
    verifier = FakeVerifier([verdicts_by_text, lambda ss: ["supported"] * len(ss)])
    plain_client, _ = client_for(make_fake_llm(script), verifier)
    sse_client, _ = client_for(
        make_fake_llm(script), FakeVerifier([verdicts_by_text, lambda ss: ["supported"] * len(ss)])
    )
    plain = (await ask(plain_client, stream=False)).json()
    done = parse_sse((await ask(sse_client, stream=True)).text)[-1][1]
    assert [a["text"] for a in plain["answer"]] == ["Sentence D is true.", "Sentence E is true."]
    assert [a["text"] for a in done["answer"]] == ["Sentence A is true.", "Sentence C is true."]
    assert plain["meta"]["generation_attempts"] == 2 and done["meta"]["generation_attempts"] == 1


# --- early exits --------------------------------------------------------------------------------


async def test_declined_sends_only_done(client_for, make_fake_llm):
    llm = make_fake_llm([analysis(intent="patient_advice", mentions=["hydroxyzin"])])
    client, _ = client_for(llm, FakeVerifier([[]]))
    events = parse_sse((await ask(client, stream=True)).text)
    assert [n for n, _ in events] == ["done"]
    assert events[0][1]["status"] == "declined"


async def test_clarification_sends_drugs_then_done_with_candidates(client_for, make_fake_llm):
    resolver = FakeResolver(
        {
            "hydroxyzin": ambiguous("hydroxyzin", [("1", "hydroxyzine", 0.88), ("2", "x", 0.86)]),
            "aspirin": resolved("aspirin", "9", "aspirin", match="exact"),
        }
    )
    llm = make_fake_llm([analysis(mentions=["aspirin", "hydroxyzin"])])
    client, _ = client_for(llm, FakeVerifier([[]]), resolver=resolver)
    q = "Can aspirin and hydroxyzin be taken together?"
    events = parse_sse((await ask(client, stream=True, question=q)).text)
    assert [n for n, _ in events] == ["drugs", "done"]
    assert [d["name"] for d in events[0][1]] == ["aspirin"]  # the one that did resolve
    assert events[1][1]["status"] == "needs_clarification"
    assert len(events[1][1]["candidates"]) == 2


async def test_not_covered_names_the_drug_in_drugs_then_done(client_for, make_fake_llm):
    resolver = FakeResolver({"hydroxyzin": not_covered("hydroxyzin", "999", "rarexamab")})
    client, _ = client_for(make_fake_llm([ASK]), FakeVerifier([[]]), resolver=resolver)
    events = parse_sse((await ask(client, stream=True)).text)
    assert [n for n, _ in events] == ["drugs", "done"]
    assert events[0][1][0]["name"] == "rarexamab"
    assert events[1][1]["status"] == "not_covered"


async def test_unrecognised_drug_sends_an_empty_drugs_event_then_done(client_for, make_fake_llm):
    client, _ = client_for(make_fake_llm([ASK]), FakeVerifier([[]]), resolver=FakeResolver({}))
    events = parse_sse((await ask(client, stream=True)).text)
    assert [n for n, _ in events] == ["drugs", "done"]
    assert events[0][1] == [] and events[1][1]["status"] == "drug_not_recognised"


async def test_writer_saying_insufficient_sends_drugs_then_done_without_sentences(
    client_for, make_fake_llm
):
    client, _ = client_for(make_fake_llm([ASK, "INSUFFICIENT"]), FakeVerifier([[]]))
    events = parse_sse((await ask(client, stream=True)).text)
    assert [n for n, _ in events] == ["drugs", "done"]
    assert events[1][1]["status"] == "insufficient_evidence" and events[1][1]["answer"] == []


async def test_verifier_off_in_eval_mode_emits_the_unverified_sentences(client_for, make_fake_llm):
    # The AT-2 baseline over SSE: no verdicts, so every cited sentence is shown, as in finalize.
    llm = make_fake_llm([ASK, "Sentence A is true. [c1]\nUncited line."])
    client, _ = client_for(llm, FakeVerifier([[]]))
    r = await ask(client, stream=True, headers={"X-Eval-Flags": "verifier=off"})
    events = parse_sse(r.text)
    assert [d["text"] for n, d in events if n == "sentence"] == ["Sentence A is true."]
    assert events[-1][1]["meta"]["verified"] is False


# --- regeneration after sentences were released (decision, spec section 6 and 10) ---------------


async def test_stream_sse_does_not_regenerate_once_a_sentence_was_released(
    client_for, make_fake_llm
):
    """SPEC decision: released sentences are final. A, C were shown, B rejected (ratio 0.67).
    A new draft would repeat or contradict what the client already has, so there is none."""
    llm = make_fake_llm([ASK, THREE])  # a second writer call would run the script dry and fail
    verifier = FakeVerifier([reject_b])
    client, _ = client_for(llm, verifier)
    events = parse_sse((await ask(client, stream=True)).text)

    done = events[-1][1]
    assert [d["text"] for n, d in events if n == "sentence"] == [
        "Sentence A is true.",
        "Sentence C is true.",
    ]
    assert [a["text"] for a in done["answer"]] == ["Sentence A is true.", "Sentence C is true."]
    assert done["meta"]["generation_attempts"] == 1 and done["meta"]["removed_sentences"] == 1
    assert len(verifier.calls) == 1


async def test_stream_sse_regenerates_when_nothing_was_released_and_shows_only_the_new_draft(
    client_for, make_fake_llm
):
    first = "Sentence A is invented. [c1]\nSentence B is invented. [c1]"
    second = "Sentence D is true. [c1]\nSentence E is true. [c1]"
    verifier = FakeVerifier([["unsupported", "unsupported"], ["supported", "supported"]])
    client, _ = client_for(make_fake_llm([ASK, first, second]), verifier)
    events = parse_sse((await ask(client, stream=True)).text)

    sent = [(d["index"], d["text"]) for n, d in events if n == "sentence"]
    assert sent == [(0, "Sentence D is true."), (1, "Sentence E is true.")]
    done = events[-1][1]
    assert [a["text"] for a in done["answer"]] == ["Sentence D is true.", "Sentence E is true."]
    assert done["meta"]["generation_attempts"] == 2 and done["meta"]["removed_sentences"] == 2
    assert "invented" not in r_text(events)


def r_text(events) -> str:
    return json.dumps([d for n, d in events if n == "sentence"])


async def test_sequential_sse_holds_back_a_draft_that_will_be_regenerated(
    client_for, make_fake_llm
):
    """Sequential mode knows the support ratio before it shows anything, so a draft that is about
    to be replaced is never shown, and the replacement is, as in the non-stream response."""
    second = "Sentence D is true. [c1]\nSentence E is true. [c1]"
    client, _ = client_for(
        make_fake_llm([ASK, THREE, second]),
        FakeVerifier([reject_b, lambda ss: ["supported"] * len(ss)]),
        mode="sequential",
    )
    events = parse_sse((await ask(client, stream=True)).text)
    sent = [d["text"] for n, d in events if n == "sentence"]
    assert sent == ["Sentence D is true.", "Sentence E is true."]
    assert [a["text"] for a in events[-1][1]["answer"]] == sent
    assert events[-1][1]["meta"]["generation_attempts"] == 2


# --- LLM failure mid-answer (item 4) --------------------------------------------------------------


class FlakyWriter:
    """analyze works; each writer call plays its script entry, a (text, error) pair: the text is
    streamed and then the error raised, or the call ends normally when the error is None."""

    model_id = "flaky"

    def __init__(self, writer_script):
        self.script = list(writer_script)
        self.writer_calls = 0

    async def structured(self, messages, schema, **kw):
        return schema.model_validate_json(ASK)

    async def stream(self, messages, **kw):
        text, error = self.script[min(self.writer_calls, len(self.script) - 1)]
        self.writer_calls += 1
        for piece in text.split(" "):
            yield piece + " "
        if error:
            raise error


OUTAGE = LLMUnavailableError("connection reset")


async def test_outage_before_any_release_is_retried_by_the_node_policy(client_for):
    llm = FlakyWriter([("Sentence A is tr", OUTAGE), ("Sentence A is true. [c1]", None)])
    client, _ = client_for(llm, FakeVerifier([["supported"]]))
    events = parse_sse((await ask(client, stream=True)).text)
    assert llm.writer_calls == 2
    assert [d["text"] for n, d in events if n == "sentence"] == ["Sentence A is true."]
    assert events[-1][1]["status"] == "answered"


async def test_outage_after_a_verified_but_unshown_sentence_is_still_retried_without_sse(
    client_for,
):
    # Non-SSE: A was verified but the response has not been sent, so a retry is invisible.
    llm = FlakyWriter(
        [("Sentence A is true. [c1]\nSentence B is tr", OUTAGE), ("Sentence C is true. [c1]", None)]
    )
    client, _ = client_for(llm, FakeVerifier([lambda ss: ["supported"] * len(ss)]))
    body = (await ask(client, stream=False)).json()
    assert llm.writer_calls == 2
    assert [a["text"] for a in body["answer"]] == ["Sentence C is true."]


async def test_outage_after_a_release_is_not_retried_and_the_answer_is_what_was_verified(
    client_for,
):
    llm = FlakyWriter(
        [
            ("Sentence A is true. [c1]\nSentence B is true. [c1]\nSentence C is tr", OUTAGE),
            ("never used. [c1]", None),
        ]
    )
    client, _ = client_for(llm, FakeVerifier([lambda ss: ["supported"] * len(ss)]))
    events = parse_sse((await ask(client, stream=True)).text)

    assert llm.writer_calls == 1  # no retry: a retried draft could contradict what was shown
    sent = [d["text"] for n, d in events if n == "sentence"]
    assert sent[0] == "Sentence A is true."
    assert "Sentence C is tr" not in sent  # the half-written line is never shown
    done = events[-1]
    assert done[0] == "done" and done[1]["status"] == "answered"
    assert [a["text"] for a in done[1]["answer"]] == sent  # done agrees with what was released
    assert done[1]["meta"]["generation_attempts"] == 1


async def test_three_failed_attempts_before_any_release_become_a_503_when_nothing_was_sent(
    client_for,
):
    class DeadAnalyze:
        model_id = "dead"

        async def structured(self, messages, schema, **kw):
            raise OUTAGE

    client, _ = client_for(DeadAnalyze(), FakeVerifier([[]]))
    r = await ask(client, stream=True)
    assert r.status_code == 503  # nothing had been sent yet, so a real status code is possible


async def test_a_fault_after_the_stream_began_ends_it_with_an_error_event_and_no_done(
    client_for,
):
    llm = FlakyWriter([("Sentence A is tr", OUTAGE)] * 3)  # retries exhausted in `answer`
    client, _ = client_for(llm, FakeVerifier([["supported"]]))
    r = await ask(client, stream=True)
    events = parse_sse(r.text)
    assert r.status_code == 200  # headers went out with the `drugs` event
    assert [n for n, _ in events] == ["drugs", "error"]
    assert events[-1][1]["status"] == 503
    assert llm.writer_calls == 3


async def test_non_stream_requests_are_untouched_by_the_sse_machinery(client_for, make_fake_llm):
    client, _ = client_for(
        make_fake_llm([ASK, "Sentence A is true. [c1]"]), FakeVerifier([["supported"]])
    )
    r = await ask(client, stream=False)
    assert r.headers["content-type"] == "application/json"
    assert r.json()["answer"][0]["text"] == "Sentence A is true."
