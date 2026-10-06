"""POST /v1/ask with a stub graph: HTTP behaviour only, no database."""

import httpx
import psycopg
import pytest

from pharmarag.app.main import create_app, get_runtime
from pharmarag.app.runtime import Runtime
from pharmarag.graph.deps import Deps
from pharmarag.llm.errors import LLMUnavailableError
from tests.graph.fakes import FakeResolver, FakeVerifier


class StubGraph:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.states: list[dict] = []

    async def ainvoke(self, state, config=None):
        self.states.append(state)
        if self.error:
            raise self.error
        return {
            "response": {
                "status": "answered",
                "drugs": [],
                "answer": [{"text": "ok", "citations": ["c1"], "support": None}],
                "sources": [],
                "notice": "From FDA drug labels via DailyMed. Not medical advice.",
                "meta": {
                    "corpus_version": "v1",
                    "latency_ms": 0,
                    "retrieval_attempts": 1,
                    "generation_attempts": 1,
                    "removed_sentences": 0,
                },
            },
            "intent": "label_qa",
            "timings": {"analyze": 1.0},
        }


@pytest.fixture
def make_client(settings, thresholds, fake_embedder, make_fake_llm):
    def _make(*, eval_mode: bool, graph: StubGraph | None = None, stages: bool = True):
        s = settings.model_copy(update={"eval_mode": eval_mode})
        deps = Deps(
            llm=make_fake_llm([]),
            embedder=fake_embedder,
            pool=None,
            settings=s,
            thresholds=thresholds,
            resolver=FakeResolver({}) if stages else None,
            verifier=FakeVerifier([[]]) if stages else None,
        )
        graph = graph or StubGraph()
        app = create_app(s)
        app.dependency_overrides[get_runtime] = lambda: Runtime(deps=deps, graph=graph)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        return client, graph

    return _make


async def test_ask_200_returns_spec_response_shape(make_client):
    client, graph = make_client(eval_mode=False)
    async with client:
        r = await client.post("/v1/ask", json={"question": "What is the dose of hydroxyzine?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert set(body) == {"status", "drugs", "answer", "sources", "notice", "meta"}
    assert set(body["meta"]) == {
        "corpus_version",
        "latency_ms",
        "retrieval_attempts",
        "generation_attempts",
        "removed_sentences",
    }
    assert r.headers["x-request-id"]
    assert graph.states[0]["question"] == "What is the dose of hydroxyzine?"
    assert graph.states[0]["request_id"] == r.headers["x-request-id"]


async def test_ask_passes_pinned_rxcuis_and_strips_question(make_client):
    client, graph = make_client(eval_mode=False)
    async with client:
        await client.post("/v1/ask", json={"question": "  dose?  ", "pinned_rxcuis": ["5553"]})
    assert graph.states[0]["question"] == "dose?"
    assert graph.states[0]["pinned_rxcuis"] == ["5553"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": 5},
        {"question": "ok", "stream": "maybe"},
        {"question": "ok", "pinned_rxcuis": "5553"},
        {"question": "ok", "surprise": 1},
    ],
)
async def test_ask_422_on_invalid_body(make_client, payload):
    client, graph = make_client(eval_mode=False)
    async with client:
        r = await client.post("/v1/ask", json=payload)
    assert r.status_code == 422
    assert graph.states == []


async def test_ask_422_on_non_json_body(make_client):
    client, _ = make_client(eval_mode=False)
    async with client:
        r = await client.post(
            "/v1/ask", content=b"not json", headers={"content-type": "application/json"}
        )
    assert r.status_code == 422


async def test_eval_flags_header_honoured_only_in_eval_mode(make_client):
    header = {"X-Eval-Flags": "resolver=off,verifier=off"}
    client, graph = make_client(eval_mode=True)
    async with client:
        await client.post("/v1/ask", json={"question": "q"}, headers=header)
    flags = graph.states[0]["flags"]
    assert flags.resolver is False and flags.verifier is False

    client, graph = make_client(eval_mode=False)
    async with client:
        r = await client.post("/v1/ask", json={"question": "q"}, headers=header)
    assert r.status_code == 200
    flags = graph.states[0]["flags"]
    assert flags.resolver is True and flags.verifier is True  # header ignored


async def test_eval_flags_partial_header_changes_only_named_flag(make_client):
    client, graph = make_client(eval_mode=True)
    async with client:
        await client.post(
            "/v1/ask", json={"question": "q"}, headers={"X-Eval-Flags": "verifier=off"}
        )
    flags = graph.states[0]["flags"]
    assert flags.resolver is True and flags.verifier is False


@pytest.mark.parametrize("bad", ["resolver=maybe", "nonsense=off", "resolver"])
async def test_malformed_eval_flags_is_422_in_eval_mode_and_ignored_otherwise(make_client, bad):
    client, _ = make_client(eval_mode=True)
    async with client:
        r = await client.post("/v1/ask", json={"question": "q"}, headers={"X-Eval-Flags": bad})
    assert r.status_code == 422
    client, _ = make_client(eval_mode=False)
    async with client:
        r = await client.post("/v1/ask", json={"question": "q"}, headers={"X-Eval-Flags": bad})
    assert r.status_code == 200


async def test_stages_this_build_lacks_are_forced_off_even_without_header(make_client):
    client, graph = make_client(eval_mode=False, stages=False)
    async with client:
        await client.post("/v1/ask", json={"question": "q"})
    flags = graph.states[0]["flags"]
    assert flags.resolver is False and flags.verifier is False


async def test_debug_flag_adds_debug_payload(make_client):
    client, _ = make_client(eval_mode=False)
    async with client:
        plain = (await client.post("/v1/ask", json={"question": "q"})).json()
        debug = (await client.post("/v1/ask", json={"question": "q", "debug": True})).json()
    assert "debug" not in plain
    assert debug["debug"]["timings_ms"] == {"analyze": 1.0}
    assert debug["debug"]["flags"] == {"resolver": True, "verifier": True}


async def test_stream_true_is_not_implemented_yet(make_client):
    client, graph = make_client(eval_mode=False)
    async with client:
        r = await client.post("/v1/ask", json={"question": "q", "stream": True})
    assert r.status_code == 501
    assert graph.states == []


@pytest.mark.parametrize(
    "error", [LLMUnavailableError("down"), psycopg.OperationalError("db down")]
)
async def test_dependency_failure_is_503(make_client, error):
    client, _ = make_client(eval_mode=False, graph=StubGraph(error=error))
    async with client:
        r = await client.post("/v1/ask", json={"question": "q"})
    assert r.status_code == 503


async def test_healthz(make_client):
    client, _ = make_client(eval_mode=False)
    async with client:
        r = await client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
