"""POST /v1/ask through the real lifespan, graph, and PostgreSQL. LLM and embedder are fakes."""

import json

import httpx
import pytest

from pharmarag.app.main import create_app
from pharmarag.app.runtime import build_runtime
from tests.graph.seed import HYDRALAZINE, HYDROXYZINE, seed_label, set_corpus_version

pytestmark = pytest.mark.pg


async def test_ask_end_to_end_with_lifespan_warmup_and_pool(
    settings, make_fake_llm, fake_embedder, pg_url, pg_conn
):
    await seed_label(pg_conn, fake_embedder, key="1", name="hydroxyzine", chunks=HYDROXYZINE)
    await seed_label(pg_conn, fake_embedder, key="2", name="hydralazine", chunks=HYDRALAZINE)
    await set_corpus_version(pg_conn, "corpus-e2e")
    await pg_conn.commit()

    llm = make_fake_llm(
        [
            json.dumps(
                {
                    "intent": "label_qa",
                    "mentions": [],
                    "sections": ["34068-7"],
                    "search_query": "hydroxyzine dosage anxiety adults",
                }
            ),
            "Hydroxyzine is dosed four times daily. [c1]",
        ]
    )
    embed_calls: list[str] = []
    real_embed = fake_embedder.embed_query
    fake_embedder.embed_query = lambda text: (embed_calls.append(text), real_embed(text))[1]

    s = settings.model_copy(update={"database_url": pg_url})

    async def factory(st):
        return await build_runtime(st, embedder=fake_embedder, llm=llm)

    app = create_app(s, runtime_factory=factory)
    async with app.router.lifespan_context(app):
        assert embed_calls == ["warm-up"]  # the model was exercised once at startup
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            r = await c.post(
                "/v1/ask",
                json={"question": "What is the hydroxyzine dosage for anxiety in adults?"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert body["answer"][0]["text"] == "Hydroxyzine is dosed four times daily."
    assert body["sources"][0]["url"].startswith(
        "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid="
    )
    assert body["meta"]["corpus_version"] == "corpus-e2e"
    assert embed_calls[1:] == [
        "hydroxyzine dosage anxiety adults"
    ]  # the analysed query, not the raw question
