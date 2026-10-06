"""GET /v1/labels/{setid} against real PostgreSQL, and the degraded app end to end."""

import json
import uuid

import httpx
import pytest

from pharmarag.app import runtime as runtime_module
from pharmarag.app.main import create_app
from pharmarag.app.runtime import build_runtime
from tests.graph.fakes import DOSE_TEXT, analysis
from tests.graph.seed import HYDRALAZINE, HYDROXYZINE, seed_label, set_corpus_version

pytestmark = pytest.mark.pg


@pytest.fixture
async def client(settings, pg_url, make_fake_llm, fake_embedder):
    s = settings.model_copy(update={"database_url": pg_url})
    llm = make_fake_llm([])

    async def factory(st):
        return await build_runtime(st, embedder=fake_embedder, llm=llm)

    app = create_app(s, runtime_factory=factory)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            c.llm = llm  # type: ignore[attr-defined]
            yield c


async def test_label_metadata_and_section_list(client, pg_conn, fake_embedder):
    setid = await seed_label(
        pg_conn, fake_embedder, key="1", name="hydroxyzine", chunks=HYDROXYZINE, spl_version=3
    )
    await seed_label(pg_conn, fake_embedder, key="2", name="hydralazine", chunks=HYDRALAZINE)
    await pg_conn.commit()

    r = await client.get(f"/v1/labels/{setid}")
    assert r.status_code == 200
    body = r.json()
    assert body["setid"] == str(setid) and body["spl_version"] == 3
    assert body["title"] == "hydroxyzine label" and body["effective_date"] == "2024-05-01"
    assert body["ingredient_key"] == "1" and body["ingredient_rxcuis"] == ["1"]
    assert body["url"] == f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
    # In label order, with the canonical section names and this label's chunk counts only.
    assert body["sections"] == [
        {"loinc": "34068-7", "title": "Dosage and administration", "chunks": 1},
        {"loinc": "34070-3", "title": "Contraindications", "chunks": 1},
        {"loinc": "34073-7", "title": "Drug interactions", "chunks": 1},
    ]


async def test_label_endpoint_returns_the_newest_version(client, pg_conn, fake_embedder):
    setid = uuid.uuid4()
    await seed_label(
        pg_conn,
        fake_embedder,
        key="1",
        name="old",
        chunks=HYDROXYZINE[:1],
        setid=setid,
        spl_version=1,
    )
    await seed_label(
        pg_conn,
        fake_embedder,
        key="1",
        name="new",
        chunks=HYDROXYZINE,
        setid=setid,
        spl_version=2,
    )
    await pg_conn.commit()
    body = (await client.get(f"/v1/labels/{setid}")).json()
    assert body["spl_version"] == 2 and body["title"] == "new label"
    assert len(body["sections"]) == 3  # version 1's single chunk is not mixed in


async def test_unknown_label_is_404_and_a_bad_setid_is_422(client):
    assert (await client.get(f"/v1/labels/{uuid.uuid4()}")).status_code == 404
    assert (await client.get("/v1/labels/not-a-uuid")).status_code == 422


async def test_degraded_app_serves_ask_as_the_baseline_and_resolve_as_503(
    settings, pg_url, make_fake_llm, fake_embedder, pg_conn, monkeypatch
):
    """Lane A's factories raise NotImplementedError: the app still starts and still answers."""

    def not_implemented(*a):
        raise NotImplementedError("cycle 2, lane A")

    monkeypatch.setattr(runtime_module, "build_resolver", not_implemented)
    monkeypatch.setattr(runtime_module, "build_rx_facts_source", not_implemented)
    await seed_label(pg_conn, fake_embedder, key="1", name="hydroxyzine", chunks=HYDROXYZINE)
    await seed_label(pg_conn, fake_embedder, key="2", name="hydralazine", chunks=HYDRALAZINE)
    await set_corpus_version(pg_conn, "corpus-degraded")
    await pg_conn.commit()

    llm = make_fake_llm([analysis(mentions=["hydroxyzin"]), f"{DOSE_TEXT} [c1]"])
    s = settings.model_copy(update={"database_url": pg_url})

    async def factory(st):
        return await build_runtime(st, embedder=fake_embedder, llm=llm)

    app = create_app(s, runtime_factory=factory)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            r = await c.post(
                "/v1/ask", json={"question": "What is the max daily dose of hydroxyzin?"}
            )
            resolve = await c.get("/v1/drugs/resolve", params={"name": "hydroxyzin"})
    assert resolve.status_code == 503
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["drugs"] == []
    assert body["meta"]["resolver"] is False and body["meta"]["verified"] is False
    assert json.dumps(body["meta"]["corpus_version"]) == '"corpus-degraded"'
