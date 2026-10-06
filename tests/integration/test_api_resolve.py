"""GET /v1/drugs/resolve with a fake resolver: HTTP behaviour only, no database."""

import httpx
import psycopg
import pytest

from pharmarag.app.main import create_app, get_runtime
from pharmarag.app.runtime import Runtime
from pharmarag.graph.deps import Deps
from tests.graph.fakes import FakeResolver, ambiguous, not_covered, resolved, unknown


@pytest.fixture
def make_client(settings, thresholds, fake_embedder, make_fake_llm):
    def _make(resolver):
        deps = Deps(
            llm=make_fake_llm([]),
            embedder=fake_embedder,
            pool=None,
            settings=settings,
            thresholds=thresholds,
            resolver=resolver,
        )
        app = create_app(settings)
        app.dependency_overrides[get_runtime] = lambda: Runtime(deps=deps, graph=None)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")

    return _make


async def test_resolve_returns_the_resolution_and_runs_the_resolver_alone(make_client):
    resolver = FakeResolver(
        {"hydroxyzin": resolved("hydroxyzin", "5553", "hydroxyzine", "approximate")}
    )
    async with make_client(resolver) as c:
        r = await c.get("/v1/drugs/resolve", params={"name": "  hydroxyzin "})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "resolved" and body["match"] == "approximate"
    assert body["ingredient_key"] == "5553" and body["name"] == "hydroxyzine"
    assert resolver.calls == [{"mentions": ["hydroxyzin"], "pinned": []}]  # stripped, no pins


async def test_resolve_shows_candidates_for_an_ambiguous_name(make_client):
    amb = ambiguous("hydrox", [("1", "hydroxyzine", 0.88), ("2", "hydroxyurea", 0.86)])
    async with make_client(FakeResolver({"hydrox": amb})) as c:
        body = (await c.get("/v1/drugs/resolve", params={"name": "hydrox"})).json()
    assert body["status"] == "ambiguous"
    assert [c["name"] for c in body["candidates"]] == ["hydroxyzine", "hydroxyurea"]


async def test_resolve_reports_unknown_and_not_covered_as_200_with_a_status(make_client):
    resolver = FakeResolver({"rarexamab": not_covered("rarexamab")})
    async with make_client(resolver) as c:
        assert (await c.get("/v1/drugs/resolve", params={"name": "rarexamab"})).json()[
            "status"
        ] == "not_covered"
        r = await c.get("/v1/drugs/resolve", params={"name": "zzz"})
    assert r.status_code == 200 and r.json()["status"] == "unknown"


async def test_resolve_is_503_when_the_build_has_no_resolver(make_client):
    async with make_client(None) as c:
        r = await c.get("/v1/drugs/resolve", params={"name": "hydroxyzin"})
    assert r.status_code == 503


@pytest.mark.parametrize("params", [{}, {"name": ""}, {"name": "   "}, {"name": "x" * 201}])
async def test_resolve_422_on_missing_blank_or_oversized_name(make_client, params):
    resolver = FakeResolver({})
    async with make_client(resolver) as c:
        r = await c.get("/v1/drugs/resolve", params=params)
    assert r.status_code == 422 and resolver.calls == []


@pytest.mark.parametrize("error", [httpx.ConnectTimeout("rxnav"), psycopg.OperationalError("db")])
async def test_resolve_dependency_failure_is_503(make_client, error):
    class Down:
        async def resolve(self, mentions, *, pinned_rxcuis=None):
            raise error

    async with make_client(Down()) as c:
        r = await c.get("/v1/drugs/resolve", params={"name": "hydroxyzin"})
    assert r.status_code == 503


async def test_resolve_surfaces_the_degraded_flag(make_client):
    local_only = resolved("hydroxyzin", "5553", "hydroxyzine", "approximate").model_copy(
        update={"degraded": True}
    )
    async with make_client(FakeResolver({"hydroxyzin": local_only, "zzz": unknown("zzz")})) as c:
        degraded = (await c.get("/v1/drugs/resolve", params={"name": "hydroxyzin"})).json()
        normal = (await c.get("/v1/drugs/resolve", params={"name": "zzz"})).json()
    assert degraded["degraded"] is True and normal["degraded"] is False
