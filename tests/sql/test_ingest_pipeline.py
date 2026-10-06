"""Seed ingredient to database, with DailyMed and RxNav mocked by respx and a real Postgres."""

from pathlib import Path

import httpx
import pytest
import respx

from pharmarag.ingest.dailymed import MARKETING_CATEGORY_BLA, MARKETING_CATEGORY_NDA, DailyMedClient
from pharmarag.ingest.pipeline import Ingestor
from pharmarag.ingest.rxnav import RxNavClient
from pharmarag.models.embedder import FakeEmbedder

pytestmark = pytest.mark.pg

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "spl"
DM = "https://dm.test/v2"
RX = "https://rx.test/REST"

NORVASC = "7367289c-b0b0-466a-83e2-558e2985c29f"
CLONIDINE = "88a8db46-0fc9-48cd-b7cc-ac9c9cdea100"
ZESTORETIC = "0d3a966f-f937-05a8-a90f-5aa52ebbd613"
TRUNCATED = "99999999-0000-0000-0000-000000000009"
OTHER_NDA = "aaaaaaaa-0000-0000-0000-00000000000a"

TRUNCATED_XML = b"""<document xmlns="urn:hl7-org:v3"><code code="34391-3"/><title>Stub</title>
<setId root="99999999-0000-0000-0000-000000000009"/><versionNumber value="1"/>
<component><structuredBody><component><section><code code="34069-5"/>
<text><paragraph>Repackaged by someone.</paragraph></text></section></component>
</structuredBody></component></document>"""


def ref(setid: str, published: str, version: int = 1) -> dict:
    return {"setid": setid, "spl_version": version, "published_date": published, "title": "T"}


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


class World:
    """Mock DailyMed + RxNav. `lists` maps a category code (or None) to label rows."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.lists: dict[str | None, list[dict]] = {}
        self.xml: dict[str, bytes] = {}
        self.fetched: list[str] = []
        router.get(f"{DM}/spls.json").mock(side_effect=self._spls)
        router.get(url__regex=rf"{DM}/spls/[0-9a-f-]+\.xml").mock(side_effect=self._xml)

    def _spls(self, request: httpx.Request) -> httpx.Response:
        rows = self.lists.get(request.url.params.get("marketing_category_code"), [])
        return httpx.Response(200, json={"data": rows, "metadata": {"next_page": "null"}})

    def _xml(self, request: httpx.Request) -> httpx.Response:
        setid = request.url.path.rsplit("/", 1)[-1].removesuffix(".xml")
        self.fetched.append(setid)
        return httpx.Response(200, content=self.xml[setid])


def mock_rxnav(router, *, ingredient: tuple[str, str], products: dict[str, list[tuple[str, str]]]):
    """`products` maps a product RxCUI to its (ingredient rxcui, name) list."""
    rxcui, name = ingredient
    router.get(f"{RX}/rxcui.json").respond(json={"idGroup": {"rxnormId": [rxcui]}})
    router.get(f"{RX}/rxcui/{rxcui}/properties.json").respond(
        json={"properties": {"rxcui": rxcui, "name": name, "tty": "IN"}}
    )
    router.get(f"{RX}/rxcui/{rxcui}/allProperties.json").respond(json={})

    def related(request: httpx.Request) -> httpx.Response:
        tty = request.url.params["tty"]
        concepts = []
        if tty == "BN":
            concepts = [{"rxcui": "b1", "name": f"Brand{name.title()}", "tty": "BN"}]
        return httpx.Response(
            200,
            json={"relatedGroup": {"conceptGroup": [{"tty": tty, "conceptProperties": concepts}]}},
        )

    router.get(f"{RX}/rxcui/{rxcui}/related.json").mock(side_effect=related)
    router.get(f"{RX}/rxcui/b1/related.json").respond(
        json={
            "relatedGroup": {
                "conceptGroup": [
                    {
                        "tty": "IN",
                        "conceptProperties": [{"rxcui": rxcui, "name": name, "tty": "IN"}],
                    }
                ]
            }
        }
    )
    for product, ings in products.items():
        router.get(f"{RX}/rxcui/{product}/related.json").respond(
            json={
                "relatedGroup": {
                    "conceptGroup": [
                        {
                            "tty": "IN",
                            "conceptProperties": [
                                {"rxcui": r, "name": n, "tty": "IN"} for r, n in ings
                            ],
                        }
                    ]
                }
            }
        )


@pytest.fixture
def router():
    with respx.mock(assert_all_called=False) as r:
        yield r


@pytest.fixture
def world(router):
    return World(router)


def make_ingestor(conn, mappings, embedder=None, per_ingredient=1) -> Ingestor:
    return Ingestor(
        rxnav=RxNavClient(RX, max_rps=0, retries=0),
        dailymed=DailyMedClient(DM, retries=0),
        mappings=mappings,
        embedder=embedder or FakeEmbedder(),
        conn=conn,
        labels_per_ingredient=per_ingredient,
    )


async def rows(conn, sql, *args):
    return await (await conn.execute(sql, args)).fetchall()


async def test_seed_is_resolved_selected_parsed_chunked_and_loaded(pg_conn, router, world):
    mock_rxnav(
        router, ingredient=("17767", "amlodipine"), products={"197361": [("17767", "amlodipine")]}
    )
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Dec 10, 2025", 10)]
    world.xml[NORVASC] = (FIXTURES / "norvasc_plr.xml").read_bytes()

    result = await make_ingestor(pg_conn, {NORVASC: {"197361"}}).ingest_seed("amlodipine")

    assert (result.status, result.ingredient_rxcui, result.setids) == ("loaded", "17767", [NORVASC])
    assert result.chunks > 20
    [label] = await rows(
        pg_conn,
        "SELECT spl_version, ingredient_key, ingredient_rxcuis, product_rxcuis, "
        "marketing_category FROM label",
    )
    assert label == (10, "17767", ["17767"], ["197361"], "NDA")
    assert await rows(pg_conn, "SELECT rxcui, name FROM ingredient") == [("17767", "amlodipine")]
    aliases = {
        r[0]: r[1:]
        for r in await rows(pg_conn, "SELECT alias, alias_tty, ingredient_key FROM drug_alias")
    }
    assert aliases["amlodipine"] == ("IN", "17767")
    assert aliases["brandamlodipine"] == ("BN", "17767")
    [(n_bad,)] = await rows(pg_conn, "SELECT count(*) FROM chunk WHERE ingredient_key <> '17767'")
    assert n_bad == 0


async def test_rerun_skips_embedding_and_changes_nothing(pg_conn, router, world):
    mock_rxnav(
        router, ingredient=("17767", "amlodipine"), products={"197361": [("17767", "amlodipine")]}
    )
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Dec 10, 2025", 10)]
    world.xml[NORVASC] = (FIXTURES / "norvasc_plr.xml").read_bytes()
    emb = CountingEmbedder()
    ingestor = make_ingestor(pg_conn, {NORVASC: {"197361"}}, emb)

    first = await ingestor.ingest_seed("amlodipine")
    ids = await rows(pg_conn, "SELECT id FROM chunk ORDER BY id")
    second = await ingestor.ingest_seed("amlodipine")

    assert (first.status, second.status) == ("loaded", "unchanged")
    assert emb.calls == 1
    assert await rows(pg_conn, "SELECT id FROM chunk ORDER BY id") == ids
    assert await rows(pg_conn, "SELECT count(*) FROM label") == [(1,)]


async def test_nda_label_beats_a_newer_anda_label(pg_conn, router, world):
    mock_rxnav(
        router, ingredient=("17767", "amlodipine"), products={"197361": [("17767", "amlodipine")]}
    )
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Jan 01, 2020", 10)]
    world.lists[None] = [ref(CLONIDINE, "Sep 01, 2026"), ref(NORVASC, "Jan 01, 2020")]
    world.xml[NORVASC] = (FIXTURES / "norvasc_plr.xml").read_bytes()
    world.xml[CLONIDINE] = (FIXTURES / "clonidine_old.xml").read_bytes()

    result = await make_ingestor(pg_conn, {NORVASC: {"197361"}, CLONIDINE: {"197361"}}).ingest_seed(
        "amlodipine"
    )

    assert result.setids == [NORVASC]
    assert CLONIDINE not in world.fetched  # the ANDA tier is never even downloaded


async def test_falls_back_to_other_categories_when_there_is_no_nda_or_bla(pg_conn, router, world):
    mock_rxnav(router, ingredient=("2599", "clonidine"), products={"1": [("2599", "clonidine")]})
    world.lists[None] = [ref(CLONIDINE, "Sep 30, 2026", 5)]
    world.xml[CLONIDINE] = (FIXTURES / "clonidine_old.xml").read_bytes()

    result = await make_ingestor(pg_conn, {CLONIDINE: {"1"}}).ingest_seed("clonidine")

    assert result.status == "loaded"
    assert await rows(pg_conn, "SELECT marketing_category FROM label") == [("ANDA",)]


async def test_combination_labels_are_not_picked_for_a_single_ingredient_seed(
    pg_conn, router, world
):
    mock_rxnav(
        router,
        ingredient=("29046", "lisinopril"),
        products={"2": [("29046", "lisinopril"), ("5487", "hydrochlorothiazide")]},
    )
    world.lists[MARKETING_CATEGORY_NDA] = [ref(ZESTORETIC, "Jun 02, 2025", 16)]
    world.xml[ZESTORETIC] = (FIXTURES / "zestoretic_old_combo.xml").read_bytes()

    result = await make_ingestor(pg_conn, {ZESTORETIC: {"2"}}).ingest_seed("lisinopril")

    assert result.status == "no_label"
    assert ZESTORETIC not in world.fetched
    assert await rows(pg_conn, "SELECT count(*) FROM label") == [(0,)]
    # the ingredient and its aliases are still recorded for the resolver
    [(n_aliases,)] = await rows(pg_conn, "SELECT count(*) FROM drug_alias")
    assert n_aliases > 0


async def test_truncated_label_is_skipped_for_the_next_candidate(pg_conn, router, world):
    mock_rxnav(
        router, ingredient=("17767", "amlodipine"), products={"197361": [("17767", "amlodipine")]}
    )
    world.lists[MARKETING_CATEGORY_NDA] = [
        ref(TRUNCATED, "Sep 01, 2026"),
        ref(NORVASC, "Dec 10, 2025", 10),
    ]
    world.xml[TRUNCATED] = TRUNCATED_XML
    world.xml[NORVASC] = (FIXTURES / "norvasc_plr.xml").read_bytes()

    result = await make_ingestor(pg_conn, {TRUNCATED: {"197361"}, NORVASC: {"197361"}}).ingest_seed(
        "amlodipine"
    )

    assert world.fetched == [TRUNCATED, NORVASC]
    assert result.setids == [NORVASC]


async def test_labels_missing_from_the_mapping_file_are_ignored(pg_conn, router, world):
    mock_rxnav(
        router, ingredient=("17767", "amlodipine"), products={"197361": [("17767", "amlodipine")]}
    )
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Dec 10, 2025", 10)]
    world.xml[NORVASC] = (FIXTURES / "norvasc_plr.xml").read_bytes()

    result = await make_ingestor(pg_conn, {}).ingest_seed("amlodipine")

    assert result.status == "no_label"
    assert world.fetched == []


async def test_unresolved_seed_writes_nothing(pg_conn, router, world):
    router.get(f"{RX}/rxcui.json").respond(json={"idGroup": {}})

    result = await make_ingestor(pg_conn, {}).ingest_seed("notadrug")

    assert result.status == "unresolved"
    for table in ("ingredient", "drug_alias", "label", "chunk"):
        assert await rows(pg_conn, f"SELECT count(*) FROM {table}") == [(0,)]


async def test_a_changed_selection_prunes_the_previously_loaded_label(pg_conn, router, world):
    mock_rxnav(
        router, ingredient=("17767", "amlodipine"), products={"197361": [("17767", "amlodipine")]}
    )
    world.xml[NORVASC] = (FIXTURES / "norvasc_plr.xml").read_bytes()
    world.xml[OTHER_NDA] = (
        (FIXTURES / "norvasc_plr.xml").read_bytes().replace(NORVASC.encode(), OTHER_NDA.encode())
    )
    mappings = {NORVASC: {"197361"}, OTHER_NDA: {"197361"}}

    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Jan 01, 2020", 10)]
    ingestor = make_ingestor(pg_conn, mappings)
    await ingestor.ingest_seed("amlodipine")

    world.lists[MARKETING_CATEGORY_NDA] = [
        ref(NORVASC, "Jan 01, 2020", 10),
        ref(OTHER_NDA, "Jan 01, 2025", 10),
    ]
    result = await ingestor.ingest_seed("amlodipine")

    assert result.setids == [OTHER_NDA]
    assert result.pruned == 1
    assert [str(r[0]) for r in await rows(pg_conn, "SELECT setid FROM label")] == [OTHER_NDA]
    assert await rows(pg_conn, "SELECT count(DISTINCT setid) FROM chunk") == [(1,)]


async def test_labels_per_ingredient_keeps_that_many(pg_conn, router, world):
    mock_rxnav(
        router, ingredient=("17767", "amlodipine"), products={"197361": [("17767", "amlodipine")]}
    )
    world.xml[NORVASC] = (FIXTURES / "norvasc_plr.xml").read_bytes()
    world.xml[OTHER_NDA] = (
        (FIXTURES / "norvasc_plr.xml").read_bytes().replace(NORVASC.encode(), OTHER_NDA.encode())
    )
    world.lists[MARKETING_CATEGORY_NDA] = [
        ref(NORVASC, "Jan 01, 2020", 10),
        ref(OTHER_NDA, "Jan 01, 2025", 10),
    ]
    world.lists[MARKETING_CATEGORY_BLA] = []

    result = await make_ingestor(
        pg_conn, {NORVASC: {"197361"}, OTHER_NDA: {"197361"}}, per_ingredient=2
    ).ingest_seed("amlodipine")

    assert result.setids == [OTHER_NDA, NORVASC]  # newest first
    assert await rows(pg_conn, "SELECT count(*) FROM label") == [(2,)]
