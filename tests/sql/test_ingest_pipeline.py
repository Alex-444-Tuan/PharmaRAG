"""Seed ingredient to database, with DailyMed and RxNav mocked by respx and a real Postgres."""

from pathlib import Path

import httpx
import pytest
import respx

from pharmarag.ingest.chunker import chunk_label
from pharmarag.ingest.dailymed import MARKETING_CATEGORY_BLA, MARKETING_CATEGORY_NDA, DailyMedClient
from pharmarag.ingest.load import LabelRow, load_label
from pharmarag.ingest.pins import Pins
from pharmarag.ingest.pipeline import Ingestor
from pharmarag.ingest.rxnav import RxNavClient
from pharmarag.ingest.spl_parser import parse_spl
from pharmarag.models.embedder import FakeEmbedder

pytestmark = pytest.mark.pg

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "spl"
DM = "https://dm.test/v2"
RX = "https://rx.test/REST"

NORVASC = "7367289c-b0b0-466a-83e2-558e2985c29f"
CLONIDINE = "88a8db46-0fc9-48cd-b7cc-ac9c9cdea100"
ZESTORETIC = "0d3a966f-f937-05a8-a90f-5aa52ebbd613"
NO_DOSAGE = "c0f7f48d-e39c-4981-9c0e-5c2eccaba424"
QLONILIK = "826de1dc-378e-4c90-86d1-e33f7a4248cd"
TRUNCATED = "99999999-0000-0000-0000-000000000009"
OTHER_NDA = "aaaaaaaa-0000-0000-0000-00000000000a"

TRUNCATED_XML = b"""<document xmlns="urn:hl7-org:v3"><code code="34391-3"/><title>Stub</title>
<setId root="99999999-0000-0000-0000-000000000009"/><versionNumber value="1"/>
<component><structuredBody><component><section><code code="34069-5"/>
<text><paragraph>Repackaged by someone.</paragraph></text></section></component>
</structuredBody></component></document>"""


def ref(setid: str, published: str, version: int = 1, title: str = "T") -> dict:
    return {
        "setid": setid,
        "spl_version": version,
        "published_date": published,
        "title": title,
    }


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
        if setid not in self.xml:
            return httpx.Response(404)
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


def make_ingestor(
    conn, mappings, embedder=None, per_ingredient=1, pins=None, repin=False, **kw
) -> Ingestor:
    return Ingestor(
        rxnav=RxNavClient(RX, max_rps=0, retries=0),
        dailymed=DailyMedClient(DM, retries=0, max_rps=0),
        mappings=mappings,
        embedder=embedder or FakeEmbedder(),
        conn=conn,
        labels_per_ingredient=per_ingredient,
        pins=pins,
        repin=repin,
        **kw,
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
    ingestor = make_ingestor(pg_conn, mappings, repin=True)
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


def norvasc_xml() -> bytes:
    return (FIXTURES / "norvasc_plr.xml").read_bytes()


def as_setid(xml: bytes, setid: str) -> bytes:
    return xml.replace(NORVASC.encode(), setid.encode())


def amlodipine(router):
    mock_rxnav(
        router, ingredient=("17767", "amlodipine"), products={"197361": [("17767", "amlodipine")]}
    )


# --- core sections --------------------------------------------------------------------------


async def test_label_without_a_dosage_section_is_skipped_for_the_next_candidate(
    pg_conn, router, world
):
    mock_rxnav(
        router, ingredient=("5553", "hydroxyzine"), products={"1": [("5553", "hydroxyzine")]}
    )
    world.lists[None] = [ref(NO_DOSAGE, "Sep 11, 2026"), ref(CLONIDINE, "Jan 01, 2020")]
    world.xml[NO_DOSAGE] = (FIXTURES / "hydroxyzine_pamoate_no_dosage.xml").read_bytes()
    world.xml[CLONIDINE] = (FIXTURES / "clonidine_old.xml").read_bytes()

    result = await make_ingestor(pg_conn, {NO_DOSAGE: {"1"}, CLONIDINE: {"1"}}).ingest_seed(
        "hydroxyzine"
    )

    assert world.fetched == [NO_DOSAGE, CLONIDINE]
    assert result.setids == [CLONIDINE]
    [(n,)] = await rows(pg_conn, "SELECT count(*) FROM chunk WHERE section_loinc = '34068-7'")
    assert n > 0


async def test_dosage_section_coded_unclassified_still_yields_dosage_chunks(pg_conn, router, world):
    mock_rxnav(router, ingredient=("2599", "clonidine"), products={"1": [("2599", "clonidine")]})
    world.lists[MARKETING_CATEGORY_NDA] = [ref(QLONILIK, "Sep 01, 2026")]
    world.xml[QLONILIK] = (FIXTURES / "qlonilik_plr_title_coded.xml").read_bytes()

    result = await make_ingestor(pg_conn, {QLONILIK: {"1"}}).ingest_seed("clonidine")

    assert result.status == "loaded"
    [(n,)] = await rows(pg_conn, "SELECT count(*) FROM chunk WHERE section_loinc = '34068-7'")
    assert n > 0


async def test_fetch_cap_stops_a_seed_with_only_unusable_labels(pg_conn, router, world):
    amlodipine(router)
    stubs = [f"aaaaaaaa-0000-0000-0000-00000000000{i}" for i in range(5)]
    world.lists[MARKETING_CATEGORY_NDA] = [ref(s, "Jan 01, 2025") for s in stubs]
    for s in stubs:
        world.xml[s] = TRUNCATED_XML

    result = await make_ingestor(
        pg_conn, {s: {"197361"} for s in stubs}, max_fetches=2
    ).ingest_seed("amlodipine")

    assert result.status == "no_label"
    assert len(world.fetched) == 2


# --- repackager and form tie-breaks ---------------------------------------------------------


async def test_manufacturer_is_tried_before_a_newer_repackager_label(pg_conn, router, world):
    amlodipine(router)
    world.lists[MARKETING_CATEGORY_NDA] = [
        ref(OTHER_NDA, "Sep 01, 2026", title="NORVASC (AMLODIPINE) TABLET [REMEDYREPACK INC.]"),
        ref(NORVASC, "Jan 01, 2020", 10, title="NORVASC (AMLODIPINE) TABLET [PFIZER]"),
    ]
    world.xml[NORVASC] = norvasc_xml()
    world.xml[OTHER_NDA] = as_setid(norvasc_xml(), OTHER_NDA)

    result = await make_ingestor(pg_conn, {NORVASC: {"197361"}, OTHER_NDA: {"197361"}}).ingest_seed(
        "amlodipine"
    )

    assert world.fetched == [NORVASC]
    assert result.setids == [NORVASC]


async def test_immediate_release_is_tried_before_a_newer_extended_release_form(
    pg_conn, router, world
):
    amlodipine(router)
    world.lists[MARKETING_CATEGORY_NDA] = [
        ref(
            OTHER_NDA,
            "Sep 01, 2026",
            title="NORVASC XR (AMLODIPINE) TABLET, EXTENDED RELEASE [PFIZER]",
        ),
        ref(NORVASC, "Jan 01, 2020", 10, title="NORVASC (AMLODIPINE) TABLET [PFIZER]"),
    ]
    world.xml[NORVASC] = norvasc_xml()
    world.xml[OTHER_NDA] = as_setid(norvasc_xml(), OTHER_NDA)

    result = await make_ingestor(pg_conn, {NORVASC: {"197361"}, OTHER_NDA: {"197361"}}).ingest_seed(
        "amlodipine"
    )

    assert result.setids == [NORVASC]


async def test_ingredient_with_only_niche_forms_still_gets_a_label(pg_conn, router, world):
    amlodipine(router)
    world.lists[MARKETING_CATEGORY_NDA] = [
        ref(
            NORVASC,
            "Jan 01, 2020",
            10,
            title="NORVASC XR (AMLODIPINE) TABLET, EXTENDED RELEASE [PFIZER]",
        )
    ]
    world.xml[NORVASC] = norvasc_xml()

    result = await make_ingestor(pg_conn, {NORVASC: {"197361"}}).ingest_seed("amlodipine")

    assert result.setids == [NORVASC]


# --- pins -----------------------------------------------------------------------------------


async def test_pin_is_written_then_kept_when_a_newer_label_appears(
    pg_conn, router, world, tmp_path
):
    amlodipine(router)
    world.xml[NORVASC] = norvasc_xml()
    world.xml[OTHER_NDA] = as_setid(norvasc_xml(), OTHER_NDA)
    mappings = {NORVASC: {"197361"}, OTHER_NDA: {"197361"}}
    pins = Pins(tmp_path / "pins.csv")

    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Jan 01, 2020", 10)]
    first = await make_ingestor(pg_conn, mappings, pins=pins).ingest_seed("amlodipine")
    assert pins.get("17767") == [NORVASC]

    world.lists[MARKETING_CATEGORY_NDA] = [
        ref(OTHER_NDA, "Sep 01, 2026", 10),  # newer: a fresh selection would take it
        ref(NORVASC, "Jan 01, 2020", 10),
    ]
    world.fetched.clear()
    second = await make_ingestor(pg_conn, mappings, pins=pins).ingest_seed("amlodipine")

    assert (first.setids, second.setids) == ([NORVASC], [NORVASC])
    assert second.status == "unchanged" and second.pruned == 0
    assert world.fetched == [NORVASC]  # the pin was checked; the newer label never downloaded


async def test_pinned_label_moves_to_its_newest_version(pg_conn, router, world, tmp_path):
    amlodipine(router)
    mappings = {NORVASC: {"197361"}}
    pins = Pins(tmp_path / "pins.csv")
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Jan 01, 2020", 10)]
    world.xml[NORVASC] = norvasc_xml()
    await make_ingestor(pg_conn, mappings, pins=pins).ingest_seed("amlodipine")
    assert await rows(pg_conn, "SELECT spl_version FROM label") == [(10,)]

    # DailyMed now serves version 11 of the same setid; the list is not consulted for a pin
    world.lists[MARKETING_CATEGORY_NDA] = []
    world.xml[NORVASC] = norvasc_xml().replace(
        b'versionNumber value="10"', b'versionNumber value="11"'
    )
    result = await make_ingestor(pg_conn, mappings, pins=pins).ingest_seed("amlodipine")

    assert result.status == "loaded"
    assert await rows(pg_conn, "SELECT spl_version FROM label") == [(11,)]
    assert await rows(pg_conn, "SELECT count(DISTINCT spl_version) FROM chunk") == [(1,)]


async def test_repin_ignores_the_pin_and_selects_afresh(pg_conn, router, world, tmp_path):
    amlodipine(router)
    world.xml[NORVASC] = norvasc_xml()
    world.xml[OTHER_NDA] = as_setid(norvasc_xml(), OTHER_NDA)
    mappings = {NORVASC: {"197361"}, OTHER_NDA: {"197361"}}
    pins = Pins(tmp_path / "pins.csv")
    pins.set("17767", "amlodipine", [NORVASC])
    world.lists[MARKETING_CATEGORY_NDA] = [
        ref(OTHER_NDA, "Sep 01, 2026", 10),
        ref(NORVASC, "Jan 01, 2020", 10),
    ]

    result = await make_ingestor(pg_conn, mappings, pins=pins, repin=True).ingest_seed("amlodipine")

    assert result.setids == [OTHER_NDA]
    assert pins.get("17767") == [OTHER_NDA]


async def test_pin_that_vanished_from_dailymed_is_replaced(pg_conn, router, world, tmp_path):
    amlodipine(router)
    gone = "dddddddd-0000-0000-0000-00000000000d"
    router.get(f"{DM}/spls/{gone}.xml").respond(404)
    world.xml[NORVASC] = norvasc_xml()
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Jan 01, 2020", 10)]
    pins = Pins(tmp_path / "pins.csv")
    pins.set("17767", "amlodipine", [gone])

    result = await make_ingestor(
        pg_conn, {gone: {"197361"}, NORVASC: {"197361"}}, pins=pins
    ).ingest_seed("amlodipine")

    assert result.setids == [NORVASC]
    assert pins.get("17767") == [NORVASC]


async def test_pin_whose_ingredient_set_changed_is_replaced(pg_conn, router, world, tmp_path):
    mock_rxnav(
        router,
        ingredient=("17767", "amlodipine"),
        products={"197361": [("17767", "amlodipine")], "999": [("17767", "a"), ("5487", "b")]},
    )
    combo = "eeeeeeee-0000-0000-0000-00000000000e"
    world.xml[NORVASC] = norvasc_xml()
    world.xml[combo] = as_setid(norvasc_xml(), combo)
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Jan 01, 2020", 10)]
    pins = Pins(tmp_path / "pins.csv")
    pins.set("17767", "amlodipine", [combo])

    result = await make_ingestor(
        pg_conn, {combo: {"999"}, NORVASC: {"197361"}}, pins=pins
    ).ingest_seed("amlodipine")

    assert result.setids == [NORVASC]
    assert combo not in world.fetched


async def test_pin_that_fails_the_core_section_check_is_replaced(pg_conn, router, world, tmp_path):
    amlodipine(router)
    world.xml[NO_DOSAGE] = (FIXTURES / "hydroxyzine_pamoate_no_dosage.xml").read_bytes()
    world.xml[NORVASC] = norvasc_xml()
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Jan 01, 2020", 10)]
    pins = Pins(tmp_path / "pins.csv")
    pins.set("17767", "amlodipine", [NO_DOSAGE])

    result = await make_ingestor(
        pg_conn, {NO_DOSAGE: {"197361"}, NORVASC: {"197361"}}, pins=pins
    ).ingest_seed("amlodipine")

    assert result.setids == [NORVASC]


# --- stale loads and chunk counts -----------------------------------------------------------


async def test_stale_load_is_reported_as_stale_and_writes_no_chunks(pg_conn, router, world):
    amlodipine(router)
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Jan 01, 2020", 10)]
    world.xml[NORVASC] = norvasc_xml()  # DailyMed serves version 10 ...
    emb = FakeEmbedder()
    chunks = chunk_label(parse_spl(norvasc_xml()), emb.count_tokens)[:3]
    await load_label(  # ... but the database already holds version 11
        pg_conn,
        LabelRow(NORVASC, 11, "T", None, "NDA", None, "17767", ["17767"], ["197361"]),
        chunks,
        emb.embed_documents([c.text for c in chunks]),
    )
    counting = CountingEmbedder()

    result = await make_ingestor(pg_conn, {NORVASC: {"197361"}}, counting).ingest_seed("amlodipine")

    assert (result.status, result.stale, result.chunks) == ("stale", 1, 0)
    assert counting.calls == 0
    assert result.pruned == 0
    assert await rows(pg_conn, "SELECT spl_version FROM label") == [(11,)]
    assert await rows(pg_conn, "SELECT count(*) FROM chunk") == [(3,)]


async def test_chunk_count_is_the_number_of_chunks_written(pg_conn, router, world):
    amlodipine(router)
    world.lists[MARKETING_CATEGORY_NDA] = [ref(NORVASC, "Jan 01, 2020", 10)]
    world.xml[NORVASC] = norvasc_xml()
    ingestor = make_ingestor(pg_conn, {NORVASC: {"197361"}})

    first = await ingestor.ingest_seed("amlodipine")
    second = await ingestor.ingest_seed("amlodipine")

    [(n,)] = await rows(pg_conn, "SELECT count(*) FROM chunk")
    assert first.chunks == n
    assert (second.chunks, second.status) == (0, "unchanged")
