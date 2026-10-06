"""DailyMed client with httpx mocked by respx. No network."""

import zipfile
from datetime import date

import httpx
import pytest
import respx

from pharmarag.ingest.dailymed import (
    RXNORM_MAPPINGS_URL,
    DailyMedClient,
    LabelRef,
    load_rxnorm_mappings,
)

BASE = "https://dm.test/services/v2"


@pytest.fixture
def client():
    return DailyMedClient(BASE, retries=2, backoff_s=0.0)


def page(rows, next_page=None):
    return {"data": rows, "metadata": {"next_page": next_page if next_page else "null"}}


ROW = {
    "setid": "7367289C-B0B0-466A-83E2-558E2985C29F",
    "spl_version": 10,
    "published_date": "Dec 10, 2025",
    "title": "NORVASC (AMLODIPINE BESYLATE) TABLET [VIATRIS SPECIALTY LLC]",
}


@respx.mock
async def test_list_labels_sends_filters_and_parses_rows(client):
    route = respx.get(f"{BASE}/spls.json").respond(json=page([ROW]))
    refs = await client.list_labels("17767", marketing_category_code="C73594")
    assert refs == [
        LabelRef(
            "7367289c-b0b0-466a-83e2-558e2985c29f",
            10,
            "NORVASC (AMLODIPINE BESYLATE) TABLET [VIATRIS SPECIALTY LLC]",
            date(2025, 12, 10),
        )
    ]
    params = route.calls.last.request.url.params
    assert params["rxcui"] == "17767"
    assert params["doctype"] == "34391-3"  # human prescription labels only
    assert params["marketing_category_code"] == "C73594"
    assert params["pagesize"] == "100"
    assert params["page"] == "1"


@respx.mock
async def test_list_labels_omits_category_filter_by_default(client):
    route = respx.get(f"{BASE}/spls.json").respond(json=page([]))
    assert await client.list_labels("17767") == []
    assert "marketing_category_code" not in route.calls.last.request.url.params


@respx.mock
async def test_list_labels_follows_pages_until_next_page_is_null(client):
    rows = [{**ROW, "setid": f"00000000-0000-0000-0000-00000000000{i}"} for i in range(3)]
    route = respx.get(f"{BASE}/spls.json")
    route.side_effect = [
        httpx.Response(200, json=page([rows[0]], next_page=2)),
        httpx.Response(200, json=page([rows[1]], next_page=3)),
        httpx.Response(200, json=page([rows[2]])),
    ]
    refs = await client.list_labels("1")
    assert [r.setid[-1] for r in refs] == ["0", "1", "2"]
    assert [c.request.url.params["page"] for c in route.calls] == ["1", "2", "3"]


@respx.mock
async def test_list_labels_stops_at_max_pages(client):
    route = respx.get(f"{BASE}/spls.json").respond(json=page([ROW], next_page=2))
    refs = await client.list_labels("1", max_pages=2)
    assert route.call_count == 2 and len(refs) == 2


@respx.mock
async def test_list_labels_tolerates_missing_or_odd_dates(client):
    bad = {**ROW, "published_date": "not a date"}
    respx.get(f"{BASE}/spls.json").respond(json=page([bad]))
    assert (await client.list_labels("1"))[0].published_date is None


@respx.mock
async def test_fetch_spl_xml_returns_raw_bytes(client):
    respx.get(f"{BASE}/spls/abc.xml").respond(content=b"<document/>")
    assert await client.fetch_spl_xml("abc") == b"<document/>"


@respx.mock
async def test_server_errors_and_429_are_retried(client):
    route = respx.get(f"{BASE}/spls/abc.xml")
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(429),
        httpx.Response(200, content=b"ok"),
    ]
    assert await client.fetch_spl_xml("abc") == b"ok"
    assert route.call_count == 3


@respx.mock
async def test_transport_errors_are_retried_then_raised(client):
    route = respx.get(f"{BASE}/spls/abc.xml").mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(httpx.ConnectTimeout):
        await client.fetch_spl_xml("abc")
    assert route.call_count == 3  # first try plus two retries


@respx.mock
async def test_404_is_not_retried(client):
    route = respx.get(f"{BASE}/spls/missing.xml").respond(404)
    with pytest.raises(httpx.HTTPStatusError):
        await client.fetch_spl_xml("missing")
    assert route.call_count == 1


def make_mapping_zip(path, lines):
    header = "SETID|SPL_VERSION|RXCUI|RXSTRING|RXTTY"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("rxnorm_mappings.txt", "\n".join([header, *lines]) + "\n")


def test_load_rxnorm_mappings_keeps_highest_version_per_setid(tmp_path):
    z = tmp_path / "m.zip"
    make_mapping_zip(
        z,
        [
            "AAAAAAAA-0000-0000-0000-000000000001|1|111|old 5 MG Oral Tablet|SCD",
            "aaaaaaaa-0000-0000-0000-000000000001|2|222|new 5 MG Oral Tablet|SCD",
            "aaaaaaaa-0000-0000-0000-000000000001|2|222|new 5 MG Oral Tablet|SY",
            'aaaaaaaa-0000-0000-0000-000000000001|2|333|new 10 MG "x" Oral Tablet|SCD',
            "bbbbbbbb-0000-0000-0000-000000000002|7|444|other|SBD",
            "short|row",
        ],
    )
    index = load_rxnorm_mappings(z)
    assert index == {
        "aaaaaaaa-0000-0000-0000-000000000001": {"222", "333"},
        "bbbbbbbb-0000-0000-0000-000000000002": {"444"},
    }


@respx.mock
async def test_download_rxnorm_mappings_streams_to_disk(client, tmp_path):
    respx.get(RXNORM_MAPPINGS_URL).respond(content=b"zipbytes" * 1000)
    dest = tmp_path / "cache" / "rxnorm_mappings.zip"
    assert await client.download_rxnorm_mappings(dest) == dest
    assert dest.read_bytes() == b"zipbytes" * 1000
    assert not dest.with_suffix(".zip.part").exists()
