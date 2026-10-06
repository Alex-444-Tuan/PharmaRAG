"""Request-time RxNav client with httpx mocked by respx. No network."""

import httpx
import pytest
import respx

from pharmarag.resolver.rxnav_client import (
    ApproxCandidate,
    Concept,
    Ingredient,
    RxNavClient,
    RxNavUnavailable,
)

BASE = "https://rx.test/REST"


def make(**kw) -> RxNavClient:
    kw.setdefault("max_rps", 0)  # no pacing in tests
    kw.setdefault("backoff_s", 0.0)
    return RxNavClient(BASE, **kw)


def related(*concepts):
    """concepts: (rxcui, name, tty), grouped by tty like the real API."""
    groups: dict[str, list] = {}
    for rxcui, name, tty in concepts:
        groups.setdefault(tty, []).append({"rxcui": rxcui, "name": name, "tty": tty})
    return {
        "relatedGroup": {
            "rxcui": None,
            "conceptGroup": [{"tty": t, "conceptProperties": c} for t, c in groups.items()],
        }
    }


# --- requests ---------------------------------------------------------------------------------


@respx.mock
async def test_find_rxcuis_uses_search_2():
    route = respx.get(f"{BASE}/rxcui.json").respond(json={"idGroup": {"rxnormId": ["5553"]}})
    assert await make().find_rxcuis("hydroxyzine") == ["5553"]
    params = route.calls.last.request.url.params
    assert params["name"] == "hydroxyzine"
    assert params["search"] == "2"


@respx.mock
async def test_find_rxcuis_no_match_is_empty():
    respx.get(f"{BASE}/rxcui.json").respond(json={"idGroup": {}})
    assert await make().find_rxcuis("xyzzy") == []


@respx.mock
async def test_approximate_asks_for_ten_active_concepts_and_dedupes_by_rxcui():
    body = {
        "approximateGroup": {
            "candidate": [
                {"rxcui": "5553", "rank": "1", "source": "GS"},
                {"rxcui": "5553", "rank": "1", "name": "hydroxyzine", "source": "RXNORM"},
                {"rxcui": "1164647", "rank": "8", "name": "hydroxyzine Pill", "source": "RXNORM"},
                {"rxcui": "1794549", "rank": "8", "name": "hydroxyzine Injection"},
            ]
        }
    }
    route = respx.get(f"{BASE}/approximateTerm.json").respond(json=body)
    got = await make().approximate("hydroxyzne")
    params = route.calls.last.request.url.params
    assert params["term"] == "hydroxyzne"
    assert params["maxEntries"] == "10"
    assert params["option"] == "1"  # active concepts only
    assert got == [
        ApproxCandidate("5553", 1, "hydroxyzine"),  # the row that carries a name wins
        ApproxCandidate("1164647", 8, "hydroxyzine Pill"),
        ApproxCandidate("1794549", 8, "hydroxyzine Injection"),
    ]


@respx.mock
async def test_approximate_with_no_candidates_is_empty():
    respx.get(f"{BASE}/approximateTerm.json").respond(json={"approximateGroup": {"candidate": []}})
    assert await make().approximate("qqqq") == []


@respx.mock
async def test_concept_properties():
    respx.get(f"{BASE}/rxcui/154987/properties.json").respond(
        json={"properties": {"rxcui": "154987", "name": "hydroxyzine hydrochloride", "tty": "PIN"}}
    )
    assert await make().concept("154987") == Concept("154987", "hydroxyzine hydrochloride", "PIN")


@respx.mock
async def test_concept_unknown_is_none():
    respx.get(f"{BASE}/rxcui/1/properties.json").respond(json={"properties": None})
    assert await make().concept("1") is None


@respx.mock
async def test_ingredients_maps_a_concept_to_its_ingredients_and_is_memoised():
    route = respx.get(f"{BASE}/rxcui/1372730/related.json").respond(
        json=related(("6809", "metformin", "IN"), ("857974", "saxagliptin", "IN"))
    )
    rx = make()
    expected = [Ingredient("6809", "metformin"), Ingredient("857974", "saxagliptin")]
    assert await rx.ingredients("1372730") == expected
    assert route.calls.last.request.url.params["tty"] == "IN"
    assert await rx.ingredients("1372730") == expected
    assert route.call_count == 1  # second call came from the memo


@respx.mock
async def test_related_reads_brands_dose_forms_and_strengths_in_one_request():
    route = respx.get(f"{BASE}/rxcui/17767/related.json").respond(
        json=related(
            ("17767", "amlodipine", "IN"),
            ("58927", "Norvasc", "BN"),
            ("317541", "Oral Tablet", "DF"),
            ("329528", "amlodipine 5 MG", "SCDC"),
        )
    )
    got = await make().related("17767")
    assert route.call_count == 1
    assert route.calls.last.request.url.params["tty"] == "IN+BN+DF+SCDC"
    assert got.ingredient == Ingredient("17767", "amlodipine")
    assert [c.name for c in got.brands] == ["Norvasc"]
    assert [c.name for c in got.dose_forms] == ["Oral Tablet"]
    assert [c.name for c in got.strengths] == ["amlodipine 5 MG"]


# --- timeout and retry policy ------------------------------------------------------------------


def test_timeout_comes_from_the_argument():
    rx = RxNavClient(BASE, timeout_s=2.0)
    assert rx._client.timeout.read == 2.0
    assert rx._client.timeout.connect == 2.0


@respx.mock
async def test_5xx_is_retried_and_the_third_attempt_wins():
    route = respx.get(f"{BASE}/rxcui.json")
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(500),
        httpx.Response(200, json={"idGroup": {"rxnormId": ["5553"]}}),
    ]
    assert await make().find_rxcuis("hydroxyzine") == ["5553"]
    assert route.call_count == 3


@respx.mock
async def test_429_is_retried():
    route = respx.get(f"{BASE}/rxcui.json")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json={"idGroup": {"rxnormId": ["1"]}}),
    ]
    assert await make().find_rxcuis("x") == ["1"]
    assert route.call_count == 2


@respx.mock
async def test_timeouts_are_retried_then_raise_unavailable():
    route = respx.get(f"{BASE}/rxcui.json")
    route.side_effect = httpx.ReadTimeout("slow")
    with pytest.raises(RxNavUnavailable):
        await make().find_rxcuis("hydroxyzine")
    assert route.call_count == 3  # three attempts, no more


@respx.mock
async def test_connection_errors_count_as_unavailable():
    route = respx.get(f"{BASE}/rxcui.json")
    route.side_effect = httpx.ConnectError("refused")
    with pytest.raises(RxNavUnavailable):
        await make().find_rxcuis("hydroxyzine")
    assert route.call_count == 3


@respx.mock
async def test_persistent_503_raises_unavailable_after_three_attempts():
    route = respx.get(f"{BASE}/rxcui.json").respond(503)
    with pytest.raises(RxNavUnavailable):
        await make().find_rxcuis("hydroxyzine")
    assert route.call_count == 3


@respx.mock
async def test_a_404_is_no_result_not_an_outage_and_is_not_retried():
    route = respx.get(f"{BASE}/rxcui.json").respond(404)
    rx = make()
    assert await rx.find_rxcuis("x") == []
    assert route.call_count == 1
    assert not rx.breaker_open


@respx.mock
async def test_a_body_that_is_not_json_is_unavailable():
    respx.get(f"{BASE}/rxcui.json").respond(200, text="<html>maintenance</html>")
    with pytest.raises(RxNavUnavailable):
        await make().find_rxcuis("x")


@respx.mock
async def test_backoff_doubles_between_attempts(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("pharmarag.ingest.http.asyncio.sleep", fake_sleep)
    respx.get(f"{BASE}/rxcui.json").respond(503)
    with pytest.raises(RxNavUnavailable):
        await make(backoff_s=0.25).find_rxcuis("x")
    assert sleeps == [0.25, 0.5]


# --- circuit breaker ---------------------------------------------------------------------------


@respx.mock
async def test_after_an_outage_calls_fail_fast_until_the_breaker_closes():
    now = [100.0]
    route = respx.get(f"{BASE}/rxcui.json").respond(503)
    rx = make(breaker_s=30.0, clock=lambda: now[0])
    with pytest.raises(RxNavUnavailable):
        await rx.find_rxcuis("x")
    assert route.call_count == 3
    assert rx.breaker_open

    # Open: no HTTP traffic at all, so a request does not wait for three more timeouts.
    with pytest.raises(RxNavUnavailable):
        await rx.find_rxcuis("x")
    assert route.call_count == 3

    # Closed again after breaker_s, and the next call really goes out.
    now[0] = 131.0
    assert not rx.breaker_open
    route.respond(200, json={"idGroup": {"rxnormId": ["5553"]}})
    assert await rx.find_rxcuis("x") == ["5553"]
    assert route.call_count == 4
