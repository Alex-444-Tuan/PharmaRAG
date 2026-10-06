"""RxNav ingestion client with httpx mocked by respx. No network."""

import time

import httpx
import pytest
import respx

from pharmarag.ingest.rxnav import DEFAULT_MAX_RPS, Alias, Ingredient, RateLimiter, RxNavClient

BASE = "https://rx.test/REST"


@pytest.fixture
def rx():
    # max_rps=0 turns the limiter off so tests run fast
    return RxNavClient(BASE, max_rps=0, retries=2, backoff_s=0.0)


def ids(*rxcuis):
    return {"idGroup": {"rxnormId": list(rxcuis)}}


def props(rxcui, name, tty):
    return {"properties": {"rxcui": rxcui, "name": name, "tty": tty}}


def related(*concepts):
    """concepts: (rxcui, name, tty). One conceptGroup per tty, like the real API."""
    groups: dict[str, list] = {}
    for rxcui, name, tty in concepts:
        groups.setdefault(tty, []).append({"rxcui": rxcui, "name": name, "tty": tty})
    return {
        "relatedGroup": {
            "rxcui": None,
            "conceptGroup": [{"tty": t, "conceptProperties": c} for t, c in groups.items()],
        }
    }


# --- ingredient lookup --------------------------------------------------------------------


@respx.mock
async def test_find_ingredient_exact_match(rx):
    name_route = respx.get(f"{BASE}/rxcui.json").respond(json=ids("17767"))
    respx.get(f"{BASE}/rxcui/17767/properties.json").respond(
        json=props("17767", "amlodipine", "IN")
    )
    assert await rx.find_ingredient("amlodipine") == Ingredient("17767", "amlodipine")
    assert name_route.calls.last.request.url.params["name"] == "amlodipine"
    assert name_route.calls.last.request.url.params["search"] == "0"


@respx.mock
async def test_find_ingredient_maps_a_precise_ingredient_to_its_base_ingredient(rx):
    respx.get(f"{BASE}/rxcui.json").respond(json=ids("104416"))
    respx.get(f"{BASE}/rxcui/104416/properties.json").respond(
        json=props("104416", "amlodipine besylate", "PIN")
    )
    respx.get(f"{BASE}/rxcui/104416/related.json").respond(
        json=related(("17767", "amlodipine", "IN"))
    )
    assert await rx.find_ingredient("amlodipine besylate") == Ingredient("17767", "amlodipine")


@respx.mock
async def test_find_ingredient_falls_back_to_normalised_search(rx):
    route = respx.get(f"{BASE}/rxcui.json")
    route.side_effect = [
        httpx.Response(200, json={"idGroup": {}}),
        httpx.Response(200, json=ids("274783")),
    ]
    respx.get(f"{BASE}/rxcui/274783/properties.json").respond(
        json=props("274783", "insulin glargine", "IN")
    )
    assert (await rx.find_ingredient("Insulin  Glargine")).rxcui == "274783"
    assert [c.request.url.params["search"] for c in route.calls] == ["0", "2"]


@respx.mock
async def test_find_ingredient_returns_none_when_nothing_matches(rx):
    respx.get(f"{BASE}/rxcui.json").respond(json={"idGroup": {}})
    assert await rx.find_ingredient("notadrug") is None


@respx.mock
async def test_find_ingredient_skips_concepts_that_are_not_single_ingredients(rx):
    # an id that maps to a combination (two ingredients) is not a seed ingredient
    respx.get(f"{BASE}/rxcui.json").respond(json=ids("999"))
    respx.get(f"{BASE}/rxcui/999/properties.json").respond(json=props("999", "x / y", "MIN"))
    respx.get(f"{BASE}/rxcui/999/related.json").respond(
        json=related(("1", "x", "IN"), ("2", "y", "IN"))
    )
    assert await rx.find_ingredient("x / y") is None


@respx.mock
async def test_product_ingredients_are_cached(rx):
    route = respx.get(f"{BASE}/rxcui/197361/related.json").respond(
        json=related(("17767", "amlodipine", "IN"))
    )
    for _ in range(3):
        assert await rx.product_ingredients("197361") == [Ingredient("17767", "amlodipine")]
    assert route.call_count == 1
    assert route.calls.last.request.url.params["tty"] == "IN"


@respx.mock
async def test_product_ingredient_key_for_a_combination_is_sorted_and_deduplicated(rx):
    respx.get(f"{BASE}/rxcui/1/related.json").respond(
        json=related(("593411", "hydrochlorothiazide", "IN"), ("6809", "metformin", "IN"))
    )
    respx.get(f"{BASE}/rxcui/2/related.json").respond(json=related(("6809", "metformin", "IN")))
    key, ingredients = await rx.product_ingredient_key({"1", "2"})
    assert key == "6809+593411"
    assert [i.rxcui for i in ingredients] == ["6809", "593411"]


# --- aliases ------------------------------------------------------------------------------


@respx.mock
async def test_aliases_cover_in_pin_sy_and_bn_with_the_right_keys(rx):
    amlodipine = Ingredient("17767", "amlodipine")

    def related_handler(request: httpx.Request) -> httpx.Response:
        tty = request.url.params["tty"]
        if tty == "PIN":
            return httpx.Response(200, json=related(("104416", "amlodipine besylate", "PIN")))
        if tty == "BN":
            return httpx.Response(
                200,
                json=related(("1", "Norvasc", "BN"), ("2", "Tribenzor", "BN")),
            )
        raise AssertionError(tty)

    respx.get(f"{BASE}/rxcui/17767/related.json").mock(side_effect=related_handler)
    respx.get(f"{BASE}/rxcui/17767/allProperties.json").respond(
        json={
            "propConceptGroup": {
                "propConcept": [
                    {"propName": "RxNorm Name", "propValue": "amlodipine"},
                    {"propName": "Tallman Synonym", "propValue": "amLODIPine"},
                    {"propName": "RXNAV_STR", "propValue": "Amlodipine Besylate Ingredient"},
                ]
            }
        }
    )
    respx.get(f"{BASE}/rxcui/1/related.json").respond(json=related(("17767", "amlodipine", "IN")))
    respx.get(f"{BASE}/rxcui/2/related.json").respond(
        json=related(
            ("17767", "amlodipine", "IN"), ("73494", "olmesartan", "IN"), ("5487", "hctz", "IN")
        )
    )
    aliases = await rx.aliases(amlodipine)
    by_name = {a.alias: a for a in aliases}
    assert by_name["amlodipine"] == Alias("amlodipine", "IN", "17767", "17767")
    assert by_name["amlodipine besylate"] == Alias("amlodipine besylate", "PIN", "104416", "17767")
    assert by_name["norvasc"] == Alias("norvasc", "BN", "1", "17767")
    # a combination brand listed under amlodipine must carry the combination's key
    assert by_name["tribenzor"] == Alias("tribenzor", "BN", "2", "5487+17767+73494")
    assert by_name["amlodipine besylate ingredient"].tty == "SY"
    # the tall-man spelling lowercases to the IN name, so it adds no duplicate row
    assert len([a for a in aliases if a.alias == "amlodipine"]) == 1
    assert all(a.alias == a.alias.lower() for a in aliases)


@respx.mock
async def test_aliases_skip_brands_with_no_known_ingredient(rx):
    respx.get(f"{BASE}/rxcui/5/related.json").mock(
        side_effect=lambda req: httpx.Response(
            200,
            json=related(("9", "Ghost", "BN")) if req.url.params["tty"] == "BN" else related(),
        )
    )
    respx.get(f"{BASE}/rxcui/5/allProperties.json").respond(json={})
    respx.get(f"{BASE}/rxcui/9/related.json").respond(json={"relatedGroup": {"rxcui": None}})
    aliases = await rx.aliases(Ingredient("5", "five"))
    assert [a.alias for a in aliases] == ["five"]


# --- retries and rate limit ---------------------------------------------------------------


@respx.mock
async def test_5xx_is_retried(rx):
    route = respx.get(f"{BASE}/rxcui/1/properties.json")
    route.side_effect = [httpx.Response(500), httpx.Response(200, json=props("1", "a", "IN"))]
    assert await rx.ingredients_of("1") == [Ingredient("1", "a")]
    assert route.call_count == 2


def test_default_rate_stays_under_the_rxnav_limit_of_20_per_second():
    assert DEFAULT_MAX_RPS < 20


async def test_rate_limiter_spaces_calls():
    limiter = RateLimiter(50.0)  # 20 ms apart
    start = time.monotonic()
    for _ in range(6):
        await limiter.wait()
    assert time.monotonic() - start >= 5 * 0.02 * 0.9


async def test_rate_limiter_disabled_does_not_sleep():
    limiter = RateLimiter(0)
    start = time.monotonic()
    for _ in range(50):
        await limiter.wait()
    assert time.monotonic() - start < 0.05


@respx.mock
async def test_client_applies_the_limiter_to_every_request():
    rx = RxNavClient(BASE, max_rps=50.0, retries=0)
    respx.get(f"{BASE}/rxcui/1/properties.json").respond(json=props("1", "a", "IN"))
    respx.get(f"{BASE}/rxcui/2/properties.json").respond(json=props("2", "b", "IN"))
    respx.get(f"{BASE}/rxcui/3/properties.json").respond(json=props("3", "c", "IN"))
    start = time.monotonic()
    for r in ("1", "2", "3"):
        await rx.ingredients_of(r)
    assert time.monotonic() - start >= 2 * 0.02 * 0.9


@respx.mock
async def test_rxnav_retries_are_paced_by_the_rate_limiter_too():
    rx = RxNavClient(BASE, max_rps=50.0, retries=2, backoff_s=0.0)
    route = respx.get(f"{BASE}/rxcui/1/properties.json")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(500),
        httpx.Response(200, json=props("1", "a", "IN")),
    ]
    start = time.monotonic()
    assert await rx.ingredients_of("1") == [Ingredient("1", "a")]
    assert route.call_count == 3
    assert time.monotonic() - start >= 2 * 0.02 * 0.9
