"""The resolution cascade (spec section 7) against a small seeded alias table, RxNav scripted.

Postgres is real (trigram and coverage are SQL); RxNav is `FakeRx`. See tests/unit/resolver/
fakes.py for the table.
"""

import json
import uuid

import pytest
from rapidfuzz.distance import JaroWinkler

from pharmarag.cache import InMemoryCache
from pharmarag.resolver.cascade import CascadeResolver
from pharmarag.resolver.local import PgLocalStore
from pharmarag.resolver.rxnav_client import ApproxCandidate, Concept, Ingredient, RxNavUnavailable

pytestmark = pytest.mark.pg

HYDROXYZINE, HYDRALAZINE = "5553", "5470"
PREDNISONE, PREDNISOLONE = "8640", "8638"


@pytest.fixture
def cache() -> InMemoryCache:
    return InMemoryCache()


@pytest.fixture
def make(pool, rx, thresholds, cache):
    def _make(*, rxnav=rx, cache=cache, th=None) -> CascadeResolver:
        return CascadeResolver(PgLocalStore(pool), rxnav, th or thresholds.resolver, cache)

    return _make


@pytest.fixture
def resolver(make) -> CascadeResolver:
    return make()


async def one(resolver, mention, **kw):
    (out,) = await resolver.resolve_detailed([mention], **kw)
    return out


# --- steps 2 and 3 stop on a hit ---


async def test_local_exact_match_resolves_and_never_calls_rxnav(resolver, rx):
    out = await one(resolver, "hydroxyzine")
    r = out.resolution
    assert (r.status, r.match, r.ingredient_key, r.name) == (
        "resolved",
        "exact",
        HYDROXYZINE,
        "hydroxyzine",
    )
    assert r.ingredient_rxcuis == [HYDROXYZINE]
    assert r.score == 1.0
    assert out.step == "local_exact"
    assert rx.calls == []


async def test_a_correctly_spelled_name_is_never_corrected_to_a_neighbour(resolver, rx):
    # RxNav would offer the more common look-alike first. Step 2 must stop before it is asked.
    rx.approx["hydralazine"] = [ApproxCandidate(HYDROXYZINE, 1, "hydroxyzine")]
    rx.exact["hydralazine"] = [HYDROXYZINE]
    r = (await one(resolver, "Hydralazine")).resolution
    assert (r.status, r.match, r.ingredient_key) == ("resolved", "exact", HYDRALAZINE)
    assert rx.calls == []


async def test_dose_and_form_words_do_not_stop_an_exact_match(resolver):
    r = (await one(resolver, "Hydroxyzine 25 mg tablets")).resolution
    assert (r.status, r.match, r.ingredient_key) == ("resolved", "exact", HYDROXYZINE)


async def test_brand_match_keeps_the_brand_rxcui_as_product_hint(resolver):
    r = (await one(resolver, "Vistaril")).resolution
    assert (r.status, r.match, r.ingredient_key, r.name) == (
        "resolved",
        "brand",
        HYDROXYZINE,
        "hydroxyzine",
    )
    assert r.product_hint == "282519"


async def test_salt_match_keeps_the_salt_rxcui_as_product_hint(resolver):
    r = (await one(resolver, "hydroxyzine HCl 25 mg tablets")).resolution
    assert (r.status, r.match, r.ingredient_key) == ("resolved", "salt", HYDROXYZINE)
    assert r.product_hint == "154987"


async def test_synonym_match_is_an_ingredient_match(resolver):
    r = (await one(resolver, "glibenclamide")).resolution
    assert (r.match, r.ingredient_key, r.name) == ("exact", "4815", "glyburide")
    assert r.product_hint is None


async def test_a_brand_with_a_release_word_matches_with_its_dose_stripped(resolver):
    r = (await one(resolver, "Glucotrol XL 10 mg")).resolution
    assert (r.status, r.match, r.ingredient_key) == ("resolved", "brand", "5454")


async def test_rxnav_exact_hit_stops_before_the_approximate_step(resolver, rx):
    rx.exact["atarax"] = ["1234"]
    rx.concepts["1234"] = Concept("1234", "Atarax", "BN")
    rx.ingredient_map["1234"] = [Ingredient(HYDROXYZINE, "hydroxyzine")]
    rx.approx["atarax"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    out = await one(resolver, "Atarax")
    r = out.resolution
    assert (r.status, r.match, r.ingredient_key, r.product_hint) == (
        "resolved",
        "brand",
        HYDROXYZINE,
        "1234",
    )
    assert out.step == "rxnav_exact"
    assert "approximate" not in rx.names_called


async def test_rxnav_exact_hit_on_a_salt_concept_is_a_salt_match(resolver, rx):
    rx.exact["hydroxyzine dihydrochloride"] = ["777"]
    rx.concepts["777"] = Concept("777", "hydroxyzine dihydrochloride", "PIN")
    rx.ingredient_map["777"] = [Ingredient(HYDROXYZINE, "hydroxyzine")]
    r = (await one(resolver, "hydroxyzine dihydrochloride")).resolution
    assert (r.match, r.ingredient_key, r.product_hint) == ("salt", HYDROXYZINE, "777")


async def test_a_misspelling_in_rxnavs_string_table_is_approximate_whether_rxnav_is_up_or_down(
    resolver, rx, make
):
    # "hydroxyzin" is a string RxNav stores for hydroxyzine, so step 3 hits. It is still the
    # user's misspelling: `exact` is only for a mention that equals one of the drug's names.
    rx.exact["hydroxyzin"] = [HYDROXYZINE]
    up = await one(resolver, "hydroxyzin")
    assert (up.resolution.status, up.resolution.match) == ("resolved", "approximate")
    assert up.resolution.ingredient_key == HYDROXYZINE
    assert up.resolution.score == round(JaroWinkler.similarity("hydroxyzin", "hydroxyzine"), 4)
    assert up.step == "rxnav_exact"  # stop-on-hit is kept: step 4 was not asked
    assert rx.names_called == ["find_rxcuis"]

    rx.down = True  # a fresh cache: the answer above must not be replayed
    down = await one(make(cache=InMemoryCache()), "hydroxyzin")
    assert (down.resolution.status, down.resolution.match) == ("resolved", "approximate")
    assert down.resolution.ingredient_key == HYDROXYZINE
    assert down.resolution.degraded


async def test_an_rxnav_hit_on_a_concept_the_alias_table_lacks_uses_the_concept_name(resolver, rx):
    rx.exact["hydroxyzin"] = ["777"]
    rx.concepts["777"] = Concept("777", "hydroxyzine", "IN")  # the name is not what was typed
    rx.ingredient_map["777"] = [Ingredient(HYDROXYZINE, "hydroxyzine")]
    r = (await one(resolver, "hydroxyzin")).resolution
    assert (r.match, r.ingredient_key) == ("approximate", HYDROXYZINE)


async def test_an_rxnav_hit_that_is_weakly_similar_does_not_bypass_the_gate(resolver, rx):
    # A string RxNav stores for something the name does not resemble: scored, then gated.
    rx.exact["flurb"] = [HYDRALAZINE]
    r = (await one(resolver, "flurb")).resolution
    assert r.status == "unknown"


async def test_an_rxnav_hit_on_two_ingredients_is_ambiguous_not_exact(resolver, rx):
    rx.exact["prednisolon"] = [PREDNISOLONE, PREDNISONE]
    r = (await one(resolver, "prednisolon")).resolution
    assert r.status == "ambiguous"
    assert {c.ingredient_key for c in r.candidates} == {PREDNISOLONE, PREDNISONE}


async def test_scores_reach_the_gate_unrounded(resolver):
    c = await resolver.cascade("hydralzine")
    assert c.scored[0].score == JaroWinkler.similarity("hydralzine", "hydralazine")
    assert c.scored[0].score != round(c.scored[0].score, 4)  # the gate sees all the digits


async def test_an_exact_name_shared_by_two_ingredients_is_ambiguous(pool, pg_conn, make):
    for rxcui, key in (("90001", HYDROXYZINE), ("90002", HYDRALAZINE)):
        await pg_conn.execute("INSERT INTO drug_alias VALUES ('hydro', 'BN', %s, %s)", (rxcui, key))
    await pg_conn.commit()
    r = (await one(make(), "hydro")).resolution
    assert r.status == "ambiguous"
    assert {c.ingredient_key for c in r.candidates} == {HYDROXYZINE, HYDRALAZINE}
    assert r.ingredient_key is None


# --- steps 4 to 8: approximate ---


async def test_misspelling_resolves_through_rxnav_approximate(resolver, rx):
    rx.approx["hydralzine"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    out = await one(resolver, "hydralzine")
    r = out.resolution
    assert (r.status, r.match, r.ingredient_key, r.name) == (
        "resolved",
        "approximate",
        HYDRALAZINE,
        "hydralazine",
    )
    # The score is the local Jaro-Winkler against the ingredient's own names, not RxNav's.
    assert r.score == round(JaroWinkler.similarity("hydralzine", "hydralazine"), 4)
    assert out.step == "approximate"
    assert rx.names_called[:2] == ["find_rxcuis", "approximate"]  # step 3 before step 4


async def test_rxnav_candidates_are_mapped_to_ingredients_and_scored_locally(resolver, rx):
    # RxNav answers with a product concept ("hydralazine Pill"); the ingredient comes from related.
    rx.approx["hydralzine"] = [ApproxCandidate("1164001", 8, "hydralazine Pill")]
    rx.ingredient_map["1164001"] = [Ingredient(HYDRALAZINE, "hydralazine")]
    r = (await one(resolver, "hydralzine")).resolution
    assert (r.status, r.ingredient_key) == ("resolved", HYDRALAZINE)


async def test_trigram_rescues_a_misspelling_when_rxnav_finds_nothing(resolver, rx):
    # RxNav does not try spelling variants for short words, and may know nothing at all.
    assert rx.approx == {}
    out = await one(resolver, "klonipin")
    r = out.resolution
    assert (r.status, r.match, r.ingredient_key, r.name) == (
        "resolved",
        "approximate",
        "2598",
        "clonazepam",
    )
    assert "approximate" in rx.names_called  # it did ask


async def test_trigram_handles_a_five_letter_brand_misspelling(resolver):
    r = (await one(resolver, "lasik")).resolution
    assert (r.status, r.ingredient_key) == ("resolved", "4603")


async def test_the_best_alias_of_a_key_scores_it_not_the_worst(resolver, rx):
    # "hydroxyzin hcl" is far from the key's IN name but close to its salt name.
    rx.approx["hydroxyzin hydrochlorid"] = [ApproxCandidate(HYDROXYZINE, 1, "hydroxyzine")]
    r = (await one(resolver, "hydroxyzin HCl")).resolution
    assert r.status == "resolved"
    assert r.score >= 0.95


async def test_two_close_ingredients_ask_for_clarification(resolver, rx):
    rx.approx["prednisolon"] = [
        ApproxCandidate(PREDNISOLONE, 1, "prednisolone"),
        ApproxCandidate(PREDNISONE, 2, "prednisone"),
    ]
    r = (await one(resolver, "prednisolon")).resolution
    assert r.status == "ambiguous"
    assert r.match is None and r.ingredient_key is None
    assert [c.ingredient_key for c in r.candidates] == [PREDNISOLONE, PREDNISONE]
    assert [c.name for c in r.candidates] == ["prednisolone", "prednisone"]
    assert r.candidates[0].score > r.candidates[1].score
    assert len(r.candidates) <= 3


async def test_a_weak_best_match_is_not_recognised(resolver, rx):
    out = await one(resolver, "xyzzy")
    assert out.resolution.status == "unknown"
    assert out.resolution.ingredient_key is None
    assert out.resolution.candidates == []


async def test_a_candidate_below_the_floor_is_not_recognised(resolver, rx):
    rx.approx["flurb"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    r = (await one(resolver, "flurb")).resolution
    assert r.status == "unknown"
    assert r.score is not None and r.score < 0.80


async def test_nothing_but_form_words_is_not_recognised_and_asks_nobody(resolver, rx):
    r = (await one(resolver, "25 mg tablets")).resolution
    assert r.status == "unknown"
    assert rx.calls == []


# --- step 9: coverage ---


async def test_a_resolved_drug_without_a_label_is_not_covered(resolver):
    r = (await one(resolver, "Ambien")).resolution
    assert (r.status, r.match, r.ingredient_key, r.name) == (
        "not_covered",
        "brand",
        "39993",
        "zolpidem",
    )


async def test_a_combination_brand_resolves_to_its_combination_key_and_is_not_covered(resolver):
    r = (await one(resolver, "Kombiglyze")).resolution
    assert (r.status, r.match, r.ingredient_key) == ("not_covered", "brand", "6809+857974")
    assert r.ingredient_rxcuis == ["6809", "857974"]
    assert r.product_hint == "1372730"
    assert r.name == "kombiglyze"  # 857974 is unknown here, so the brand is the display name


async def test_an_ingredient_outside_the_alias_table_is_not_covered(resolver, rx):
    rx.approx["warfarn"] = [ApproxCandidate("11289", 1, "warfarin")]
    rx.ingredient_map["11289"] = [Ingredient("11289", "warfarin")]
    r = (await one(resolver, "warfarn")).resolution
    assert (r.status, r.match, r.ingredient_key, r.name) == (
        "not_covered",
        "approximate",
        "11289",
        "warfarin",
    )


async def test_a_rxnav_exact_combination_hit_is_not_covered(resolver, rx):
    rx.exact["tribenzor"] = ["1372732"]
    rx.concepts["1372732"] = Concept("1372732", "Tribenzor", "BN")
    rx.ingredient_map["1372732"] = [
        Ingredient("17767", "amlodipine"),
        Ingredient("118463", "olmesartan"),
        Ingredient("5487", "hydrochlorothiazide"),
    ]
    r = (await one(resolver, "Tribenzor")).resolution
    assert (r.status, r.ingredient_key) == ("not_covered", "5487+17767+118463")
    assert r.name == "hydrochlorothiazide / amlodipine / olmesartan"  # in key order


# --- pinned_rxcuis ---


@pytest.fixture
def prednisol(rx):
    rx.approx["prednisolon"] = [
        ApproxCandidate(PREDNISOLONE, 1, "prednisolone"),
        ApproxCandidate(PREDNISONE, 2, "prednisone"),
    ]


async def test_a_pinned_ingredient_in_the_candidate_set_wins(resolver, prednisol):
    r = (await one(resolver, "prednisolon", pinned_rxcuis=[PREDNISONE])).resolution
    assert (r.status, r.match, r.ingredient_key, r.name) == (
        "resolved",
        "pinned",
        PREDNISONE,
        "prednisone",
    )
    assert r.candidates == []


async def test_a_pin_outside_the_candidate_set_changes_nothing(resolver, prednisol):
    r = (await one(resolver, "prednisolon", pinned_rxcuis=[HYDROXYZINE])).resolution
    assert r.status == "ambiguous"


async def test_a_pin_never_overrides_a_clear_resolution_to_another_drug(resolver, rx):
    # hydralazine is typed correctly; a pin for hydroxyzine is not among its candidates.
    r = (await one(resolver, "hydralazine", pinned_rxcuis=[HYDROXYZINE])).resolution
    assert (r.match, r.ingredient_key) == ("exact", HYDRALAZINE)


async def test_a_pin_for_a_candidate_the_user_could_not_have_been_offered_is_ignored(resolver, rx):
    # hydralazine is a trigram/RxNav candidate for "flurb" but scores below ambiguous_min: it was
    # never on offer, so a pin for it must not turn a rejected name into a resolution.
    rx.approx["flurb"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    r = (await one(resolver, "flurb", pinned_rxcuis=[HYDRALAZINE])).resolution
    assert r.status == "unknown"


async def test_a_pin_still_goes_through_the_coverage_check(resolver, rx):
    rx.approx["zolpidim"] = [ApproxCandidate("39993", 1, "zolpidem")]
    r = (await one(resolver, "zolpidim", pinned_rxcuis=["39993"])).resolution
    assert (r.status, r.match, r.ingredient_key) == ("not_covered", "pinned", "39993")


async def test_a_pin_matches_a_combination_only_when_it_names_the_whole_set(resolver):
    one_ingredient = (await one(resolver, "kombiglyze", pinned_rxcuis=["6809"])).resolution
    assert one_ingredient.match == "brand"  # shares metformin, but it is not "just metformin"
    whole = (await one(resolver, "kombiglyze", pinned_rxcuis=["857974+6809"])).resolution
    assert whole.match == "pinned"
    assert whole.ingredient_key == "6809+857974"


async def test_malformed_pins_are_ignored(resolver, prednisol):
    r = (await one(resolver, "prednisolon", pinned_rxcuis=["", "abc", "8640+"])).resolution
    assert r.status == "ambiguous"


async def test_pins_apply_to_every_mention(resolver, prednisol):
    out = await resolver.resolve(["prednisolon", "vistaril"], pinned_rxcuis=[PREDNISONE])
    assert [(r.match, r.ingredient_key) for r in out] == [
        ("pinned", PREDNISONE),
        ("brand", HYDROXYZINE),
    ]


# --- many mentions ---


async def test_one_resolution_per_mention_in_order(resolver):
    out = await resolver.resolve(["Vistaril", "xyzzy", "hydralazine", "Vistaril"])
    assert [r.mention for r in out] == ["Vistaril", "xyzzy", "hydralazine", "Vistaril"]
    assert [r.status for r in out] == ["resolved", "unknown", "resolved", "resolved"]
    assert await resolver.resolve([]) == []


# --- NFR-4: RxNav down ---


async def test_rxnav_down_falls_back_to_the_alias_table_and_marks_the_result_degraded(resolver, rx):
    rx.down = True
    out = await one(resolver, "klonipin")
    assert out.degraded
    assert out.resolution.degraded  # contract v2
    assert (out.resolution.status, out.resolution.ingredient_key) == ("resolved", "2598")
    assert out.resolution.match == "approximate"


async def test_rxnav_down_exact_alias_still_works_and_is_not_degraded(resolver, rx):
    rx.down = True
    out = await one(resolver, "Vistaril")
    assert (out.resolution.status, out.resolution.match) == ("resolved", "brand")
    assert not out.degraded
    assert not out.resolution.degraded
    assert rx.calls == []


async def test_rxnav_down_does_not_raise_for_unknown_names(resolver, rx):
    rx.down = True
    out = await one(resolver, "xyzzy")
    assert out.resolution.status == "unknown"
    assert out.degraded and out.resolution.degraded


async def test_a_failure_after_a_successful_call_discards_the_rxnav_candidates(
    resolver, rx, monkeypatch
):
    # Step 3 answers, step 4 fails: the whole RxNav contribution is dropped, not half of it.
    rx.approx["hydralzine"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]

    async def boom(term, max_entries=10):
        raise RxNavUnavailable("step 4 failed")

    monkeypatch.setattr(rx, "approximate", boom)
    out = await one(resolver, "hydralzine")
    assert out.degraded
    assert out.resolution.ingredient_key == HYDRALAZINE  # trigram alone still finds it


async def test_local_only_mode_makes_no_rxnav_calls_and_is_not_degraded(make, cache):
    out = await one(make(rxnav=None), "klonipin")
    assert (out.resolution.status, out.resolution.ingredient_key) == ("resolved", "2598")
    assert not out.degraded
    assert await cache.get("rx:resolve:klonipin") is None


# --- step 10: cache ---


async def test_rxnav_based_results_are_cached_as_json_under_the_spec_key(resolver, rx, cache):
    rx.approx["hydralzine"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    first = (await one(resolver, "Hydralzine 25 mg")).resolution
    raw = await cache.get("rx:resolve:hydralzine")  # the normalised mention
    assert raw is not None
    body = json.loads(raw)
    assert body["norm"] == "hydralzine"
    assert any(s["key"] == HYDRALAZINE for s in body["scored"])

    rx.calls.clear()
    again = await one(resolver, "hydralzine")
    assert again.cached
    assert again.resolution.model_dump() == first.model_dump() | {"mention": "hydralzine"}
    assert rx.calls == []


async def test_the_cache_ttl_is_24_hours(pool, rx, thresholds):
    now = [0.0]
    cache = InMemoryCache(clock=lambda: now[0])
    resolver = CascadeResolver(PgLocalStore(pool), rx, thresholds.resolver, cache)
    rx.approx["hydralzine"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    await one(resolver, "hydralzine")
    now[0] = 24 * 3600 - 1
    assert await cache.get("rx:resolve:hydralzine") is not None
    now[0] = 24 * 3600
    assert await cache.get("rx:resolve:hydralzine") is None
    rx.calls.clear()
    assert not (await one(resolver, "hydralzine")).cached
    assert rx.calls  # expired: RxNav is asked again


async def test_local_exact_results_are_not_cached(resolver, cache):
    await one(resolver, "Vistaril")
    assert await cache.get("rx:resolve:vistaril") is None


async def test_degraded_results_are_not_cached(resolver, rx, cache):
    rx.down = True
    await one(resolver, "klonipin")
    assert await cache.get("rx:resolve:klonipin") is None
    rx.down = False
    out = await one(resolver, "klonipin")  # RxNav is back: the next request uses it
    assert not out.degraded
    assert await cache.get("rx:resolve:klonipin") is not None


async def test_coverage_is_checked_on_every_request_not_cached(resolver, rx, pg_conn):
    rx.approx["warfarn"] = [ApproxCandidate("11289", 1, "warfarin")]
    rx.ingredient_map["11289"] = [Ingredient("11289", "warfarin")]
    assert (await one(resolver, "warfarn")).resolution.status == "not_covered"
    # A label is ingested while the entry is cached: no flush needed.
    await pg_conn.execute(
        "INSERT INTO label (setid, spl_version, title, ingredient_key, ingredient_rxcuis, "
        "product_rxcuis) VALUES (%s, 1, 'warfarin label', '11289', '{11289}', '{}')",
        (uuid.uuid4(),),
    )
    await pg_conn.commit()
    out = await one(resolver, "warfarn")
    assert out.cached
    assert out.resolution.status == "resolved"


async def test_the_gate_is_applied_after_the_cache_so_thresholds_take_effect(
    make, rx, cache, thresholds
):
    rx.approx["hydralzine"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    assert (await one(make(), "hydralzine")).resolution.status == "resolved"
    strict = thresholds.resolver.model_copy(update={"resolved_min": 0.999})
    out = await one(make(th=strict), "hydralzine")
    assert out.cached
    assert out.resolution.status == "ambiguous"


async def test_pins_work_on_a_cached_entry(make, rx):
    rx.approx["prednisolon"] = [
        ApproxCandidate(PREDNISOLONE, 1, "prednisolone"),
        ApproxCandidate(PREDNISONE, 2, "prednisone"),
    ]
    first = await one(make(), "prednisolon")
    assert first.resolution.status == "ambiguous"
    rx.calls.clear()
    pinned = await one(make(), "prednisolon", pinned_rxcuis=[PREDNISONE])
    assert pinned.cached
    assert (pinned.resolution.match, pinned.resolution.ingredient_key) == ("pinned", PREDNISONE)
    assert rx.calls == []


async def test_a_corrupt_cache_entry_is_ignored(resolver, rx, cache):
    await cache.set("rx:resolve:hydralzine", b"{not json", 60)
    rx.approx["hydralzine"] = [ApproxCandidate(HYDRALAZINE, 1, "hydralazine")]
    out = await one(resolver, "hydralzine")
    assert not out.cached
    assert out.resolution.ingredient_key == HYDRALAZINE
