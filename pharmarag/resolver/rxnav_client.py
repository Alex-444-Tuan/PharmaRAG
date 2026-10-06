"""Request-time RxNav client (spec sections 7 and 8).

Not the ingestion client (`pharmarag.ingest.rxnav`): that one runs in batch, with a 10 s
timeout. This one runs inside a user request, so every call has a 2 s budget and a failure
must degrade the answer, not fail it (NFR-4).

SPEC: section 6 puts the RxNav retry in a LangGraph `RetryPolicy` on the node. It lives here,
in the client, so the retry also covers the cascade's several calls and a node policy cannot
double-retry on top of it. Three attempts, exponential backoff, on timeouts, connection errors,
HTTP 429 and 5xx. When the attempts run out the client raises `RxNavUnavailable`.

A circuit breaker sits on top: after 3 calls in a row that exhausted their retries the client
fails fast for `breaker_s` seconds; any answer resets the count. Without it every request
during an outage would wait three 2 s timeouts per call before falling back, which breaks the
latency target (NFR-1) for as long as RxNav is down. One exhausted call alone does not trip
it: a single slow reply must not switch RxNav off for 30 s.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from pharmarag.ingest.http import RateLimiter, get_with_retry

log = logging.getLogger(__name__)

# RxNav allows 20 requests per second per IP. 15 leaves headroom for other processes.
DEFAULT_MAX_RPS = 15.0
ATTEMPTS = 3
BACKOFF_S = 0.25
BREAKER_S = 30.0
# The breaker opens after this many calls in a row that exhausted their retries. One bad call
# (a slow reply, a blip) must not switch RxNav off for everyone for 30 s.
BREAKER_THRESHOLD = 3
_MEMO_SIZE = 4096


class RxNavUnavailable(Exception):
    """RxNav did not answer after the retries (or the circuit breaker is open)."""


@dataclass(frozen=True)
class Ingredient:
    rxcui: str
    name: str


@dataclass(frozen=True)
class Concept:
    rxcui: str
    name: str
    tty: str


@dataclass(frozen=True)
class ApproxCandidate:
    rxcui: str
    rank: int
    name: str | None


@dataclass
class RelatedConcepts:
    """Concepts related to one ingredient, by term type."""

    ingredient: Ingredient | None = None
    brands: list[Concept] = field(default_factory=list)  # BN
    branded_drugs: list[Concept] = field(
        default_factory=list
    )  # SBD, e.g. "x 5 MG Oral Tablet [Brand]"
    dose_forms: list[Concept] = field(default_factory=list)  # DF
    strengths: list[Concept] = field(default_factory=list)  # SCDC


class RxNavSource(Protocol):
    """What the cascade and the facts source need from RxNav. Tests stub this."""

    async def find_rxcuis(self, name: str) -> list[str]: ...

    async def approximate(self, term: str, max_entries: int = 10) -> list[ApproxCandidate]: ...

    async def concept(self, rxcui: str) -> Concept | None: ...

    async def ingredients(self, rxcui: str) -> list[Ingredient]: ...

    async def related(self, rxcui: str) -> RelatedConcepts: ...


class RxNavClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 2.0,
        client: httpx.AsyncClient | None = None,
        max_rps: float = DEFAULT_MAX_RPS,
        attempts: int = ATTEMPTS,
        backoff_s: float = BACKOFF_S,
        breaker_s: float = BREAKER_S,
        breaker_threshold: int = BREAKER_THRESHOLD,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None
        self._limiter = RateLimiter(max_rps)
        self._retries = max(attempts - 1, 0)
        self._backoff_s = backoff_s
        self._breaker_s = breaker_s
        self._clock = clock
        self._breaker_threshold = breaker_threshold
        self._failures = 0  # consecutive calls that exhausted their retries
        self._down_until = 0.0
        # rxcui -> ingredients. A concept's ingredients never change within a process
        # lifetime that matters, and the cascade asks for the same few concepts over and over.
        self._ingredients: OrderedDict[str, list[Ingredient]] = OrderedDict()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def breaker_open(self) -> bool:
        return self._clock() < self._down_until

    async def _json(self, path: str, **params: str) -> dict[str, Any]:
        """GET `path`. {} for a 4xx other than 429 (RxNav says "no such thing"), never raises
        anything but `RxNavUnavailable`."""
        # SPEC: the circuit breaker is not in the spec. See the module docstring for why.
        if self.breaker_open:
            raise RxNavUnavailable("RxNav circuit breaker open")
        try:
            resp = await get_with_retry(
                self._client,
                f"{self._base}{path}",
                retries=self._retries,
                backoff_s=self._backoff_s,
                before_request=self._limiter.wait,
                params=params or None,
            )
            body = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 and e.response.status_code != 429:
                # RxNav answered, so it is up: a 404 is "no such thing". A 400 usually means
                # this client sent a malformed request, which is a bug worth seeing.
                self._failures = 0
                log.warning(
                    "rxnav %s answered HTTP %s; treating as no result", path, e.response.status_code
                )
                return {}
            self._trip(path, e)
        except (httpx.TransportError, ValueError) as e:  # ValueError: body was not JSON
            self._trip(path, e)
        else:
            self._failures = 0
            return body
        raise AssertionError("unreachable")  # pragma: no cover

    def _trip(self, path: str, error: Exception) -> None:
        self._failures += 1
        if self._failures >= self._breaker_threshold:
            self._down_until = self._clock() + self._breaker_s
            self._failures = 0
        log.warning("rxnav unavailable (%s): %s: %s", path, type(error).__name__, error)
        raise RxNavUnavailable(f"{path}: {type(error).__name__}") from error

    # --- cascade endpoints ---------------------------------------------------------------

    async def find_rxcuis(self, name: str) -> list[str]:
        """Step 3: `rxcui.json?name=&search=2`. Exact, then normalised match. [] if none."""
        body = await self._json("/rxcui.json", name=name, search="2")
        return list((body.get("idGroup") or {}).get("rxnormId") or [])

    async def approximate(self, term: str, max_entries: int = 10) -> list[ApproxCandidate]:
        """Step 4: `approximateTerm.json`, active concepts only (`option=1`).

        The response has one row per source vocabulary, so the same rxcui repeats. Rows are
        de-duplicated by rxcui, in rank order. RxNav's scores are ignored (spec section 7).
        """
        # SPEC: "restricted to active concepts" is read as RxNav's `option=1` (active concepts).
        body = await self._json(
            "/approximateTerm.json", term=term, maxEntries=str(max_entries), option="1"
        )
        out: dict[str, ApproxCandidate] = {}
        for row in (body.get("approximateGroup") or {}).get("candidate") or []:
            rxcui = str(row.get("rxcui") or "")
            if not rxcui:
                continue
            try:
                rank = int(row.get("rank", 1_000_000))
            except (TypeError, ValueError):
                rank = 1_000_000
            known = out.get(rxcui)
            if known is None:
                out[rxcui] = ApproxCandidate(rxcui, rank, row.get("name"))
            elif known.name is None and row.get("name"):
                out[rxcui] = ApproxCandidate(rxcui, known.rank, row["name"])
        return sorted(
            out.values(), key=lambda c: (c.rank, int(c.rxcui) if c.rxcui.isdigit() else 0)
        )

    async def concept(self, rxcui: str) -> Concept | None:
        body = await self._json(f"/rxcui/{rxcui}/properties.json")
        props = body.get("properties") or {}
        if not props.get("rxcui"):
            return None
        return Concept(props["rxcui"], props.get("name", ""), props.get("tty", ""))

    async def ingredients(self, rxcui: str) -> list[Ingredient]:
        """Ingredients (TTY IN) behind any concept. An ingredient maps to itself. Memoised."""
        hit = self._ingredients.get(rxcui)
        if hit is not None:
            self._ingredients.move_to_end(rxcui)
            return hit
        found = [
            Ingredient(c["rxcui"], c.get("name", ""))
            for c in await self._related_concepts(rxcui, "IN")
        ]
        self._ingredients[rxcui] = found
        while len(self._ingredients) > _MEMO_SIZE:
            self._ingredients.popitem(last=False)
        return found

    # --- facts endpoint ------------------------------------------------------------------

    async def related(self, rxcui: str) -> RelatedConcepts:
        """Brands, dose forms and strengths of one ingredient in a single request."""
        # The tty list is space-separated: httpx sends a space as "+", which RxNav reads as a
        # separator. A literal "+" in the value would be sent as %2B, and RxNav answers HTTP 400.
        body = await self._json(f"/rxcui/{rxcui}/related.json", tty="IN BN DF SCDC SBD")
        out = RelatedConcepts()
        for group in (body.get("relatedGroup") or {}).get("conceptGroup") or []:
            tty = group.get("tty")
            concepts = [
                Concept(c["rxcui"], c.get("name", ""), tty or "")
                for c in group.get("conceptProperties") or []
            ]
            if tty == "IN" and concepts:
                out.ingredient = Ingredient(concepts[0].rxcui, concepts[0].name)
            elif tty == "BN":
                out.brands = concepts
            elif tty == "DF":
                out.dose_forms = concepts
            elif tty == "SCDC":
                out.strengths = concepts
            elif tty == "SBD":
                out.branded_drugs = concepts
        return out

    async def _related_concepts(self, rxcui: str, tty: str) -> list[dict[str, Any]]:
        body = await self._json(f"/rxcui/{rxcui}/related.json", tty=tty)
        out: list[dict[str, Any]] = []
        for group in (body.get("relatedGroup") or {}).get("conceptGroup") or []:
            out.extend(group.get("conceptProperties") or [])
        return out
