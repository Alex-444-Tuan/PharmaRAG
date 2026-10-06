"""Look-alike misinterpretation eval (spec section 11.1, acceptance target AT-1).

Each dataset row is one single-drug question with a gold ingredient key. Two arms run on it:

* baseline: resolver and ingredient filter off. The question is answered by unfiltered
  retrieval over the whole corpus.
* system: resolver on, so retrieval is filtered to the resolved ingredient (or the request
  stops with a clarification).

Definitions (section 11.1):

* **Misinterpreted**: the answer cites any chunk whose ingredient_key differs from gold, or the
  status is `not_covered` or `drug_not_recognised` for a drug that is in the corpus.
* **Clarified**: `needs_clarification` with the gold drug among the candidates. Counted on its
  own and not a misinterpretation: the system did not act on a wrong drug.
* Metric: (B - S) / B, with B and S the misinterpreted counts of baseline and system. Raw counts
  are always shown beside it.

Two modes. `full` runs the whole graph through `run_ask` (needs an LLM). `resolver-only` needs no
LLM: the system arm is the resolver alone, taking the drug `mention` from the dataset, and the
baseline arm is a PROXY (unfiltered dense and lexical retrieval fused, then a check of which
drugs the top evidence belongs to). A proxy is not the AT-1 number: it never writes an answer.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from psycopg import AsyncConnection

from pharmarag.app.settings import RetrievalThresholds
from pharmarag.graph.state import Flags, Resolution
from pharmarag.models.embedder import Embedder
from pharmarag.resolver.base import Resolver
from pharmarag.retrieval.dense import dense_search
from pharmarag.retrieval.fusion import fuse
from pharmarag.retrieval.lexical import lexical_search

DATASET_DIR = Path(__file__).parent / "datasets"
KINDS = ("misspelling", "brand", "correct_partner")

# Outcomes. The first group is not a misinterpretation, the second is.
CORRECT = "correct"
CLARIFIED = "clarified"
OTHER = "other"  # full mode only: answered nothing (insufficient_evidence, declined)
WRONG_DRUG = "wrong_drug"
NOT_RECOGNISED = "not_recognised"
NOT_COVERED = "not_covered"
CLARIFY_WITHOUT_GOLD = "clarify_without_gold"
MISINTERPRETED = frozenset({WRONG_DRUG, NOT_RECOGNISED, NOT_COVERED, CLARIFY_WITHOUT_GOLD})
OUTCOMES = (
    CORRECT,
    CLARIFIED,
    OTHER,
    WRONG_DRUG,
    NOT_RECOGNISED,
    NOT_COVERED,
    CLARIFY_WITHOUT_GOLD,
)


@dataclass(frozen=True)
class Row:
    question: str
    mention: str
    gold_ingredient_key: str
    gold_name: str
    partner_name: str
    kind: Literal["misspelling", "brand", "correct_partner"]


def dataset_path(split: str) -> Path:
    return DATASET_DIR / f"lookalike_{split}.jsonl"


def load_dataset(split: str) -> list[Row]:
    rows = [
        Row(**json.loads(line))
        for line in dataset_path(split).read_text().splitlines()
        if line.strip()
    ]
    return rows


# --- classification (pure) ---


def classify_resolution(res: Resolution, gold_key: str) -> str:
    """Resolver-only system arm: what happened to the gold drug."""
    if res.status in ("resolved", "not_covered"):
        if res.ingredient_key == gold_key:
            return CORRECT if res.status == "resolved" else NOT_COVERED
        return WRONG_DRUG  # acted on another drug (even one without a label)
    if res.status == "ambiguous":
        gold_offered = any(c.ingredient_key == gold_key for c in res.candidates)
        return CLARIFIED if gold_offered else CLARIFY_WITHOUT_GOLD
    return NOT_RECOGNISED


def _candidate_matches(candidate: Any, gold_key: str, gold_name: str) -> bool:
    """Whether one entry of a response's `candidates` is the gold drug.

    SPEC: section 10 says a `needs_clarification` response carries `candidates` (up to 3) and
    does not give the entry's shape. Entries are read as objects with `ingredient_key` or
    `rxcui`, or `name`; a bare string is a key or a name.
    """
    if isinstance(candidate, str):
        return candidate == gold_key or candidate.lower() == gold_name.lower()
    if not isinstance(candidate, dict):
        return False
    return (
        candidate.get("ingredient_key") == gold_key
        or candidate.get("rxcui") == gold_key
        or str(candidate.get("name", "")).lower() == gold_name.lower()
    )


def cited_keys(response: dict, key_of_setid: dict[str, str]) -> list[str]:
    """Ingredient key of every cited source. A label chunk maps through its setid, an RxNorm fact
    line (an `rxcui` entry) is its own ingredient's key. A source of unknown origin is "?"."""
    keys: list[str] = []
    for src in response.get("sources", []):
        if "setid" in src:
            keys.append(key_of_setid.get(str(src["setid"]), "?"))
        elif "rxcui" in src:
            keys.append(str(src["rxcui"]))
        else:
            keys.append("?")
    return keys


def classify_response(
    response: dict, row: Row, key_of_setid: dict[str, str]
) -> tuple[str, list[str]]:
    """Full mode: (outcome, cited ingredient keys) for one `/v1/ask` response."""
    status = response.get("status")
    if status == "needs_clarification":
        offered = any(
            _candidate_matches(c, row.gold_ingredient_key, row.gold_name)
            for c in response.get("candidates", [])
        )
        return (CLARIFIED if offered else CLARIFY_WITHOUT_GOLD), []
    if status == "drug_not_recognised":
        return NOT_RECOGNISED, []
    if status == "not_covered":
        return NOT_COVERED, []
    if status == "answered":
        keys = cited_keys(response, key_of_setid)
        if not keys:
            return OTHER, keys
        if any(k != row.gold_ingredient_key for k in keys):
            return WRONG_DRUG, keys
        return CORRECT, keys
    return OTHER, []  # insufficient_evidence, declined


def counts(outcomes: Iterable[str]) -> dict[str, int]:
    c = Counter(outcomes)
    out = {o: c.get(o, 0) for o in OUTCOMES}
    out["misinterpreted"] = sum(c.get(o, 0) for o in MISINTERPRETED)
    out["total"] = sum(c.values())
    return out


def reduction(baseline: int, system: int) -> float | None:
    """(B - S) / B. None when B is 0: there is nothing to reduce."""
    return None if baseline == 0 else (baseline - system) / baseline


# --- resolver-only arms ---


@dataclass(frozen=True)
class SystemRow:
    mention: str
    gold_ingredient_key: str
    kind: str
    outcome: str
    status: str
    match: str | None
    resolved_key: str | None
    resolved_name: str | None
    score: float | None
    candidates: list[dict]
    degraded: bool


async def run_resolver_system(rows: Sequence[Row], resolver: Resolver) -> list[SystemRow]:
    """Resolve each row's `mention` and classify it. `Resolution.degraded` says whether RxNav
    was down for that row."""
    out: list[SystemRow] = []
    for row in rows:
        (res,) = await resolver.resolve([row.mention])
        out.append(
            SystemRow(
                mention=row.mention,
                gold_ingredient_key=row.gold_ingredient_key,
                kind=row.kind,
                outcome=classify_resolution(res, row.gold_ingredient_key),
                status=res.status,
                match=res.match,
                resolved_key=res.ingredient_key,
                resolved_name=res.name,
                score=res.score,
                candidates=[c.model_dump() for c in res.candidates],
                degraded=res.degraded,
            )
        )
    return out


@dataclass(frozen=True)
class ProxyRow:
    question: str
    gold_ingredient_key: str
    kind: str
    top1_key: str | None
    top8_keys: list[str]
    top1_wrong: bool
    any_top8_wrong: bool


async def run_baseline_proxy(
    rows: Sequence[Row],
    conn: AsyncConnection,
    embedder: Embedder,
    retrieval: RetrievalThresholds,
) -> list[ProxyRow]:
    """PROXY for the baseline arm, not the AT-1 number: unfiltered dense and lexical retrieval
    of the question, fused with RRF. A row counts when the top evidence chunk (`top1_wrong`), or
    any of the top `evidence_top_k` (`any_top8_wrong`), belongs to another ingredient key. No
    answer is written, so nothing here says whether a writer would have cited that chunk."""
    out: list[ProxyRow] = []
    for row in rows:
        vec = embedder.embed_query(row.question)
        hits = await dense_search(
            conn,
            vec,
            ingredient_key=None,
            sections=None,
            top_k=retrieval.top_k_per_branch,
            attempt=1,
        )
        hits += await lexical_search(
            conn,
            row.question,
            ingredient_key=None,
            sections=None,
            top_k=retrieval.top_k_per_branch,
            attempt=1,
        )
        evidence, _ok = fuse(hits, attempt=1, ingredient_keys=[], thresholds=retrieval)
        keys = [c.ingredient_key for c in evidence]
        gold = row.gold_ingredient_key
        out.append(
            ProxyRow(
                question=row.question,
                gold_ingredient_key=gold,
                kind=row.kind,
                top1_key=keys[0] if keys else None,
                top8_keys=keys,
                top1_wrong=bool(keys) and keys[0] != gold,
                any_top8_wrong=any(k != gold for k in keys),
            )
        )
    return out


def proxy_counts(rows: Sequence[ProxyRow]) -> dict[str, int]:
    return {
        "total": len(rows),
        "top1_wrong": sum(r.top1_wrong for r in rows),
        "any_top8_wrong": sum(r.any_top8_wrong for r in rows),
        "no_evidence": sum(r.top1_key is None for r in rows),
    }


# --- full mode (whole graph) ---

Arm = Literal["baseline", "system"]
AskFn = Callable[[str, Flags], Awaitable[dict]]


@dataclass(frozen=True)
class FullRow:
    question: str
    gold_ingredient_key: str
    kind: str
    outcome: str
    status: str
    cited_keys: list[str]


async def key_of_setids(conn: AsyncConnection, setids: Iterable[str]) -> dict[str, str]:
    ids = sorted(set(setids))
    if not ids:
        return {}
    cur = await conn.execute(
        "SELECT DISTINCT setid::text, ingredient_key FROM label WHERE setid::text = ANY(%s)",
        (ids,),
    )
    return {r[0]: r[1] for r in await cur.fetchall()}


async def run_full_arm(
    rows: Sequence[Row], arm: Arm, ask: AskFn, conn: AsyncConnection
) -> list[FullRow]:
    """Send every question through the graph. The baseline arm switches the resolver off
    (section 7, "Baseline for AT-1"); the system arm leaves everything on."""
    flags = Flags(resolver=(arm == "system"))
    out: list[FullRow] = []
    for row in rows:
        response = await ask(row.question, flags)
        setids = [s["setid"] for s in response.get("sources", []) if "setid" in s]
        outcome, keys = classify_response(response, row, await key_of_setids(conn, setids))
        out.append(
            FullRow(
                question=row.question,
                gold_ingredient_key=row.gold_ingredient_key,
                kind=row.kind,
                outcome=outcome,
                status=str(response.get("status")),
                cited_keys=keys,
            )
        )
    return out


def rows_as_dicts(rows: Sequence[Any]) -> list[dict]:
    return [asdict(r) for r in rows]
