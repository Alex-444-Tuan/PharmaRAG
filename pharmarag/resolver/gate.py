"""Decision gate (spec section 7). Pure: scores and thresholds in, a verdict out.

Outcome    Rule
resolved   exact match, or top >= resolved_min and top - next distinct ingredient >= margin
ambiguous  two distinct ingredients within `margin`, or ambiguous_min <= top < resolved_min
unknown    top < ambiguous_min

The gate does not know about coverage (step 9): "not covered" is decided after it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pharmarag.app.settings import ResolverThresholds

Outcome = Literal["resolved", "ambiguous", "unknown"]

# Scores are sums and differences of floats: 0.95 - 0.90 is 0.04999999999999993. Comparing
# at nine decimals makes "at least 0.05 above" and "at least 0.90" mean what they say.
_DECIMALS = 9


@dataclass(frozen=True)
class KeyScore:
    """The best similarity between the mention and any name of one ingredient key."""

    key: str
    score: float


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    top: KeyScore | None  # the winner when resolved, else the best candidate (if any)
    candidates: tuple[KeyScore, ...]  # ranked, at most `max_candidates` (ambiguous only)


def _r(x: float) -> float:
    return round(x, _DECIMALS)


def rank(scores: Sequence[KeyScore]) -> list[KeyScore]:
    """Best first, one entry per key (its best score), ties broken by key for determinism."""
    best: dict[str, float] = {}
    for s in scores:
        best[s.key] = max(best.get(s.key, float("-inf")), s.score)
    return sorted((KeyScore(k, v) for k, v in best.items()), key=lambda s: (-_r(s.score), s.key))


def decide(scores: Sequence[KeyScore], th: ResolverThresholds, *, exact: bool = False) -> Verdict:
    """Apply the gate to the scored candidates of one mention.

    `exact` means the candidates came from a local alias or RxNav exact match (steps 2 and 3).
    One exact ingredient is resolved whatever its score. Several distinct exact ingredients
    (a name two drugs share) fall through to the margin rule and come out ambiguous.
    """
    ranked = rank(scores)
    if not ranked:
        return Verdict("unknown", None, ())
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    if exact and second is None:
        return Verdict("resolved", top, ())
    if _r(top.score) < th.ambiguous_min:
        return Verdict("unknown", top, ())

    contested = second is not None and _r(top.score - second.score) < th.margin
    if _r(top.score) >= th.resolved_min and not contested:
        return Verdict("resolved", top, ())

    # SPEC: section 7 says "up to 3 candidates" without a floor. Candidates below ambiguous_min
    # are not offered: a clarification that lists a drug the name does not resemble is noise.
    options = [s for s in ranked if _r(s.score) >= th.ambiguous_min]
    return Verdict("ambiguous", top, tuple(options[: th.max_candidates]))
