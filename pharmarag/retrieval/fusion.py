"""Reciprocal rank fusion and evidence grading (spec section 8, "Fusion and grading").

Pure functions over `Hit` lists, so they are unit-testable without a database.
"""

from collections import defaultdict
from collections.abc import Collection, Sequence
from uuid import UUID

from pharmarag.app.settings import RetrievalThresholds
from pharmarag.graph.state import Chunk, Hit


def _rrf(hits: Sequence[Hit], k: int) -> dict[int, float]:
    """Sum 1 / (k + rank) over the branches a chunk appears in.

    `hits` come from at most one dense and one lexical list per ingredient key, and chunks
    belong to exactly one key, so a chunk has at most one rank per branch. If a caller passes
    duplicates anyway, only the best rank per (chunk, branch) counts.
    """
    best: dict[tuple[int, str], int] = {}
    for h in hits:
        key = (h.chunk_id, h.source)
        best[key] = min(best.get(key, h.rank), h.rank)
    scores: dict[int, float] = defaultdict(float)
    for (chunk_id, _source), rank in best.items():
        scores[chunk_id] += 1.0 / (k + rank)
    return scores


def fuse(
    hits: Sequence[Hit],
    *,
    attempt: int,
    ingredient_keys: Sequence[str],
    thresholds: RetrievalThresholds,
    preferred_setids: Collection[UUID] = (),
) -> tuple[list[Chunk], bool]:
    """Return (evidence chunks with ids c1..cN, evidence_ok).

    1. Keep only hits from `attempt`. The `hits` reducer is append-only, so after a retry
       the state holds both attempts and fusing both would mix stale results into the answer.
    2. RRF over the dense and lexical lists, k = thresholds.rrf_k.
    3. When fused scores tie, a chunk from a label in `preferred_setids` (the product_hint
       label) goes first. Then chunk id, for determinism.
    4. Top `evidence_top_k`, or `evidence_top_k_per_drug` per drug when there are several keys.
    5. evidence_ok: at least one chunk, and either the best dense score is >= dense_min or at
       least `overlap_min` evidence chunks were returned by both branches.
    """
    current = [h for h in hits if h.retrieval_attempt == attempt]
    scores = _rrf(current, thresholds.rrf_k)

    by_chunk: dict[int, Hit] = {}
    for h in current:
        by_chunk.setdefault(h.chunk_id, h)
    preferred = set(preferred_setids)

    def order(chunk_id: int) -> tuple[float, int, int]:
        # Rounded so float noise from summing in different orders cannot break a true tie.
        h = by_chunk[chunk_id]
        return (-round(scores[chunk_id], 12), 0 if h.setid in preferred else 1, chunk_id)

    ranked = sorted(by_chunk, key=order)
    if len(ingredient_keys) > 1:
        selected: list[int] = []
        for key in ingredient_keys:
            of_key = [c for c in ranked if by_chunk[c].ingredient_key == key]
            selected += of_key[: thresholds.evidence_top_k_per_drug]
    else:
        selected = ranked[: thresholds.evidence_top_k]

    evidence = [
        Chunk(
            id=f"c{i}",
            chunk_id=cid,
            setid=by_chunk[cid].setid,
            spl_version=by_chunk[cid].spl_version,
            ingredient_key=by_chunk[cid].ingredient_key,
            section_loinc=by_chunk[cid].section_loinc,
            section_title=by_chunk[cid].section_title,
            subsection_title=by_chunk[cid].subsection_title,
            text=by_chunk[cid].text,
            spans=by_chunk[cid].spans,
            effective_date=by_chunk[cid].effective_date,
            fused_score=scores[cid],
        )
        for i, cid in enumerate(selected, start=1)
    ]

    # SPEC: section 8 step 5 does not say whether "best dense score" ranges over all dense hits
    # or only the kept evidence, nor whether the multi-drug case is graded per drug. Simplest
    # reading: best over all current-attempt dense hits, one global verdict.
    best_dense = max((h.score for h in current if h.source == "dense"), default=0.0)
    dense_ids = {h.chunk_id for h in current if h.source == "dense"}
    lexical_ids = {h.chunk_id for h in current if h.source == "lexical"}
    overlap = sum(1 for c in evidence if c.chunk_id in dense_ids and c.chunk_id in lexical_ids)
    ok = bool(evidence) and (
        best_dense >= thresholds.dense_min or overlap >= thresholds.overlap_min
    )
    return evidence, ok
