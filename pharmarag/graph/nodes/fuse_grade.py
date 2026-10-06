"""fuse_grade: evidence grader node (spec section 8, "Fusion and grading")."""

from uuid import UUID

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import RAGState
from pharmarag.retrieval.fusion import fuse


async def _preferred_setids(deps: Deps, state: RAGState) -> set[UUID]:
    """Labels that list a brand/salt RxCUI the user typed. A tie-break input, never a filter."""
    hints = [r.product_hint for r in state.get("resolutions", []) if r.product_hint]
    if not hints:
        return set()
    async with deps.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT DISTINCT setid FROM label WHERE product_rxcuis && %s::text[]", (hints,)
        )
        return {row[0] for row in await cur.fetchall()}


async def fuse_grade(state: RAGState, deps: Deps) -> dict:
    evidence, ok = fuse(
        state.get("hits", []),
        attempt=state.get("retrieval_attempt", 1),
        ingredient_keys=state.get("ingredient_keys", []),
        thresholds=deps.thresholds.retrieval,
        preferred_setids=await _preferred_setids(deps, state),
    )
    return {"evidence": evidence, "evidence_ok": ok}
