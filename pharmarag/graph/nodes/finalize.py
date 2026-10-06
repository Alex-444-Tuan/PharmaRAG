"""finalize: response builder node (spec sections 6 and 10). The single exit of the graph.

The notice and the sources are added here, never by the model. This node is also the last
gate for FR-6: a sentence reaches the response only if it is `supported`, or `unverified`
on a request that has the verifier switched off (the AT-2 baseline).
"""

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Chunk, Fact, Flags, RAGState, Sentence, Status

NOTICE = "From FDA drug labels via DailyMed. Not medical advice."
DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
DECLINED_INTENTS = ("patient_advice", "out_of_scope")


def split_sentences(
    sentences: list[Sentence], *, verifier_on: bool
) -> tuple[list[Sentence], list[Sentence]]:
    """(shown, removed). Fail closed: with the verifier on, only `supported` is shown."""
    shown: list[Sentence] = []
    removed: list[Sentence] = []
    for s in sentences:
        ok = s.verdict == "supported" or (s.verdict == "unverified" and not verifier_on)
        (shown if ok else removed).append(s)
    return shown, removed


def _source_entries(shown: list[Sentence], evidence: list[Chunk], facts: list[Fact]) -> list[dict]:
    cited = list(dict.fromkeys(cid for s in shown for cid in s.citations))
    by_chunk = {c.id: c for c in evidence}
    by_fact = {f.id: f for f in facts}
    out: list[dict] = []
    for cid in cited:
        if cid in by_chunk:
            c = by_chunk[cid]
            out.append(
                {
                    "id": c.id,
                    "setid": str(c.setid),
                    "spl_version": c.spl_version,
                    "section": c.section_title,
                    "effective_date": c.effective_date.isoformat() if c.effective_date else None,
                    "url": DAILYMED_URL.format(setid=c.setid),
                }
            )
        elif cid in by_fact:
            f = by_fact[cid]
            out.append({"id": f.id, "rxcui": f.rxcui, "text": f.text})
    return out


async def _corpus_version(deps: Deps) -> str:
    async with deps.pool.connection() as conn:
        cur = await conn.execute("SELECT corpus_version FROM corpus_meta WHERE id = 1")
        row = await cur.fetchone()
    return row[0] if row else "unknown"


def _derive_status(state: RAGState, shown: list[Sentence]) -> Status:
    preset = state.get("status")
    if preset and preset != "answered":
        return preset
    if state.get("intent") in DECLINED_INTENTS:
        return "declined"
    if not state.get("evidence_ok", False):
        return "insufficient_evidence"
    return "answered" if shown else "insufficient_evidence"


async def finalize(state: RAGState, deps: Deps) -> dict:
    flags = state.get("flags") or Flags()
    shown, removed_now = split_sentences(state.get("sentences", []), verifier_on=flags.verifier)
    status = _derive_status(state, shown)
    if status != "answered":
        shown = []
    removed = [*state.get("removed_sentences", []), *removed_now]

    evidence = state.get("evidence", [])
    facts = state.get("rx_facts", [])
    drugs = [
        {"mention": r.mention, "name": r.name, "rxcui": r.ingredient_key, "match": r.match}
        for r in state.get("resolutions", [])
        if r.status == "resolved"
    ]
    answer = [
        {
            "text": s.text,
            "citations": s.citations,
            "support": {"span": s.span, "score": s.score} if s.span else None,
        }
        for s in shown
    ]
    response = {
        "status": status,
        "drugs": drugs,
        "answer": answer,
        "sources": _source_entries(shown, evidence, facts),
        "notice": NOTICE,
        "meta": {
            "corpus_version": await _corpus_version(deps),
            # SPEC: node time summed, which over-counts the two parallel retrieval branches.
            # `run_ask` replaces it with wall-clock time; state has no start timestamp.
            "latency_ms": round(sum(state.get("timings", {}).values())),
            "retrieval_attempts": state.get("retrieval_attempt", 1) if "evidence" in state else 0,
            "generation_attempts": state.get("generation_attempt", 1)
            if "sentences" in state
            else 0,
            "removed_sentences": len(removed),
        },
    }
    return {"status": status, "response": response, "removed_sentences": removed}
