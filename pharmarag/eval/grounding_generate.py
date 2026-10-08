"""Generate the 40 natural test claims (spec section 11.2, "Claim set").

Runs the real graph (real retrieval, real LLM) on the 14 fixed questions with the verifier off,
and keeps every writer sentence with a citation, together with the evidence the writer saw.

The only step in the grounding eval that calls the LLM. Cost is one `analyze` call and one writer
call per question per pass, counted and reported by `CountingLLM`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pharmarag.eval.grounding_data import Claim, Question
from pharmarag.graph.deps import Deps
from pharmarag.graph.run import effective_flags, initial_state
from pharmarag.graph.state import Flags
from pharmarag.llm.base import LLMClient, Message

TARGET_CLAIMS = 40


class CountingLLM:
    """Wraps an LLMClient and counts calls and characters (a cost estimate, not billing data)."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self.calls = 0
        self.input_chars = 0
        self.output_chars = 0

    def _count_in(self, messages: Sequence[Message], system: str | None) -> None:
        self.calls += 1
        self.input_chars += sum(len(m["content"]) for m in messages) + len(system or "")

    async def complete(self, messages, *, system=None, temperature=0.0, max_tokens=500) -> str:
        self._count_in(messages, system)
        out = await self._inner.complete(
            messages, system=system, temperature=temperature, max_tokens=max_tokens
        )
        self.output_chars += len(out)
        return out

    async def stream(
        self, messages, *, system=None, temperature=0.0, max_tokens=500
    ) -> AsyncIterator[str]:
        self._count_in(messages, system)
        async for piece in self._inner.stream(
            messages, system=system, temperature=temperature, max_tokens=max_tokens
        ):
            self.output_chars += len(piece)
            yield piece

    async def structured(self, messages, schema, *, system=None, temperature=0.0, max_tokens=500):
        self._count_in(messages, system)
        out = await self._inner.structured(
            messages, schema, system=system, temperature=temperature, max_tokens=max_tokens
        )
        self.output_chars += len(out.model_dump_json())
        return out

    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "approx_input_tokens": self.input_chars // 4,
            "approx_output_tokens": self.output_chars // 4,
            "note": "tokens estimated as characters / 4; thinking tokens are not counted",
        }


async def ask_with_evidence(graph: Any, deps: Deps, question: str) -> dict:
    """One request through the graph with the verifier off. Returns the final graph state."""
    flags = effective_flags(Flags(resolver=True, verifier=False), deps)
    if flags.verifier:
        raise RuntimeError("claim generation needs the verifier off (AT-2 baseline)")
    state = initial_state(question, flags=flags)
    return await graph.ainvoke(
        state, config={"recursion_limit": deps.thresholds.graph.recursion_limit}
    )


def claims_from_state(
    q: Question, final: dict, max_facts: int, model_id: str, pass_no: int
) -> list[Claim]:
    """The cited sentences of one answer, each with the evidence the writer was given.

    Mirrors what the `answer` node passed to the writer: the chunks in `evidence` and the facts
    of the final retrieval attempt, capped at `max_facts`. Sentences with no valid citation are
    skipped here (FR-5 removes them in every mode, so they cannot reach a user).
    """
    attempt = final.get("retrieval_attempt", 1)
    facts = [f for f in final.get("rx_facts", []) if f.retrieval_attempt == attempt][:max_facts]
    evidence = final.get("evidence", [])
    out = []
    for i, s in enumerate(final.get("sentences", [])):
        if not s.citations:
            continue
        out.append(
            Claim(
                id="",
                question=q.question,
                sentence=s.text,
                citations=s.citations,
                evidence=evidence,
                facts=facts,
                kind="natural",
                meta={
                    "question_id": q.id,
                    "answerable": q.answerable,
                    "sentence_index": i,
                    "pass": pass_no,
                    "llm_model_id": model_id,
                    "status": (final.get("response") or {}).get("status"),
                },
            )
        )
    return out


def select_claims(candidates: Sequence[Claim], target: int = TARGET_CLAIMS) -> list[Claim]:
    """Up to `target` claims, round-robin over questions so no question dominates, with
    duplicate sentences (same question, same text) dropped. Ids g01, g02, ... in selection order.
    """
    by_q: dict[str, list[Claim]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for c in candidates:
        key = (c.meta["question_id"], c.sentence.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        by_q[c.meta["question_id"]].append(c)
    chosen: list[Claim] = []
    queues = [list(v) for _, v in sorted(by_q.items())]
    while len(chosen) < target and any(queues):
        for qu in queues:
            if qu and len(chosen) < target:
                chosen.append(qu.pop(0))
    width = max(2, len(str(target)))
    for i, c in enumerate(chosen, 1):
        c.id = f"g{i:0{width}d}"
    return chosen


async def generate(
    graph: Any,
    deps: Deps,
    questions: Sequence[Question],
    *,
    llm: CountingLLM,
    target: int = TARGET_CLAIMS,
    max_passes: int = 3,
) -> tuple[list[Claim], list[dict]]:
    """Ask every question; repeat passes (new samples) until `target` distinct claims exist."""
    candidates: list[Claim] = []
    log: list[dict] = []
    for pass_no in range(1, max_passes + 1):
        for q in questions:
            final = await ask_with_evidence(graph, deps, q.question)
            got = claims_from_state(
                q, final, deps.thresholds.retrieval.max_facts, llm.model_id, pass_no
            )
            candidates.extend(got)
            log.append(
                {
                    "pass": pass_no,
                    "question_id": q.id,
                    "status": (final.get("response") or {}).get("status"),
                    "evidence_ok": final.get("evidence_ok"),
                    "sentences": len(final.get("sentences", [])),
                    "cited_sentences": len(got),
                }
            )
        chosen = select_claims(candidates, target)
        if len(chosen) >= target:
            return chosen, log
    return select_claims(candidates, target), log
