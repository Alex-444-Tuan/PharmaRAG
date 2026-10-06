"""answer: writer node (spec section 9).

Cycle 1 runs the writer only. The verifier and the regeneration edge arrive in cycle 3, so
sentences leave this node as "unverified" (or "unsupported" for lines with no citation).
"""

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import RAGState
from pharmarag.grounding.writer import write_answer


async def answer(state: RAGState, deps: Deps) -> dict:
    attempt = state.get("retrieval_attempt", 1)
    facts = [f for f in state.get("rx_facts", []) if f.retrieval_attempt == attempt][
        : deps.thresholds.retrieval.max_facts
    ]
    result = await write_answer(
        deps.llm,
        state["question"],
        state.get("evidence", []),
        facts,
        temperature=deps.settings.llm_temperature,
        max_tokens=deps.settings.llm_max_tokens,
    )
    return {"sentences": result.sentences}
