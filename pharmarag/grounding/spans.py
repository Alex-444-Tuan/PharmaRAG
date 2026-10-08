"""Evidence spans and candidate selection (spec section 9, "Verifier, per sentence", step 1).

A span is a sentence or table row of a cited chunk (offsets are stored with the chunk), or a whole
RxNorm fact line. For one sentence the candidates are all spans of the chunks it cites plus the
`nearest_spans` spans closest to it by bi-encoder cosine across all evidence, trimmed to the
`candidate_spans` closest. The NLI model sees each span with one neighbour sentence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pharmarag.graph.state import Chunk, Fact


@dataclass(frozen=True)
class SpanRecord:
    id: str  # "c2:4" for span 4 of chunk c2, "r1" for a RxNorm fact
    source: str  # the citation id this span belongs to: "c2" or "r1"
    text: str  # the span itself: the number check and the embedding use this
    context: str  # span plus one neighbour sentence: the NLI premise
    section: str = ""  # section title of the chunk, as filed in the label
    subsection: str = ""  # subsection title, or ""


def build_spans(evidence: Sequence[Chunk], facts: Sequence[Fact]) -> list[SpanRecord]:
    """All spans of the evidence chunks, then the facts, in citation order.

    The neighbour is the previous span of the same chunk, or the next one for the first span.
    A chunk with no stored spans (should not happen) becomes one span covering its text.
    """
    out: list[SpanRecord] = []
    for chunk in evidence:
        pieces: list[tuple[int, str]] = []
        for sp in chunk.spans:
            text = chunk.text[sp.start : sp.end].strip()
            if text:
                pieces.append((sp.i, text))
        if not pieces and chunk.text.strip():
            pieces = [(0, chunk.text.strip())]
        for k, (i, text) in enumerate(pieces):
            if len(pieces) == 1:
                context = text
            elif k > 0:
                context = f"{pieces[k - 1][1]} {text}"
            else:
                context = f"{text} {pieces[k + 1][1]}"
            out.append(
                SpanRecord(
                    id=f"{chunk.id}:{i}",
                    source=chunk.id,
                    text=text,
                    context=context,
                    section=chunk.section_title or "",
                    subsection=chunk.subsection_title or "",
                )
            )
    for fact in facts:
        text = fact.text.strip()
        if text:
            out.append(SpanRecord(id=fact.id, source=fact.id, text=text, context=text))
    return out


def select_candidates(
    sentence_vec: np.ndarray,
    span_matrix: np.ndarray,
    records: Sequence[SpanRecord],
    citations: Sequence[str],
    *,
    nearest_spans: int,
    candidate_spans: int,
) -> list[int]:
    """Indices into `records` of the candidate spans, best cosine first.

    Union of (a) every span whose source is in `citations` and (b) the `nearest_spans` highest
    cosine spans over all evidence, then the top `candidate_spans` of that union by cosine.
    Vectors are L2-normalised, so cosine is a dot product. Ties keep evidence order.
    """
    if not records:
        return []
    sims = span_matrix @ sentence_vec
    cited = set(citations)
    chosen = {i for i, r in enumerate(records) if r.source in cited}
    nearest = sorted(range(len(records)), key=lambda i: (-float(sims[i]), i))[:nearest_spans]
    chosen.update(nearest)
    ranked = sorted(chosen, key=lambda i: (-float(sims[i]), i))
    return ranked[:candidate_spans]
