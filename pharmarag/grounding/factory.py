"""Builder for the request-time grounding verifier.

The signature is the contract: the app lifespan calls it. Lane A owns the body (cycle 3).
"""

from pharmarag.app.settings import Settings, Thresholds
from pharmarag.graph.state import Chunk, Sentence, Span
from pharmarag.grounding.base import Verifier
from pharmarag.grounding.verifier import (
    ContextMode,
    CrossEncoderNLI,
    GroundingVerifier,
    HeadingMode,
)
from pharmarag.models.embedder import Embedder

# SPEC: section 9 step 3 scores (span plus neighbour, sentence) pairs in one batch. The shipped
# default is "cascade": that batch first, then (only when it does not entail) a second batch with
# the span alone. Its verdicts equal "scored both premises", which on the dev set cut the
# false-reject rate (AT-2b) from 9.1% to 0% (reports/grounding_tune_dev_*.md). The heading is the
# label's subsection title put in front of the premise, so "in adults" is supported by "2.1 Adult
# Dosage". Set DEFAULT_CONTEXT = "neighbour" and DEFAULT_HEADING = "none" for the plain spec design.
DEFAULT_CONTEXT: ContextMode = "cascade"
DEFAULT_HEADING: HeadingMode = "sub"

_WARMUP_TEXT = "Take one tablet by mouth once daily."


def build_verifier(settings: Settings, thresholds: Thresholds, embedder: Embedder) -> Verifier:
    """Load the NLI cross-encoder (`settings.nli_model`) once, run one warm-up inference,
    and reuse `embedder` for span embeddings.

    The warm-up pushes a real sentence through the whole path (span embeddings, candidate
    selection, one NLI batch), so the first user request does not pay one-off setup costs.
    """
    nli = CrossEncoderNLI(settings.nli_model)
    verifier = GroundingVerifier(
        embedder,
        nli,
        thresholds.grounding,
        context=DEFAULT_CONTEXT,
        heading=DEFAULT_HEADING,
    )

    chunk = Chunk(
        id="c1",
        chunk_id=0,
        setid="00000000-0000-0000-0000-000000000000",
        spl_version=1,
        ingredient_key="0",
        section_loinc="34068-7",
        section_title="Dosage and administration",
        text=_WARMUP_TEXT,
        spans=[Span(i=0, start=0, end=len(_WARMUP_TEXT))],
        fused_score=0.0,
    )

    # Sync internals, not asyncio.run: the app builds the verifier from inside a running loop.
    # The warm-up sentence is a near-miss ("two" for "one"), so the cascade's second stage
    # normally runs too and both batch shapes are warm.
    prepared = verifier._prepare_sync([chunk], [])
    verifier._analyse_sync(
        Sentence(text="Take two tablets by mouth every day.", citations=["c1"]), prepared
    )
    return verifier
