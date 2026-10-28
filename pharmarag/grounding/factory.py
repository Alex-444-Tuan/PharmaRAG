"""Builder for the request-time grounding verifier.

The signature is the contract: the app lifespan calls it. Lane A owns the body (cycle 3).
"""

from pharmarag.app.settings import Settings, Thresholds
from pharmarag.graph.state import Chunk, Sentence, Span
from pharmarag.grounding.base import Verifier
from pharmarag.grounding.verifier import CrossEncoderNLI, GroundingVerifier
from pharmarag.models.embedder import Embedder

_WARMUP_TEXT = "Take one tablet by mouth once daily."


def build_verifier(settings: Settings, thresholds: Thresholds, embedder: Embedder) -> Verifier:
    """Load the NLI cross-encoder (`settings.nli_model`) once, run one warm-up inference,
    and reuse `embedder` for span embeddings.

    The premise construction (`premise_context`, `premise_heading`) and the candidate count come
    from `thresholds.grounding`, i.e. from the frozen `config/thresholds.yaml`. SPEC: section 9
    step 3 scores (span plus neighbour, sentence) in one batch; the frozen default is a cascade
    (that batch, then the span alone on a miss) with the subsection title in front, which the
    dev set showed is needed to keep the false-reject rate under the AT-2b limit (spec review
    log, round 5).

    The warm-up pushes a real sentence through the whole path (span embeddings, candidate
    selection, NLI batches), so the first user request does not pay one-off setup costs.
    """
    nli = CrossEncoderNLI(settings.nli_model, revision=settings.nli_model_revision or None)
    verifier = GroundingVerifier(embedder, nli, thresholds.grounding)

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
    verifier._analyse_many_sync(
        [Sentence(text="Take two tablets by mouth every day.", citations=["c1"])], prepared
    )
    return verifier
