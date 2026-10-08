"""Builder for the request-time grounding verifier.

The signature is the contract: the app lifespan calls it. Lane A owns the body (cycle 3).
"""

from pharmarag.app.settings import Settings, Thresholds
from pharmarag.grounding.base import Verifier
from pharmarag.models.embedder import Embedder


def build_verifier(settings: Settings, thresholds: Thresholds, embedder: Embedder) -> Verifier:
    """Load the NLI cross-encoder (`settings.nli_model`) once, run one warm-up inference,
    and reuse `embedder` for span embeddings."""
    raise NotImplementedError("cycle 3, lane A")
