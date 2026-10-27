"""Model loads pass the pinned Hugging Face commit (Settings.*_model_revision)."""

import sentence_transformers

from pharmarag.grounding import factory
from pharmarag.grounding.verifier import CrossEncoderNLI
from pharmarag.models.embedder import FakeEmbedder
from tests.unit.grounding.test_verifier import FakeNLI


def test_factory_passes_the_pinned_nli_revision(settings, thresholds, monkeypatch):
    built = []

    def make(model_name, revision=None):
        nli = FakeNLI(lambda p, h: (0.9, 0.0))
        nli.model_name, nli.revision = model_name, revision
        built.append(nli)
        return nli

    monkeypatch.setattr(factory, "CrossEncoderNLI", make)
    factory.build_verifier(
        settings.model_copy(update={"nli_model_revision": "abc123"}), thresholds, FakeEmbedder()
    )
    factory.build_verifier(
        settings.model_copy(update={"nli_model_revision": ""}), thresholds, FakeEmbedder()
    )
    assert built[0].revision == "abc123"
    assert built[1].revision is None  # an empty setting means "not pinned"


def test_cross_encoder_nli_passes_revision_to_sentence_transformers(monkeypatch):
    seen = {}

    class Config:
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    class Stub:
        def __init__(self, name, **kwargs):
            seen.update(kwargs, name=name)
            self.model = type("M", (), {"config": Config})()

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Stub)
    CrossEncoderNLI("some/model", revision="deadbeef")
    assert seen["name"] == "some/model" and seen["revision"] == "deadbeef"


def test_embedder_passes_revision_to_sentence_transformers(monkeypatch):
    from pharmarag.models.embedder import SentenceTransformerEmbedder

    seen = {}

    class Stub:
        def __init__(self, name, **kwargs):
            seen.update(kwargs, name=name)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Stub)
    SentenceTransformerEmbedder("some/model", 384, revision="cafe")
    assert seen["revision"] == "cafe"
