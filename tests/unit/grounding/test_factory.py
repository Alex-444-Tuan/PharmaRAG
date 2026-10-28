"""build_verifier: models load once, one warm-up inference, no event-loop surprises."""

import pytest

from pharmarag.graph.state import Chunk, Sentence, Span
from pharmarag.grounding import factory
from pharmarag.grounding.verifier import CrossEncoderNLI, GroundingVerifier
from pharmarag.models.embedder import FakeEmbedder
from tests.unit.grounding.test_verifier import FakeNLI


@pytest.fixture
def fake_nli_class(monkeypatch):
    """Replace the real cross-encoder with a scripted one; record how it was built."""
    built = []

    def make(model_name, revision=None):
        nli = FakeNLI(lambda p, h: (0.9, 0.0))
        nli.model_name = model_name
        nli.revision = revision
        built.append(nli)
        return nli

    monkeypatch.setattr(factory, "CrossEncoderNLI", make)
    return built


def test_models_load_once_and_the_settings_model_is_used(settings, thresholds, fake_nli_class):
    s = settings.model_copy(update={"nli_model": "some/nli-model"})
    v = factory.build_verifier(s, thresholds, FakeEmbedder())
    assert isinstance(v, GroundingVerifier)
    assert len(fake_nli_class) == 1
    assert fake_nli_class[0].model_name == "some/nli-model"
    assert v.nli is fake_nli_class[0]


def test_one_warmup_inference_before_returning(settings, thresholds, fake_nli_class):
    factory.build_verifier(settings, thresholds, FakeEmbedder())
    nli = fake_nli_class[0]
    assert 1 <= len(nli.batches) <= 2  # one NLI pass, plus the cascade's second stage at most


def test_the_embedder_is_shared_not_reloaded(settings, thresholds, fake_nli_class):
    emb = FakeEmbedder()
    v = factory.build_verifier(settings, thresholds, emb)
    assert v.embedder is emb


def test_grounding_thresholds_are_the_frozen_ones(settings, thresholds, fake_nli_class):
    v = factory.build_verifier(settings, thresholds, FakeEmbedder())
    assert v.thresholds == thresholds.grounding


def test_premise_mode_and_candidates_come_from_the_frozen_thresholds(
    settings, thresholds, fake_nli_class
):
    v = factory.build_verifier(settings, thresholds, FakeEmbedder())
    g = thresholds.grounding
    assert (v.context, v.heading) == (g.premise_context, g.premise_heading)
    assert (g.premise_context, g.premise_heading, g.candidate_spans) == ("cascade", "sub", 3)
    # not constants in the factory: change the thresholds and the verifier follows
    other = thresholds.model_copy(
        update={
            "grounding": g.model_copy(
                update={
                    "premise_context": "neighbour",
                    "premise_heading": "none",
                    "candidate_spans": 4,
                }
            )
        }
    )
    w = factory.build_verifier(settings, other, FakeEmbedder())
    assert (w.context, w.heading, w.thresholds.candidate_spans) == ("neighbour", "none", 4)
    assert not hasattr(factory, "DEFAULT_CONTEXT") and not hasattr(factory, "DEFAULT_HEADING")


async def test_can_be_built_from_inside_a_running_event_loop(settings, thresholds, fake_nli_class):
    # The app builds the verifier in its async lifespan; asyncio.run() there would raise.
    v = factory.build_verifier(settings, thresholds, FakeEmbedder())
    chunk = Chunk(
        id="c1",
        chunk_id=1,
        setid="00000000-0000-0000-0000-000000000001",
        spl_version=1,
        ingredient_key="1",
        section_loinc="34068-7",
        section_title="Dosage",
        text="Take with food.",
        spans=[Span(i=0, start=0, end=15)],
        fused_score=0.1,
    )
    out = await v.verify([Sentence(text="Take with food.", citations=["c1"])], [chunk], [])
    assert out[0].verdict == "supported"


# --- the real model, when it is already in the local Hugging Face cache ---


def _cached_nli_path() -> str | None:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download("cross-encoder/nli-deberta-v3-small", local_files_only=True)
    except Exception:
        return None


@pytest.fixture(scope="module")
def real_nli():
    path = _cached_nli_path()
    if path is None:
        pytest.skip("cross-encoder/nli-deberta-v3-small is not in the local cache")
    return CrossEncoderNLI(path)


def test_real_model_label_order_is_read_from_its_config(real_nli):
    # This model lists contradiction, entailment, neutral. Nothing in our code says so.
    assert real_nli._model.model.config.id2label == {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }
    assert (real_nli.entail_idx, real_nli.contra_idx) == (1, 0)


def test_real_model_separates_entailment_from_contradiction(real_nli):
    premise = "Do not use in patients with severe renal impairment."
    e, c = real_nli.score(
        [
            (premise, "The drug should not be used in patients with severe kidney disease."),
            (premise, "The drug is safe in patients with severe renal impairment."),
        ]
    )
    assert e[0] > 0.9 and c[0] < 0.1
    assert c[1] > 0.8 and e[1] < 0.1
