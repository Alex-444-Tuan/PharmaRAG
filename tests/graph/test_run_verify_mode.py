"""VERIFY_MODE=off and the request flags: who may switch verification off (FR-6, FR-9)."""

import pytest

from pharmarag.graph.deps import Deps
from pharmarag.graph.nodes.answer import request_verify_mode
from pharmarag.graph.run import effective_flags
from pharmarag.graph.state import Flags
from tests.graph.fakes import FakeResolver, FakeVerifier


@pytest.fixture
def make(settings, thresholds, fake_embedder, make_fake_llm):
    def _make(*, eval_mode: bool, verify_mode: str, verifier: bool = True) -> Deps:
        return Deps(
            llm=make_fake_llm([]),
            embedder=fake_embedder,
            pool=None,
            settings=settings.model_copy(
                update={"eval_mode": eval_mode, "verify_mode": verify_mode}
            ),
            thresholds=thresholds,
            resolver=FakeResolver({}),
            verifier=FakeVerifier([[]]) if verifier else None,
        )

    return _make


def test_verify_mode_off_switches_the_verifier_off_in_eval_mode_only(make):
    dev = make(eval_mode=True, verify_mode="off")
    assert effective_flags(Flags(), dev).verifier is False
    # Production: the setting is ignored, the request cannot turn verification off either.
    prod = make(eval_mode=False, verify_mode="off")
    assert effective_flags(Flags(), prod).verifier is True
    assert effective_flags(Flags(verifier=False), prod).verifier is True


@pytest.mark.parametrize("mode", ["sequential", "stream"])
def test_the_other_modes_leave_the_flag_alone(make, mode):
    deps = make(eval_mode=True, verify_mode=mode)
    assert effective_flags(Flags(), deps).verifier is True
    assert effective_flags(Flags(verifier=False), deps).verifier is False


def test_the_mode_a_request_runs_in(make):
    state = {"flags": Flags(verifier=True)}
    assert request_verify_mode(state, make(eval_mode=True, verify_mode="stream")) == "stream"
    assert request_verify_mode(state, make(eval_mode=True, verify_mode="sequential")) == (
        "sequential"
    )
    assert request_verify_mode(state, make(eval_mode=True, verify_mode="off")) == "off"
    assert (
        request_verify_mode(
            {"flags": Flags(verifier=False)}, make(eval_mode=True, verify_mode="stream")
        )
        == "off"
    )
    # No verifier in this build: the writer runs alone, and the flag was forced off by run_ask.
    assert request_verify_mode(
        state, make(eval_mode=True, verify_mode="stream", verifier=False)
    ) == ("off")
    prod = make(eval_mode=False, verify_mode="off")
    assert request_verify_mode(state, prod) == "stream"
