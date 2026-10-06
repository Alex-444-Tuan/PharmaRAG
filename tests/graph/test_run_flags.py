"""Effective flags: the one place that decides whether a request runs with resolver/verifier."""

import pytest

from pharmarag.graph.deps import Deps
from pharmarag.graph.run import effective_flags
from pharmarag.graph.state import Flags


@pytest.fixture
def make(settings, thresholds, fake_embedder, make_fake_llm):
    def _make(*, eval_mode: bool, resolver: bool, verifier: bool) -> Deps:
        return Deps(
            llm=make_fake_llm([]),
            embedder=fake_embedder,
            pool=None,
            settings=settings.model_copy(update={"eval_mode": eval_mode}),
            thresholds=thresholds,
            resolver_available=resolver,
            verifier_available=verifier,
        )

    return _make


OFF = Flags(resolver=False, verifier=False)


def test_outside_eval_mode_a_request_can_never_run_with_verifier_or_resolver_off(make):
    deps = make(eval_mode=False, resolver=True, verifier=True)
    # Even a caller that bypasses the API header check cannot switch the stages off.
    assert effective_flags(OFF, deps) == Flags(resolver=True, verifier=True)
    assert effective_flags(Flags(verifier=False), deps).verifier is True


def test_in_eval_mode_flags_are_honoured(make):
    deps = make(eval_mode=True, resolver=True, verifier=True)
    assert effective_flags(OFF, deps) == OFF
    assert effective_flags(Flags(), deps) == Flags()
    assert effective_flags(Flags(verifier=False), deps) == Flags(resolver=True, verifier=False)


def test_stage_missing_from_this_build_is_off_whatever_is_requested(make):
    for eval_mode in (True, False):
        deps = make(eval_mode=eval_mode, resolver=False, verifier=False)
        assert effective_flags(Flags(), deps) == OFF
    deps = make(eval_mode=False, resolver=True, verifier=False)
    assert effective_flags(Flags(), deps) == Flags(resolver=True, verifier=False)
