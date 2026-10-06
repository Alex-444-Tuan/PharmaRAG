"""Process-wide resources, built once at startup (spec section 3: models load at startup)."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pharmarag.app.settings import Settings, load_thresholds
from pharmarag.cache import Cache, InMemoryCache
from pharmarag.db import create_pool
from pharmarag.graph.build import build_graph
from pharmarag.graph.deps import Deps
from pharmarag.grounding.base import Verifier
from pharmarag.llm.base import LLMClient
from pharmarag.llm.factory import get_llm
from pharmarag.models.embedder import Embedder, SentenceTransformerEmbedder
from pharmarag.resolver.base import Resolver, RxFactsSource
from pharmarag.resolver.factory import build_resolver, build_rx_facts_source, close_rxnav

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    deps: Deps
    graph: Any


class _Build:
    """Sentinel: "build this stage with its factory" (as opposed to None: "no such stage")."""


BUILD = _Build()


def _from_factory(build, what: str, settings: Settings):
    """Run a stage factory. NotImplementedError means "not built in this version"; it is
    tolerated only in eval mode, where a build without the stage is a legitimate baseline."""
    try:
        return build()
    except NotImplementedError:
        if not settings.eval_mode:
            raise
        logger.warning("%s is not implemented in this build; running without it", what)
        return None


async def build_runtime(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    cache: Cache | None = None,
    resolver: Resolver | None | _Build = BUILD,
    rx_facts: RxFactsSource | None | _Build = BUILD,
    verifier: Verifier | None = None,
) -> Runtime:
    """Load the embedder, open the DB pool, create the LLM client, compile the graph.

    The injectable arguments let tests and the eval harness skip the model download and the
    network. `resolver` and `rx_facts` default to the real ones from
    `pharmarag.resolver.factory`; pass a fake, or None for "no such stage". The verifier is None
    until cycle 3.

    Outside EVAL_MODE a build with no resolver does not start (RuntimeError). Without a resolver
    the request flag is forced off and every query runs unfiltered over the whole corpus (FR-3),
    which is the AT-1 baseline and must never be what production serves. In eval mode a missing
    resolver is allowed, so the baseline can be measured.
    """
    if embedder is None:
        embedder = await asyncio.to_thread(
            SentenceTransformerEmbedder, settings.embedding_model, settings.embedding_dim
        )
    # The first inference pays one-off setup costs (weights paging in, thread pools).
    # Doing it here keeps that off the first user's latency.
    await asyncio.to_thread(embedder.embed_query, "warm-up")
    pool = await create_pool(settings.database_url)
    thresholds = load_thresholds(settings.thresholds_path)
    cache = cache or InMemoryCache()  # Redis arrives in cycle 4
    if isinstance(resolver, _Build):
        resolver = _from_factory(
            lambda: build_resolver(settings, thresholds, pool, cache), "resolver", settings
        )
    if resolver is None and not settings.eval_mode:
        await pool.close()
        raise RuntimeError(
            "no drug resolver: refusing to start, because requests would run unfiltered "
            "(FR-3). Set EVAL_MODE=true to run the resolver-off baseline on purpose."
        )
    if isinstance(rx_facts, _Build):
        rx_facts = _from_factory(
            lambda: build_rx_facts_source(settings, thresholds, cache),
            "RxNorm facts source",
            settings,
        )
    deps = Deps(
        llm=llm or get_llm(settings),
        embedder=embedder,
        pool=pool,
        settings=settings,
        thresholds=thresholds,
        resolver=resolver,
        rx_facts=rx_facts,
        verifier=verifier,
        cache=cache,
    )
    return Runtime(deps=deps, graph=build_graph(deps))


async def close_runtime(runtime: Runtime) -> None:
    try:
        await runtime.deps.pool.close()
    finally:
        # Releases the shared RxNav HTTP client. A no-op when none was built.
        await close_rxnav(runtime.deps.settings)
