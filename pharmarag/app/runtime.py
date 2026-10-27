"""Process-wide resources, built once at startup (spec section 3: models load at startup)."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pharmarag.app.observability import (
    Metrics,
    MissProbeEmbedder,
    attach_rxnav_error_counter,
    detach_rxnav_error_counter,
)
from pharmarag.app.settings import Settings, load_thresholds
from pharmarag.cache import Cache, InMemoryCache, RedisCache
from pharmarag.caching.factory import (
    build_answer_cache,
    build_query_embedding_cache,
    build_rate_limiter,
)
from pharmarag.db import create_pool
from pharmarag.graph.build import build_graph
from pharmarag.graph.deps import Deps
from pharmarag.grounding.base import Verifier
from pharmarag.grounding.factory import build_verifier
from pharmarag.llm.base import LLMClient
from pharmarag.llm.factory import get_llm
from pharmarag.models.embedder import Embedder, SentenceTransformerEmbedder
from pharmarag.ops import AnswerCache, QueryEmbeddingCache, RateLimiter
from pharmarag.resolver.base import Resolver, RxFactsSource
from pharmarag.resolver.factory import build_resolver, build_rx_facts_source, close_rxnav

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    deps: Deps
    graph: Any
    # None: this build has no rate limit / answer cache (EVAL_MODE without the factory, or
    # ANSWER_CACHE=false). The middleware and the endpoint skip the stage.
    rate_limiter: RateLimiter | None = None
    answer_cache: AnswerCache | None = None
    # Set when build_runtime made the Redis client itself, so close_runtime closes it.
    owns_cache: bool = False
    rxnav_error_handler: logging.Handler | None = None


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


def make_cache(settings: Settings) -> tuple[Cache, bool]:
    """(cache, owned). Redis unless CACHE_BACKEND=memory. `RedisCache` connects lazily and is
    fail-soft, so a Redis that is down at startup slows requests down and shows up in
    `/readyz`, but does not stop the process from starting."""
    if settings.cache_backend == "redis":
        return RedisCache(settings.redis_url), True
    return InMemoryCache(), False


async def build_runtime(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    cache: Cache | None = None,
    resolver: Resolver | None | _Build = BUILD,
    rx_facts: RxFactsSource | None | _Build = BUILD,
    verifier: Verifier | None | _Build = BUILD,
    rate_limiter: RateLimiter | None | _Build = BUILD,
    answer_cache: AnswerCache | None | _Build = BUILD,
    query_embeddings: QueryEmbeddingCache | None | _Build = BUILD,
    metrics: Metrics | None = None,
) -> Runtime:
    """Load the embedder, open the DB pool, create the LLM client, compile the graph.

    The injectable arguments let tests and the eval harness skip the model download and the
    network. `resolver` and `rx_facts` default to the real ones from
    `pharmarag.resolver.factory`, and the verifier the real one from
    `pharmarag.grounding.factory`; pass a fake, or None for "no such stage".

    Outside EVAL_MODE a build with no resolver does not start (RuntimeError). Without a resolver
    the request flag is forced off and every query runs unfiltered over the whole corpus (FR-3),
    which is the AT-1 baseline and must never be what production serves. In eval mode a missing
    resolver is allowed, so the baseline can be measured.

    The verifier follows the same rule (FR-6). Outside EVAL_MODE a build with no verifier does
    not start, and neither does VERIFY_MODE=off: answers would reach users unchecked. In eval
    mode a missing verifier is the AT-2 baseline.

    The cache comes from CACHE_BACKEND (redis or memory) unless one is passed in. The rate
    limiter, the answer cache (only when ANSWER_CACHE=true) and the query-embedding cache come
    from `pharmarag.caching.factory`. A factory that raises NotImplementedError is tolerated only
    in EVAL_MODE, like the other stages, so a production build never silently runs without a rate
    limit.
    """
    if embedder is None:
        embedder = await asyncio.to_thread(
            SentenceTransformerEmbedder,
            settings.embedding_model,
            settings.embedding_dim,
            settings.embedding_model_revision,  # a pinned commit, so the weights cannot drift
        )
    # The first inference pays one-off setup costs (weights paging in, thread pools).
    # Doing it here keeps that off the first user's latency.
    await asyncio.to_thread(embedder.embed_query, "warm-up")
    owns_cache = False
    if cache is None:
        cache, owns_cache = make_cache(settings)
    metrics = metrics or Metrics()
    pool = await create_pool(settings.database_url)
    try:
        thresholds = load_thresholds(settings.thresholds_path)
        if isinstance(resolver, _Build):
            resolver = _from_factory(
                lambda: build_resolver(settings, thresholds, pool, cache), "resolver", settings
            )
        if resolver is None and not settings.eval_mode:
            raise RuntimeError(
                "no drug resolver: refusing to start, because requests would run unfiltered "
                "(FR-3). Set EVAL_MODE=true to run the resolver-off baseline on purpose."
            )
        if isinstance(verifier, _Build):
            # Loads the NLI model and runs a warm-up inference: off the event loop.
            verifier = await asyncio.to_thread(
                _from_factory,
                lambda: build_verifier(settings, thresholds, embedder),
                "verifier",
                settings,
            )
        if not settings.eval_mode and (verifier is None or settings.verify_mode == "off"):
            raise RuntimeError(
                "no grounding verifier: refusing to start, because answers would reach users "
                "unchecked (FR-6). Set EVAL_MODE=true to run the verifier-off baseline on "
                "purpose."
            )
        if isinstance(rx_facts, _Build):
            rx_facts = _from_factory(
                lambda: build_rx_facts_source(settings, thresholds, cache),
                "RxNorm facts source",
                settings,
            )
        if isinstance(rate_limiter, _Build):
            rate_limiter = _from_factory(
                lambda: build_rate_limiter(settings, cache), "rate limiter", settings
            )
        if isinstance(answer_cache, _Build):
            answer_cache = (
                _from_factory(lambda: build_answer_cache(settings, cache), "answer cache", settings)
                if settings.answer_cache
                else None
            )
        if isinstance(query_embeddings, _Build):
            # The wrapper only marks hit vs miss for the metric; the embedder is unchanged.
            query_embeddings = _from_factory(
                lambda: build_query_embedding_cache(MissProbeEmbedder(embedder), cache),
                "query-embedding cache",
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
            query_embeddings=query_embeddings,
            metrics=metrics,
        )
        graph = build_graph(deps)
    except BaseException:
        await pool.close()
        if owns_cache:
            await cache.aclose()  # type: ignore[attr-defined]
        raise
    return Runtime(
        deps=deps,
        graph=graph,
        rate_limiter=rate_limiter,
        answer_cache=answer_cache,
        owns_cache=owns_cache,
        rxnav_error_handler=attach_rxnav_error_counter(metrics),
    )


async def close_runtime(runtime: Runtime) -> None:
    if runtime.rxnav_error_handler is not None:
        detach_rxnav_error_counter(runtime.rxnav_error_handler)
    try:
        await runtime.deps.pool.close()
    finally:
        try:
            if runtime.owns_cache:
                await runtime.deps.cache.aclose()  # type: ignore[attr-defined]
        finally:
            # Releases the shared RxNav HTTP client. A no-op when none was built.
            await close_rxnav(runtime.deps.settings)
