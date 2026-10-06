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
from pharmarag.llm.base import LLMClient
from pharmarag.llm.factory import get_llm
from pharmarag.models.embedder import Embedder, SentenceTransformerEmbedder
from pharmarag.resolver.factory import build_resolver, build_rx_facts_source

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    deps: Deps
    graph: Any


async def build_runtime(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    cache: Cache | None = None,
) -> Runtime:
    """Load the embedder, open the DB pool, create the LLM client, compile the graph.

    `embedder`, `llm` and `cache` can be injected so tests and the eval harness skip the model
    download. The resolver and the RxNorm facts source come from the factories in
    `pharmarag.resolver.factory`. A factory that raises NotImplementedError (Lane A's code is
    not merged yet) leaves that stage None, and the app degrades to the baseline for it: the
    resolver flag is forced off and `meta.resolver` says so. The verifier stays None until
    cycle 3.
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
    try:
        resolver = build_resolver(settings, thresholds, pool, cache)
    except NotImplementedError:
        logger.warning("resolver is not implemented in this build; running without it")
        resolver = None
    try:
        rx_facts = build_rx_facts_source(settings, thresholds, cache)
    except NotImplementedError:
        logger.warning("RxNorm facts source is not implemented in this build; running without it")
        rx_facts = None
    deps = Deps(
        llm=llm or get_llm(settings),
        embedder=embedder,
        pool=pool,
        settings=settings,
        thresholds=thresholds,
        resolver=resolver,
        rx_facts=rx_facts,
        verifier=None,  # cycle 3
        cache=cache,
    )
    return Runtime(deps=deps, graph=build_graph(deps))


async def close_runtime(runtime: Runtime) -> None:
    await runtime.deps.pool.close()
