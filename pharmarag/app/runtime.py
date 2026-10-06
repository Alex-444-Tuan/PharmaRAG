"""Process-wide resources, built once at startup (spec section 3: models load at startup)."""

import asyncio
from dataclasses import dataclass
from typing import Any

from pharmarag.app.settings import Settings, load_thresholds
from pharmarag.db import create_pool
from pharmarag.graph.build import build_graph
from pharmarag.graph.deps import Deps
from pharmarag.llm.base import LLMClient
from pharmarag.llm.factory import get_llm
from pharmarag.models.embedder import Embedder, SentenceTransformerEmbedder


@dataclass
class Runtime:
    deps: Deps
    graph: Any


async def build_runtime(
    settings: Settings, *, embedder: Embedder | None = None, llm: LLMClient | None = None
) -> Runtime:
    """Load the embedder, open the DB pool, create the LLM client, compile the graph.

    `embedder` and `llm` can be injected so tests and the eval harness skip the model download.
    """
    if embedder is None:
        embedder = await asyncio.to_thread(
            SentenceTransformerEmbedder, settings.embedding_model, settings.embedding_dim
        )
    # The first inference pays one-off setup costs (weights paging in, thread pools).
    # Doing it here keeps that off the first user's latency.
    await asyncio.to_thread(embedder.embed_query, "warm-up")
    pool = await create_pool(settings.database_url)
    deps = Deps(
        llm=llm or get_llm(settings),
        embedder=embedder,
        pool=pool,
        settings=settings,
        thresholds=load_thresholds(settings.thresholds_path),
    )
    return Runtime(deps=deps, graph=build_graph(deps))


async def close_runtime(runtime: Runtime) -> None:
    await runtime.deps.pool.close()
