import pytest
from langgraph.types import RetryPolicy

from pharmarag.app.settings import Thresholds
from pharmarag.db import create_pool
from pharmarag.graph.build import build_graph
from pharmarag.graph.deps import Deps
from pharmarag.llm.errors import LLMUnavailableError
from tests.graph.seed import HYDRALAZINE, HYDROXYZINE, seed_label, set_corpus_version


@pytest.fixture
async def pool(pg_url):
    p = await create_pool(pg_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
def graph_thresholds(thresholds) -> Thresholds:
    """The frozen thresholds, with the grading values the route tests depend on pinned.

    The yaml values are tuned on dev sets and move. These tests use word-overlap fake
    embeddings, so they pin the grader to values their fixtures were written for.
    """
    retrieval = thresholds.retrieval.model_copy(update={"dense_min": 0.5, "overlap_min": 2})
    grounding = thresholds.grounding.model_copy(update={"support_ratio_min": 0.8})
    return thresholds.model_copy(update={"retrieval": retrieval, "grounding": grounding})


@pytest.fixture
def make_deps(pool, settings, graph_thresholds, fake_embedder):
    def _make(llm, **kw) -> Deps:
        return Deps(
            llm=llm,
            embedder=fake_embedder,
            pool=pool,
            settings=settings,
            thresholds=graph_thresholds,
            **kw,
        )

    return _make


FAST_RETRY = RetryPolicy(
    initial_interval=0.0, jitter=False, max_attempts=3, retry_on=LLMUnavailableError
)


@pytest.fixture
def make_graph():
    """build_graph with the production retry policy's shape but no backoff sleep."""

    def _make(deps: Deps):
        return build_graph(deps, llm_retry=FAST_RETRY)

    return _make


@pytest.fixture
async def corpus(pg_conn, fake_embedder):
    """Two look-alike drugs, committed so pooled connections can see them."""
    ids = {
        "hydroxyzine": await seed_label(
            pg_conn, fake_embedder, key="1", name="hydroxyzine", chunks=HYDROXYZINE
        ),
        "hydralazine": await seed_label(
            pg_conn, fake_embedder, key="2", name="hydralazine", chunks=HYDRALAZINE
        ),
    }
    await set_corpus_version(pg_conn, "corpus-abc123")
    await pg_conn.commit()
    return ids
