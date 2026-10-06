import pytest

from pharmarag.db import create_pool
from pharmarag.graph.build import build_graph
from pharmarag.graph.deps import Deps
from tests.graph.seed import HYDRALAZINE, HYDROXYZINE, seed_label, set_corpus_version


@pytest.fixture
async def pool(pg_url):
    p = await create_pool(pg_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
def make_deps(pool, settings, thresholds, fake_embedder):
    def _make(llm, **kw) -> Deps:
        return Deps(
            llm=llm,
            embedder=fake_embedder,
            pool=pool,
            settings=settings,
            thresholds=thresholds,
            **kw,
        )

    return _make


@pytest.fixture
def make_graph():
    return build_graph


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
