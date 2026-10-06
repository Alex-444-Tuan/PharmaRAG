import pytest

from pharmarag.db import create_pool
from tests.unit.resolver.fakes import FakeRx, seed


@pytest.fixture
async def pool(pg_url, pg_conn):
    """A pool on the throwaway database, with the small alias table of fakes.py committed.

    Depends on `pg_conn` so the tables are truncated after the test.
    """
    await seed(pg_conn)
    p = await create_pool(pg_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
def rx() -> FakeRx:
    return FakeRx()
