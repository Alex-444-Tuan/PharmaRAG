"""PostgreSQL connection helpers. Contract file: owned by the orchestrator.

Runtime code uses psycopg 3 with raw SQL. The schema lives in migrations/ (spec section 5).
"""

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)


async def create_pool(
    database_url: str, *, min_size: int = 1, max_size: int = 10
) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        database_url, min_size=min_size, max_size=max_size, configure=_configure, open=False
    )
    await pool.open()
    return pool


async def connect(database_url: str) -> psycopg.AsyncConnection:
    conn = await psycopg.AsyncConnection.connect(database_url)
    await _configure(conn)
    return conn
