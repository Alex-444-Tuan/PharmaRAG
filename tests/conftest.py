"""Shared fixtures. Contract file: owned by the orchestrator.

Postgres tests get a throwaway database per test session, so two lanes (or two
worktrees) running tests at the same time never collide. They are skipped when
Postgres is unreachable. Start it with `docker compose up -d postgres`.
"""

import os
import uuid

import psycopg
import pytest
from alembic import command
from alembic.config import Config

from pharmarag.app.settings import REPO_ROOT, Settings, Thresholds, load_thresholds
from pharmarag.db import connect
from pharmarag.llm.base import FakeLLM
from pharmarag.models.embedder import FakeEmbedder

ADMIN_URL = os.environ.get(
    "TEST_DATABASE_ADMIN_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)
TABLES = ("chunk", "label", "drug_alias", "ingredient", "corpus_meta")


@pytest.fixture
def settings() -> Settings:
    """Fake LLM, eval mode on. Override with settings.model_copy(update=...)."""
    return Settings(_env_file=None, llm_provider="fake", llm_model="fake-llm", eval_mode=True)


@pytest.fixture(scope="session")
def thresholds() -> Thresholds:
    return load_thresholds(REPO_ROOT / "config" / "thresholds.yaml")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make_fake_llm():
    """Factory: make_fake_llm(["resp1", "resp2"], token_delay_s=0.0)."""
    return FakeLLM


@pytest.fixture(scope="session")
def pg_url() -> str:
    """URL of a fresh, migrated database for this test session."""
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d postgres`")
    name = f"pharmarag_test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    url = ADMIN_URL.rsplit("/", 1)[0] + f"/{name}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    yield url
    admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture
async def pg_conn(pg_url):
    """Async connection to the session database. All tables are truncated after each test."""
    conn = await connect(pg_url)
    yield conn
    await conn.rollback()
    await conn.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    await conn.commit()
    await conn.close()


CORPUS_URL = os.environ.get(
    "CORPUS_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pharmarag_dev"
)


@pytest.fixture(scope="session")
def corpus_url() -> str:
    """URL of the ingested dev corpus (read-only use). Skips if missing or empty."""
    try:
        with psycopg.connect(CORPUS_URL, connect_timeout=2) as conn:
            n = conn.execute("SELECT count(*) FROM label").fetchone()[0]
    except psycopg.Error:
        pytest.skip("dev corpus not available; run the ingest into pharmarag_dev")
    if n == 0:
        pytest.skip("dev corpus is empty; run the ingest into pharmarag_dev")
    return CORPUS_URL


@pytest.fixture
async def corpus_conn(corpus_url):
    """Read-only async connection to the dev corpus. Never write through it."""
    conn = await connect(corpus_url)
    await conn.set_read_only(True)
    yield conn
    await conn.close()
