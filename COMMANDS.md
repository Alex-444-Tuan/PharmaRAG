# Commands

Local dev cheat sheet. Entries marked *(planned)* come from the spec and will be confirmed once the code exists.

## Setup

```bash
uv sync                                                   # creates .venv with Python 3.12 + dev deps
cp .env.example .env                                      # then set LLM_* vars
```

## Docker

```bash
docker compose up -d                                       # api, postgres, redis
docker compose --profile ingest run --rm ingest            # (planned) load the corpus
docker compose --profile eval run --rm eval lookalike      # (planned) also: grounding, latency
docker compose logs -f api
docker compose down -v                                     # also wipes the postgres volume
```

## Tests

```bash
uv run pytest -q             # everything (~780 tests, ~30 s)
uv run pytest tests/unit -q
uv run pytest tests/graph -q  # route tests with FakeLLM
uv run pytest tests/sql -q    # filter-leak and EXPLAIN plan tests (needs postgres)
uv run pytest -m corpus -q    # tests against the real pharmarag_dev corpus (incl. graph-level leak test)
uv run ruff check . && uv run ruff format --check .
```

## Ingest and eval

```bash
uv run python -m pharmarag.ingest.run --seed data/seed_ingredients.csv             # uses pins, ~3.5 min if unchanged
uv run python -m pharmarag.ingest.run --seed data/seed_ingredients.csv --limit 10  # smoke run
uv run python -m pharmarag.ingest.run --seed data/seed_ingredients.csv --repin     # re-select every label, ~22 min
uv run python -m pharmarag.ingest.run --seed data/seed_ingredients.csv --only paroxetine,sertraline  # re-ingest named seeds
uv run python -m pharmarag.ingest.audit                   # flag odd label picks (special-indication brands, repackagers)

# Evals need DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pharmarag_dev
uv run python -m pharmarag.eval.run lookalike --split test --mode both --runs 3 --resolver-only  # no LLM needed
uv run python -m pharmarag.eval.run lookalike --split test --mode both --runs 5   # full AT-1, needs LLM key
uv run python -m pharmarag.eval.gate_sweep                # 2,192 dev misspellings vs the resolver gate
uv run python scripts/calibrate_evidence_thresholds.py    # dense_min / overlap_min calibration
uv run python -m pharmarag.eval.run grounding             # (planned, cycle 3)
uv run python -m pharmarag.eval.run latency               # (planned, cycle 3)
```

## Postgres

```bash
# Local dev uses the Homebrew Postgres (pgvector installed). Dev DB: pharmarag_dev
psql postgresql://postgres:postgres@localhost:5432/pharmarag_dev
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pharmarag_dev uv run alembic upgrade head
# Tests create and drop their own pharmarag_test_* databases
```

```sql
SELECT count(*) FROM label;
SELECT ingredient_key, count(*) FROM chunk GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
SELECT * FROM drug_alias WHERE alias % 'hydroxyzin' ORDER BY similarity(alias, 'hydroxyzin') DESC;
```

## Redis

```bash
docker compose exec redis redis-cli
KEYS rx:resolve:*
TTL rx:resolve:hydroxyzine
FLUSHDB                      # clear all caches
```

## API

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question":"What is the max daily dose of hydroxyzin for anxiety?"}' | jq
curl -s 'localhost:8000/v1/drugs/resolve?name=hydroxyzin' | jq   # match, score, degraded
curl -s localhost:8000/v1/labels/<setid> | jq
open http://localhost:8000/docs
```
