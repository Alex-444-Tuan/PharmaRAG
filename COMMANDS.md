# Commands

Local dev cheat sheet. Entries marked *(planned)* come from the spec and will be confirmed once the code exists.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # (planned)
pip install -e ".[dev]"                                   # (planned)
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
pytest                       # everything
pytest tests/unit -q
pytest tests/graph -q        # route tests with FakeLLM
pytest tests/sql -q          # filter-leak and EXPLAIN plan tests (needs postgres)
ruff check . && ruff format --check .
```

## Ingest and eval

```bash
python -m ingest.run --seed data/seed_ingredients.csv     # (planned)
python -m eval.run lookalike                              # (planned) writes reports/
python -m eval.run grounding                              # (planned)
python -m eval.run latency                                # (planned)
```

## Postgres

```bash
docker compose exec postgres psql -U postgres pharmarag
alembic upgrade head
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
curl -s 'localhost:8000/v1/drugs/resolve?name=hydroxyzin' | jq
open http://localhost:8000/docs
```
