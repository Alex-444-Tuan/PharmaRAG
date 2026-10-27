# PharmaRAG

PharmaRAG answers questions about prescription drugs from FDA DailyMed labels. It cites a label span for every factual sentence, and it never ranks text from a drug the user did not name.

It is a FastAPI service. A LangGraph state machine runs over PostgreSQL with pgvector, Redis caches and rate-limits, and two local SentenceTransformers models do the embedding and the grounding check (an NLI cross-encoder). A hosted LLM writes the answer.

Three properties are enforced in code and measured by scripts in this repository:

1. **No look-alike mix-ups.** Every drug mention is resolved to an RxNorm ingredient key before retrieval. Retrieval is restricted to that key in SQL, before any similarity ranking, so a question about hydroxyzine cannot retrieve hydralazine text (FR-2, FR-3).
2. **No unsupported sentences.** Each sentence the LLM writes is checked against the retrieved label spans by an NLI model and a number check. A sentence that is not supported never reaches the user, including over server-sent events (FR-6).
3. **No added latency from the check.** The verifier runs while the LLM is still writing the next sentence (AT-3).

**This is not medical advice.** The service reports what an FDA label says. It does not diagnose, recommend doses for a patient, or replace a pharmacist or physician. Patient-specific requests are declined. Every response carries the notice "From FDA drug labels via DailyMed. Not medical advice."

## Architecture

```mermaid
flowchart LR
    client["Client<br/>POST /v1/ask"]
    evalh["Eval harness<br/>scripted evals"]

    subgraph api["FastAPI service"]
        graph["LangGraph state machine<br/>agent nodes, parallel branches, retry edges"]
        emb["Embedding model<br/>SentenceTransformers"]
        nli["NLI cross-encoder<br/>grounding verifier"]
    end

    llm["LLM API<br/>analyze, write"]
    rxnav["RxNav REST API<br/>name to RxCUI"]
    pg[("Postgres + pgvector<br/>labels, chunks, vectors")]
    redis[("Redis<br/>cache and rate limits")]
    ingest["Ingestion CLI<br/>parse, chunk, embed"]
    dm["DailyMed<br/>label XML, mappings"]

    client --> graph
    evalh --> graph
    graph --> llm
    graph --> rxnav
    graph --> pg
    graph --> redis
    ingest --> pg
    ingest --> dm
    ingest --> rxnav
```

Arrows point from caller to callee. The full design is in [PharmaRAG-Project-Specification.md](PharmaRAG-Project-Specification.md); requirement IDs (FR-x, NFR-x, AT-x) in this file refer to it.

## Quickstart

### With Docker Compose

Requirements: Docker with Compose v2. The image bakes in both models, so the first build downloads about 700 MB of weights and the container starts with no network access to Hugging Face.

```bash
# 1. Settings. The LLM model ID must be a pinned ID, not an alias.
cat > .env <<'EOF'
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-5-5
LLM_API_KEY=<your key>
EOF

# 2. Start the API, PostgreSQL (pgvector 0.8) and Redis. The api container applies the
#    database migrations before it starts.
docker compose up -d --build
curl http://localhost:8000/readyz        # 200 when the database, Redis and both models are ready

# 3. Load the label corpus (calls DailyMed and RxNav; takes several minutes).
docker compose --profile ingest run --rm ingest
```

Host ports can be changed with `API_PORT`, `POSTGRES_PORT` and `REDIS_PORT` if 8000, 5432 or 6379 are taken. `LLM_PROVIDER=fake` starts the stack without an API key, but answers nothing; CI uses it to check that the stack comes up.

The Swagger page is at `http://localhost:8000/docs`.

### Local, with uv

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), a PostgreSQL 16+ with pgvector and `pg_trgm`, and Redis (or `CACHE_BACKEND=memory`).

```bash
uv sync
docker compose up -d postgres redis                      # or your own Postgres and Redis
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pharmarag
uv run alembic upgrade head
uv run python -m pharmarag.ingest.run --seed data/seed_ingredients.csv --limit 10   # smoke ingest
uv run uvicorn pharmarag.app.main:app --port 8000
```

Tests: `uv run pytest -q` (tests that need PostgreSQL skip themselves when it is unreachable; `corpus` tests need the ingested dev corpus and skip without it). Lint: `uv run ruff check . && uv run ruff format --check .`.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL`, `REDIS_URL` | local URLs (Compose sets the service URLs) | Connections |
| `CACHE_BACKEND` | `redis` | `redis`, or `memory` for one process with no Redis |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` | `anthropic`, none, none | Hosted LLM. The model ID is pinned, not an alias. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Bi-encoder |
| `NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | Verifier |
| `VERIFY_MODE` | `stream` | `off`, `sequential` or `stream`. `off` is honoured only in `EVAL_MODE`. |
| `RXNAV_BASE_URL` | `https://rxnav.nlm.nih.gov/REST` | Point at RxNav-in-a-Box to remove the external dependency |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per client IP, `/v1/*` only |
| `ANSWER_CACHE` | `true` | Cache `answered` JSON responses for 24 h |
| `TRUST_FORWARDED_FOR` | `false` | Key the rate limit on the rightmost `X-Forwarded-For` entry. Turn on only behind a proxy that sets it. |
| `EVAL_MODE` | `false` | Allows the `X-Eval-Flags` header and the baselines |
| `LOG_QUESTIONS` | `false` | When false, logs hold a hash of the question only |

## API

| Method and path | Purpose |
| --- | --- |
| `POST /v1/ask` | Answer a question |
| `GET /v1/drugs/resolve?name=` | Run the resolver alone |
| `GET /v1/labels/{setid}` | Label metadata and section list |
| `GET /healthz` | The process is up |
| `GET /readyz` | Database, Redis (when it is the cache) and both models are ready. `503` with the failing dependencies named. |
| `GET /metrics` | Prometheus metrics |

### Ask a question

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question": "What is the max daily dose of hydroxyzin for anxiety?"}'
```

```json
{
  "status": "answered",
  "drugs": [
    {"mention": "hydroxyzin", "name": "hydroxyzine", "rxcui": "<IN RxCUI>",
     "ingredient_key": "<key>", "match": "approximate"}
  ],
  "answer": [
    {"text": "<sentence>", "citations": ["c2"], "support": {"span": "c2:4", "score": 0.93}}
  ],
  "sources": [
    {"id": "c2", "setid": "<uuid>", "spl_version": 12, "section": "Dosage and administration",
     "effective_date": "<date>",
     "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=<uuid>"}
  ],
  "notice": "From FDA drug labels via DailyMed. Not medical advice.",
  "meta": {"corpus_version": "<hash>", "latency_ms": 0, "retrieval_attempts": 1,
           "generation_attempts": 1, "removed_sentences": 0, "cache_hit": false}
}
```

The resolved drug name is listed first in every response, because a valid name for the wrong drug cannot be detected (spec section 7, "Known limit").

### Statuses

Every outcome other than a fault is HTTP 200 with a `status` the client switches on.

| `status` | Meaning | Client action |
| --- | --- | --- |
| `answered` | At least one supported sentence | Show the answer and sources |
| `needs_clarification` | A name matched more than one drug. `candidates` lists up to 3 as `{ingredient_key, name, score}`; `ambiguous_mention` names the mention. | Resend with `pinned_rxcuis` set to the chosen `ingredient_key` (a combination is `a+b`) |
| `drug_not_recognised` | No candidate scored high enough | Ask the user to retype |
| `not_covered` | The drug resolved, but no label is in the corpus | Say so |
| `insufficient_evidence` | Retrieval or grounding failed after the retries | Say the label does not answer this |
| `declined` | Patient-specific advice or out of scope | Show the notice |

Faults: `422` invalid request, `429` over the rate limit (with `Retry-After`), `503` a dependency is down.

### Clarification

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question": "What is the dose of hydroxazine?", "pinned_rxcuis": ["<ingredient_key from candidates>"]}'
```

### Streaming

With `"stream": true` the response is server-sent events: `drugs`, then one `sentence` event per verified sentence, then `done` with the full response. `done` is authoritative. If a fault happens after the first byte, the last event is `error` and no `done` follows. A client should treat a stream without `done` as failed.

### Operations

- **Request IDs.** Every response has an `X-Request-ID` header. A client-supplied one (letters, digits, `.`, `_`, `-`, up to 64 characters) is echoed; otherwise one is generated.
- **Rate limit.** Fixed one-minute window per client IP on `/v1/*`; `/healthz`, `/readyz`, `/metrics` and `/docs` are exempt. Over the limit: `429` with `Retry-After`. By default the client IP is the socket peer; `X-Forwarded-For` is ignored unless `TRUST_FORWARDED_FOR=true`, because otherwise a client could pick its own bucket.
- **Answer cache.** Only `answered` JSON responses are cached, keyed on `corpus_version`, the effective flags, the question and the pinned RxCUIs, so a label update invalidates old answers. `meta.cache_hit` says which one you got. SSE and `debug` requests bypass it.
- **Logs.** One JSON line per graph node and per request, with `request_id`, `node`, `duration_ms`, `status`, `retrieval_attempts`, `generation_attempts`, `removed_sentences` and `cache_hit`. The question appears only as `question_hash`, unless `LOG_QUESTIONS=true`.
- **Metrics.** `pharmarag_http_requests_total{path,code}`, `pharmarag_ask_total{status}`, `pharmarag_node_duration_seconds{node}` (histogram), `pharmarag_cache_requests_total{cache,result}` (`answer` and `query_embedding`), `pharmarag_rxnav_errors_total` and `pharmarag_rate_limited_total`.
- **Degradation.** If RxNav is down the resolver falls back to the local alias table, so exact-name queries still work (NFR-4). If Redis is down the caches act as misses and the rate limiter fails open; `/readyz` reports it.

## Evaluation

Three scripted evaluations compare a switched-off baseline with the full system on a frozen test set (spec section 11). Each writes a JSON report and a Markdown summary to [`reports/`](reports/), recording the pinned model ID, the `corpus_version`, the git commit of `config/thresholds.yaml` and the number of runs. Thresholds are tuned on dev sets only and frozen by commit before a test-set run.

```bash
# Look-alike misinterpretation (AT-1). Needs the ingested corpus and an LLM key.
uv run python -m pharmarag.eval.run lookalike --split test --mode both --runs 5
uv run python -m pharmarag.eval.run lookalike --split test --mode both --runs 3 --resolver-only   # no LLM

# Hallucinated claims (AT-2, AT-2b). Scoring is deterministic and needs no LLM.
uv run python -m pharmarag.eval.run grounding --help

# Latency (AT-3), replay mode: recorded LLM timings, real verifier.
uv run python -m pharmarag.eval.latency --replay --verifier real

# In Docker, with the stack up. Reports land in ./reports on the host.
docker compose --profile eval run --rm eval lookalike --split test --mode both --runs 3 --resolver-only
```

Run the evals from a checkout with `uv run` when the report must record the thresholds commit: the Docker image has no `.git`, so those fields are empty there.

### Measured results

Every number below is copied from a committed report. The caveats are part of the result.

**AT-1, look-alike misinterpretation** (target: at least 85% fewer). Report: [`reports/lookalike_test_20261008T161806Z.md`](reports/lookalike_test_20261008T161806Z.md). Full graph, test split of 15 queries (8 misspellings, 4 brand look-alikes, 3 correctly spelled less-common partners), `claude-sonnet-5-5`, 5 runs, corpus `24bdf02ae716c922`, thresholds commit `980367d8`.

| | Misinterpreted per run (of 15) |
| --- | --- |
| Baseline (resolver and filter off) | 2, 1, 1, 2, 2 |
| System | 0, 0, 0, 0, 0 |

The reduction is 100% on the mean counts (1.6 falling to 0). In every run the system also returned `needs_clarification` for 3 queries; these are counted separately and are not misinterpretations, because the system did not act on a wrong drug. Caveat: the baseline fails on only 1 or 2 of the 15 queries, so one query moves the percentage by a large step, and "85%" cannot be read off these counts. A resolver-only report with a retrieval proxy ([`reports/lookalike_test_20261006T193004Z.md`](reports/lookalike_test_20261006T193004Z.md)) shows the top unfiltered evidence chunk belonging to another drug for 7 of 15 queries; that is a proxy, not the AT-1 number, because no answer is written.

**AT-2 and AT-2b, hallucinated claims** (targets: at least 73% fewer; at most 10% of supported claims wrongly rejected). Report: [`reports/grounding_test_20261023T213941Z.md`](reports/grounding_test_20261023T213941Z.md). 40 labelled claims: 30 supported, 10 hallucinated.

| Metric | Result | Counts |
| --- | --- | --- |
| AT-2 reduction | 90.0% | 10 hallucinated claims with the verifier off, 1 passed with it on |
| AT-2b false-reject rate | 10.0% | 3 of 30 supported claims rejected (at the limit) |

Caveats. Only 1 of the 10 hallucinated claims is a natural hallucination; the other 9 are perturbed (changed dose, negation, a fact from another drug), because the top-up rule fired. The labels were drafted by an LLM judge (Claude Opus 5.5) and reviewed by a pharmacist, who approved all 40 unchanged. The AT-2b rate sits exactly on its limit, with 3 rejections in 30. The verifier is deterministic, so the report is one run; the claim set is small.

**AT-3, no added latency** (target: stream p50 within 5% and p95 within 10% of verifier off). Report: [`reports/latency_replay_real_20261016T002648Z.md`](reports/latency_replay_real_20261016T002648Z.md). Replay mode: recorded `claude-sonnet-5-5` token timings, the real NLI verifier, 30 questions x 5 repetitions x 3 modes with the first repetition discarded, 0 errors. The headline set is the 80 paired requests (20 questions) that need no regeneration.

| Mode | p50 | p95 |
| --- | --- | --- |
| Verifier off | 4150 ms | 6435 ms |
| Sequential | 4626 ms | 7550 ms |
| Stream | 4306 ms | 6636 ms |

Stream is **+3.8% at p50 and +3.1% at p95** relative to off, within both limits. Caveat: the bootstrap 95% interval for the p50 difference is +1.7% to **+5.2%**, so its upper end is above the 5% limit and the data cannot rule out a failure at that percentile. Sequential verification costs 476 ms at p50, which is what the in-stream design avoids. Replay isolates verifier overhead from network noise; it uses the local alias table for the resolver and no RxNorm facts, and it is not a live-LLM measurement.

## Data and attribution

Label text comes from [DailyMed](https://dailymed.nlm.nih.gov/), the FDA's label repository at the U.S. National Library of Medicine. Drug names are resolved with the RxNav / RxNorm API.

> This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.

The same statement is on the Swagger page.

## Repository layout

```text
pharmarag/
  app/        FastAPI app, middleware, runtime wiring, logs and metrics, settings
  graph/      state, build, routing, nodes/ (one file per node)
  resolver/   cascade, RxNav client, decision gate
  retrieval/  dense, lexical, fusion
  grounding/  writer, verifier, numeric check, stream
  caching/    rate limiter, answer cache, query-embedding cache
  ingest/     DailyMed client, SPL parser, chunker, loader
  eval/       the three evaluations and their datasets
config/       frozen thresholds
migrations/   Alembic
reports/      committed evaluation reports
```
