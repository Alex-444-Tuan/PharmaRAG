#!/usr/bin/env bash
# Restore the CI corpus fixture into an EMPTY database: migrate the schema, then load the data.
#
#   scripts/load_ci_fixture.sh postgresql://postgres:postgres@localhost:5432/pharmarag_ci_x
#
# The argument (or DATABASE_URL) names a database that already exists and has no tables. The
# fixture is data only; the schema comes from `alembic upgrade head`, so a migration that the
# fixture no longer fits fails here, loudly, instead of hiding in a stale dump.
# Needs: psql, gunzip, uv.  Written by scripts/export_ci_fixture.py.
set -euo pipefail

URL="${1:-${DATABASE_URL:-}}"
if [[ -z "$URL" ]]; then
  echo "usage: $0 <database-url>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="${CI_FIXTURE:-$ROOT/data/ci_fixture.sql.gz}"

# Refuse a database that already has the schema's tables: loading twice would double the rows
# (or fail half-way on the primary keys) and leave a corpus nobody can describe.
EXISTING="$(psql "$URL" -v ON_ERROR_STOP=1 -Atc \
  "select count(*) from information_schema.tables where table_schema = 'public' and table_name <> 'alembic_version'")"
if [[ "$EXISTING" != "0" ]]; then
  echo "refusing: $EXISTING tables already exist in the target database; use a fresh one" >&2
  exit 1
fi

cd "$ROOT"
DATABASE_URL="$URL" uv run alembic upgrade head
# The dump brackets itself in BEGIN/COMMIT; ON_ERROR_STOP aborts at the first error and the open
# transaction rolls back when psql exits, so it is all of the data or none of it.
gunzip -c "$FIXTURE" | psql "$URL" -v ON_ERROR_STOP=1 -q -f - >/dev/null
psql "$URL" -Atc \
  "select 'loaded: ' || (select count(*) from label) || ' labels, ' || (select count(*) from chunk) || ' chunks, ' || (select count(*) from drug_alias) || ' aliases, corpus_version ' || (select corpus_version from corpus_meta)"
