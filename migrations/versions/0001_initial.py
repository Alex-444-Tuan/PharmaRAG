"""Initial schema (spec section 5).

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

UPGRADE = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE ingredient (
  rxcui TEXT PRIMARY KEY,
  name  TEXT NOT NULL
);

CREATE TABLE drug_alias (
  alias          TEXT NOT NULL,
  alias_tty      TEXT NOT NULL,
  source_rxcui   TEXT NOT NULL,
  ingredient_key TEXT NOT NULL,
  PRIMARY KEY (alias, source_rxcui)
);
CREATE INDEX drug_alias_trgm ON drug_alias USING gin (alias gin_trgm_ops);
CREATE INDEX drug_alias_key  ON drug_alias (ingredient_key);

CREATE TABLE label (
  setid              UUID NOT NULL,
  spl_version        INT  NOT NULL,
  title              TEXT NOT NULL,
  labeler            TEXT,
  marketing_category TEXT,
  effective_date     DATE,
  ingredient_key     TEXT NOT NULL,
  ingredient_rxcuis  TEXT[] NOT NULL,
  product_rxcuis     TEXT[] NOT NULL,
  PRIMARY KEY (setid, spl_version)
);
CREATE INDEX label_ingredient_key ON label (ingredient_key);

CREATE TABLE chunk (
  id               BIGSERIAL PRIMARY KEY,
  setid            UUID NOT NULL,
  spl_version      INT  NOT NULL,
  ingredient_key   TEXT NOT NULL,
  section_loinc    TEXT NOT NULL,
  section_title    TEXT NOT NULL,
  subsection_title TEXT,
  ordinal          INT  NOT NULL,
  text             TEXT NOT NULL,
  spans            JSONB NOT NULL,
  tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  embedding        vector(384) NOT NULL,
  FOREIGN KEY (setid, spl_version) REFERENCES label ON DELETE CASCADE
);
CREATE INDEX chunk_key_section ON chunk (ingredient_key, section_loinc);
CREATE INDEX chunk_tsv         ON chunk USING gin (tsv);
CREATE INDEX chunk_hnsw        ON chunk USING hnsw (embedding vector_cosine_ops);

CREATE TABLE corpus_meta (
  id             INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  corpus_version TEXT NOT NULL,
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

DOWNGRADE = """
DROP TABLE IF EXISTS corpus_meta;
DROP TABLE IF EXISTS chunk;
DROP TABLE IF EXISTS label;
DROP TABLE IF EXISTS drug_alias;
DROP TABLE IF EXISTS ingredient;
"""


def upgrade() -> None:
    op.execute(UPGRADE)


def downgrade() -> None:
    op.execute(DOWNGRADE)
