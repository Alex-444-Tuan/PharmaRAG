"""Report metadata and writers shared by the evals (spec section 11, "Rules for all three").

Every report records the pinned model ID, the corpus_version, the git commit of
config/thresholds.yaml (and whether the file has uncommitted changes), the run count and the
timestamp. JSON for machines, Markdown for people.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from psycopg import AsyncConnection

from pharmarag.app.settings import REPO_ROOT, Thresholds

THRESHOLDS_FILE = "config/thresholds.yaml"


def _git(*args: str, repo: Path) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def thresholds_provenance(repo: Path = REPO_ROOT) -> dict:
    """The commit that last changed config/thresholds.yaml, and whether it has local edits.

    Section 11 says this commit must predate any test-set run. `dirty` true means the numbers
    in the report were produced with thresholds that are not committed, so they are not
    reproducible from that commit.
    """
    commit = _git("log", "-1", "--format=%H", "--", THRESHOLDS_FILE, repo=repo)
    status = _git("status", "--porcelain", "--", THRESHOLDS_FILE, repo=repo)
    return {
        "thresholds_commit": commit or None,
        "thresholds_dirty": None if status is None else bool(status),
        "git_head": _git("rev-parse", "HEAD", repo=repo),
    }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def corpus_version(conn: AsyncConnection) -> str:
    cur = await conn.execute("SELECT corpus_version FROM corpus_meta WHERE id = 1")
    row = await cur.fetchone()
    return row[0] if row else "unknown"


def base_meta(
    *,
    eval_name: str,
    split: str,
    mode: str,
    runs: int,
    model_id: str | None,
    corpus: str,
    thresholds: Thresholds,
    dataset: Path,
    repo: Path = REPO_ROOT,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    return {
        "eval": eval_name,
        "split": split,
        "mode": mode,
        "runs": runs,
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": model_id,
        "corpus_version": corpus,
        **thresholds_provenance(repo),
        "thresholds": {
            "resolver": thresholds.resolver.model_dump(),
            "retrieval": thresholds.retrieval.model_dump(),
        },
        "dataset": {"file": dataset.name, "sha256": sha256_of(dataset)},
    }


def write_report(report: dict, out_dir: Path, markdown: str) -> tuple[Path, Path]:
    """Write reports/<eval>_<split>_<timestamp>.json and the .md summary next to it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace("-", "").replace(":", "")
    stem = f"{report['eval']}_{report['split']}_{stamp}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    md_path.write_text(markdown)
    return json_path, md_path
