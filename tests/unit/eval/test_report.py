"""Report provenance: the thresholds commit and its dirty flag (spec section 11)."""

import subprocess
from datetime import UTC, datetime

import pytest

from pharmarag.eval import lookalike as la
from pharmarag.eval.report import base_meta, thresholds_provenance, write_report


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "thresholds.yaml").write_text("resolver: {}\n")
    (tmp_path / "other.txt").write_text("a\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "thresholds v1")
    return tmp_path


def head(repo) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_clean_thresholds_report_their_commit(repo):
    p = thresholds_provenance(repo)
    assert p["thresholds_commit"] == head(repo)
    assert p["thresholds_dirty"] is False


def test_uncommitted_threshold_edits_are_flagged_dirty(repo):
    (repo / "config" / "thresholds.yaml").write_text("resolver: {margin: 0.1}\n")
    assert thresholds_provenance(repo)["thresholds_dirty"] is True


def test_the_commit_is_the_one_that_last_touched_the_thresholds_not_head(repo):
    first = head(repo)
    (repo / "other.txt").write_text("b\n")
    git(repo, "commit", "-q", "-am", "unrelated")
    p = thresholds_provenance(repo)
    assert head(repo) != first
    assert p["thresholds_commit"] == first  # section 11: must predate the test run
    assert p["git_head"] == head(repo)
    (repo / "config" / "thresholds.yaml").write_text("resolver: {margin: 0.1}\n")
    git(repo, "commit", "-q", "-am", "retune")
    assert thresholds_provenance(repo)["thresholds_commit"] == head(repo)


def test_unrelated_dirty_files_do_not_mark_the_thresholds_dirty(repo):
    (repo / "other.txt").write_text("changed\n")
    assert thresholds_provenance(repo)["thresholds_dirty"] is False


def test_outside_a_git_repo_the_fields_are_unknown_not_an_error(tmp_path):
    p = thresholds_provenance(tmp_path)
    assert p["thresholds_commit"] is None
    assert p["thresholds_dirty"] is None


def test_base_meta_records_everything_section_11_asks_for(thresholds):
    now = datetime(2026, 10, 6, 12, 30, 5, tzinfo=UTC)
    meta = base_meta(
        eval_name="lookalike",
        split="dev",
        mode="both",
        runs=3,
        model_id="claude-sonnet-5-5",
        corpus="abc123",
        thresholds=thresholds,
        dataset=la.dataset_path("dev"),
        now=now,
    )
    assert meta["model_id"] == "claude-sonnet-5-5"
    assert meta["corpus_version"] == "abc123"
    assert meta["runs"] == 3
    assert meta["timestamp"] == "2026-10-06T12:30:05Z"
    assert "thresholds_commit" in meta and "thresholds_dirty" in meta
    assert meta["thresholds"]["resolver"]["resolved_min"] == 0.90
    assert len(meta["dataset"]["sha256"]) == 64


def test_report_files_are_named_by_split_and_timestamp(tmp_path):
    report = {"eval": "lookalike", "split": "test", "timestamp": "2026-10-06T12:30:05Z"}
    json_path, md_path = write_report(report, tmp_path / "reports", "# hi\n")
    assert json_path.name == "lookalike_test_20261006T123005Z.json"
    assert md_path.name == "lookalike_test_20261006T123005Z.md"
    assert md_path.read_text() == "# hi\n"
