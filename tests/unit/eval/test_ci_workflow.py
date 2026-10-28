"""The eval workflow (.github/workflows/eval.yml) cannot be run here, so check that it parses, that
it is wired as intended, and that every gate command it runs is accepted by the real argument
parser."""

import shlex

import pytest
import yaml

from pharmarag.app.settings import REPO_ROOT
from pharmarag.eval import run as eval_run

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "eval.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def triggers(workflow) -> dict:
    return workflow[True] if True in workflow else workflow["on"]  # YAML 1.1 reads `on` as True


def test_triggers(workflow):
    on = triggers(workflow)
    assert on["push"]["branches"] == ["main"]
    assert "pull_request" in on and "workflow_dispatch" in on
    assert on["schedule"][0]["cron"]


def test_the_real_verifier_job_never_runs_on_a_pull_request_and_the_gates_job_not_nightly(workflow):
    nightly = workflow["jobs"]["at3-real"]["if"]
    assert "schedule" in nightly and "workflow_dispatch" in nightly
    assert "pull_request" not in nightly and "push" not in nightly
    assert workflow["jobs"]["gates"]["if"] == "github.event_name != 'schedule'"


def test_postgres_service_is_the_pinned_pgvector_image(workflow):
    for job in ("gates", "at3-real"):
        assert workflow["jobs"][job]["services"]["postgres"]["image"] == (
            "pgvector/pgvector:0.8.0-pg17"
        )


def test_every_step_runs_or_uses_something(workflow):
    for job in ("gates", "at3-real"):
        for step in steps(workflow, job):
            assert ("run" in step) != ("uses" in step), step


def test_dependencies_come_from_the_lockfile_and_the_cache_is_in_memory(workflow):
    assert workflow["env"]["CACHE_BACKEND"] == "memory"
    for job in ("gates", "at3-real"):
        runs = [s.get("run", "") for s in steps(workflow, job)]
        assert "uv sync --frozen" in runs  # plain: CPU torch is routed in pyproject.toml


def test_models_are_downloaded_at_the_pinned_revisions_and_cached_by_them(workflow):
    for job in ("gates", "at3-real"):
        by_name = {s.get("name", ""): s for s in steps(workflow, job)}
        cache = next(s for n, s in by_name.items() if n.startswith("Cache Hugging Face"))
        assert "steps.pins.outputs.embedding" in cache["with"]["key"]
        assert "steps.pins.outputs.nli" in cache["with"]["key"]
        download = next(s for n, s in by_name.items() if n.startswith("Download the pinned"))
        assert "revision=s.embedding_model_revision" in download["run"]
        assert "revision=s.nli_model_revision" in download["run"]
        pins = next(s for n, s in by_name.items() if n.startswith("Read the model pins"))
        assert "embedding_model_revision" in pins["run"] and "nli_model_revision" in pins["run"]


def gate_commands(workflow, job) -> list[list[str]]:
    out = []
    for step in steps(workflow, job):
        run = step.get("run", "")
        if "pharmarag.eval.run" in run:
            words = shlex.split(run.replace('"$REPORT_DIR', '"/tmp/reports'))
            out.append(words[words.index("pharmarag.eval.run") + 1 :])
            assert step["env"]["HF_HUB_OFFLINE"] == "1"  # models come from the pinned cache
    return out


def test_pr_gates_are_lookalike_grounding_and_a_fake_verifier_latency_replay(workflow):
    parsed = [eval_run.parse_args(c) for c in gate_commands(workflow, "gates")]
    assert [p.eval for p in parsed] == ["lookalike", "grounding", "latency"]
    assert all(p.ci for p in parsed)
    assert parsed[0].resolver_only and parsed[0].offline and parsed[0].split == "test"
    assert parsed[1].action == "score"
    assert (parsed[2].verifier, parsed[2].questions, parsed[2].reps) == ("fake", 10, 3)


def test_nightly_is_the_full_real_verifier_protocol_with_the_same_limits(workflow):
    (cmd,) = gate_commands(workflow, "at3-real")
    p = eval_run.parse_args(cmd)
    assert (p.eval, p.verifier, p.questions, p.reps) == ("latency", "real", 30, 5)
    # No knob exists to loosen the limits: the gate takes them from pharmarag.eval.latency.
    assert not any("tolerance" in w or "limit" in w for w in cmd)
    uploads = [s for s in steps(workflow, "at3-real") if "upload-artifact" in s.get("uses", "")]
    assert uploads and uploads[0]["if"] == "always()"


def test_the_fixture_is_loaded_before_the_gates_run(workflow):
    for job in ("gates", "at3-real"):
        runs = [s.get("run", "") for s in steps(workflow, job)]
        load = next(i for i, r in enumerate(runs) if "load_ci_fixture.sh" in r)
        first_gate = next(i for i, r in enumerate(runs) if "pharmarag.eval.run" in r)
        assert load < first_gate
    assert (REPO_ROOT / "scripts" / "load_ci_fixture.sh").exists()
    assert (REPO_ROOT / "data" / "ci_fixture.sql.gz").exists()
