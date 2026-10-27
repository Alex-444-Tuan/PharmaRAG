"""The eval workflow (.github/workflows/eval.yml) cannot be run here, so check that it parses and
that every gate command it runs is accepted by the real argument parser."""

import shlex

import pytest
import yaml

from pharmarag.app.settings import REPO_ROOT
from pharmarag.eval import run as eval_run

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "eval.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def test_triggers_are_push_to_main_and_pull_request(workflow):
    on = workflow[True] if True in workflow else workflow["on"]  # YAML 1.1 reads `on` as True
    assert on["push"]["branches"] == ["main"]
    assert "pull_request" in on


def test_postgres_service_is_the_pinned_pgvector_image(workflow):
    svc = workflow["jobs"]["gates"]["services"]["postgres"]
    assert svc["image"] == "pgvector/pgvector:0.8.0-pg17"


def test_every_step_runs_or_uses_something(workflow):
    for step in workflow["jobs"]["gates"]["steps"]:
        assert ("run" in step) != ("uses" in step), step


def gate_commands(workflow) -> list[list[str]]:
    out = []
    for step in workflow["jobs"]["gates"]["steps"]:
        run = step.get("run", "")
        if "pharmarag.eval.run" in run:
            words = shlex.split(run.replace('"$REPORT_DIR', '"/tmp/reports'))
            out.append(words[words.index("pharmarag.eval.run") + 1 :])
    return out


def test_the_three_gates_are_present_and_parse(workflow):
    cmds = gate_commands(workflow)
    parsed = [eval_run.parse_args(c) for c in cmds]
    assert [p.eval for p in parsed] == ["lookalike", "grounding", "latency"]
    assert all(p.ci for p in parsed)
    assert parsed[0].resolver_only and parsed[0].offline and parsed[0].split == "test"
    assert parsed[1].action == "score"


def test_the_fixture_is_loaded_before_the_gates_run(workflow):
    runs = [s.get("run", "") for s in workflow["jobs"]["gates"]["steps"]]
    load = next(i for i, r in enumerate(runs) if "load_ci_fixture.sh" in r)
    first_gate = next(i for i, r in enumerate(runs) if "pharmarag.eval.run" in r)
    assert load < first_gate
    assert (REPO_ROOT / "scripts" / "load_ci_fixture.sh").exists()
    assert (REPO_ROOT / "data" / "ci_fixture.sql.gz").exists()
