"""The deployment files, checked offline (no Docker daemon needed).

These do not prove the image builds; the `compose` job in CI does. They pin the properties the
review found missing, so an edit that drops one fails here first.
"""

import re

import pytest
import yaml

from pharmarag.app.settings import REPO_ROOT, Settings

DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
README = (REPO_ROOT / "README.md").read_text()
COMPOSE = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
CI = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())


def dockerfile_arg(name: str) -> str:
    return re.search(rf"^ARG {name}=(\S+)$", DOCKERFILE, re.M).group(1)


# --- the image ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arg", "field"),
    [
        ("EMBEDDING_MODEL", "embedding_model"),
        ("EMBEDDING_MODEL_REVISION", "embedding_model_revision"),
        ("NLI_MODEL", "nli_model"),
        ("NLI_MODEL_REVISION", "nli_model_revision"),
    ],
)
def test_the_models_baked_into_the_image_are_the_ones_the_app_asks_for(arg, field):
    """If these drift, the offline container cannot find the model it was configured with."""
    assert dockerfile_arg(arg) == getattr(Settings(_env_file=None), field)


def test_models_are_downloaded_at_the_pinned_revisions_and_loaded_offline_at_build_time():
    assert "snapshot_download(repo, revision=rev" in DOCKERFILE
    assert "HF_HUB_OFFLINE=1 /opt/venv/bin/python" in DOCKERFILE  # the build-time load check
    assert re.search(r"^\s+HF_HUB_OFFLINE=1", DOCKERFILE, re.M)  # and offline at run time


def test_uvicorn_runs_without_proxy_headers_and_with_json_logs():
    cmd = DOCKERFILE[DOCKERFILE.rindex("CMD ") :]
    assert "uvicorn" in cmd and "--no-proxy-headers" in cmd
    assert "--no-access-log" in cmd
    path = re.search(r'"--log-config", "([^"]+)"', cmd).group(1)
    assert path.startswith("/app/") and (REPO_ROOT / path.removeprefix("/app/")).is_file()


def test_the_image_runs_as_a_non_root_user():
    assert "useradd" in DOCKERFILE and re.search(r"^USER app$", DOCKERFILE, re.M)


def test_torch_comes_from_the_lock_file_not_a_cuda_filter():
    """Contract v4 routes torch to the CPU index in pyproject.toml; the image just syncs."""
    assert "uv sync --frozen --no-dev --no-install-project" in DOCKERFILE
    assert "nvidia" not in DOCKERFILE


# --- compose ------------------------------------------------------------------------------------


def services():
    return COMPOSE["services"]


def test_nothing_overrides_the_image_command_for_the_api():
    """The api uses the image CMD, so --no-proxy-headers applies; a compose `command` would
    have to repeat it."""
    api = services()["api"]
    assert "command" not in api or "--no-proxy-headers" in " ".join(api["command"])


def test_a_one_shot_migrate_service_gates_api_ingest_and_eval():
    assert services()["migrate"]["command"] == ["alembic", "upgrade", "head"]
    assert services()["migrate"]["restart"] == "no"
    for name in ("api", "ingest", "eval"):
        dep = services()[name]["depends_on"]
        assert dep["migrate"]["condition"] == "service_completed_successfully", name
        assert dep["redis"]["condition"] == "service_healthy", name
        assert "api" not in dep, f"{name} must not need the API up"


def test_the_api_service_no_longer_runs_migrations_itself():
    assert "alembic" not in " ".join(services()["api"].get("command", []))


def test_ingest_and_eval_commands():
    assert services()["ingest"]["command"][-3:] == [
        "pharmarag.ingest.run",
        "--seed",
        "data/seed_ingredients.csv",
    ]
    assert services()["eval"]["entrypoint"] == ["python", "-m", "pharmarag.eval.run"]
    assert services()["ingest"]["profiles"] == ["ingest"]
    assert services()["eval"]["profiles"] == ["eval"]


@pytest.mark.parametrize("name", ["ingest", "eval"])
def test_bind_mounted_one_shot_services_run_as_a_configurable_uid(name):
    user = services()[name]["user"]
    assert "PHARMARAG_UID" in user and "10001" in user  # defaults to the image user


@pytest.mark.parametrize("name", ["postgres", "redis"])
def test_database_and_redis_are_published_on_loopback_only(name):
    (port,) = services()[name]["ports"]
    assert port.startswith("127.0.0.1:")


def test_every_long_running_service_has_a_healthcheck():
    for name in ("postgres", "redis", "api"):
        assert "healthcheck" in services()[name], name


# --- CI -----------------------------------------------------------------------------------------


def step_names(job):
    return [s.get("name") or s.get("uses") for s in CI["jobs"][job]["steps"]]


def test_ci_installs_with_plain_uv_sync_and_caches_models_by_revision():
    steps = CI["jobs"]["test"]["steps"]
    assert any(s.get("run") == "uv sync --frozen" for s in steps)
    cache = next(s for s in steps if s.get("id") == "hf")
    assert "steps.models.outputs.key" in cache["with"]["key"]
    key_step = next(s for s in steps if s.get("id") == "models")
    assert "embedding_model_revision" in key_step["run"] and "nli_model_revision" in key_step["run"]


def test_ci_has_postgres_and_redis_services_and_a_compose_job_after_the_tests():
    assert {"postgres", "redis"} <= set(CI["jobs"]["test"]["services"])
    assert CI["jobs"]["compose"]["needs"] == "test"
    assert "Wait for /readyz" in step_names("compose")


# --- README -------------------------------------------------------------------------------------


def test_readme_run_commands_disable_uvicorn_proxy_headers():
    for block in re.findall(r"```bash\n(.*?)```", README, re.S):
        if "uvicorn pharmarag" in block:
            assert "--no-proxy-headers" in block


@pytest.mark.parametrize(
    "path", ["README.md", "Dockerfile", "docker-compose.yml", ".github/workflows/ci.yml"]
)
def test_public_files_do_not_point_at_the_private_spec_or_agent_instructions(path):
    text = (REPO_ROOT / path).read_text()
    assert "PharmaRAG-Project-Specification" not in text
    assert "CLAUDE.md" not in text
    assert not re.search(r"\bspec(ification)? section\b", text, re.I)
