# syntax=docker/dockerfile:1.7
#
# PharmaRAG API image. Three stages:
#   builder  the Python environment, with CPU-only PyTorch
#   models   the embedding and NLI models, downloaded once at build time at pinned commits
#   runtime  python:3.12-slim + the environment + the models + the code, non-root
# Startup needs no network: the models are in the image and HF_HUB_OFFLINE=1.
ARG PYTHON_IMAGE=python:3.12-slim

# --- builder -----------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock ./
# Exactly the versions in uv.lock, without the dev group and without the project itself (the
# code is copied into the runtime stage and run from /app). pyproject.toml routes torch to the
# PyTorch CPU index on Linux, so the lock file holds `torch==2.14.1+cpu` and none of the roughly
# 3 GB of CUDA packages that the PyPI wheel would pull in.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# --- models ------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS models
# The defaults must equal the Settings defaults (tests/unit/app/test_deploy_files.py checks it).
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG EMBEDDING_MODEL_REVISION=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
ARG NLI_MODEL=cross-encoder/nli-deberta-v3-small
ARG NLI_MODEL_REVISION=fa2804872c3b4bd748f38c0185cc85775361e735
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    EMBEDDING_MODEL_REVISION=${EMBEDDING_MODEL_REVISION} \
    NLI_MODEL=${NLI_MODEL} \
    NLI_MODEL_REVISION=${NLI_MODEL_REVISION} \
    HF_HOME=/opt/hf
COPY --from=builder /opt/venv /opt/venv
# Download the PyTorch/safetensors weights and the config and tokenizer files at the pinned
# commits: not the ONNX, OpenVINO, TensorFlow or Flax copies the same repositories carry.
# A download by commit hash writes no `refs/main`, and the app (offline) may ask for the default
# revision, so the pinned commit is recorded as `main` as well.
RUN /opt/venv/bin/python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

ignore = ["onnx/*", "openvino/*", "*.onnx", "*.bin", "*.h5", "*.msgpack", "*.ot", "*.tflite"]
for repo, rev in (
    (os.environ["EMBEDDING_MODEL"], os.environ["EMBEDDING_MODEL_REVISION"]),
    (os.environ["NLI_MODEL"], os.environ["NLI_MODEL_REVISION"]),
):
    snapshot_download(repo, revision=rev, ignore_patterns=ignore)
    refs = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(rev)
PY
# Load both through the classes the app uses, with the network switched off: once at the pinned
# revision, once by name alone. A missing file fails the build here, not the first container start.
RUN HF_HUB_OFFLINE=1 /opt/venv/bin/python -c "import os; \
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer(os.environ['EMBEDDING_MODEL'], device='cpu', revision=os.environ['EMBEDDING_MODEL_REVISION']); \
CrossEncoder(os.environ['NLI_MODEL'], device='cpu', revision=os.environ['NLI_MODEL_REVISION']); \
CrossEncoder(os.environ['NLI_MODEL'], device='cpu')"

# --- runtime -----------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false
RUN useradd --system --uid 10001 --home-dir /home/app --create-home app
COPY --from=builder /opt/venv /opt/venv
COPY --from=models /opt/hf /opt/hf
WORKDIR /app
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app config ./config
COPY --chown=app:app data ./data
COPY --chown=app:app pharmarag ./pharmarag
RUN mkdir -p /app/reports && chown app:app /app/reports
USER app
EXPOSE 8000
# /readyz, not /healthz: the container is "healthy" when it can serve, which includes the models,
# the database and Redis. start-period covers loading and warming both models.
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=6 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/readyz', timeout=4)"]
# --no-proxy-headers: uvicorn otherwise rewrites the client address from X-Forwarded-For for
#   peers in 127.0.0.1, which lets a client pick its own rate-limit bucket. The app parses the
#   header itself, and only when TRUST_FORWARDED_FOR=true.
# --log-config / --no-access-log: every log line is JSON (uvicorn's included), and the per-request
#   JSON line from the app, which carries the request ID, replaces the access log.
CMD ["uvicorn", "pharmarag.app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--no-proxy-headers", "--no-access-log", \
     "--log-config", "/app/pharmarag/app/log_config.json"]
