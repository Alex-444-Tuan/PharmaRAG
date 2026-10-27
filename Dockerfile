# syntax=docker/dockerfile:1.7
#
# PharmaRAG API image (spec section 12). Three stages:
#   builder  the Python environment, with CPU-only PyTorch
#   models   the embedding and NLI models, downloaded once at build time
#   runtime  python:3.12-slim + the environment + the models + the code, non-root
# Startup needs no network: the models are in the image and HF_HUB_OFFLINE=1.
ARG PYTHON_IMAGE=python:3.12-slim

# --- builder -----------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /build
COPY pyproject.toml uv.lock ./

# Install exactly the versions in uv.lock, without the dev group and without the project itself
# (the code is copied into the runtime stage and run from /app).
#
# PyTorch is installed from the CPU index. On Linux x86_64 the PyPI wheel pulls about 3 GB of
# CUDA libraries (nvidia-*, cuda-*, triton) that a CPU-only service never loads. They are
# filtered out of the export, and torch comes from download.pytorch.org/whl/cpu at the version
# the lock file pins, so image and lock file cannot drift apart. `--no-deps` is safe because the
# export lists every transitive dependency with an exact pin.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt \
 && grep -iE '^torch==' requirements.txt > torch.txt \
 && grep -viE '^(torch==|nvidia-|cuda-|triton)' requirements.txt > requirements-cpu.txt \
 && uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python --no-deps \
      --index-url https://download.pytorch.org/whl/cpu -r torch.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps -r requirements-cpu.txt

# --- models ------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS models
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG NLI_MODEL=cross-encoder/nli-deberta-v3-small
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} NLI_MODEL=${NLI_MODEL} HF_HOME=/opt/hf
COPY --from=builder /opt/venv /opt/venv
# Only the PyTorch/safetensors weights and the config and tokenizer files: not the ONNX,
# OpenVINO, TensorFlow or Flax copies that the same repositories also carry.
RUN /opt/venv/bin/python -c "import os; \
from huggingface_hub import snapshot_download; \
ignore = ['onnx/*', 'openvino/*', '*.onnx', '*.bin', '*.h5', '*.msgpack', '*.ot', '*.tflite']; \
[snapshot_download(m, ignore_patterns=ignore) for m in (os.environ['EMBEDDING_MODEL'], os.environ['NLI_MODEL'])]"
# Load both through the classes the app uses, with the network switched off. A missing file
# fails the build here, not the first container start.
RUN HF_HUB_OFFLINE=1 /opt/venv/bin/python -c "import os; \
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer(os.environ['EMBEDDING_MODEL'], device='cpu'); \
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
CMD ["uvicorn", "pharmarag.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
