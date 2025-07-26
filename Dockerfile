# syntax=docker/dockerfile:1.6
FROM python:3.10-slim AS base

ARG ROCM_WHEEL_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/"
ARG NEMO_REPO="https://github.com/NVIDIA/NeMo.git"
ARG NEMO_TAG="v2.2.0"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    TZ=Europe/Amsterdam \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64

# ---- System deps ----
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git ffmpeg libsndfile1 sox build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Install PDM ----
RUN python -m pip install --upgrade pip && pip install pdm==2.15.4

# ---- Clone NeMo ----
RUN git clone --depth 1 --branch ${NEMO_TAG} ${NEMO_REPO} /app/NeMo

# ---- Copy project ----
COPY pyproject.toml .
# COPY project-overview.md README.md .
COPY parakeet_nemo_asr_rocm/ parakeet_nemo_asr_rocm/
COPY scripts/ scripts/

# ---- Lock & export deps ----
RUN pdm lock && pdm export -G rocm --no-hashes -o requirements-all.txt

# ---- Install all deps (ROCm wheels via find-links) ----
RUN pip install --no-cache-dir --find-links "$ROCM_WHEEL_URL" -r requirements-all.txt

# ---- Install NeMo (no deps) ----
WORKDIR /app/NeMo
RUN pip install --no-deps -e ".[asr]"

# ---- Install our project (no deps) ----
WORKDIR /app
RUN pip install --no-deps -e .

CMD ["python", "-m", "parakeet_nemo_asr_rocm.app"]
