FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    TZ=Europe/Amsterdam \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64

# ---- System deps ----
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 sox build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Copy project files ----
COPY pyproject.toml requirements-all.txt ./
COPY parakeet_nemo_asr_rocm/ parakeet_nemo_asr_rocm/
COPY scripts/ scripts/

# ---- Install all deps (ROCm wheels via find-links) ----
RUN pip install --no-cache-dir -r requirements-all.txt

# ---- Install project ----
RUN pip install --no-deps -e .

CMD ["parakeet-rocm", "transcribe", "--output-format", "srt", "--word-timestamps", "--stream", "--batch-size", "42", "--quiet", "data/samples/"]
