# Parakeet NeMo ASR ROCm implementation

---

## Assumptions (as requested)

1. **Host**: Arch Linux (x86\_64) with **ROCm 6.4.1** installed under `/opt/rocm`.
2. **GPU**: AMD Radeon **RX 6600 (GFX1032 / RDNA2)**.
3. **Inference only**, no training/fine‑tuning.
4. **PyTorch ROCm 2.7.0** wheels (and torchaudio 2.7.0) are available at your ROCm `--find-links`.
5. **NeMo version = 2.2** (per the Parakeet HF card). We’ll clone/tag NeMo **v2.2.0** inside the container.
6. Python 3.10.
7. You want **exact deps** from NeMo’s `requirements.txt` and `requirements_asr.txt` (we hardcode them in `pyproject.toml`).
8. Container uses **bind-mount of host ROCm** and passes `/dev/kfd`, `/dev/dri`.
9. We use **PDM** to export a flattened `requirements-all.txt`.

If any of these are off, let me know and I’ll adjust.

---

## 0. Key Links

* **Hugging Face model (Parakeet-TDT 0.6B v2):**
  [https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)

* **NVIDIA NeMo (GitHub):**
  [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

* **NeMo Requirements (we mirror these):**

  * Core: [https://github.com/NVIDIA/NeMo/blob/main/requirements/requirements.txt](https://github.com/NVIDIA/NeMo/blob/main/requirements/requirements.txt)
  * ASR:  [https://github.com/NVIDIA/NeMo/blob/main/requirements/requirements\_asr.txt](https://github.com/NVIDIA/NeMo/blob/main/requirements/requirements_asr.txt)

---

## 1. Host Prerequisites

1. **Install ROCm 6.4.1** (Arch repo/AUR). Ensure `/opt/rocm` exists and libraries are there.

2. Add yourself to GPU-access groups and re-login:

   ```bash
   sudo usermod -aG video,render $USER
   ```

3. Check GPU visibility:

   ```bash
   rocminfo | grep Name -A3
   ```

   If RX6600 isn’t recognized, we’ll export `HSA_OVERRIDE_GFX_VERSION=10.3.0` in the container.

4. **Docker + docker compose v2** installed.

---

## 2. Repository Structure

Create your repo **`parakeet_nemo_asr_rocm/`** like this:

```txt
parakeet_nemo_asr_rocm/
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
├── README.md
├── .gitignore
├── .dockerignore
├── pdm.lock                 # created after first `pdm lock` (commit it)
├── .pdm.toml                # created by PDM (commit it)
│
├── parakeet_nemo_asr_rocm/  # your Python package
│   ├── __init__.py
│   ├── app.py               # entrypoint: python -m parakeet_nemo_asr_rocm.app
│   ├── cli.py               # CLI entry point
│   ├── transcribe.py        # batch transcription script
│   ├── utils/
│   │   ├── __init__.py
│   │   └── audio_io.py
│   └── models/
│       ├── __init__.py
│       └── parakeet.py
│
├── scripts/
│   ├── export_requirements.sh
│   └── dev_shell.sh
│
├── data/
│   ├── samples/sample.wav
│   └── output/
└── tests/
    ├── __init__.py
    └── test_transcribe.py
```

> Rename/adjust modules however you like; just make sure the Docker `CMD` matches.

---

## 3. `pyproject.toml`

Hardcoded NeMo reqs + torch pin; ROCm extras (torchaudio, onnxruntime-rocm) in an optional group.

```toml
[project]
name = "parakeet-nemo-asr-rocm"
version = "0.1.0"
description = "ASR inference service for NVIDIA Parakeet-TDT 0.6B v2 on AMD ROCm GPUs"
authors = [{ name = "elvee" }]
requires-python = ">=3.10,<3.11"

dependencies = [
  # ---- NeMo requirements.txt ----
  "fsspec==2024.12.0",
  "huggingface_hub>=0.24",
  "numba",
  "numpy>=1.22",
  "onnx>=1.7.0",
  "protobuf~=5.29.5",
  "python-dateutil",
  "ruamel.yaml",
  "scikit-learn",
  "setuptools>=70.0.0",
  "tensorboard",
  "text-unidecode",
  "torch==2.7.0",
  "tqdm>=4.41.0",
  "wget",
  "wrapt",

  # ---- NeMo requirements_asr.txt ----
  "braceexpand",
  "editdistance",
  "einops",
  "jiwer>=3.1.0,<4.0.0",
  "kaldi-python-io",
  "lhotse!=1.31.0",
  "librosa>=0.10.1",
  "marshmallow",
  "optuna",
  "packaging",
  "pyannote.core",
  "pyannote.metrics",
  "pydub",
  "pyloudnorm",
  "resampy",
  "scipy>=0.14",
  "soundfile",
  "sox<=1.5.0",
  "texterrors<1.0.0",
  "whisper_normalizer",
]

[project.optional-dependencies]
rocm = [
  "torchaudio==2.7.0",
  "onnxruntime-rocm==1.18.0",
]

[project.scripts]
parakeet-nemo-asr-rocm = "parakeet_nemo_asr_rocm.cli:main"
transcribe = "parakeet_nemo_asr_rocm.transcribe:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["parakeet_nemo_asr_rocm"]
```

---

## 4. `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.10-slim AS base

ARG ROCM_WHEEL_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/"
ARG NEMO_REPO="https://github.com/NVIDIA/NeMo.git"
ARG NEMO_TAG="v2.2.0"   # NeMo 2.2 tag (adjust if tag name differs)

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

# ---- Clone NeMo (source, for transparency/fixes) ----
RUN git clone --depth 1 --branch ${NEMO_TAG} ${NEMO_REPO} /app/NeMo

# ---- Copy project ----
COPY pyproject.toml .
COPY . .

# ---- Lock & export deps with PDM ----
RUN pdm lock
RUN pdm export --no-hashes -o requirements-all.txt

# ---- Install ROCm Torch stack first ----
RUN pip install --no-cache-dir --find-links "$ROCM_WHEEL_URL" \
      torch==2.7.0 torchaudio==2.7.0 onnxruntime-rocm==1.18.0

# ---- Install rest of deps (respect already-installed torch stack) ----
RUN PIP_IGNORE_INSTALLED=0 pip install --no-cache-dir -r requirements-all.txt

# ---- Install NeMo (no deps) ----
WORKDIR /app/NeMo
RUN pip install --no-deps -e ".[asr]"

# ---- Install our project (no deps) ----
WORKDIR /app
RUN pip install --no-deps -e .

# ---- Default command ----
CMD ["python", "-m", "parakeet_nemo_asr_rocm.app"]
```

---

## 5. `docker-compose.yaml`

```yaml
version: "3.9"

services:
  parakeet-asr:
    build:
      context: .
      dockerfile: Dockerfile
    image: parakeet-nemo-asr-rocm:latest
    container_name: parakeet-asr-rocm
    ports:
      - "8000:8000"        # adjust if you expose a service
    environment:
      HSA_OVERRIDE_GFX_VERSION: "10.3.0"
      LD_LIBRARY_PATH: "/opt/rocm/lib:/opt/rocm/lib64"
      TZ: "Europe/Amsterdam"
    devices:
      - "/dev/kfd"
      - "/dev/dri"
    group_add:
      - "video"
      - "render"
    volumes:
      - "/opt/rocm:/opt/rocm:ro"
      - "./data:/data"
    command: ["python", "-m", "parakeet_nemo_asr_rocm.app"]
    restart: unless-stopped
```

---

## 6. Build & Run

```bash
# From repo root
docker compose build
docker compose up
```

Manual run alternative:

```bash
docker build -t parakeet-nemo-asr-rocm:latest .
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v /opt/rocm:/opt/rocm:ro \
  -v "$(pwd)/data:/data" \
  parakeet-nemo-asr-rocm:latest
```

---

## 7. Smoke Test (inside container)

```python
import torch
import nemo.collections.asr as nemo_asr

print("HIP version:", torch.version.hip)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt-0.6b-v2"
).eval().to("cuda")

with torch.inference_mode():
    out = asr_model.transcribe(["/data/samples/sample.wav"], batch_size=1)

print(out[0])
```

You should see transcription text and GPU usage (`watch -n1 rocm-smi` on host).

---

## 8. Troubleshooting

* **ROCm libs not found**: Ensure `/opt/rocm` bind-mount and `LD_LIBRARY_PATH` env are correct.
* **Device permission errors**: Confirm `group_add` in compose and your user groups on host.
* **OOM**: Convert to FP16: `asr_model = asr_model.half()` after `.to("cuda")`.
* **Version mismatches**: Check `torch.version.hip` vs ROCm version. Update pinned wheels as needed.
* **NeMo 2.2 changes**: If NeMo 2.2 alters deps beyond those two files, you may need to update `pyproject.toml`. Here we stayed faithful to the files you specified.

---
