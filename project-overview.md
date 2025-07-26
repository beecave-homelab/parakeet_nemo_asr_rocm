# Project Overview – parakeet_nemo_asr_rocm [![Version](https://img.shields.io/badge/Version-v0.1.1-informational)](./VERSIONS.md)

This repository provides a containerised, GPU-accelerated Automatic Speech Recognition (ASR) inference service for the NVIDIA **Parakeet-TDT 0.6B v2** model, running on **AMD ROCm** GPUs.

---

## Table of Contents

- [Directory layout](#directory-layout)
- [Key technology choices](#key-technology-choices)
- [Build & run (quick)](#build--run-quick)
- [Next steps / TODO](#next-steps--todo)

---

## Directory layout

```txt
parakeet_nemo_asr_rocm/
├── Dockerfile                  # Build image with ROCm, NeMo 2.2, project code
├── docker-compose.yaml         # Orchestrate container with /opt/rocm bind-mounts
├── pyproject.toml              # Exact, pinned Python dependencies (PDM-managed)
├── README.md                   # Quick-start & usage
├── .gitignore                  # Common ignores
├── .dockerignore               # Ignore build context cruft
│
├── parakeet_nemo_asr_rocm/     # Python package
│   ├── __init__.py
│   ├── app.py                  # Module entry point (uvicorn/fastapi or CLI)
│   ├── cli.py                  # Console-script entry
│   ├── transcribe.py           # Batch transcription helper
│   ├── utils/
│   │   ├── __init__.py
│   │   └── audio_io.py         # WAV/PCM helpers
│   └── models/
│       ├── __init__.py
│       └── parakeet.py         # Model wrapper (load & cache)
│
├── scripts/
│   ├── export_requirements.sh  # PDM -> requirements-all.txt
│   └── dev_shell.sh            # Helper to exec into running container
│
├── data/
│   ├── samples/sample.wav      # Example audio
│   └── output/                 # Transcription outputs
│
└── tests/
    ├── __init__.py
    └── test_transcribe.py
```

## Key technology choices

| Area                 | Choice |
|----------------------|--------|
| GPU runtime          | ROCm 6.4.1 (host bind-mount) |
| Deep-learning stack  | PyTorch 2.7.0 ROCm wheels + torchaudio 2.7.0 |
| Model hub            | Hugging Face `nvidia/parakeet-tdt-0.6b-v2` |
| Framework            | NVIDIA NeMo 2.2 (ASR collection) |
| Package manager      | PDM 2.15 – generates lockfile + requirements-all.txt |
| Container base       | `python:3.10-slim` |

## Build & run (quick)

```bash
# Build
$ docker compose build

# Run detached
docker compose up -d

# Inside container
$ docker exec -it parakeet-asr-rocm python -m parakeet_nemo_asr_rocm.transcribe /data/samples/sample.wav
```

## Next steps / TODO

1. Implement minimal package code (`app.py`, `cli.py`, etc.).
2. Add helper scripts and sample data.
3. Write tests and ensure CI passes.
4. Update documentation when new features land.
