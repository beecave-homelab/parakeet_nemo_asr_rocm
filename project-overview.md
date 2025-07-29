# Project Overview – parakeet_nemo_asr_rocm [![Version](https://img.shields.io/badge/Version-v0.2.1-informational)](./VERSIONS.md)

This repository provides a containerised, GPU-accelerated Automatic Speech Recognition (ASR) inference service for the NVIDIA **Parakeet-TDT 0.6B v2** model, running on **AMD ROCm** GPUs.

---

## Table of Contents

- [Directory layout](#directory-layout)
- [Key technology choices](#key-technology-choices)
- [Build & run (quick)](#build--run-quick)
- [Configuration & environment variables](#configuration--environment-variables)
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
│   ├── cli.py                  # Typer-based CLI entry point
│   ├── transcribe.py           # Batch transcription helper
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_io.py         # WAV/PCM helpers
│   │   └── file_utils.py       # File naming and overwrite protection
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
    ├── test_transcribe.py
    └── test_file_utils.py      # Tests for file utilities
```

## Audio format support

The service now accepts and automatically decodes **WAV, MP3, AAC, FLAC and MP4** audio inputs. Decoding first attempts `libsndfile` (via `soundfile`) and transparently falls back to **pydub + ffmpeg** for formats not natively supported.

## Configuration & environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PYTORCH_HIP_ALLOC_CONF` | Mitigate ROCm GPU memory fragmentation | `expandable_segments:True` |
| `NEUTRON_NUMBA_DISABLE_JIT` | Optionally disable Numba JIT to save VRAM | `1` |
| `CHUNK_LEN_SEC` | Length (s) of audio chunks for segmented inference | `20` |

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
$ docker exec -it parakeet-asr-rocm parakeet-nemo-asr-rocm /data/samples/sample.wav
```

## File Overwrite Protection

The CLI now includes automatic file overwrite protection. When transcribing audio files, output files are automatically renamed with numbered suffixes (e.g., `audio.txt`, `audio - 1.txt`, `audio - 2.txt`) to prevent accidental overwrites. Use the `--overwrite` flag to force overwriting existing files.

## Next steps / TODO

1. Implement minimal package code (`app.py`, `cli.py`, etc.).
2. Add helper scripts and sample data.
3. Write tests and ensure CI passes.
4. Update documentation when new features land.
