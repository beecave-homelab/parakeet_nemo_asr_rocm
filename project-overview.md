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
│   ├── cli.py                  # Typer-based CLI entry point with rich progress
│   ├── transcribe.py           # Batch transcription with timestamp support
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py          # Sliding-window chunker for long audio
│   ├── timestamps/
│   │   ├── __init__.py
│   │   ├── adapt.py            # NeMo timestamp adaptation
│   │   ├── segmentation.py     # Intelligent subtitle segmentation
│   │   └── models.py           # Data models for aligned results
│   ├── formatting/
│   │   ├── __init__.py
│   │   └── formatters.py       # SRT, VTT, JSON, TXT output formatters
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_io.py         # WAV/PCM helpers
│   │   ├── file_utils.py       # File naming and overwrite protection
│   │   ├── constant.py         # Environment variables and constants
│   │   └── env_loader.py       # Environment configuration loader
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

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_CHUNK_LEN_SEC` | `30` | Segment length for chunked transcription |
| `DEFAULT_BATCH_SIZE` | `1` | Batch size for inference |
| `MAX_LINE_CHARS` | `42` | Maximum characters per subtitle line |
| `MAX_LINES_PER_BLOCK` | `2` | Maximum lines per subtitle block |
| `MAX_CPS` | `17` | Maximum characters per second for reading speed |
| `MAX_BLOCK_CHARS` | `84` | Hard character limit per subtitle block |
| `MAX_BLOCK_CHARS_SOFT` | `90` | Soft character limit for merging segments |
| `MIN_SEGMENT_DURATION_SEC` | `1.2` | Minimum subtitle display duration |
| `MAX_SEGMENT_DURATION_SEC` | `5.5` | Maximum subtitle display duration |
| `DISPLAY_BUFFER_SEC` | `0.2` | Additional display buffer after last word |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | ROCm memory management |
| `NEUTRON_NUMBA_DISABLE_JIT` | `1` | Optionally disable Numba JIT to save VRAM |
| `CHUNK_LEN_SEC` | `20` | Length (s) of audio chunks for segmented inference |

Copy `.env.example` → `.env` and adjust as needed.

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
$ docker exec -it parakeet-asr-rocm parakeet-rocm /data/samples/sample.wav
```

## CLI Features

### Commands

- `transcribe`: Transcribe one or more audio files with rich progress reporting

### Options

- `--model`: Model name/path (default: nvidia/parakeet-tdt-0.6b-v2)
- `--output-dir`: Output directory (default: ./output)
- `--output-format`: Output format: txt, srt, vtt, json (default: txt)
- `--batch-size`: Batch size for inference (default: 1)
- `--chunk-len-sec`: Segment length for long audio (default: 30)
- `--overlap-duration`: Overlap between chunks (default: 15)
- `--word-timestamps`: Enable word-level timestamps
- `--highlight-words`: Highlight words in SRT/VTT outputs
- `--overwrite`: Overwrite existing files
- `--verbose`: Enable verbose logging
- `--fp32` / `--fp16`: Precision control for inference

## Advanced Features

### Long Audio Processing

Automatic sliding-window chunking for audio files longer than 20 minutes with timestamp offsetting and duplicate-word merging.

### Subtitle Readability

Intelligent segmentation that respects:

- Character limits (42 chars per line, 84 chars per block)
- Reading speed (12-17 characters per second)
- Natural clause boundaries with backtracking
- Prevention of orphan words

### File Overwrite Protection

Automatic file renaming with numbered suffixes to prevent accidental overwrites. Use `--overwrite` to force replacement.

## Next steps / TODO

1. Add streaming transcription support (if feasible)
2. Performance optimizations for very long audio
3. Additional output format support
4. Batch processing optimizations
