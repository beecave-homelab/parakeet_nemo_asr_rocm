# Project Overview – parakeet-rocm [![Version](https://img.shields.io/badge/Version-v0.3.0-informational)](./VERSIONS.md)

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
├── .env.example                # Example environment variables
├── .gitignore                  # Common ignores
├── .dockerignore               # Ignore build context cruft
│
├── .github/                    # GitHub Actions and PR templates
│   └── ...
│
├── parakeet_rocm/     # Python package
│   ├── __init__.py
│   ├── cli.py                  # Typer-based CLI entry point with rich progress
│   ├── transcribe.py           # Batch transcription with timestamp support
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── merge.py            # Overlap-aware merging of transcribed segments
│   ├── timestamps/
│   │   ├── __init__.py
│   │   ├── segmentation.py     # Intelligent subtitle segmentation
│   │   └── models.py           # Data models for aligned results
│   ├── formatting/
│   │   ├── __init__.py
│   │   ├── _srt.py             # SRT formatting logic
│   │   ├── _txt.py             # TXT formatting logic
│   │   └── ...                 # Other formatters (VTT, JSON)
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
    ├── test_merge.py
    ├── test_segmentation_and_formatters.py
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
- `--merge-strategy`: How to merge overlapping chunks: `none`, `contiguous`, `lcs` (default: lcs)
- `--word-timestamps`: Enable word-level timestamps
- `--highlight-words`: Highlight words in SRT/VTT outputs
- `--overwrite`: Overwrite existing files
- `--verbose`: Enable verbose logging
- `--quiet`: Suppress console output except progress bar
- `--no-progress`: Disable the Rich progress bar while still showing created file paths (combine with `--quiet` for fully silent operation)
- `--fp32` / `--fp16`: Precision control for inference

## Advanced Features

### Long Audio Processing

Automatic sliding-window chunking for long audio files. Overlapping segments are merged using one of two strategies:

- **Contiguous**: A fast, simple merge that stitches segments at the midpoint of the overlap.
- **LCS (Longest Common Subsequence)**: A more sophisticated, text-aware merge that aligns tokens in the overlap region to produce more natural transitions. This is the default and recommended strategy.

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
