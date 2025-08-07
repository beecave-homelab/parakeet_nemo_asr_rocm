# Parakeet-ROCm - Version History

## Table of Contents

- [v0.4.0 (Current)](#v040-current---06-08-2025)
- [v0.3.0](#v030---31-07-2025)
- [v0.2.2](#v022---28-07-2025)
- [v0.2.1](#v021---27-07-2025)
- [v0.2.0](#v020---27-07-2025)
- [v0.1.1](#v011---27-07-2025)
- [v0.1.0](#v010---26-07-2025)

---

## **v0.4.0** (Current) - *06-08-2025*

### ✨ **Feature Release** – Directory Watching & Media Format Expansion

#### ✨ **New Features in v0.4.0**

- **`--watch` Flag**: Continuously monitors directories or wildcard pattern(s) and auto-transcribes newly detected media.
- **Verbose Watcher Logs**: With `--verbose`, the watcher now prints per-scan stats and skip reasons.
- **Broader Media Support**: Added common audio and video extensions (e.g., `.m4a`, `.ogg`, `.mp4`, `.mkv`). Any FFmpeg-decodable file is accepted.

#### 🔧 **Improvements in v0.4.0**

- **Documentation**: Updated `README.md`, `project-overview.md`, and `AGENTS.md` to document the watcher and media support.
- **Refactoring**: Centralised extension list and wildcard resolver in `utils/file_utils.py`.

#### 📝 **Key Commits in v0.4.0**

- `06fbf0c` - Bump version to 0.4.0 and enhance transcription functionality
- `6e4a6df` - Add directory watching and media format expansion features
- `ce354db` - Enhance logging configuration in transcribe.py
- `31d9c2f` - Update .env.example and remove Makefile for ROCm support
- `b197025` - Refactor Dockerfile and update dependencies for ROCm support

---

## **v0.3.0** - *31-07-2025*

### ✨ **Chunking, Merging & Timestamping Overhaul**

This release introduces a complete overhaul of long-form audio processing, featuring sophisticated chunking and merging strategies to significantly improve transcription accuracy and readability.

### ✨ **New Features**

- **Advanced Chunk Merging**: Added a new `chunking` module with two overlap-aware merging strategies:
  - `lcs`: A text-aware merge using Longest Common Subsequence to produce natural transitions (default).
  - `contiguous`: A faster, simpler merge that stitches segments at the midpoint.
- **FFmpeg Fallback**: Integrated an FFmpeg fallback for robust audio decoding when `soundfile` encounters unsupported formats.
- **New Output Formatters**: Added support for `CSV`, `JSONL`, and `TSV` output formats.

### 🐛 **Bug Fixes**

- **Fixed**: Cumulative timestamp drift in long audio transcriptions.
  - **Issue**: Successive audio chunks would accumulate small timing errors, causing timestamps to become progressively inaccurate over time.
  - **Root Cause**: The previous merging logic did not account for minor discrepancies in silence detection or token timing at the boundaries of chunks.
  - **Solution**: The new `lcs` merging strategy computes a time offset based on the first aligned token pair in the overlapping region and applies it to the subsequent chunk, ensuring perfect alignment and eliminating drift.

### 🔧 **Improvements**

- **Refactored Transcription Pipeline**: The main `transcribe.py` script was refactored to integrate the new chunking and merging system, controlled via the `--merge-strategy` CLI argument.
- **Enhanced Timestamping**: Word-level timestamp generation is now more accurate due to the improved merging logic.

### 📝 **Key Commits**

`ca829de`, `dc50eb0`, `d14104c`, `9d42c34`, `567d34a`, `6e28c16`, `024a236`, `0921c95`, `5b7b0f8`, `2e7e423`

---

## **v0.2.2** - *28-07-2025*

### ♻️ **Refactoring & Cleanup**

- **Removed**: Obsolete segmentation helpers `_is_clause_boundary` and `_segment_words` from `timestamps/adapt.py`.
- **Refactored**: Streamlined timestamp adaptation logic to use the new `segment_words` API exclusively.
- **Documentation**: Enhanced README with detailed features and badges.

---

## **v0.2.1** - *27-07-2025*

### 🐛 **Bug Fixes & Style Compliance**

- **Fixed**: Resolved `Unexpected keyword argument` errors by standardizing on `chunk_len_sec` across `app.py` and `transcribe.py`.
- **Fixed**: Corrected a `F841 Local variable ... is assigned to but never used` error in `transcribe.py`.
- **Style**: Enforced strict coding standards across the entire codebase, including Google-style docstrings, PEP 8 compliance, absolute imports, and consistent type hinting.
- **Style**: Corrected constant naming conventions in `utils/env_loader.py`.

### 📝 **Key Commits in v0.2.1**

`625c674`, `3477724`, `a4318c2`, `ec491be`, `766daea`

---

## **v0.2.0** - *27-07-2025*

### ✨ **New Features in v0.2.0**

- **Added**: Chunked inference support in `transcribe.py` for efficient long audio processing
- **Added**: Project-level environment overrides and constants for flexible configuration
- **Added**: Support for pydub in `audio_io.py` (broader audio format compatibility)
- **Added**: Initial Parakeet NeMo ASR ROCm implementation

### 🔧 **Improvements in v0.2.0**

- **Refactored**: Codebase for flexibility, maintainability, and clearer environment variable handling
- **Improved**: Documentation, configuration, and dependency management

### 📝 **Key Commits in v0.2.0**

`3477724`, `a4318c2`, `ec491be`, `766daea`, `8532a5f`

---

## **v0.1.1** - *27-07-2025*

### 🐛 **Bug Fix**

- **Fixed**: Stereo WAV files caused shape mismatch (`[1, T, 2]`) inside NeMo dataloader leading to `TypeError: Output shape mismatch for audio_signal`.
  - **Issue**: `transcribe.py` passed file paths to `model.transcribe()`, which then loaded audio without down-mixing.
  - **Root Cause**: Stereo signals keep channel dimension; model expects mono.
  - **Solution**: `transcribe_paths()` now pre-loads audio, down-mixes to mono and passes numpy waveforms directly, ensuring shape `(time,)`.

### 🔧 **Improvements in v0.1.1**

- Always converts input audio to mono automatically; users no longer need to pre-process.

### 📝 **Key Commits in v0.1.1**

`<pending>` (commit hashes will be added when committed)

---

## **v0.1.0** - *26-07-2025*

### 🎉 **Initial Release**

Minimal but functional ROCm-enabled ASR inference stack for NVIDIA Parakeet-TDT 0.6B v2.

#### ✨ **Features**

- Docker image and `docker-compose.yaml` with ROCm 6.4.1 and NeMo 2.4 pre-installed.
- Python package skeleton (`parakeet_nemo_asr_rocm`) with CLI entry-point and FastAPI stub.
- Batch transcription helper `transcribe.py` and sample stereo WAV.
- PDM-managed `pyproject.toml` with exact dependency pins + optional `rocm` extras.
- Smoke test and CI scaffold.

### 📝 **Key Commits in v0.1.0**

`<initial>`
