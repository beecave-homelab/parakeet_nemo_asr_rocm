# parakeet_nemo_asr_rocm

---

## Table of Contents

- [v0.1.1 (Current)](#v011-current---27-07-2025)
- [v0.1.0](#v010---26-07-2025)

---

## **v0.1.1** (Current) - *27-07-2025*

### 🐛 **Bug Fix**

- **Fixed**: Stereo WAV files caused shape mismatch (`[1, T, 2]`) inside NeMo dataloader leading to `TypeError: Output shape mismatch for audio_signal`.
  - **Issue**: `transcribe.py` passed file paths to `model.transcribe()`, which then loaded audio without down-mixing.
  - **Root Cause**: Stereo signals keep channel dimension; model expects mono.
  - **Solution**: `transcribe_paths()` now pre-loads audio, down-mixes to mono and passes numpy waveforms directly, ensuring shape `(time,)`.

### 🔧 **Improvements**

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
