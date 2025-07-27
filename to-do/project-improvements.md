# Development Plan — Reusing `parakeet-mlx` CLI features in `parakeet_nemo_asr_rocm`

**Role:** ASR Tooling & CLI Architect

## Assumptions

- We will **keep NeMo as the inference backend** (ROCm/PyTorch), only reusing the *CLI UX patterns and output formatting* from `parakeet-mlx`.
- We can depend on `ffmpeg` (for consistent audio decoding) and accept it as a runtime prerequisite.
- We can add lightweight runtime deps: `typer` (CLI), `rich` (progress), optional `pydantic` for config validation.
- NeMo’s `ASRModel.transcribe(..., timestamps=True)` returns char/word/segment timestamps; this is sufficient to implement SRT/VTT/JSON outputs.
- We will **not** port MLX-specific features (e.g., MLX dtypes/attention internals). We’ll provide analogous toggles where NeMo supports them (precision, chunking, streaming).

---

## 1) Current State (target repo)

- `parakeet_nemo_asr_rocm/app.py` — minimal `argparse` CLI: `audio` + `--batch-size`, prints plain text.
- `parakeet_nemo_asr_rocm/transcribe.py` — batch helper; no structured outputs/timestamps.
- `pyproject.toml` exposes `parakeet-nemo-asr-rocm` and `transcribe` entry points.
- No word-level timestamps, no SRT/VTT/JSON outputs, no progress UI, no multi-file output management.

---

## 2) Source of Reusable Ideas (from `parakeet-mlx` CLI)

- **Typer**-based CLI with subcommand (`transcribe`) and rich help.
- **Options**:
  - `--model` (HF repo id), `--output-dir`, `--output-format (txt|srt|vtt|json|all)`, `--output-template` with variables `{parent, filename, index, date}`.
  - `--highlight-words` (word-level emphasis in srt/vtt).
  - Chunking: `--chunk-duration`, `--overlap-duration`.
  - Precision flag: `--fp32/--bf16` (MLX); maps to `.half()`/`.float()` in NeMo.
  - Attention mode switches (MLX local vs full) — **not directly portable**, see mapping.
- **Formatters**: `to_txt`, `to_srt`, `to_vtt`, `to_json`.
- **Progress**: `rich.progress` with spinner/bar and per-file status.
- **FFmpeg** audio loader (consistent decode).

---

## 3) Mapping of Features (Reuse vs Adapt)

| `parakeet-mlx` feature | Action in ROCm/NeMo | Notes |
|---|---|---|
| Typer app & subcommand | **Reuse (adapt)** | Port structure & help text; keep `parakeet-nemo-asr-rocm transcribe` UX. |
| `--model` HF repo id | **Reuse (adapt)** | Pass to `nemo_asr.models.ASRModel.from_pretrained`. |
| `--output-dir` | **Reuse** | Same behavior. |
| `--output-format` (txt/srt/vtt/json/all) | **Reuse** | Implement formatters using NeMo timestamps. |
| `--output-template` with `{parent, filename, index, date}` | **Reuse** | Same templating. |
| `--highlight-words` | **Reuse** | Bold/underline current token in SRT/VTT lines. |
| `--chunk-duration`, `--overlap-duration` | **Adapt** | Prefer NeMo built-in long-audio handling; otherwise implement external chunker with overlap and merge. |
| `--verbose/-v` | **Reuse** | Sentence/time logging & debug prints. |
| `--fp32/--bf16` | **Adapt** | Map to `.float()` vs `.half()` on model/inputs; expose `--fp32/--fp16`. |
| `--local-attention` | **Defer/Optional** | Only if NeMo exposes compatible controls for TDT/FastConformer at inference. |
| Rich progress | **Reuse** | Per-file progress + chunk progress. |
| FFmpeg requirement | **Reuse** | Validate presence; document in README. |

---

## 4) CLI Spec (target)

**Command:**

```bash
parakeet-nemo-asr-rocm transcribe <audio_files...> \
  [--model nvidia/parakeet-tdt-0.6b-v2] \
  [--output-dir ./out] \
  [--output-format txt|srt|vtt|json|all] \
  [--output-template "{filename}"] \
  [--highlight-words] \
  [--chunk-duration 120 --overlap-duration 15] \
  [--batch-size 1] \
  [--fp32/--fp16] \
  [-v/--verbose]
```

**Output template vars:** `{parent}`, `{filename}`, `{index}`, `{date}` (YYYYMMDD).

---

## 5) Implementation Plan (Phased)

### Phase 1 — CLI Skeleton & Wiring

1. Add `parakeet_nemo_asr_rocm/cli_ty.py` with Typer app.
2. Move current argparse entry (`app.py`) behind Typer or keep as “quick path”; update `pyproject.toml` script to point to Typer main.
3. Implement options: `audio`, `--batch-size`, `--model`, `--output-dir`, `--output-format`, `--output-template`, `--verbose`.
4. Add `rich` progress for per-file loop.

**Deliverables:** parity with current behavior + directories/files ready.

### Phase 2 — Timestamps & Formatters

1. Call `model.transcribe([...], timestamps=True)` (NeMo) per batch/file.
2. Build adapters to a neutral `AlignedResult`-like structure:
   - `Result.text`
   - `Result.sentences: [ {text, start, end, tokens:[{text,start,end}]} ]`
   - Construct from NeMo’s `output[0].timestamp['word'|'segment']`.
3. Implement `to_txt`, `to_json`, `to_srt`, `to_vtt` with `format_timestamp(hh:mm:ss,ms)`; support `--highlight-words`.
4. Write outputs to `--output-dir` using `--output-template`.

**Deliverables:** TXT/SRT/VTT/JSON files; examples verified.

### Phase 3 — Chunking for Long Audio

1. Option A (preferred if feasible): Use NeMo long-form APIs/args if available to keep internal timestamp continuity.
2. Option B: External chunker with `--chunk-duration` + `--overlap-duration`:
   - Slice audio (ffmpeg decode to mono 16k float32), generate overlapping windows.
   - Transcribe each chunk with timestamps; **offset** chunk timestamps; **merge** with simple overlap reconciliation (take tokens until next chunk’s first token start > last token end - tolerance).
   - Drive progress bar using `(file_index * total_chunks + current_chunk)`.
3. Pluggable chunk callback for progress updates (mirroring `parakeet-mlx`).

**Deliverables:** reliable long-audio transcription without OOM; timing continuity smoke tests.

### Phase 4 — Precision & Performance

1. Implement `--fp32/--fp16`:
   - After loading, call `model.float()` or `model.half()` and ensure inputs are cast accordingly.
2. `--local-attention` **(optional)**:
   - Only expose if NeMo TDT inference supports an equivalent flag; otherwise hide.

**Deliverables:** doc + benchmarks on VRAM/RTFx trade‑offs.

### Phase 5 — Streaming (Optional, parity with MLX doc section)

- Expose `transcribe-stream` subcommand if a NeMo streaming API exists for TDT.
- Provide parameters (context, depth) **only** when supported.

---

## 6) Key Functions/Modules to Add

```text
parakeet_nemo_asr_rocm/
  cli_ty.py            # Typer app: transcribe command, options, progress UI
  formatting/
    __init__.py
    srt.py             # to_srt(result, highlight_words=False)
    vtt.py             # to_vtt(result, highlight_words=False)
    jsonfmt.py         # to_json(result)
    txt.py             # to_txt(result)
  timestamps/
    adapt.py           # build AlignedResult from NeMo outputs
  audio_io/
    ffmpeg_decode.py   # ensure consistent mono/16k decode; presence check
  chunking/
    chunker.py         # sliding window with overlap + merge utilities
```

**Utility signatures (sketch):**

```python
# timestamps/adapt.py
def nemo_to_aligned(nemo_sample) -> AlignedResult: ...
```

```python
# formatting/srt.py
def to_srt(result, highlight_words: bool = False) -> str: ...
```

```python
# common
def format_timestamp(seconds: float, include_hours: bool = True, decimal: str=",") -> str: ...
```

---

## 7) Example: SRT Line Assembly (word-highlight)

```python
def _srt_entries(sent):
    # sentence-level entry
    yield (sent.start, sent.end, sent.text.strip())

def _srt_entries_wordwise(sent):
    for i, tok in enumerate(sent.tokens):
        start = tok.start
        end = sent.tokens[i+1].start if i < len(sent.tokens)-1 else tok.end
        text = "".join(
            f"<u>{t.text.strip()}</u>" if j == i else t.text
            for j, t in enumerate(sent.tokens)
        ).strip()
        yield (start, end, text)
```

---

## 8) Testing Strategy

### **Unit tests**

- Formatting: SRT/VTT timestamp formatting, word highlighting on/off.
- JSON structure: sentences/tokens consistent and numeric rounding.
- Template rendering: `{filename}`, `{index}`, `{date}`, absolute vs relative output paths.
- Chunk merge logic: continuity across overlaps; off‑by‑one boundaries.

### **Integration tests**

- Short file (≤30s) all formats.
- Long file (≥20m) with chunking+overlap.
- Precision modes (fp16/fp32) on supported GPU; compare text equality.

### **Smoke tests**

- FFmpeg presence; fallback messages.
- Graceful errors when output path invalid or disk full.

---

## 9) Documentation Updates

- README:
  - Install (Python & Docker), **ffmpeg prerequisite**.
  - CLI usage examples covering all options.
  - Output format samples (short snippets).
  - Performance notes (fp16 vs fp32, batch size).
- `--help` examples in README using Typer auto‑help.
- Troubleshooting: ROCm permissions, OOM hints, slow I/O.

---

## 10) Backlog / Nice‑to‑Have

- `--diarize` (future; pyannote integration).
- `--language` / domain prompts (when available).
- `--verbose-json` format compatible with Whisper‑style schemas.
- `--num-workers` for parallel file preprocessing.
- `--max-duration` guardrail to avoid accidental multi‑hour jobs.

---

## 11) Milestones & Estimates

| Milestone | Scope | Estimate |
|---|---|---|
| M1 | Typer CLI + progress + basic TXT output | 0.5–1 day |
| M2 | Timestamps adapter + SRT/VTT/JSON + template | 1.5–2 days |
| M3 | Chunking + overlap + merge + tests | 2–3 days |
| M4 | Precision flag + benchmarks + docs | 0.5–1 day |
| M5 | (Optional) Streaming subcommand (if supported) | 1–2 days |

---

## 12) Risk & Mitigation

- **Timestamp API differences** (NeMo): validate early with sample outputs; add adapters/tests.
- **Chunk merge artifacts**: conservative overlap (≥10–15s), token‑boundary merge, unit tests.
- **ROCm memory variability**: expose `--batch-size` and `--fp16`; document fallback.
- **FFmpeg availability**: CLI pre‑check with actionable error.

---

## 13) Acceptance Criteria

- Users can run:
  - `parakeet-nemo-asr-rocm transcribe a.wav --output-format srt`
  - `... --output-format all --highlight-words --output-template "{filename}_{date}_{index}"`
- Outputs created per file in `--output-dir`; timestamps correct and monotonic.
- Progress bar reflects per‑file & per‑chunk progress.
- README documents all flags with examples; CI tests pass.
