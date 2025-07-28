# Development Plan — Reusing `parakeet-mlx` CLI features in `parakeet_nemo_asr_rocm`

**Role:** ASR Tooling & CLI Architect

## Assumptions

- We will **keep NeMo as the inference backend** (ROCm/PyTorch), only reusing the *CLI UX patterns and output formatting* from `parakeet-mlx`.
- We can depend on `ffmpeg` (for consistent audio decoding) and accept it as a runtime prerequisite.
- We can add lightweight runtime deps: `typer` (CLI), `rich` (progress), optional `pydantic` for config validation.
- NeMo’s `ASRModel.transcribe(..., timestamps=True)` returns char/word/segment timestamps; this is sufficient to implement SRT/VTT/JSON outputs (see **HF model card**: [Transcribing with timestamps](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2#transcribing-with-timestamps)).
- We will **not** port MLX-specific features (e.g., MLX dtypes/attention internals). We’ll provide analogous toggles where NeMo supports them (precision, chunking, streaming).

---

## 1) Current State (target repo)

- `parakeet_nemo_asr_rocm/app.py` — minimal `argparse` CLI: `audio` + `--batch-size`, prints plain text.  
  ↳ Source: `app.py` ([link](https://github.com/beecave-homelab/parakeet_nemo_asr_rocm/blob/605bf7063eb5cbe0e629655fcd044e8fad23c1c7/parakeet_nemo_asr_rocm/app.py))
- `parakeet_nemo_asr_rocm/transcribe.py` — batch helper; no structured outputs/timestamps.  
  ↳ Source: `transcribe.py` ([link](https://github.com/beecave-homelab/parakeet_nemo_asr_rocm/blob/605bf7063eb5cbe0e629655fcd044e8fad23c1c7/parakeet_nemo_asr_rocm/transcribe.py))
- `pyproject.toml` exposes `parakeet-nemo-asr-rocm` and `transcribe` entry points.  
  ↳ Source: `pyproject.toml` ([link](https://github.com/beecave-homelab/parakeet_nemo_asr_rocm/blob/605bf7063eb5cbe0e629655fcd044e8fad23c1c7/pyproject.toml))
- Thin console wrapper `parakeet_nemo_asr_rocm/cli.py` forwarding to module main.  
  ↳ Source: `cli.py` ([link](https://github.com/beecave-homelab/parakeet_nemo_asr_rocm/blob/605bf7063eb5cbe0e629655fcd044e8fad23c1c7/parakeet_nemo_asr_rocm/cli.py))
- No word-level timestamps, no SRT/VTT/JSON outputs, no progress UI, no multi-file output management.

---

## 2) Source of Reusable Ideas (from `parakeet-mlx` CLI)

- **Typer**-based CLI with subcommand (`transcribe`) and rich help.  
  ↳ Source: `parakeet_mlx/cli.py` ([link](https://github.com/senstella/parakeet-mlx/blob/master/parakeet_mlx/cli.py))
- **Options**:
  - `--model` (HF repo id), `--output-dir`, `--output-format (txt|srt|vtt|json|all)`, `--output-template` with variables `{parent, filename, index, date}`.
  - `--highlight-words` (word-level emphasis in srt/vtt).
  - Chunking: `--chunk-duration`, `--overlap-duration`.
  - Precision flag: `--fp32/--bf16` (MLX); maps to `.half()`/`.float()` in NeMo.
  - Attention mode switches (MLX local vs full) — **not directly portable**, see mapping.
- **Formatters**: `to_txt`, `to_srt`, `to_vtt`, `to_json`.  
  ↳ Source: `parakeet_mlx/cli.py` ([link](https://github.com/senstella/parakeet-mlx/blob/master/parakeet_mlx/cli.py))
- **Progress**: `rich.progress` with spinner/bar and per-file status.  
  ↳ Source: `parakeet_mlx/cli.py` ([link](https://github.com/senstella/parakeet-mlx/blob/master/parakeet_mlx/cli.py))
- **FFmpeg** audio loader (consistent decode).  
  ↳ Source: `parakeet_mlx/audio.py` ([link](https://github.com/senstella/parakeet-mlx/blob/master/parakeet_mlx/audio.py)) and README ([link](https://github.com/senstella/parakeet-mlx/blob/master/README.md))

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
| `--chunk-duration`, `--overlap-duration` | **Adapt** | Prefer NeMo long-audio handling; else external chunker + merge. |
| `--verbose/-v` | **Reuse** | Sentence/time logging & debug prints. |
| `--fp32/--bf16` | **Adapt** | Map to `.float()` vs `.half()` on model/inputs; expose `--fp32/--fp16`. |
| `--local-attention` | **Defer/Optional** | Only if NeMo exposes compatible controls at inference. |
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

**Deliverables:** parity met; directory structure ready.

### Phase 2 — Timestamps & Formatters

1. Call NeMo: `model.transcribe([...], timestamps=True)` (see HF card: [link](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2#transcribing-with-timestamps)).
2. Adapter to `AlignedResult`-like structure:
   - `Result.text`
   - `Result.sentences: [ {text, start, end, tokens:[{text,start,end}]} ]`
3. Implement `to_txt`, `to_json`, `to_srt`, `to_vtt` with `format_timestamp(hh:mm:ss,ms)`; support `--highlight-words`.
4. Write to `--output-dir` using `--output-template`.

**Deliverables:** TXT/SRT/VTT/JSON verified on samples.

### Phase 3 — Chunking for Long Audio

1. Prefer NeMo long-form options if available (keep timestamps continuous).
2. Else external chunker:
   - Decode with ffmpeg (mono 16k float32), sliding windows with `--chunk-duration`, `--overlap-duration`.
   - Transcribe each chunk; **offset** timestamps; **merge** overlaps (token-boundary rule).
   - Progress updates per chunk.
3. Internal callback similar to MLX CLI progress updates.

**Deliverables:** stable long-audio results; continuity tests.

### Phase 4 — Precision & Performance

1. `--fp32/--fp16`:
   - After load: `model.float()` vs `model.half()`, ensure input dtype alignment.
2. `--local-attention`:
   - Only expose when NeMo TDT inference provides equivalent control.

**Deliverables:** doc + VRAM/latency trade-off notes.

### Phase 5 — Streaming (Optional)

- Add `transcribe-stream` if NeMo streaming API exists for TDT.
- Parameters (context/depth) only when supported.

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
    ffmpeg_decode.py   # consistent mono/16k decode; presence check
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
- JSON structure: sentences/tokens consistent + numeric rounding.
- Template rendering: `{filename}`, `{index}`, `{date}`, absolute vs relative output paths.
- Chunk merge logic: continuity across overlaps; off-by-one boundaries.

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
  - CLI usage with examples for all flags.
  - Output format samples (snippets).
  - Performance notes (fp16 vs fp32, batch size).
- `--help` examples via Typer.
- Troubleshooting: ROCm permissions, OOM hints, slow I/O.

---

## 10) Backlog / Nice-to-Have

- `--diarize` (future; pyannote integration).
- `--language` / domain prompts (when available).
- `--verbose-json` format compatible met Whisper.
- `--num-workers` for parallel preprocessing.
- `--max-duration` guardrail to avoid accidental multi-hour jobs.

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
- **Chunk merge artifacts**: conservative overlap (≥10–15s), token-boundary merge, unit tests.
- **ROCm memory variability**: expose `--batch-size` and `--fp16`; document fallback.
- **FFmpeg availability**: CLI pre-check with actionable error.

---

## 13) Acceptance Criteria

- Users can run:
  - `parakeet-nemo-asr-rocm transcribe a.wav --output-format srt`
  - `... --output-format all --highlight-words --output-template "{filename}_{date}_{index}"`
- Outputs created per file in `--output-dir`; timestamps correct and monotonic.
- Progress bar reflects per-file & per-chunk progress.
- README documents all flags with examples; CI tests pass.

---

## 14) Quick Reference — Source Links

### **Upstream (ideas to reuse)**

- `parakeet-mlx` README — CLI overview & options: <https://github.com/senstella/parakeet-mlx/blob/master/README.md>  
- `parakeet-mlx/parakeet_mlx/cli.py` — Typer app, options, formatters, progress: <https://github.com/senstella/parakeet-mlx/blob/master/parakeet_mlx/cli.py>  
- `parakeet-mlx/parakeet_mlx/audio.py` — ffmpeg decode helper: <https://github.com/senstella/parakeet-mlx/blob/master/parakeet_mlx/audio.py>

### **Target repo (current state)**

- `parakeet_nemo_asr_rocm/app.py`: <https://github.com/beecave-homelab/parakeet_nemo_asr_rocm/blob/605bf7063eb5cbe0e629655fcd044e8fad23c1c7/parakeet_nemo_asr_rocm/app.py>  
- `parakeet_nemo_asr_rocm/transcribe.py`: <https://github.com/beecave-homelab/parakeet_nemo_asr_rocm/blob/605bf7063eb5cbe0e629655fcd044e8fad23c1c7/parakeet_nemo_asr_rocm/transcribe.py>  
- `parakeet_nemo_asr_rocm/cli.py`: <https://github.com/beecave-homelab/parakeet_nemo_asr_rocm/blob/605bf7063eb5cbe0e629655fcd044e8fad23c1c7/parakeet_nemo_asr_rocm/cli.py>  
- `pyproject.toml`: <https://github.com/beecave-homelab/parakeet_nemo_asr_rocm/blob/605bf7063eb5cbe0e629655fcd044e8fad23c1c7/pyproject.toml>

### **Reference**

- HF model card (timestamps): <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2#transcribing-with-timestamps>
