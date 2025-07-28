# To-Do: Clean-Up & Deduplicate CLI Refactor Plan

This plan outlines the steps to finish refactoring the `parakeet_nemo_asr_rocm` CLI, consolidate duplicate entries, and complete the remaining advanced features. This plan is based on the `parakeet-mlx` project mentioned in the `project-improvements.md` file.

## Tasks

- [x] **Analysis Phase:**
  - [x] Research NeMo long-audio handling and decide between native support or custom chunker.
    - Path: `parakeet_nemo_asr_rocm/transcribe.py`, NeMo ASR docs
    - Action: Investigate `ASRModel` capabilities for continuous transcription of >20-min audio.
    - Analysis Results:
      - NeMo lacks a built-in continuous long-audio API for Conformer/Parakeet models; decision: build custom sliding-window chunker with timestamp offset merge.
    - Accept Criteria: Decision recorded above and task checked off.

  - [x] Evaluate streaming transcription feasibility.
    - Path: `parakeet_nemo_asr_rocm/cli_ty.py`
    - Action: Check NeMo for real-time streaming APIs compatible with Parakeet models.
    - Analysis Results:
      - Streaming inference unavailable for Parakeet-TDT; decision: mark streaming as not feasible and defer.
    - Accept Criteria: Decision recorded above and task checked off.

- [ ] **Implementation Phase:**
  - [x] Improve caption length thresholds (completed)
    - Path: `parakeet_nemo_asr_rocm/timestamps/segmentation.py`
    - Action: Introduced `MAX_BLOCK_CHARS_SOFT` (90), minimum caption size guard, and stretched captions below `MIN_SEGMENT_DURATION_SEC` to the minimum.
    - Status: Completed ✅

  - [ ] Expose soft/hard block char limits via env vars
    - Path: `utils/constant.py`, `.env.example`
    - Action: Add `MAX_BLOCK_CHARS` (hard) and `MAX_BLOCK_CHARS_SOFT` (soft) with sensible defaults and document in `.env.example`.
    - Status: Pending

  - [x] Typer CLI skeleton & wiring (basic `typer.Typer` app, `transcribe` command).
    - Path: `parakeet_nemo_asr_rocm/cli_ty.py`
    - Status: Completed

  - [x] Update project entry point to Typer main.
    - Path: `pyproject.toml`
    - Status: Completed

  - [x] Add basic progress reporting with `rich`.
    - Path: `parakeet_nemo_asr_rocm/cli_ty.py`
    - Status: Completed

  - [x] Enable timestamp generation in NeMo (`timestamps=True`).
    - Path: `parakeet_nemo_asr_rocm/transcribe.py`
    - Status: Completed

  - [ ] Implement long-audio processing (if custom approach chosen).
    - Path: `parakeet_nemo_asr_rocm/chunking/chunker.py`
    - Action: Create sliding-window chunker, offset & merge timestamps.
    - Status: In progress

  - [x] Add subtitle readability constraints (industry-standard SRT/VTT rules).
    - Path: `parakeet_nemo_asr_rocm/timestamps/adapt.py`, `utils/constant.py`, `.env.example`
    - Implementation details:
      1. **Character & line limits**
         - 1–2 lines per subtitle block (never 3).
         - ≤ 42 characters per line (including spaces).
      2. **Reading-speed (CPS)**
         - Target 12–17 characters per second.
         - Minimum display time = `(char_count / 17) + 0.2 s` buffer.
         - Clamp absolute display time to 1 s – 7 s.
      3. **Segmentation logic in `nemo_to_aligned()`**
         - Build a segment while ALL are true:
           a) `(chars / duration) ≤ 17 cps`,
           b) `chars ≤ 84` (2 × 42),
           c) `duration ≤ 6 s`,
           d) next word **does not** start a new clause (punctuation boundary).
      4. **Post-processing**
         - If a single rendered line exceeds 42 chars, split into two at nearest space.
      5. **Linguistic clean-up**
         - Drop fillers ("well", "you know"), stutters, hesitations.
         - Use sentence-case with normal punctuation; ellipsis only for unfinished/trailing sentences.
      6. **Constants to expose via `utils.constant`**
         - `MAX_CPS`, `MAX_LINE_CHARS`, `MAX_LINES_PER_BLOCK`,
           `MAX_SEGMENT_DURATION_SEC`, `MIN_SEGMENT_DURATION_SEC`, `DISPLAY_BUFFER_SEC`.
    - Status: Completed

  - [ ] Refine sentence-boundary segmentation in subtitles. (completed)
    - Path: `parakeet_nemo_asr_rocm/timestamps/adapt.py`
    - Action: Backtrack to nearest punctuation before limits and cut at clause boundaries. Prevented orphan words by tightening merge rules.
    - Status: In progress (validated on `voice-sample-13.srt`, but not working yet for `output/Exploring the Paranoid Country with 374142 Bunkers to Hide Everyone-2.srt`)

  - [ ] Finalise `AlignedResult` model and converters.
    - Path: `parakeet_nemo_asr_rocm/timestamps/adapt.py`
    - Action: Finish `nemo_to_aligned()` implementation.
    - Status: In progress

  - [ ] Complete output formatters (`to_srt`, `to_vtt`, `to_json`, `to_txt`).
    - Path: `parakeet_nemo_asr_rocm/formatting/`
    - Action: Render formatted strings, support `--highlight-words` in SRT/VTT.
    - Status: In progress

  - [ ] Add precision flags.
    - Path: `parakeet_nemo_asr_rocm/cli_ty.py`
    - Action: Implement `--fp32` / `--fp16`, call `model.float()` / `model.half()`.
    - Status: Pending

- [ ] **Testing Phase:**
  - [ ] Unit tests for formatting & chunking.
    - Path: `tests/unit/`
    - Action: `test_formatting.py`, `test_chunking.py`.
    - Accept Criteria: All unit tests pass.

  - [x] Integration tests for CLI.
    - Path: `tests/integration/`
    - Action: Smoke tests on short (≤30 s) and long (≥20 m) audio files.
    - Accept Criteria: Correct output files generated; transcript non-empty.

- [ ] **Documentation Phase:**
  - [ ] Update docs to reflect new CLI and features.
    - Path: `project-overview.md`, `README.md`
    - Action: Document commands, options, examples, ffmpeg prerequisite, precision trade-offs.
    - Accept Criteria: Documentation builds cleanly and is accurate.

## Related Files

- `parakeet_nemo_asr_rocm/cli_ty.py`
- `parakeet_nemo_asr_rocm/formatting/`
- `parakeet_nemo_asr_rocm/timestamps/`
- `parakeet_nemo_asr_rocm/chunking/`
- `parakeet_nemo_asr_rocm/transcribe.py`
- `utils/constant.py`
- `tests/`

## Future Enhancements

- [ ] Investigate and, if feasible, implement `transcribe-stream` subcommand for real-time transcription.
