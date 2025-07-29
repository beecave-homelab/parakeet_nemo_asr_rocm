# To-Do: Refactor CLI and Add Advanced Features from parakeet-mlx

This plan outlines the steps to refactor the existing `parakeet_nemo_asr_rocm` command-line interface (CLI) to incorporate the user experience, output formatting, and advanced features from the `parakeet-mlx` project, while retaining NeMo as the core inference engine.

## Tasks

- [x] **Phase 1: CLI Skeleton & Wiring**
  - [x] Create the basic Typer CLI application and wire it up.
    - Path: `parakeet_nemo_asr_rocm/cli_ty.py`
    - Action: Initialize a `typer.Typer()` app. Create a `transcribe` command that accepts a list of audio files. Implement the following options: `--model`, `--batch-size`, `--output-dir`, `--output-format`, `--output-template`, and `--verbose`.
    - Status: `Completed`
  - [x] Update the project's entry point.
    - Path: `pyproject.toml`
    - Action: Change the `[project.scripts]` entry point from the old `app.py` to the new `cli_ty.py` main function.
    - Status: `Completed`
  - [x] Add basic progress reporting.
    - Path: `parakeet_nemo_asr_rocm/cli_ty.py`
    - Action: Use the `rich` library to add a progress bar that tracks the processing of multiple input audio files.
    - Status: `Completed`
- [x] **Phase 2: Timestamps & Formatters**
  - Path: `parakeet_nemo_asr_rocm/timestamps/adapt.py`, `parakeet_nemo_asr_rocm/utils/constant.py`, `.env.example`
  - Action: Implement character-per-second (CPS) and max-characters-per-line checks while grouping words into segments. Default limits (configurable via env): `MAX_CPS=17`, `MAX_LINE_CHARS=42`, `MAX_LINES_PER_BLOCK=2`, `MAX_SEGMENT_DURATION_SEC=6`, `MIN_SEGMENT_DURATION_SEC=1`.
  - Action: Update grouping algorithm to: a) respect CPS & character limits, b) split long lines at nearest space, c) honour punctuation boundaries.
  - Action: Expose env vars through `utils.constant` and document them in `.env.example`.
  - Status: `Pending`
  - [x] Enable timestamp generation in NeMo.
    - Path: `parakeet_nemo_asr_rocm/transcribe.py` (or equivalent logic in `cli_ty.py`)
    - Action: Modify the `model.transcribe(...)` call to include `timestamps=True` to retrieve word and segment-level timing information.
    - Status: `Completed`
  - [-] Create a standardized data structure for transcription results.
    - Path: `parakeet_nemo_asr_rocm/timestamps/adapt.py`
    - Action: Implement a function `nemo_to_aligned` that converts the raw NeMo timestamp output into a consistent `AlignedResult` class (e.g., a Pydantic or dataclass model) containing text, sentences, and token-level timestamps.
    - Status: `In progress`
  - [-] Implement output formatters.
    - Path: `parakeet_nemo_asr_rocm/formatting/`
    - Action: Create the formatting module with `srt.py`, `vtt.py`, `jsonfmt.py`, and `txt.py`. Each file will contain a function (e.g., `to_srt`) that takes the `AlignedResult` and returns a formatted string. Implement support for the `--highlight-words` option in SRT and VTT formats.
    - Status: `In progress`
- [ ] **Phase 3: Chunking for Long Audio**
  - [ ] **Analysis:** Investigate NeMo's native long-audio handling capabilities.
    - Path: NeMo ASR documentation, `parakeet_nemo_asr_rocm/transcribe.py`
    - Action: Research if NeMo's `ASRModel` provides a built-in, robust method for transcribing long audio files with continuous timestamps. This is preferred over a manual implementation.
    - Accept Criteria: A clear decision on whether to use a native NeMo feature or proceed with a custom chunker.
  - [ ] **Implementation:** Implement long-audio processing based on the analysis.
    - Path: `parakeet_nemo_asr_rocm/chunking/chunker.py` (if manual approach is chosen)
    - Action: If NeMo does not provide a suitable internal method, implement a manual chunker. This involves creating overlapping audio segments, transcribing each, offsetting the timestamps, and merging the results to produce a single, coherent transcript.
    - Status: `Pending`
- [ ] **Phase 4: Precision & Performance**
  - [ ] Implement precision controls.
    - Path: `parakeet_nemo_asr_rocm/cli_ty.py`
    - Action: Add `--fp32` and `--fp16` flags to the CLI. After loading the model, call `model.float()` or `model.half()` based on the user's selection.
    - Status: `Pending`
- [ ] **Testing Phase:**
  - [ ] Implement unit tests for new components.
    - Path: `tests/unit/`
    - Action: Create `test_formatting.py` to verify SRT/VTT/JSON output correctness, including timestamp formats and word highlighting. Create `test_chunking.py` to test the overlap and merge logic. Create `test_cli.py` to validate output template rendering.
    - Accept Criteria: Unit tests pass for all core formatting and chunking logic.
  - [x] Implement integration tests for the CLI. (basic smoke tests added for txt & srt)
    - Path: `tests/integration/`
    - Action: Create tests that run the full `transcribe` command on a short audio file (<=30s) and a long audio file (>=20m). Verify that the correct output files are generated in all formats and that the content is as expected.
    - Accept Criteria: The CLI runs end-to-end without errors and produces valid, non-empty output files.
- [ ] **Documentation Phase:**
  - [ ] Update project documentation.
    - Path: `project-overview.md`, `README.md`
    - Action: Document the new CLI interface, including all commands, options, and output formats. Update the installation instructions to include new dependencies like `typer` and `rich`. Add a note about requiring `ffmpeg`.
    - Accept Criteria: Documentation is clear, complete, and accurately reflects the new functionality.

## Implementation Plan (Phased)

### Phase 1 — CLI Skeleton & Wiring

- [x] **1.1** Add `parakeet_nemo_asr_rocm/cli_ty.py` with a Typer app.
- [x] **1.2** Update `pyproject.toml` script to point to the new Typer main function.
- [x] **1.3** Implement CLI options: `audio_files`, `--model`, `--output-dir`, `--output-format`, `--output-template`, `--batch-size`, `--verbose`.
- [x] **1.4** Add a `rich.progress` bar for the main file processing loop.
- [x] **1.5** Move or adapt the existing transcription logic from `transcribe.py` into the new Typer command.
**Deliverables:** A functional CLI that mirrors the old tool's behavior but is built with Typer and includes basic progress reporting.

### Phase 2 — Timestamps & Formatters

- [x] **2.1** Enable timestamp generation by calling `model.transcribe(..., return_hypotheses=True)`.
- [x] **2.2** Create a new module `parakeet_nemo_asr_rocm/timestamps/adapt.py` to convert NeMo's output into a standardized `AlignedResult` data structure.
- [x] **2.3** Create the `parakeet_nemo_asr_rocm/formatting/` module with formatters (`to_txt`, `to_json`, `to_srt`, `to_vtt`).
- [x] **2.4** Implement the logic to save formatted transcriptions to the specified `--output-dir` using the `--output-template`.
**Deliverables:** The CLI can generate TXT, SRT, VTT, and JSON output files with accurate timestamps.

- [ ] **Subtitle Readability Enhancements**
  - Path: `parakeet_nemo_asr_rocm/timestamps/adapt.py`, `parakeet_nemo_asr_rocm/utils/constant.py`, `.env.example`
  - Action: Implement character-per-second (CPS) and max-characters-per-line checks while grouping words into segments. Default limits (configurable via env): `MAX_CPS=17`, `MAX_LINE_CHARS=42`, `MAX_LINES_PER_BLOCK=2`, `MAX_SEGMENT_DURATION_SEC=6`, `MIN_SEGMENT_DURATION_SEC=1`.
  - Action: Update grouping algorithm to: a) respect CPS & character limits, b) split long lines at nearest space, c) honour punctuation boundaries.
  - Action: Expose env vars through `utils.constant` and document them in `.env.example`.
  - Status: `Pending`

### Phase 3 — Chunking for Long Audio

- [ ] **3.1** Investigate and implement NeMo's built-in long-audio processing features if available and suitable.
- [ ] **3.2** If native support is insufficient, create an external chunker in `parakeet_nemo_asr_rocm/chunking/chunker.py` that uses a sliding window with overlap.
- [ ] **3.3** Implement timestamp-aware merging logic for overlapping transcriptions.
- [ ] **3.4** Integrate chunking progress into the `rich` progress bar.
**Deliverables:** The tool can reliably transcribe audio files longer than the model's context window without OOM errors.

### Phase 4 — Precision & Performance

- [ ] **4.1** Implement `--fp32` and `--fp16` flags to control model precision via `model.float()` or `model.half()`.
- [ ] **4.2** Document the performance trade-offs (VRAM, speed) for each precision mode.
**Deliverables:** Users can control the inference precision to balance performance and resource usage.

### Phase 5 — Streaming (Optional)

- [ ] **5.1** Investigate NeMo's streaming API availability for the Parakeet model.
- [ ] **5.2** If supported, implement a `transcribe-stream` subcommand.
**Deliverables:** (Optional) A streaming transcription mode for real-time use cases.

## Documentation Updates

- [ ] **9.1** Update `project-overview.md` to reflect the new CLI structure, features, and modules.
- [ ] **9.2** Update `README.md` with:
  - [ ] Instructions for the `ffmpeg` prerequisite.
  - [ ] New CLI usage examples covering all options.
  - [ ] Sample snippets for each output format.
  - [ ] Performance notes (fp16 vs fp32, batch size).
- [ ] **9.3** Ensure the `--help` output is clear and add examples to the README.
- [ ] **9.4** Add a troubleshooting section for common issues (ROCm, OOM, etc.).

## Related Files

- `parakeet_nemo_asr_rocm/cli_ty.py` (new)
- `parakeet_nemo_asr_rocm/formatting/` (new module)
- `parakeet_nemo_asr_rocm/timestamps/` (new module)
- `parakeet_nemo_asr_rocm/audio_io/` (new module)
- `parakeet_nemo_asr_rocm/chunking/` (new module)
- `parakeet_nemo_asr_rocm/transcribe.py` (modified)
- `parakeet_nemo_asr_rocm/app.py` (to be deprecated/removed)
- `pyproject.toml` (modified)
- `README.md` (modified)
- `project-overview.md` (modified)
- `tests/` (new unit and integration tests)

## Debugging Log (As of 2025-07-27)

**Resolved Issue (2025-07-27):** Fixed `AttributeError` by replacing direct `conv_subsampling` access with a flexible `_calc_time_stride` helper that handles Conformer encoders.
**Next Steps:**

1. ~~Verify timestamp output accuracy across several sample files.~~ ✅ (sample passed)
2. Proceed to Phase 3 — long-audio chunking analysis.

- **Find the Correct Stride Calculation:** We need to investigate the `ConformerEncoder` object in NeMo to find the correct way to determine its total subsampling factor. This might involve inspecting the model's configuration or source code to understand how the audio is downsampled through its layers.
- **Update `cli_ty.py`:** Once the correct method is found, we will update the `time_stride` calculation in `cli_ty.py` to use the correct attributes from the `ConformerEncoder`.
- **Rerun Smoke Test:** After applying the fix, we will run the smoke test again to verify that timestamp generation and SRT output work correctly.

### Debugging Update — 2025-07-27 (evening)

*Implemented*

- Added word-grouping segmentation logic in `timestamps/adapt.py`; captions now split into ~5-6 s blocks.
- Exposed segmentation thresholds via env vars (`SEGMENT_MAX_GAP_SEC`, `SEGMENT_MAX_DURATION_SEC`, `SEGMENT_MAX_WORDS`) and documented them in `.env.example`.
- Integrated constants through single-load `utils.constant` as per env-handling policy.

*Observed*

- SRT blocks have correct segmentation but **timestamps are ~10× too large** (≈5 min for 26 s clip).
  Investigation shows NeMo `Hypothesis.timestamp` returns *frame indices*, not seconds, for Parakeet-TDT.

*Planned Fix*

1. Calculate global `time_stride = subsampling_factor × window_stride` once via `_calc_time_stride()`.
2. Pass this stride into `get_word_timestamps` and always scale indices `* time_stride` there.
3. Remove all other ad-hoc scaling to avoid double multiplication.
4. Re-run smoke test on `voice-sample.wav`; expect final caption end ~00:00:26.

*Result (2025-07-27 night)*

- Implemented `_calc_time_stride()` fallback chain (QuartzNet / Conformer / cfg).
- Passed computed `time_stride` into `get_word_timestamps`; removed all other scaling.
- Smoke test on `voice-sample.wav` now ends at **00:00:26.3**, matching clip length.
- Confirmed SRT/VTT timestamps are accurate (<50 ms drift) and no longer ~10× oversized.
- Phase 2 timestamp-scaling sub-task **completed ✅**.

---

## Future Enhancements

- [ ] **Streaming Transcription:** Investigate and expose a `transcribe-stream` subcommand if a suitable real-time streaming API exists in NeMo for the target models.
