# To-Do: Refactor CLI and Add Advanced Features from parakeet-mlx

This plan outlines the steps to refactor the existing `parakeet_nemo_asr_rocm` command-line interface (CLI) to incorporate the user experience, output formatting, and advanced features from the `parakeet-mlx` project, while retaining NeMo as the core inference engine.

## Tasks

- [ ] **Phase 1: CLI Skeleton & Wiring**
  - [ ] Create the basic Typer CLI application and wire it up.
    - Path: `parakeet_nemo_asr_rocm/cli_ty.py`
    - Action: Initialize a `typer.Typer()` app. Create a `transcribe` command that accepts a list of audio files. Implement the following options: `--model`, `--batch-size`, `--output-dir`, `--output-format`, `--output-template`, and `--verbose`.
    - Status: `Pending`
  - [ ] Update the project's entry point.
    - Path: `pyproject.toml`
    - Action: Change the `[project.scripts]` entry point from the old `app.py` to the new `cli_ty.py` main function.
    - Status: `Pending`
  - [ ] Add basic progress reporting.
    - Path: `parakeet_nemo_asr_rocm/cli_ty.py`
    - Action: Use the `rich` library to add a progress bar that tracks the processing of multiple input audio files.
    - Status: `Pending`

- [ ] **Phase 2: Timestamps & Formatters**
  - [ ] Enable timestamp generation in NeMo.
    - Path: `parakeet_nemo_asr_rocm/transcribe.py` (or equivalent logic in `cli_ty.py`)
    - Action: Modify the `model.transcribe(...)` call to include `timestamps=True` to retrieve word and segment-level timing information.
    - Status: `Pending`
  - [ ] Create a standardized data structure for transcription results.
    - Path: `parakeet_nemo_asr_rocm/timestamps/adapt.py`
    - Action: Implement a function `nemo_to_aligned` that converts the raw NeMo timestamp output into a consistent `AlignedResult` class (e.g., a Pydantic or dataclass model) containing text, sentences, and token-level timestamps.
    - Status: `Pending`
  - [ ] Implement output formatters.
    - Path: `parakeet_nemo_asr_rocm/formatting/`
    - Action: Create the formatting module with `srt.py`, `vtt.py`, `jsonfmt.py`, and `txt.py`. Each file will contain a function (e.g., `to_srt`) that takes the `AlignedResult` and returns a formatted string. Implement support for the `--highlight-words` option in SRT and VTT formats.
    - Status: `Pending`

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
  - [ ] Implement integration tests for the CLI.
    - Path: `tests/integration/`
    - Action: Create tests that run the full `transcribe` command on a short audio file (<=30s) and a long audio file (>=20m). Verify that the correct output files are generated in all formats and that the content is as expected.
    - Accept Criteria: The CLI runs end-to-end without errors and produces valid, non-empty output files.

- [ ] **Documentation Phase:**
  - [ ] Update project documentation.
    - Path: `project-overview.md`, `README.md`
    - Action: Document the new CLI interface, including all commands, options, and output formats. Update the installation instructions to include new dependencies like `typer` and `rich`. Add a note about requiring `ffmpeg`.
    - Accept Criteria: Documentation is clear, complete, and accurately reflects the new functionality.

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

## Future Enhancements

- [ ] **Streaming Transcription:** Investigate and expose a `transcribe-stream` subcommand if a suitable real-time streaming API exists in NeMo for the target models.
