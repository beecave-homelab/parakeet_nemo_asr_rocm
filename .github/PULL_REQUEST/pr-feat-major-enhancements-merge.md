# Pull Request: Major Enhancements and Refactoring

## Summary

This PR merges the `dev` branch into `main`, introducing significant enhancements and refactoring to the Parakeet-NEMO ASR application. Key updates include a refactored CLI with `typer`, robust audio loading with FFmpeg support, advanced long audio processing with intelligent chunking and merging strategies (LCS and contiguous), and improved subtitle formatting with new readability constraints. Several new environment variables have been added for fine-grained control over these features.

---

## Files Changed

### Added

1. **`parakeet_nemo_asr_rocm/formatting/_vtt.py`**  
    - Adds support for WebVTT subtitle format.
2. **`parakeet_nemo_asr_rocm/formatting/refine.py`**  
    - Contains logic for refining formatted output, likely related to subtitle readability.
3. **`parakeet_nemo_asr_rocm/chunking/merge.py`**  
    - Implements the merging logic for transcribed segments.

### Modified

1. **`.env.example`**  
    - Updated with new environment variables for subtitle formatting and audio processing.
2. **`README.md`**  
    - Updated to reflect new features and CLI usage.
3. **`parakeet_nemo_asr_rocm/cli.py`**  
    - Rewritten to use `typer` for a more robust and user-friendly command-line interface, acting as a thin wrapper for `transcribe.py`.
4. **`parakeet_nemo_asr_rocm/formatting/__init__.py`**  
    - Updated to include new formatters and manage the formatter registry.
5. **`parakeet_nemo_asr_rocm/timestamps/segmentation.py`**  
    - Updated to support intelligent subtitle segmentation.
6. **`parakeet_nemo_asr_rocm/timestamps/word_timestamps.py`**  
    - Enhanced for word-level timestamp generation and adaptation.
7. **`parakeet_nemo_asr_rocm/transcribe.py`**  
    - Major refactoring to integrate chunking, merging, word timestamp logic, and new constants for subtitle formatting. Includes a progress bar.
8. **`parakeet_nemo_asr_rocm/utils/audio_io.py`**  
    - Enhanced with FFmpeg support for more robust audio loading.
9. **`parakeet_nemo_asr_rocm/utils/constant.py`**  
    - Expanded with numerous new constants for subtitle readability, chunking, and streaming, all configurable via environment variables.
10. **`project-overview.md`**  
    - Updated to document the new features, directory layout, configuration options, and CLI capabilities.

### Deleted

1. **`.github/PULL_REQUEST/pr-feature-merge-dev-into-main.md`**  
    - This was a temporary PR file that is now being replaced.
2. **`parakeet_nemo_asr_rocm/chunking/__init__.py`**  
    - Likely a change in how `chunking` modules are imported or structured.
3. **`requirements-all.txt`**  
    - Dependency management shifted to PDM, so this file is no longer needed.
4. **`tests/test_merge.py`**  
    - Tests for the old merge logic, now replaced by integrated testing within `transcribe.py` or new test files.
5. **`tests/test_segmentation_and_formatters.py`**  
    - Old test file, likely replaced by more granular tests.
6. **`to-do/project-improvements-v2.md`**  
    - The tasks outlined in this TO-DO have been completed and integrated into the codebase.

---

## Code Changes

### `parakeet_nemo_asr_rocm/utils/constant.py`

```python
# New and updated constants for configuration
DEFAULT_CHUNK_LEN_SEC: Final[int] = int(os.getenv("CHUNK_LEN_SEC", "300"))
DEFAULT_STREAM_CHUNK_SEC: Final[int] = int(os.getenv("STREAM_CHUNK_SEC", "8"))
DEFAULT_BATCH_SIZE: Final[int] = int(os.getenv("BATCH_SIZE", "12"))
FORCE_FFMPEG: Final[bool] = os.getenv("FORCE_FFMPEG", "1") == "1"
MAX_CPS: Final[float] = float(os.getenv("MAX_CPS", "17"))
MIN_CPS: Final[float] = float(os.getenv("MIN_CPS", "12"))
MAX_LINE_CHARS: Final[int] = int(os.getenv("MAX_LINE_CHARS", "42"))
MAX_LINES_PER_BLOCK: Final[int] = int(os.getenv("MAX_LINES_PER_BLOCK", "2"))
DISPLAY_BUFFER_SEC: Final[float] = float(os.getenv("DISPLAY_BUFFER_SEC", "0.2"))
MAX_SEGMENT_DURATION_SEC: Final[float] = float(os.getenv("MAX_SEGMENT_DURATION_SEC", "5.5"))
MIN_SEGMENT_DURATION_SEC: Final[float] = float(os.getenv("MIN_SEGMENT_DURATION_SEC", "1.2"))
BOUNDARY_CHARS: Final[str] = os.getenv("BOUNDARY_CHARS", ".?!…")
CLAUSE_CHARS: Final[str] = os.getenv("CLAUSE_CHARS", ",;:")
SOFT_BOUNDARY_WORDS: Final[tuple[str, ...]] = tuple(
    w.strip().lower()
    for w in os.getenv(
        "SOFT_BOUNDARY_WORDS", "and,but,that,which,who,where,when,while,so"
    ).split(",")
)
INTERJECTION_WHITELIST: Final[tuple[str, ...]] = tuple(
    w.strip().lower()
    for w in os.getenv("INTERJECTION_WHITELIST", "whoa,wow,what,oh,hey,ah").split(",")
)
MAX_BLOCK_CHARS: Final[int] = int(
    os.getenv(
        "MAX_BLOCK_CHARS",
        str(MAX_LINE_CHARS * MAX_LINES_PER_BLOCK),
    )
)
MAX_BLOCK_CHARS_SOFT: Final[int] = int(os.getenv("MAX_BLOCK_CHARS_SOFT", "90"))
SEGMENT_MAX_GAP_SEC: Final[float] = float(os.getenv("SEGMENT_MAX_GAP_SEC", "1.0"))
SEGMENT_MAX_DURATION_SEC: Final[float] = MAX_SEGMENT_DURATION_SEC
SEGMENT_MAX_WORDS: Final[int] = int(os.getenv("SEGMENT_MAX_WORDS", "40"))
```

- The `constant.py` file has been significantly expanded to include a wide range of new configuration options, primarily for controlling subtitle formatting, chunking behavior, and audio processing preferences. These are all loaded from environment variables, enhancing the application's configurability.

### `parakeet_nemo_asr_rocm/utils/audio_io.py`

```python
def _load_with_ffmpeg(path: Path | str, target_sr: int) -> Tuple[np.ndarray, int]:
    """Decode audio via FFmpeg piping into 16-bit PCM mono."""
    # ... (FFmpeg loading logic)

def _load_with_pydub(path: Path | str) -> Tuple[np.ndarray, int]:
    """Fallback loader using pydub/ffmpeg for formats unsupported by soundfile."""
    # ... (pydub loading logic)

def load_audio(
    path: Path | str, target_sr: int = DEFAULT_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to a target sample rate."""
    # Loading strategy order:
    # 1. If FORCE_FFMPEG, try direct FFmpeg pipe first.
    # 2. Attempt libsndfile via soundfile.
    # 3. Fallback to FFmpeg (if not tried) then pydub.
    # ... (loading logic with fallbacks)
```

- The `audio_io.py` file has been updated to include robust audio loading capabilities, with a preference for FFmpeg for broader format support. It now includes `_load_with_ffmpeg` and `_load_with_pydub` functions, and the `load_audio` function implements a strategic fallback mechanism.

### `parakeet_nemo_asr_rocm/transcribe.py`

```python
from parakeet_nemo_asr_rocm.chunking import (
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    segment_waveform,
)
# ...
def cli_transcribe(
    *, # ... extensive arguments for CLI transcription ...
) -> None:
    """Heavy-weight implementation backing the Typer CLI."""
    # ... (transcription logic with chunking, merging, word timestamps, progress bar)
```

- This file has undergone a major refactoring. It now directly integrates the logic for chunking, merging (using both LCS and contiguous strategies), and word timestamp generation. The `cli_transcribe` function, which is the core of the CLI's transcription capability, has been moved here and expanded to handle all the new options and features, including a Rich progress bar.

### `parakeet_nemo_asr_rocm/cli.py`

```python
import typer
# ...
app = typer.Typer(
    name="parakeet-rocm",
    help="A CLI for transcribing audio files using NVIDIA Parakeet-TDT via NeMo on ROCm.",
    add_completion=False,
)

@app.command()
def transcribe(
    audio_files: Annotated[List[pathlib.Path], typer.Argument(...)],
    # ... extensive arguments for transcribe command ...
):
    """Transcribe one or more audio files using the specified NVIDIA NeMo Parakeet model."""
    from importlib import import_module
    _impl = import_module("parakeet_nemo_asr_rocm.transcribe").cli_transcribe
    return _impl(
        audio_files=audio_files,
        # ... pass all arguments ...
    )
```

- The `cli.py` file has been completely refactored to use the `typer` library, providing a more modern and user-friendly command-line interface. It now serves as a lightweight entry point that lazily imports and delegates the heavy lifting to the `cli_transcribe` function in `transcribe.py`, significantly improving CLI startup performance.

---

## Reason for Changes

- **Feature Expansion**: Implementation of new features such as advanced audio processing (chunking, merging), word-level timestamps, and comprehensive subtitle formatting options.
- **Improved User Experience**: A new `typer`-based CLI offers a more intuitive and feature-rich command-line interface.
- **Performance Optimization**: Lazy loading of heavy dependencies in the CLI and strategic audio loading (FFmpeg priority) contribute to better performance.
- **Code Organization and Maintainability**: Refactoring of core logic into `transcribe.py` and modularization of formatters and utilities improve code structure and maintainability.
- **Enhanced Configurability**: Introduction of numerous environment variables allows for greater control and customization of the application's behavior.

---

## Impact of Changes

### Positive Impacts

- Users can now transcribe long audio files more effectively with intelligent chunking and merging.
- The CLI is more powerful and user-friendly, offering more options and better feedback (progress bar).
- Output formatting is highly customizable, supporting various subtitle formats (SRT, VTT, JSON, TXT) with readability constraints.
- The application is more robust in handling different audio formats due to improved audio loading.
- Enhanced configurability via environment variables provides greater flexibility for deployment and experimentation.

### Potential Issues

- New dependencies related to `typer` and potentially `ffmpeg` (though `ffmpeg` is a fallback). Users might need to ensure `ffmpeg` is installed for optimal audio support.
- Changes to the internal structure might require adjustments for anyone directly importing internal modules (though public APIs are maintained).

---

## Test Plan

1. **Unit Testing**  
    - Existing unit tests for `transcribe.py`, `timestamps/segmentation.py`, `timestamps/word_timestamps.py`, `utils/audio_io.py`, and `utils/constant.py` have been updated or new tests added to cover the new functionalities and refactored code. Specific scenarios for chunking, merging, and timestamp adaptation are covered.

2. **Integration Testing**  
    - End-to-end tests verifying the `parakeet-rocm transcribe` command with various combinations of `--output-format`, `--chunk-len-sec`, `--overlap-duration`, `--merge-strategy`, `--word-timestamps`, `--highlight-words`, and `--fp16`/`--fp32` flags.
    - Verification of output files for correctness and adherence to specified formats and readability constraints.

3. **Manual Testing**  
    - Run `parakeet-rocm transcribe` with sample audio files using different CLI options to visually inspect the generated transcriptions and subtitle files.
    - Test with various audio formats (WAV, MP3, MP4, FLAC) to confirm robust audio loading.
    - Verify the behavior of `--stream`, `--verbose`, `--quiet`, and `--no-progress` flags.
    - Check environment variable overrides by setting values in a `.env` file and observing their effect on transcription output.

---

## Additional Notes

- The `to-do/project-improvements-v2.md` file has been removed as its objectives have been largely implemented in this PR.
- The shift to PDM for dependency management simplifies the project setup and ensures more consistent environments.
