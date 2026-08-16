# Parakeet-ROCm - Version History

## Table of Contents

- [v0.15.0 (Current)](#v0150-current---06-07-2026)
- [v0.14.0](#v0140---21-02-2026)
- [v0.13.0](#v0130---february-2026)
- [v0.12.0](#v0120---february-2026)
- [v0.11.0](#v0110---january-2026)
- [v0.10.0](#v0100---december-2025)
- [v0.9.0](#v090---december-2025)
- [v0.8.2](#v082---october-2025)
- [v0.8.1](#v081---october-2025)
- [v0.8.0](#v080---october-2025)
- [v0.7.0](#v070---october-2025)
- [v0.6.0](#v060---october-2025)
- [v0.5.2](#v052---08-08-2025)
- [v0.5.1](#v051---08-08-2025)
- [v0.5.0](#v050---07-08-2025)
- [v0.4.0](#v040---06-08-2025)
- [v0.3.0](#v030---31-07-2025)
- [v0.2.2](#v022---28-07-2025)
- [v0.2.1](#v021---27-07-2025)
- [v0.2.0](#v020---27-07-2025)
- [v0.1.1](#v011---27-07-2025)
- [v0.1.0](#v010---26-07-2025)

______________________________________________________________________

## **v0.15.0** (Current) - *06-07-2026*

### ✨ **Feature Release - Thread-Safe Cache, Writeability Checks & Logging Overhaul**

### ✨ **New Features in v0.15.0**

- **Added**: `ensure_dir_writable()` utility in `file_utils.py` for atomic directory writeability validation using `NamedTemporaryFile` probes.
- **Added**: CLI pre-flight validation of output directory writeability before transcription starts, with benchmark collector integration.
- **Added**: Centralized logging configuration for API model warmup and transcription, with structured request lifecycle logging.
- **Added**: `pr-draft.sh` script for creating GitHub draft PRs with interactive file picker and dry-run mode.

### 🐛 **Bug Fixes in v0.15.0**

- **Fixed**: Thread-safe model cache operations — `unload_model_to_cpu()` no longer triggers spurious CPU→GPU→CPU round-trips; `threading.Lock` (`_cache_lock`) protects LRU insert and cache clear operations (closes #31).
  - **Issue**: `unload_model_to_cpu()` called `get_model()` which reloaded the model to GPU; race condition with `clear_model_cache()`.
  - **Root Cause**: No locking around cache access; `get_model()` used instead of a no-load peek.
  - **Solution**: Added `_cache_lock`, replaced `get_model()` with `_cached_keys` set for no-load checks, added device guard before `model.to("cpu")`.
- **Fixed**: Subtitle timestamp drift over long videos — LCS merge now uses median/MAD-based robust offset estimation from all anchor pairs instead of anchoring to only the first match (closes #10).
  - **Issue**: Cumulative linear drift in SRT timestamps, especially in watch/systemd usage.
  - **Root Cause**: Single-anchor time offset in `merge_longest_common_subsequence` was fragile with repeated tokens.
  - **Solution**: Estimate chunk-b time shift from all LCS anchor pairs using median-based offset.
- **Fixed**: Watch-mode SIGINT handler — replaced `sys.exit(0)` with cooperative shutdown, closed race window and test state leak, prevented reentrant `BufferedWriter` crash and duplicate "Stopping…" messages (PRs #37/#38).
- **Fixed**: Pre-flight filename validation before ASR starts to catch invalid filenames early (closes #32).
- **Fixed**: API logging — `response_format` now uses validated value from `transcription_request`; auth check and request tracking ordering corrected; `finish_api_request()` no longer called when tracking was never started.
- **Fixed**: Logging configuration properly honors `NEMO_LOG_LEVEL` and `TRANSFORMERS_VERBOSITY` environment overrides; third-party logger suppression expanded.
- **Fixed**: `set_nemo_verbose` deterministic reset and debug logging centralization (PR #34 review).

### 🔧 **Improvements in v0.15.0**

- **Improved**: Centralized logging across `job_manager.py`, transcription modules, and collector — all logging now routed through `utils/logging_config`.
- **Improved**: API request logging moved after validation; client origin and file details removed from log messages for privacy.
- **Improved**: Template exception handling widened with Google-style `Args:` docstring format.
- **Improved**: Test naming standardized with double-underscore convention; `noqa` comments removed.
- **Updated**: `AGENTS.md` logging policy and testing guidance; `.env.example` formatting fixes.

### 📝 **Key Commits in v0.15.0**

`06dcb15`, `613ed6c`, `47da44c`, `fc776e3`, `a284697`, `dcadecd`

______________________________________________________________________

## **v0.14.0** - *21-02-2026*

### ✨ **Feature Release - API Auth & Model Controls**

### ✨ **New Features in v0.14.0**

- **Added**: API bearer token authentication helper and enforced request validation for protected routes.
- **Added**: API-only model override via `API_MODEL_NAME` with `whisper-1` alias resolving to the API model.
- **Added**: API warmup/idle offload now targets the API model and warmup runs asynchronously at startup.
- **Added**: WebUI theme/CSS applied consistently for mounted and standalone UI, plus config wiring for unsafe filenames.

### 🐛 **Bug Fixes in v0.14.0**

- **Fixed**: API routes now pass API-specific default batch and chunk sizing into transcription.
- **Fixed**: Logging defaults now quiet dependency verbosity while honoring explicit environment overrides.

### 🔧 **Improvements in v0.14.0**

- **Updated**: Docker Compose runtime commands and `.dockerignore` coverage for safer build context.
- **Improved**: Unit/integration tests for API app, routes, logging, and constants.

### 📝 **Key Commits in v0.14.0**

`9b8ff84`, `652fb7f`, `71b319a`, `598dffb`, `bcce74c`

______________________________________________________________________

## **v0.13.0** - *February 2026*

### ✨ **Feature Release - OpenAI-Compatible Transcription API**

### ✨ **New Features in v0.13.0**

- **Added**: OpenAI Whisper-compatible `/v1/audio/transcriptions` API contract with request/response schemas and parameter mapping utilities.
- **Added**: Unified FastAPI + Gradio runtime plus API-only app mode and CLI command support.

### 🐛 **Bug Fixes in v0.13.0**

- **Fixed**: API validation and error payload handling to return concise OpenAI-style error responses.
- **Fixed**: API runtime now reuses shared transcription defaults (batch/chunk/overlap) instead of hardcoded route values.
- **Fixed**: Verbose transcription responses now return coherent BCP-47 language codes (`en` for v2, `und` otherwise).
- **Fixed**: API service host binding is decoupled from WebUI host configuration via `API_SERVER_NAME`.
- **Fixed**: API request debug logging no longer records auth/API-key header presence metadata, preventing security-scan findings for sensitive logging.

### 🔧 **Improvements in v0.13.0**

- **Improved**: Debug diagnostics for API requests with origin metadata and effective transcription settings while avoiding secret leakage.
- **Added**: API smoke-test script and expanded unit/integration test coverage for schemas, mapping, routes, app factory, and CLI wiring.
- **Updated**: Architecture guidance in `AGENTS.md` and `project-overview.md` to enforce modular-first feature extension (thin entrypoints, layered boundaries, protocol/strategy extension points).

### 📝 **Key Commits in v0.13.0**

`a64337d`, `532d767`, `ac993bf`, `e688946`

______________________________________________________________________

## **v0.12.0** - *February 2026*

### ✨ **Feature Release - Filename Flexibility & Security Updates**

### ✨ **New Features in v0.12.0**

- **Enhanced**: `--allow-unsafe-filenames` CLI flag (and `ALLOW_UNSAFE_FILENAMES` env var) support across the transcription pipeline, with expanded test coverage

### 🐛 **Bug Fixes in v0.12.0**

- **Fixed**: Audio I/O support for relaxed filename validation when explicitly enabled

### 🔧 **Improvements in v0.12.0**

- **Updated**: Dependency constraints and lockfile to address high-severity security advisories (see issue #20)
- **Updated**: Requirements export behavior to use `--pyproject` for reproducible, source-of-truth dependency exports
- **Improved**: Test coverage for sentence chunking edge cases and filename validation modes

### 📝 **Key Commits in v0.12.0**

`6ff1ecf`, `072759c`, `8dbadef`, `c3c4338`, `0794544`

______________________________________________________________________

## **v0.11.0** - *January 2026*

### ✨ **Feature Release - CodeQL & Security Hardening**

### ✨ **New Features in v0.11.0**

- **Added**: CodeQL analysis workflow for automated security scanning
- **Added**: WebUI, benchmarks, and pipeline refinements bundled into a consolidated feature update
- **Added**: Opt-in `--allow-unsafe-filenames` CLI flag (and `ALLOW_UNSAFE_FILENAMES` env var) to allow relaxed filename validation (spaces, brackets, quotes, non-ASCII) in output filenames while preserving all security invariants

### 🐛 **Bug Fixes in v0.11.0**

- **Fixed**: SRT base directory validation and safe root boundary checks
- **Fixed**: Path validation to prevent URL injection and traversal in audio/subtitle I/O

### 🔧 **Improvements in v0.11.0**

- **Improved**: Centralized SRT path error handling and validation utilities
- **Updated**: Documentation for environment variables and security-related configuration

### 📝 **Key Commits in v0.11.0**

`22f7a4a`, `b3f7aea`, `21c21ea`, `ccc7f02`, `8a29bc0`

______________________________________________________________________

## **v0.10.0** - *December 2025*

### ✨ **Feature Release - WebUI CLI Refactor & Text Deduplication**

### ✨ **New Features in v0.10.0**

- **Added**: Boundary-aware text deduplication for chunk merging (`_dedupe_text_near_boundary`, `_dedupe_nearby_repeats`, `_fuzzy_overlap_skip_tokens`, `_merge_text_pair`)
- **Added**: Fuzzy overlap detection to handle minor transcription variations at chunk boundaries
- **Added**: Token usage analyzer script for AGENTS.md instruction files
- **Added**: Batch progress callback support for benchmark quality analysis

### 🐛 **Bug Fixes in v0.10.0**

- **Fixed**: Merge strategy dropdown now greyed out when word timestamps disabled (UI consistency)
- **Fixed**: Exception handling for torch initialization errors in GPU availability check
- **Fixed**: Missing space in ASR batch completion log message
- **Fixed**: Patterns parameter now mutable list in watch mode

### ♻️ **Refactoring in v0.10.0**

- **Refactored**: WebUI CLI extracted into separate `webui/cli.py` module following export-only `__init__.py` convention
- **Added**: `__main__.py` entry points for `python -m parakeet_rocm` and `python -m parakeet_rocm.webui`
- **Improved**: Slug sanitization with path traversal protection in benchmark collector
- **Updated**: Docker setup migrated from script-based to CLI-based WebUI launch

### 📝 **Key Commits in v0.10.0**

`b818d6d`, `43f9504`, `4608179`, `9506feb`, `6bf7792`

______________________________________________________________________

## **v0.9.0** - *December 2025*

### ✨ **Feature Release - WebUI, Benchmarks, and Test/Docs Refinements**

### ✨ **New Features in v0.9.0**

- **Added**: Gradio WebUI (`parakeet_rocm.webui`) with presets, validation, and a Benchmarks tab.
- **Added**: Benchmark metrics collection (`parakeet_rocm.benchmarks`) with JSON artifacts and optional AMD GPU telemetry.

### 🐛 **Bug Fixes** in v0.9.0

- **Fixed**: Pydantic v2 compatibility by replacing deprecated `.copy()` with `.model_copy()`.
- **Fixed**: Dependency compatibility by pinning `nemo-toolkit` to `<2.5.0`.

### 🔧 **Improvements in v0.9.0**

- **Improved**: CLI integration tests with consistent module-level markers/skip gates and better GPU detection.
- **Updated**: Developer docs (`project-overview.md`, `docs/test-markers.md`) to reflect the latest test organization and WebUI/benchmark submodules.

### 📝 **Key Commits in v0.9.0**

`a787382`, `0e592c3`, `4e4ed60`, `4b89f10`, `26c3c6d`

______________________________________________________________________

## **v0.8.2** - *October 2025*

### ♻️ **Refactoring & Code Quality Release - Phase 5 Completion**

#### ✨ **New Features in v0.8.2**

- **Added**: Watch mode helper function (`_setup_watch_mode()` in `cli.py`)

  - Extracted 55 lines of watch mode setup logic into dedicated function
  - Handles base directory resolution, callback creation, and watcher initialization
  - Improved CLI code organization and maintainability
  - Reduces complexity in main `transcribe()` function

- **Added**: Advanced Ruff cleanup script (`scripts/clean_codebase_sorted.sh`)

  - Runs Ruff checks in organized passes by rule category
  - Supports `--keep-going` flag to run all passes even on failures
  - Supports `--preview` flag for DOC/FA preview rules
  - Enhanced error handling with colored output
  - Better development workflow for code quality checks

#### 🔧 **Improvements in v0.8.2**

- **Refactored**: `adapt_nemo_hypotheses()` function (`timestamps/adapt.py`)

  - Reduced from 175-line monolithic function to 54-line orchestration
  - Extracted 5 focused helper functions:
    - `_merge_short_segments_pass()` - Merges segments that are too short
    - `_fix_segment_overlaps()` - Adjusts end times to prevent overlaps
    - `_forward_merge_small_leading_words()` - Moves orphan words to previous segment
    - `_merge_tiny_leading_captions()` - Merges captions with very short first lines
    - `_ensure_punctuation_endings()` - Merges segments lacking proper punctuation
  - Each helper function has single, clear responsibility
  - Improved testability with isolated, focused functions
  - Better adherence to Single Responsibility Principle (SRP)

- **Improved**: Configuration management (`pyproject.toml`)

  - Migrated from standalone `isort` to Ruff's built-in import sorting
  - Added comprehensive Ruff configuration with all AGENTS.md rule categories
  - Added `[tool.pytest.ini_options]` for consistent test execution
  - Fixed typer dependency duplication
  - Added section comments for better organization
  - Removed deprecated `[tool.isort]` configuration

- **Enhanced**: Test coverage and reliability

  - Updated integration tests to use existing `sample_mono.wav` file
  - Fixed 2 previously skipped integration tests (now passing)
  - Improved test pass rate from 97% (108/111) to 99% (110/111)
  - Better test file organization and documentation

- **Improved**: Code style and formatting

  - Applied Ruff auto-formatting across test files
  - Fixed docstring first-line capitalization (imperative mood)
  - Removed unnecessary blank lines
  - Improved code consistency and readability

- **Enhanced**: Development tooling

  - Updated `clean_codebase.sh` with better usage instructions
  - Added help flag (`-h`, `--help`) support
  - Improved error handling with `set -euo pipefail`
  - Support for custom target paths
  - Better output messages

#### 📝 **Key Commits in v0.8.2**

`88c57d0`, `f07a96c`, `dcca02b`, `41487d8`, `7cea718`

#### 🎯 **SOLID Compliance Achievement**

- **Phase 5 Complete**: All large functions refactored
- **SRP Improvements**: Functions now have single, clear responsibilities
- **OCP Improvements**: Easier to extend without modifying existing code
- **Target Grade Achieved**: A- (90+/100) for SOLID compliance ✅

______________________________________________________________________

## **v0.8.1** - *October 2025*

### 🐛 **Bug Fix & Refactoring Release - Phase 4 Completion**

#### 🔧 **Improvements in v0.8.1**

- **Added**: Merge strategy registry pattern (`chunking/merge.py`)

  - New `MERGE_STRATEGIES` dictionary maps strategy names to merge functions
  - Registry contains `"contiguous"` and `"lcs"` strategies
  - Exported in `chunking/__init__.py` for public use
  - Follows Open/Closed Principle (easy to add new strategies)

- **Refactored**: Replaced conditional logic with registry lookup (`file_processor.py`)

  - Removed duplicate `if/else` conditionals checking merge strategy
  - Simplified from 8 lines of conditionals to 2 lines of registry lookup
  - Single source of truth for available merge strategies
  - Improved maintainability and extensibility

- **Fixed**: Line length linting errors in `file_processor.py`

  - Resolved 3 line-too-long warnings (lines 304, 312, 462)
  - Extracted conditional expressions into variables for readability
  - Split long f-strings across multiple lines
  - All lines now comply with 88-character limit (PEP 8)

- **Improved**: Exception documentation in `transcribe_file()` docstring

  - Added `FileNotFoundError` and `RuntimeError` to Raises section
  - Better documentation of error conditions for API consumers
  - Follows Google-style docstring conventions

#### 📝 **Documentation in v0.8.1**

- **Updated**: `project-overview.md` with merge strategy registry pattern
  - Added registry pattern extension to Design Patterns section
  - Documented before/after code examples
  - Explained Open/Closed Principle benefits
  - Maintains comprehensive architecture documentation

#### 🧪 **Testing in v0.8.1**

- **Verified**: All existing tests continue to pass
  - Total: 108 tests passing, 3 skipped
  - No breaking changes to existing functionality
  - Registry pattern fully backward compatible

#### 📝 **Key Commits in v0.8.1**

`133f6dc`, `62a6a45`, `d5916d9`, `f38506c`, `afba39e`

______________________________________________________________________

## **v0.8.0** - *October 2025*

### ♻️ **Refactor Release - SOLID Principles Implementation (Phases 2 & 3)**

#### 🔧 **Improvements in v0.8.0**

- **Refactored**: `transcribe_file()` function decomposed into modular helper functions (Phase 2)

  - Extracted `_load_and_prepare_audio()` for audio loading logic (lines 194-238)
  - Extracted `_apply_stabilization()` for stable-ts refinement (lines 241-388)
  - Extracted `_format_and_save_output()` for output handling (lines 391-474)
  - Main function reduced from 320 lines to ~155 lines of orchestration
  - Improved Single Responsibility Principle compliance
  - Enhanced testability with isolated, focused functions
  - Existing helper functions `_transcribe_batches()` and `_merge_word_segments()` already in place

- **Added**: FormatterSpec metadata system for output formats (Phase 3)

  - New `FormatterSpec` dataclass encapsulates formatter metadata
  - Attributes: `format_func`, `requires_word_timestamps`, `supports_highlighting`, `file_extension`
  - All 7 formatters (txt, json, jsonl, csv, tsv, srt, vtt) registered with metadata
  - Replaced hard-coded format checks with metadata queries
  - Improved Open/Closed Principle compliance (easy to add new formats)
  - Enhanced maintainability with centralized format requirements

- **Improved**: Formatter function signatures standardized

  - All formatters now accept `**kwargs: object` parameter
  - Graceful handling of unsupported parameters (e.g., `highlight_words` for txt/json)
  - Uniform interface across all output formats
  - Better extensibility for future formatter enhancements

- **Improved**: Code organization and separation of concerns

  - Format validation logic moved to metadata layer
  - File extension handling centralized in FormatterSpec
  - Highlighting support determined by metadata, not hard-coded lists
  - Reduced coupling between file_processor and formatting modules

#### 🧪 **Testing in v0.8.0**

- **Added**: Unit tests for `file_processor` helper functions (`tests/test_file_processor.py`)

  - 10 comprehensive tests for extracted helper functions
  - Tests for `_load_and_prepare_audio()` (basic and verbose modes)
  - Tests for `_apply_stabilization()` (enabled, disabled, error handling)
  - Tests for `_format_and_save_output()` (basic, highlighting, templates, watch mode, unique filenames)
  - All tests use proper mocking and follow TDD best practices

- **Added**: Unit tests for FormatterSpec system (`tests/test_formatting.py`)

  - 17 comprehensive tests for formatter metadata
  - Tests for FormatterSpec dataclass creation and defaults
  - Tests for registry completeness and type validation
  - Tests for metadata queries (srt, vtt, txt, json)
  - Tests for formatter retrieval (case-insensitive, error handling)
  - Tests for \*\*kwargs support across all formatters
  - All tests pass with 100% coverage of new code

- **Verified**: All existing tests continue to pass

  - Total: 108 tests passing, 3 skipped
  - Phase 2: Added 10 tests (91 → 101 tests)
  - Phase 3: Added 17 tests (101 → 108 tests)
  - No breaking changes to existing functionality
  - Coverage maintained at 68% overall

#### 📝 **Key Commits in v0.8.0**

`62a6a45`, `d5916d9`, `f38506c`, `afba39e`, `9f4ec03`

______________________________________________________________________

## **v0.7.0** - *October 2025*

### ✨ **Feature Release - SOLID Refactoring & Configuration Objects**

#### ✨ **New Features in v0.7.0**

- **Added**: Configuration dataclasses for improved code organization (SOLID principles)
  - New `config.py` module with four configuration dataclasses
  - `TranscriptionConfig`: Groups transcription settings (batch_size, chunk_len_sec, etc.)
  - `StabilizationConfig`: Groups stable-ts refinement settings
  - `OutputConfig`: Groups output-related settings
  - `UIConfig`: Groups UI and logging settings
  - Reduces function parameter count from 24 to 11 parameters
  - Improves Interface Segregation compliance (SOLID principle)
  - Configuration defaults honor project constants from `utils/constant.py`
- **Added**: Comprehensive SOLID principles analysis
  - Detailed evaluation of codebase against SOLID principles
  - Overall grade: B+ (85/100)
  - Prioritized recommendations for improvements
  - Analysis document: `to-do/solid-principles-analysis.md`

#### 🔧 **Improvements in v0.7.0**

- **Improved**: Function signatures reduced for better maintainability
  - `transcribe_file()` signature simplified using config objects
  - `cli_transcribe()` updated to construct and use config objects
  - All call sites updated to use new configuration pattern
- **Improved**: Enhanced documentation and architecture guides
  - Added comprehensive architecture overview with Mermaid diagrams
  - Documented design patterns (Protocol-Oriented, Registry, Configuration Objects)
  - Updated `project-overview.md` with configuration objects section
  - Enhanced `AGENTS.md` with configuration management rules (Section 16)
  - Added Table of Contents to `AGENTS.md` for better navigation

#### 🧪 **Testing in v0.7.0**

- **Added**: Unit tests for configuration objects (`tests/test_config.py`)
  - 11 comprehensive tests covering all config classes
  - Tests for default values, custom values, and dataclass behavior
  - 100% coverage of config module
  - All 81 tests passing

#### 📝 **Key Commits in v0.7.0**

`27a0b19`, `b9e9a38`, `648e9ac`, `6f2d483`, `848c42e`, `6138ab0`, `ac0949e`, `48c2408`, `bd0bb2f`, `db71d55`

______________________________________________________________________

## **v0.6.0** - *October 2025*

### ✨ **Feature Release - Timestamp Refinement & Enhanced Tooling**

#### ✨ **New Features in v0.6.0**

- **Added**: Optional stable-ts timestamp refinement integration
  - Enables word-level timestamp refinement using stable-ts library
  - Supports VAD (Voice Activity Detection) and Demucs denoising
  - Configurable VAD threshold for fine-tuning
  - Graceful degradation when stable-ts is unavailable
- **Added**: Hugging Face model/cache manager CLI tool
  - New `hf-models` command for managing downloaded models
  - List, inspect, and clean model cache
  - Simplifies model management workflow
- **Added**: Enhanced SRT readability reporting and analysis tools
  - Comprehensive readability metrics and scoring
  - Percentile statistics (P50/P90/P95) for duration and CPS
  - Configurable score weights via `--weights` CLI option
  - CI-friendly exit codes via `--fail-below-score` and `--fail-delta-below`
  - Detailed violation reporting with `--show-violations`
- **Added**: Environment variable for model name configuration
  - Model name now configurable via `PARAKEET_MODEL_NAME` env var
  - Allows easy switching between model versions
- **Added**: Idle unload and subdirectory mirroring features
  - Automatic model unloading after idle timeout
  - Output subdirectory structure mirrors input structure

#### 🐛 **Bug Fixes in v0.6.0**

- **Fixed**: Dockerfile now includes all project requirements
  - **Issue**: Missing dependencies in Docker image
  - **Root Cause**: Requirements not properly copied during build
  - **Solution**: Updated Dockerfile CMD and dependency installation
- **Fixed**: Parakeet ASR model name updated to v3
  - **Issue**: Outdated model reference
  - **Solution**: Added new constant for v3 model name
- **Fixed**: Type annotations in file_processor.py
  - **Issue**: Inconsistent type hints causing linting errors
  - **Root Cause**: Missing or incorrect type annotations
  - **Solution**: Refactored type annotations and updated imports

#### 🔧 **Improvements in v0.6.0**

- **Improved**: Extensive code refactoring for clarity and consistency
  - Refactored chunking and merge functions
  - Refactored timestamp adaptation and word timestamp refinement
  - Refactored formatting functions across all output formats
  - Refactored audio I/O and environment loading utilities
  - Enhanced Gradio and Streamlit web applications
- **Improved**: Enhanced test coverage and quality
  - Added tests for stable-ts integration
  - Added tests for SRT diff report functionality
  - Added tests for transcribe-and-diff workflow
  - Refactored test fixtures and type annotations
- **Improved**: Documentation enhancements
  - Updated coding style guide with enhanced linting rules
  - Reorganized README sections for better clarity
  - Enhanced project overview with new features
  - Added stable-ts refinement documentation

#### 📝 **Key Commits in v0.6.0**

`48c2408`, `bd0bb2f`, `db71d55`, `6be7484`, `a9a54d5`, `651692c`, `86d1987`, `ca63342`, `8e38e90`, `265788b`

______________________________________________________________________

## **v0.5.2** - *08-08-2025*

### 🐛 **Patch Release - Testing and Docs**

#### 🧪 Testing Improvements in v0.5.2

- Added: Focused unit tests for timestamps modules to raise coverage
  - `tests/test_refine.py`: SRT roundtrip, merging rules, gap enforcement, wrapping
  - `tests/test_segmentation_core.py`: `split_lines()` and `segment_words()` constraints and long-sentence splits
  - `tests/test_adapt_core.py`: Lightweight `adapt_nemo_hypotheses()` coverage via monkeypatching
- Result: Overall coverage improved to ~83%

#### 📝 Documentation in v0.5.2

- Updated: Version badges in `README.md` and `project-overview.md`

#### 📝 **Key Commits in v0.5.2**

`9fce3e6`, `f2dffb2`, `b31cb0d`

______________________________________________________________________

## **v0.5.1** - *08-08-2025*

### 🐛 **Bug Fixes in v0.5.1**

- Fixed: Pylint constant naming issues across CLI modules
  - Issue: Constants not defined in capital letters caused lint failures
  - Solution: Renamed attributes to uppercase (e.g., `RESOLVE_INPUT_PATHS`) and aligned usages

### 📝 **Documentation**

- Updated: `AGENTS.md` and `project-overview.md` to reflect CLI options and configurations

### 📝 **Key Commits in v0.5.1**

`882f4c4`, `439751b`, `0b3d54b`, `121ebc4`, `d91e499`

______________________________________________________________________

## **v0.5.0** - *07-08-2025*

### ✨ **Feature Release** - Web UI and Enhanced Testing Suite

#### ✨ **New Features in v0.5.0**

- **Streamlit Web UI**: Added a new web-based interface for easier interaction with the ASR system
- **Gradio Integration**: Included Gradio-based interface for quick testing and demonstrations
- **Docker Compose Setup**: Added configuration for running web interfaces in containers

#### 🧪 **Testing Improvements in v0.5.0**

- **Expanded Test Coverage**: Added comprehensive unit tests for core functionality
- **Test Utilities**: Improved test infrastructure and utilities
- **CI/CD**: Enhanced testing in continuous integration

#### 🔧 **Code Quality in v0.5.0**

- **Code Style**: Applied consistent formatting and linting across the codebase
- **Documentation**: Updated documentation for new features and improvements
- **Dependencies**: Updated and organized project dependencies

#### 📝 **Key Commits in v0.5.0**

- `7ab6305` - docs: Update VERSIONS.md with v0.4.0 commit hashes
- `237be60` - docs: Update AGENTS.md to include Pull Request Workflow and Rules
- `437d4a7` - Add extensive unit tests to raise coverage
- `58a1472` - Add docstring and style fixes across package
- `8ee71a3` - test: Update CLI invocation to explicitly invoke 'transcribe' subcommand
- `6f1a22f` - feat: Add Streamlit Web UI for Parakeet-NEMO ASR
- `80b4e55` - feat: Update Docker Compose configuration for Gradio and Streamlit

______________________________________________________________________

## **v0.4.0** - *06-08-2025*

### ✨ **Feature Release** - Directory Watching & Media Format Expansion

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

______________________________________________________________________

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

______________________________________________________________________

## **v0.2.2** - *28-07-2025*

### ♻️ **Refactoring & Cleanup**

- **Removed**: Obsolete segmentation helpers `_is_clause_boundary` and `_segment_words` from `timestamps/adapt.py`.
- **Refactored**: Streamlined timestamp adaptation logic to use the new `segment_words` API exclusively.
- **Documentation**: Enhanced README with detailed features and badges.

______________________________________________________________________

## **v0.2.1** - *27-07-2025*

### 🐛 **Bug Fixes & Style Compliance**

- **Fixed**: Resolved `Unexpected keyword argument` errors by standardizing on `chunk_len_sec` across `app.py` and `transcribe.py`.
- **Fixed**: Corrected a `F841 Local variable ... is assigned to but never used` error in `transcribe.py`.
- **Style**: Enforced strict coding standards across the entire codebase, including Google-style docstrings, PEP 8 compliance, absolute imports, and consistent type hinting.
- **Style**: Corrected constant naming conventions in `utils/env_loader.py`.

### 📝 **Key Commits in v0.2.1**

`625c674`, `3477724`, `a4318c2`, `ec491be`, `766daea`

______________________________________________________________________

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

______________________________________________________________________

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

______________________________________________________________________

## **v0.1.0** - *26-07-2025*

### 🎉 **Initial Release**

Minimal but functional ROCm-enabled ASR inference stack for NVIDIA Parakeet-TDT 0.6B v2.

#### ✨ **Features**

- Docker image and `docker-compose.yaml` with ROCm 6.4.1 and NeMo 2.4 pre-installed.
- Python package skeleton (`parakeet_rocm`) with CLI entry-point and FastAPI stub.
- Batch transcription helper `transcribe.py` and sample stereo WAV.
- PDM-managed `pyproject.toml` with exact dependency pins + optional `rocm` extras.
- Smoke test and CI scaffold.

### 📝 **Key Commits in v0.1.0**

`<initial>`
