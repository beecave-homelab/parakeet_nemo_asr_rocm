# Testing Guide

This document describes the testing strategy and best practices for the Parakeet ROCm ASR project, following the `/test-suite` workflow principles.

**Updated**: December 7, 2025 - Test organization and counts verified with current test suite state.

## Quick Start

```bash
# Run fast unit tests only (default)
pdm run pytest tests/unit/

# Run unit tests with coverage
pdm run pytest tests/unit/ --cov=parakeet_rocm --cov-report=term-missing:skip-covered

# Run all tests (unit + integration + e2e)
pdm run pytest

# Run specific test suites
pdm run pytest tests/unit/          # Unit tests only
pdm run pytest tests/integration/   # Integration tests only
pdm run pytest tests/e2e/           # E2E tests only

# Run by marker
pdm run pytest -m integration       # Integration tests only
pdm run pytest -m "gpu and e2e"     # GPU end-to-end tests only
pdm run pytest -m slow              # Slow tests only

# Exclude heavy tests
pdm run pytest -m "not (gpu or slow or e2e)"  # Fast tests only
```

## Test Organization

### Directory Structure

```txt
tests/
├── unit/                # Fast, hermetic unit tests (56 tests)
│   ├── test_adapt_core.py
│   ├── test_chunker.py
│   ├── test_cli_unit.py
│   ├── test_config.py
│   ├── test_env_loader.py
│   ├── test_formatting.py
│   ├── test_merge.py
│   ├── test_refine.py
│   ├── test_segmentation_and_formatters.py
│   ├── test_segmentation_core.py
│   ├── test_transcription_utils.py
│   └── test_word_timestamps.py
├── integration/         # Cross-boundary tests (20 tests)
│   ├── test_audio_io.py
│   ├── test_cli.py      # CLI E2E tests (GPU-heavy)
│   ├── test_file_processor.py
│   ├── test_file_utils.py
│   ├── test_stable_ts.py
│   └── test_watch_and_file_utils.py
├── e2e/                 # End-to-end workflow tests (19 tests, 1 skipped)
│   ├── test_srt_diff_report.py
│   ├── test_transcribe.py          # Placeholder, skipped
│   └── test_transcribe_and_diff.py
├── slow/                # Resource-intensive tests (reserved)
└── __init__.py
```

### Test Markers

We use pytest markers to categorize tests by resource requirements:

- **`@pytest.mark.integration`** - Crosses process, filesystem, network, or GPU boundaries
- **`@pytest.mark.slow`** - Long-running tests (>5 seconds)
- **`@pytest.mark.e2e`** - Full workflow smoke tests
- **`@pytest.mark.gpu`** - Requires GPU or accelerator hardware

### Test Types

**Unit Tests** (default, fast)

- Pure logic testing
- No external I/O
- Deterministic and hermetic
- Run by default with `pytest`

**Integration Tests** (`@pytest.mark.integration`)

- Cross filesystem, subprocess, or network boundaries
- Use `tmp_path` for filesystem isolation
- Opt-in via `-m integration`

**GPU Tests** (`@pytest.mark.gpu`)

- Require GPU hardware
- Skip gracefully in CI or when GPU unavailable
- Always combined with `@pytest.mark.slow` and `@pytest.mark.e2e`

**End-to-End Tests** (`@pytest.mark.e2e`)

- Full workflow validation
- CLI integration tests
- Black-box user-level scenarios

## Writing Tests

### Test Naming Convention

Follow the pattern: `test_<unit_under_test>__<expected_behavior>`

```python
def test_segment_words__splits_long_sentence() -> None:
    """Verify long sentences split into multiple constrained segments."""
    # Arrange
    words = [...]
    
    # Act
    segments = segment_words(words)
    
    # Assert
    assert len(segments) >= 2
```

### Arrange-Act-Assert Pattern

All tests should follow the AAA pattern for clarity:

```python
def test_load_audio__handles_mono_wav() -> None:
    """Test loading a mono WAV file."""
    # Arrange - set up test data
    audio_path = tmp_path / "test.wav"
    create_sample_wav(audio_path)
    
    # Act - invoke the function under test
    audio, sr = load_audio(audio_path)
    
    # Assert - verify expected behavior
    assert sr == 16000
    assert len(audio) > 0
```

### Using Fixtures

```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    """Create a sample audio file for testing."""
    audio_file = tmp_path / "sample.wav"
    # Create minimal test audio
    create_test_audio(audio_file)
    return audio_file

def test_transcribe__processes_audio(sample_audio: Path) -> None:
    """Test transcription with sample audio."""
    result = transcribe(sample_audio)
    assert result.text
```

### Parametrized Tests

Use `@pytest.mark.parametrize` to test multiple inputs:

```python
@pytest.mark.parametrize(
    "text,max_cps,expected_merge",
    [
        ("Fast words.", 5, True),   # High CPS -> merge
        ("Slow words.", 20, False), # Normal CPS -> no merge
    ],
)
def test_merge_logic(text: str, max_cps: int, expected_merge: bool) -> None:
    """Test merge behavior with different CPS thresholds."""
    result = should_merge(text, max_cps)
    assert result == expected_merge
```

### Type Annotations

All test functions must have type annotations:

```python
from pathlib import Path

def test_example(tmp_path: Path) -> None:
    """Example test with proper type hints."""
    # Test implementation
    pass
```

## Test Isolation & Determinism

### Filesystem Isolation

Use `tmp_path` fixture for file operations:

```python
def test_save_output(tmp_path: Path) -> None:
    """Test file saving to temporary directory."""
    output_file = tmp_path / "output.txt"
    save_file(output_file, "content")
    assert output_file.exists()
    assert output_file.read_text() == "content"
```

### Mocking External Dependencies

Mock external boundaries, not internal logic:

```python
from unittest.mock import Mock, patch

def test_api_client__retries_on_failure() -> None:
    """Test API client retry logic."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = [ConnectionError(), Mock(status_code=200)]
        response = api_client.fetch()
        assert mock_get.call_count == 2
```

### Deterministic Data

- Use fixed seeds for random data
- Normalize timestamps, UUIDs, and unordered collections
- Create minimal test fixtures programmatically

## GPU Tests

GPU tests must:

1. Be marked with all three: `@pytest.mark.gpu`, `@pytest.mark.slow`, `@pytest.mark.e2e`
2. Skip gracefully when GPU unavailable
3. Include proper docstrings explaining hardware requirements

```python
import os
import pytest

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.e2e
def test_model_inference(tmp_path: Path) -> None:
    """Test full model inference pipeline.
    
    Requires GPU hardware and loads full model pipeline.
    
    Args:
        tmp_path: Pytest fixture providing temporary directory.
    """
    if os.getenv("CI") == "true":
        pytest.skip("GPU test skipped in CI environment")
    
    # Test implementation
    result = run_inference()
    assert result
```

## Coverage Goals

- **Target**: ≥85% line coverage for critical business logic
- **Focus**: Core algorithms, utilities, and formatting functions
- **Exclude**: Integration-heavy code requiring GPU/external services

Run coverage reports:

```bash
# Terminal report
pdm run pytest --cov=parakeet_rocm --cov-report=term-missing:skip-covered

# HTML report (detailed)
pdm run pytest --cov=parakeet_rocm --cov-report=html
open htmlcov/index.html

# XML report (for CI)
pdm run pytest --cov=parakeet_rocm --cov-report=xml
```

## Common Patterns

### Testing CLI Commands

```python
from typer.testing import CliRunner, Result

def test_cli_help() -> None:
    """Test CLI help output."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
```

### Testing with Temporary Files

```python
def test_audio_processing(tmp_path: Path) -> None:
    """Test audio file processing."""
    # Create test input
    input_file = tmp_path / "input.wav"
    create_sample_audio(input_file)
    
    # Process
    output_file = tmp_path / "output.txt"
    process_audio(input_file, output_file)
    
    # Verify
    assert output_file.exists()
    content = output_file.read_text()
    assert len(content) > 0
```

### Testing Error Handling

```python
import pytest

def test_load_audio__raises_on_invalid_file() -> None:
    """Test error handling for invalid audio files."""
    with pytest.raises(ValueError, match="Invalid audio format"):
        load_audio("/nonexistent/file.wav")
```

## CI Integration

In CI environments:

- GPU tests are automatically skipped (checks `CI=true` env var)
- Only fast unit tests run by default
- Coverage reports are generated and uploaded

## Best Practices Summary

✅ **DO:**

- Write fast, deterministic, hermetic unit tests by default
- Use proper test markers for integration/GPU/slow tests
- Follow AAA pattern and one assertion per test concept
- Use `tmp_path` for filesystem isolation
- Mock external boundaries (I/O, GPU, APIs)
- Include type annotations on all test functions
- Use parametrized tests for multiple input scenarios

❌ **DON'T:**

- Download large assets or models in unit tests
- Rely on real GPUs or networks during default runs
- Use absolute paths or hardcoded file locations
- Leave tests non-deterministic (random without seeds)
- Mock internal business logic
- Skip writing docstrings for test functions

## Running Specific Test Suites

```bash
# Fast unit tests only (default)
pdm run pytest tests/unit/

# Integration tests only
pdm run pytest tests/integration/

# E2E tests only
pdm run pytest tests/e2e/

# Run all tests (unit + integration + e2e)
pdm run pytest

# Exclude specific markers
pdm run pytest -m "not (gpu or e2e)"

# Run tests for specific module
pdm run pytest tests/unit/test_formatting.py

# Run specific test function
pdm run pytest tests/unit/test_formatting.py::test_srt_formatter_basic

# Verbose output
pdm run pytest -v

# Stop after first failure
pdm run pytest --maxfail=1

# Show local variables on failure
pdm run pytest -l

# Collect and show tests without running
pdm run pytest --collect-only

# Run unit tests with coverage (recommended for CI)
pdm run pytest tests/unit/ --cov=parakeet_rocm --cov-report=term-missing
```

## Debugging Tests

```bash
# Run with detailed output
pdm run pytest -vv

# Show print statements
pdm run pytest -s

# Drop into debugger on failure
pdm run pytest --pdb

# Detailed traceback
pdm run pytest --tb=long
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- Project coding standards: `AGENTS.md`
- Test suite workflow: `iterate-tests.md`
- Model offloading validation: `docs/validating-model-offloading.md`
