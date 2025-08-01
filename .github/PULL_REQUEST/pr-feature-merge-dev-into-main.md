# Pull Request: Merge dev into main

## Summary

This PR merges the `dev` branch into `main`, bringing in significant improvements to the project's structure, functionality, and documentation. The changes include a major refactor of the timestamp handling logic, enhanced CLI features, and comprehensive code organization.

---

## Files Changed

### Added

1. **`.windsurf/rules/`**  
   - Added coding style guide and environment variables rules for consistent development practices.

2. **`parakeet_nemo_asr_rocm/chunking/`**  
   - Introduced new module for audio chunking functionality.

3. **`parakeet_nemo_asr_rocm/formatting/`**  
   - Added formatters for multiple output formats (JSON, SRT, TXT, VTT).
   - Implemented refinement logic for better output quality.

4. **`parakeet_nemo_asr_rocm/timestamps/`**  
   - New module for timestamp handling with improved segmentation and word alignment.

5. **`scripts/`**  
   - Added utility scripts for codebase maintenance and SRT diff reporting.

6. **`tests/`**  
   - Added integration tests for CLI and new test utilities.

7. **`to-do/`**  
   - Added project improvement and refactoring documentation.

### Modified

1. **`parakeet_nemo_asr_rocm/`**  
   - Refactored core modules for better organization and maintainability.
   - Updated CLI with new features and improved user experience.

2. **`README.md` & `project-overview.md`**  
   - Enhanced documentation with new features and setup instructions.

3. **`pyproject.toml` & `pdm.lock`**  
   - Updated dependencies and project metadata.

### Deleted

1. **`parakeet_nemo_asr_rocm/app.py`**  
   - Removed in favor of the new CLI-based approach.

---

## Code Changes

### `parakeet_nemo_asr_rocm/timestamps/`

```python
# Example of new timestamp handling
def adapt_nemo_hypotheses(hypotheses: List[Hypothesis], model: ASRModel) -> AlignedResult:
    word_timestamps = get_word_timestamps(hypotheses, model)
    segments = segment_words(word_timestamps)
    return AlignedResult(segments=segments, word_segments=word_timestamps)
```

### `parakeet_nemo_asr_rocm/formatting/`

```python
# Example of new formatter interface
def format_as_srt(segments: List[Segment], word_segments: List[WordSegment]) -> str:
    """Format segments and word timestamps as SRT."""
    # Implementation details...
```

---

## Reason for Changes

This merge brings several important improvements:

- **Code Organization**: Better modular structure with clear separation of concerns.
- **Enhanced Functionality**: New formatting options and improved timestamp handling.
- **Maintainability**: Comprehensive tests and documentation.
- **Developer Experience**: Better tooling and development guidelines.

---

## Impact of Changes

### Positive Impacts

- More accurate timestamp generation
- Support for multiple output formats
- Improved code maintainability
- Better test coverage
- Comprehensive documentation

### Potential Issues

- Breaking changes in the API surface
- May require updates to existing integrations

---

## Test Plan

1. **Unit Testing**
   - Run `pytest tests/` to ensure all tests pass.
   - Verify timestamp accuracy with various input types.

2. **Integration Testing**
   - Test CLI with different combinations of arguments.
   - Verify output formats match expected specifications.

3. **Manual Testing**
   - Process sample audio files and verify output quality.
   - Check edge cases (short clips, various speakers, background noise).

---

## Additional Notes

- This merge includes all changes up to version 0.2.2.
- Documentation has been updated to reflect all changes.
- Follow-up tasks are documented in the `to-do/` directory.
