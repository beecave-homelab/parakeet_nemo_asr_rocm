# Pull Request: Refactor transcription pipeline

## Summary

Refactors the monolithic transcription implementation into a modular package. The CLI now delegates to a dedicated `transcription` subpackage that isolates environment setup, per-file processing and shared helpers.

---

## Files Changed

### Added

1. **`parakeet_nemo_asr_rocm/transcription/__init__.py`**
   - Exposes `cli_transcribe` from the new subpackage.
2. **`parakeet_nemo_asr_rocm/transcription/cli.py`**
   - Orchestrates batch transcription and argument handling.
3. **`parakeet_nemo_asr_rocm/transcription/file_processor.py`**
   - Handles per-file transcription, merging and formatting.
4. **`parakeet_nemo_asr_rocm/transcription/utils.py`**
   - Provides environment configuration and time-stride utilities.

### Modified

1. **`parakeet_nemo_asr_rocm/transcribe.py`**
   - Slimmed to a compatibility wrapper re-exporting the new CLI.
2. **`README.md`**
   - Documents the new modular transcription pipeline.
3. **`project-overview.md`**
   - Updates directory layout and architecture notes.

### Deleted

- None

---

## Code Changes

### `parakeet_nemo_asr_rocm/transcription/cli.py`

```python
model = get_model(model_name)
model = model.half() if fp16 else model.float()
...
output_path = transcribe_file(
    audio_path,
    model=model,
    formatter=formatter,
    file_idx=file_idx,
    output_dir=output_dir,
    output_format=output_format,
    output_template=output_template,
    batch_size=batch_size,
    chunk_len_sec=chunk_len_sec,
    overlap_duration=overlap_duration,
    highlight_words=highlight_words,
    word_timestamps=word_timestamps,
    merge_strategy=merge_strategy,
    overwrite=overwrite,
    verbose=verbose,
    quiet=quiet,
    no_progress=no_progress,
    progress=progress,
    main_task=main_task,
)
```

- Moves orchestration into the `transcription` package and delegates per-file work to `transcribe_file`.

---

## Reason for Changes

- Improve maintainability by separating CLI, per-file processing and shared helpers.
- Enable future extensions without expanding a single monolithic module.

---

## Impact of Changes

### Positive Impacts

- Clearer module boundaries and easier navigation for new contributors.
- Documentation reflects updated structure.

### Potential Issues

- Downstream imports of `parakeet_nemo_asr_rocm.transcribe` now reference a wrapper; ensure external code uses the same API.

---

## Test Plan

1. **Unit Testing**
   - `pdm run pytest`
2. **Integration Testing**
   - Existing CLI tests exercise the new modular pipeline.
3. **Manual Testing**
   - Verified sample transcriptions to ensure output parity.

---

## Additional Notes

- Future work can further split `file_processor.py` if additional features are added.
