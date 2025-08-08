# Pull Request: Add optional stable-ts timestamp refinement

## Summary

This PR integrates optional stable-ts post-processing to refine word timestamps and expose new CLI flags for stabilization, VAD and Demucs.

---

## Files Changed

### Added

1. **`parakeet_nemo_asr_rocm/integrations/stable_ts.py`**
   - Provides helper to refine timestamps using stable-ts with optional VAD and Demucs.

2. **`tests/test_stable_ts.py`**
   - Covers fallback behavior when transcribe_any fails.

### Modified

1. **`parakeet_nemo_asr_rocm/cli.py`**
   - Adds CLI flags `--stabilize`, `--vad`, `--demucs` and `--vad-threshold`.
2. **`parakeet_nemo_asr_rocm/transcription/cli.py`**
   - Propagates stabilization options through batch workflow.
3. **`parakeet_nemo_asr_rocm/transcription/file_processor.py`**
   - Invokes refinement step before segmentation.
4. **`README.md`**
   - Documents new stabilization options.

### Deleted

- None

---

## Code Changes

### `parakeet_nemo_asr_rocm/integrations/stable_ts.py`
```python
segment = {
    "start": words[0].start,
    "end": words[-1].end,
    "text": " ".join(w.word for w in words),
    "words": [
        {"word": w.word, "start": w.start, "end": w.end} for w in words
    ],
}
```
- Builds a stable-ts compatible structure before calling `transcribe_any` or `postprocess_word_timestamps`.

---

## Reason for Changes
- Improve subtitle alignment by leveraging stable-ts for audio-aware refinement.

---

## Impact of Changes

### Positive Impacts
- More accurate word timings.
- Optional noise reduction and silence suppression.

### Potential Issues
- Requires additional dependencies when stabilization is used.

---

## Test Plan

1. **Unit Testing**
   - `pdm run pytest` – all tests including new stable-ts test pass.
2. **Integration Testing**
   - Not performed.
3. **Manual Testing**
   - Not performed.

---

## Additional Notes
- Attempted to add dependencies with `pdm add` but network access was denied.
