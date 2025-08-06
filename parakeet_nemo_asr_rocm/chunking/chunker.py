"""Sliding-window chunker for long-audio transcription.

This module provides utilities to split long mono waveforms into overlapping
windows and later restore global timestamps by offsetting NeMo hypotheses.
Further merge helpers will be added in follow-up iterations.

The logic is intentionally kept free of any NeMo or torch imports so that it
can be reused for offline testing without GPU dependencies.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = [
    "segment_waveform",
]


def segment_waveform(
    wav: np.ndarray,
    sr: int,
    chunk_len_sec: int,
    overlap_sec: int = 0,
) -> List[Tuple[np.ndarray, float]]:
    """Split ``wav`` into overlapping windows.

    Args:
        wav: Mono waveform as a 1-D ``float32`` NumPy array.
        sr: Sample rate of the waveform (Hz).
        chunk_len_sec: Desired window length in **seconds**. If ``<=0`` the full
            signal is returned as a single chunk.
        overlap_sec: Overlap between successive windows in **seconds**. Must be
            strictly smaller than ``chunk_len_sec`` to make progress.

    Returns:
        A list of ``(segment, offset_sec)`` tuples where ``offset_sec`` is the
        starting position of the segment relative to the original audio.

    """
    if chunk_len_sec <= 0 or wav.size == 0:
        return [(wav, 0.0)]

    if overlap_sec < 0:
        raise ValueError("overlap_sec must be >= 0")
    if overlap_sec >= chunk_len_sec:
        raise ValueError("overlap_sec must be < chunk_len_sec")

    window_samples = int(chunk_len_sec * sr)
    step_samples = int(max(chunk_len_sec - overlap_sec, 1) * sr)

    segments: list[tuple[np.ndarray, float]] = []
    for start in range(0, len(wav), step_samples):
        seg = wav[start : start + window_samples]
        if seg.size == 0:
            break
        offset_sec = start / sr
        segments.append((seg, offset_sec))
        if seg.size < window_samples:
            # reached tail; avoid duplicate empty segment due to range()
            break
    return segments
