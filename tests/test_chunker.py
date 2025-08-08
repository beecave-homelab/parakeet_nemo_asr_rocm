"""Unit tests for the waveform chunking utility.

These tests validate basic segmentation, input validation, and full-signal
handling for `segment_waveform`.
"""

import numpy as np
import pytest

from parakeet_nemo_asr_rocm.chunking.chunker import segment_waveform


def test_segment_waveform_basic() -> None:
    """Regular overlapping chunks should be produced at correct offsets.

    Returns:
        None: This is a pytest test function.
    """

    wav = np.arange(10, dtype=np.float32)
    segs = segment_waveform(wav, sr=1, chunk_len_sec=4, overlap_sec=2)
    assert segs[0][1] == 0.0
    assert segs[1][1] == 2.0
    assert len(segs) == 5


def test_segment_waveform_invalid_overlap() -> None:
    """Invalid overlap values should raise ``ValueError``.

    Returns:
        None: This is a pytest test function.
    """

    wav = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError):
        segment_waveform(wav, sr=1, chunk_len_sec=2, overlap_sec=2)
    with pytest.raises(ValueError):
        segment_waveform(wav, sr=1, chunk_len_sec=2, overlap_sec=-1)


def test_segment_waveform_full_signal() -> None:
    """Zero chunk length should return the full signal as one segment.

    Returns:
        None: This is a pytest test function.
    """

    wav = np.arange(3, dtype=np.float32)
    segs = segment_waveform(wav, sr=1, chunk_len_sec=0)
    assert segs == [(wav, 0.0)]
