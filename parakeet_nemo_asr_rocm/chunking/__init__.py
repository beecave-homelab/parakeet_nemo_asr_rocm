"""Chunking and merging utilities for long-form ASR.

This module provides tools for splitting long audio into overlapping chunks and
merging the transcription results back together while handling overlaps.
"""

from .chunker import segment_waveform
from .merge import (
    merge_longest_common_subsequence,
    merge_longest_contiguous,
)

__all__ = [
    "segment_waveform",
    "merge_longest_contiguous",
    "merge_longest_common_subsequence",
]
