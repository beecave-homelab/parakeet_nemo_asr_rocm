"""Unit tests for timestamps segmentation core.

Covers `split_lines()` and `segment_words()` against readability constraints.
"""
from __future__ import annotations

from typing import List

import pytest

from parakeet_nemo_asr_rocm.timestamps.models import Segment, Word
from parakeet_nemo_asr_rocm.timestamps.segmentation import (
    segment_words,
    split_lines,
)
from parakeet_nemo_asr_rocm.utils.constant import (
    MAX_CPS,
    MAX_LINE_CHARS,
    MAX_SEGMENT_DURATION_SEC,
)


def _w(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end)


def test_split_lines_balanced_and_within_limit() -> None:
    text = "This is a reasonably long line that should be split in two parts."
    out = split_lines(text)
    parts = out.split("\n")
    assert 1 <= len(parts) <= 2
    # Ensure each line respects MAX_LINE_CHARS
    assert all(len(p) <= MAX_LINE_CHARS for p in parts)


def test_segment_words_single_sentence_ok_limits() -> None:
    # A sentence that should stay one segment and respect limits
    words: List[Word] = [
        _w("This", 0.00, 0.20),
        _w("is", 0.21, 0.35),
        _w("fine", 0.36, 0.80),
        _w("enough.", 0.81, 1.40),
    ]
    segs = segment_words(words)
    assert len(segs) == 1
    seg = segs[0]
    plain = seg.text.replace("\n", " ")
    dur = seg.end - seg.start
    cps = len(plain) / max(dur, 1e-3)
    assert cps <= MAX_CPS
    assert dur <= MAX_SEGMENT_DURATION_SEC


def test_segment_words_splits_long_sentence() -> None:
    # Construct a long sentence that exceeds character limit, forcing a split
    t = 0.0
    words: List[Word] = []
    for token in (
        "This", "sentence", "is", "intentionally", "made", "very",
        "very", "long", "to", "exceed", "the", "character", "limits,",
        "so", "that", "the", "algorithm", "needs", "to", "split",
        "at", "clause", "boundaries", "and", "fallback", "as",
        "needed."
    ):
        words.append(_w(token, t, t + 0.15))
        t += 0.18
    segs = segment_words(words)
    # Expect >1 segments due to length/duration
    assert len(segs) >= 2
    for seg in segs:
        plain = seg.text.replace("\n", " ")
        dur = seg.end - seg.start
        # Duration within hard cap
        assert dur <= MAX_SEGMENT_DURATION_SEC
        # Each line within MAX_LINE_CHARS
        for line in seg.text.split("\n"):
            assert len(line) <= MAX_LINE_CHARS
