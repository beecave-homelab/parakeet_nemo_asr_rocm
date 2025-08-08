"""Unit tests for overlapping token merging utilities.

These tests cover both contiguous and LCS-based merging to ensure
chronological ordering and duplicate removal in overlapped regions.
"""

from __future__ import annotations

from parakeet_nemo_asr_rocm.chunking import (
    merge_longest_common_subsequence,
    merge_longest_contiguous,
)
from parakeet_nemo_asr_rocm.timestamps.models import Word


def _make_overlap_samples() -> tuple[list[Word], list[Word]]:
    """Construct two overlapping word lists with duplicate tokens.

    Returns:
        tuple[list[Word], list[Word]]: Two sequences ``a`` and ``b`` where
        ``b`` starts within ``a`` to create an overlap region.
    """

    a = [
        Word(word="Hello", start=0.0, end=0.3),
        Word(word="world", start=0.3, end=0.6),
        Word(word="this", start=0.6, end=0.9),
        Word(word="is", start=0.9, end=1.2),
    ]
    # Second chunk starts inside first (~0.6 s overlap)
    b = [
        Word(word="this", start=0.6, end=0.9),
        Word(word="is", start=0.9, end=1.2),
        Word(word="a", start=1.2, end=1.4),
        Word(word="test", start=1.4, end=1.6),
    ]
    return a, b


def test_merge_longest_contiguous() -> None:
    """Contiguous merge should preserve order and maintain monotonic timing.

    Returns:
        None: This is a pytest test function.
    """
    a, b = _make_overlap_samples()
    merged = merge_longest_contiguous(a, b, overlap_duration=0.6)
    # Check that basic sequence ordering is chronological and no missing words
    words = [w.word for w in merged]
    assert words[0:2] == ["Hello", "world"]
    assert words[-2:] == ["a", "test"]
    # Ensure timeline monotonicity
    for prev, nxt in zip(merged, merged[1:]):
        assert prev.end <= nxt.start + 1e-6


def test_merge_longest_common_subsequence() -> None:
    """LCS merge should deduplicate overlap and keep natural order.

    Returns:
        None: This is a pytest test function.
    """
    a, b = _make_overlap_samples()
    merged = merge_longest_common_subsequence(a, b, overlap_duration=0.6)
    words = [w.word for w in merged]
    assert words == ["Hello", "world", "this", "is", "a", "test"]
    # Ensure no duplicates
    assert len(words) == len(set(words)), "Duplicates found in merged words"
