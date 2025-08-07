"""Token-based merging utilities for overlapping ASR chunks.

These helpers are inspired by the MLX implementation and operate purely on
`Word` objects as defined in :pymod:`parakeet_nemo_asr_rocm.timestamps.models`.
They are deliberately backend-agnostic so they can be unit-tested without
requiring NeMo or GPU dependencies.

Two strategies are provided:

* ``merge_longest_contiguous`` – fast heuristic that keeps tokens from the first
  chunk up to the midpoint of the *time* overlap and tokens from the second
  chunk after that point. It assumes that token order is correct and simply
  removes duplicates in the overlap zone.
* ``merge_longest_common_subsequence`` – closer match to MLX logic; performs an
  LCS on the *text* (case-insensitive) of the tokens inside the overlap window
  and combines gaps by preferring the longer gap segment.

Both functions accept two pre-sorted token lists and the *overlap duration* in
seconds. They return a **new** merged list (original inputs remain unmodified).

.. note::
   The NeMo RNNT Hypothesis tokens are wordpiece-level and may split words. We
   therefore match on the *normalised token text* rather than integer IDs.
"""

from __future__ import annotations

import string
from typing import List

from parakeet_nemo_asr_rocm.timestamps.models import Word

__all__ = [
    "merge_longest_contiguous",
    "merge_longest_common_subsequence",
]


def _normalise(text: str) -> str:
    """Normalise token text for matching.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Lower-case.
    3. Remove punctuation characters so that minor punctuation differences do not
       break longest-common-subsequence matching.
    """
    text = text.strip().lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def merge_longest_contiguous(
    a: List[Word],
    b: List[Word],
    *,
    overlap_duration: float,
) -> List[Word]:
    """Merge two token lists using midpoint split within the overlap.

    Parameters
    ----------
    a, b
        Chronologically sorted token lists from successive chunks.
    overlap_duration
        How many **seconds** of audio overlapped between the two parent chunks.

    """
    if not a:
        return b.copy()
    if not b:
        return a.copy()

    # Determine time boundaries
    a_end = a[-1].end
    b_start = b[0].start

    # Non-overlapping fast-path
    if b_start >= a_end:
        return a + b

    # Midpoint of overlapping area
    cutoff = (a_end + b_start) / 2.0

    merged: List[Word] = [w for w in a if w.end <= cutoff]
    merged.extend(w for w in b if w.start >= cutoff)

    return merged


def _shift_words(words: List[Word], offset: float) -> List[Word]:
    """Return **new** Word objects with start/end shifted by *offset* seconds."""
    return [
        Word(word=w.word, start=w.start + offset, end=w.end + offset, score=w.score)
        for w in words
    ]


def merge_longest_common_subsequence(
    a: List[Word],
    b: List[Word],
    *,
    overlap_duration: float,
) -> List[Word]:
    """Merge two token sequences via time-tolerant LCS.

    This is a simplified, text-based adaptation of the MLX implementation. The
    algorithm:
    1. Identify the sub-lists of *a* and *b* that fall inside the temporal
       overlap window.
    2. Compute an LCS matrix on normalised token text.
    3. Walk back the matrix to get matching indices, then stitch the sequences
       keeping the longer of each gap.
    """
    if not a:
        return b.copy()
    if not b:
        return a.copy()

    a_end_time = a[-1].end
    b_start_time = b[0].start

    # If no temporal overlap just concatenate
    if b_start_time >= a_end_time:
        return a + b

    # Extract overlapping slices
    overlap_a = [t for t in a if t.start >= b_start_time - overlap_duration]
    overlap_b = [t for t in b if t.end <= a_end_time + overlap_duration]

    # Build LCS DP table on token text
    m, n = len(overlap_a), len(overlap_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if _normalise(overlap_a[i].word) == _normalise(overlap_b[j].word):
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    # Recover LCS indices
    i = j = 0
    lcs_pairs: list[tuple[int, int]] = []
    while i < m and j < n:
        if _normalise(overlap_a[i].word) == _normalise(overlap_b[j].word):
            lcs_pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1

        # If LCS is empty fall back to midpoint heuristic
    if not lcs_pairs:
        return merge_longest_contiguous(a, b, overlap_duration=overlap_duration)

    # ------------------------------------------------------------------
    # Time-drift correction: compute offset so that the *first* LCS pair
    # aligns perfectly, then shift **all** tokens coming from chunk *b*.
    # ------------------------------------------------------------------
    first_a_idx_overlap, first_b_idx_overlap = lcs_pairs[0]
    anchor_a_time = overlap_a[first_a_idx_overlap].start
    anchor_b_time = overlap_b[first_b_idx_overlap].start
    time_offset = anchor_a_time - anchor_b_time

    # Shift complete *b* list (non-mutating)
    b_shifted = _shift_words(b, time_offset)

    # Map overlapping indices back to original arrays (for a and shifted b)
    a_offset = len(a) - len(overlap_a)
    lcs_indices_a = [a_offset + p[0] for p in lcs_pairs]
    lcs_indices_b = [p[1] for p in lcs_pairs]

    merged: List[Word] = []
    merged.extend(a[: lcs_indices_a[0]])

    for idx in range(len(lcs_pairs)):
        ia = lcs_indices_a[idx]
        ib = lcs_indices_b[idx]
        merged.append(a[ia])  # shared token (use *a* instance to keep original attrs)

        if idx < len(lcs_pairs) - 1:
            next_ia = lcs_indices_a[idx + 1]
            next_ib = lcs_indices_b[idx + 1]

            gap_a = a[ia + 1 : next_ia]
            gap_b = b_shifted[ib + 1 : next_ib]
            merged.extend(gap_a if len(gap_a) >= len(gap_b) else gap_b)

    merged.extend(b_shifted[lcs_indices_b[-1] + 1 :])
    return merged
