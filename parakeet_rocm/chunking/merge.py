"""Token-based merging utilities for overlapping ASR chunks.

These helpers are inspired by the MLX implementation and operate purely on
`Word` objects as defined in :pymod:`parakeet_rocm.timestamps.models`.
They are deliberately backend-agnostic so they can be unit-tested without
requiring NeMo or GPU dependencies.

Two strategies are provided:

* ``merge_longest_contiguous`` - fast heuristic that keeps tokens from the first
  chunk up to the midpoint of the *time* overlap and tokens from the second
  chunk after that point. It assumes that token order is correct and simply
  removes duplicates in the overlap zone.
* ``merge_longest_common_subsequence`` - closer match to MLX logic; performs an
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
from collections.abc import Callable
from statistics import median

from parakeet_rocm.timestamps.models import Word

__all__ = [
    "merge_longest_contiguous",
    "merge_longest_common_subsequence",
    "MERGE_STRATEGIES",
]


def _normalise(text: str) -> str:
    """Normalise token text for matching.

    The text is prepared by trimming whitespace, converting to lowercase, and
    removing punctuation.

    Returns:
        str: The normalised text with leading/trailing whitespace removed, all
            characters lowercased, and punctuation characters removed.
    """
    text = text.strip().lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def merge_longest_contiguous(
    a: list[Word],
    b: list[Word],
    *,
    overlap_duration: float,
) -> list[Word]:
    """Merge two chronologically-sorted word lists at the overlap midpoint.

    This function resolves any temporal overlap at its midpoint. It assumes that
    token order is correct and simply removes duplicates in the overlap zone.

    Parameters:
        a (list[Word]): First (earlier) token sequence, expected pre-sorted
            by time.
        b (list[Word]): Second (later) token sequence, expected pre-sorted
            by time.
        overlap_duration (float): Duration in seconds of the overlap between
            the two parent chunks.

    Returns:
        list[Word]: Merged sequence of ``Word`` objects combining the first
            chunk up to the overlap midpoint and the second chunk from the
            midpoint onward.
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

    merged: list[Word] = [w for w in a if w.end <= cutoff]
    merged.extend(w for w in b if w.start >= cutoff)

    return merged


def _shift_words(words: list[Word], offset: float) -> list[Word]:
    """Create new word objects whose times are shifted by a given offset.

    This function creates new ``Word`` objects with start and end times
    adjusted by the provided offset. The original objects are not modified.

    Parameters:
        words (list[Word]): Sequence of ``Word`` objects to copy and shift.
        offset (float): Seconds to add to each word's start and end times
            (can be negative).

    Returns:
        list[Word]: New ``Word`` objects with start and end times adjusted by
            ``offset``.
    """
    return [
        Word(word=w.word, start=w.start + offset, end=w.end + offset, score=w.score) for w in words
    ]


def _estimate_time_offset_from_lcs(
    overlap_a: list[Word],
    overlap_b: list[Word],
    lcs_pairs: list[tuple[int, int]],
) -> float:
    """Estimate robust time offset between overlapping token sequences.

    The merge pipeline uses LCS matches as temporal anchors to shift chunk ``b``
    onto chunk ``a``. In repetitive phrases, the first LCS match can be an
    ambiguous anchor and produce an over-aggressive shift that accumulates across
    many chunk boundaries. This helper computes per-anchor offsets and returns a
    robust median-based estimate.

    Parameters:
        overlap_a (list[Word]): Overlap slice from the earlier chunk.
        overlap_b (list[Word]): Overlap slice from the later chunk.
        lcs_pairs (list[tuple[int, int]]): Matched index pairs `(ia, ib)` in
            overlap-local coordinates.

    Returns:
        float: Estimated offset (seconds) to add to chunk ``b`` timestamps.
    """
    if not lcs_pairs:
        return 0.0

    offsets = [overlap_a[ia].start - overlap_b[ib].start for ia, ib in lcs_pairs]
    if len(offsets) == 1:
        return offsets[0]

    offset_median = float(median(offsets))
    abs_deviations = [abs(value - offset_median) for value in offsets]
    mad = float(median(abs_deviations))

    # Keep anchors close to the median. Use a small floor so the filter still
    # works when MAD is ~0 due to quantized timestamps.
    tolerance = max(mad * 3.0, 0.05)
    filtered_offsets = [value for value in offsets if abs(value - offset_median) <= tolerance]
    if not filtered_offsets:
        return offset_median
    return float(median(filtered_offsets))


def merge_longest_common_subsequence(
    a: list[Word],
    b: list[Word],
    *,
    overlap_duration: float,
) -> list[Word]:
    """Merge two token sequences using a time-tolerant LCS on text.

    This function finds tokens in the temporal overlap between the two sequences,
    computes an LCS over normalised token text to identify matching tokens, aligns
    the second sequence to the first using a robust median-based offset from all
    LCS anchors (filtering outliers via MAD), and stitches the sequences by
    preferring the longer token gaps between matches. If no LCS anchors are found
    within the overlap, the function falls back to a midpoint-based contiguous
    merge.

    Parameters:
        a (list[Word]): First (earlier) token sequence, expected pre-sorted
            by time.
        b (list[Word]): Second (later) token sequence, expected pre-sorted
            by time.
        overlap_duration (float): Temporal tolerance in seconds used to
            expand the overlap window when selecting tokens for LCS matching.

    Returns:
        list[Word]: A new merged token list. Original input lists are not
            mutated; shared tokens are taken from ``a``, and tokens from
            ``b`` may be time-shifted to align with ``a``.
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
    # Time-drift correction: estimate offset from all LCS anchors and shift
    # **all** tokens coming from chunk *b* using a robust median estimate.
    # ------------------------------------------------------------------
    time_offset = _estimate_time_offset_from_lcs(overlap_a, overlap_b, lcs_pairs)

    # Shift complete *b* list (non-mutating)
    b_shifted = _shift_words(b, time_offset)

    # Map overlapping indices back to original arrays (for a and shifted b)
    a_offset = len(a) - len(overlap_a)
    lcs_indices_a = [a_offset + p[0] for p in lcs_pairs]
    lcs_indices_b = [p[1] for p in lcs_pairs]

    merged: list[Word] = []
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


# Registry of available merge strategies. The callable signature includes the
# keyword-only ``overlap_duration`` parameter for type checkers.
MERGE_STRATEGIES: dict[str, Callable[[list[Word], list[Word], float], list[Word]]] = {
    "contiguous": merge_longest_contiguous,
    "lcs": merge_longest_common_subsequence,
}
