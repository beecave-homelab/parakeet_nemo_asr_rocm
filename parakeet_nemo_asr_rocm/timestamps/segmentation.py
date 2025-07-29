"""Utilities for converting word-level timestamps into readability-compliant
subtitle segments.

This module implements the main sentence/clause segmentation algorithm that
was previously embedded in `timestamps/adapt.py`. It now lives in a dedicated
module so that the logic can be unit-tested in isolation and imported by
both the NeMo adaptor and formatting layers.
"""

from __future__ import annotations

from typing import List

from parakeet_nemo_asr_rocm.timestamps.models import Segment, Word
from parakeet_nemo_asr_rocm.utils.constant import (
    DISPLAY_BUFFER_SEC,
    MAX_CPS,
    MAX_LINE_CHARS,
    MAX_SEGMENT_DURATION_SEC,
    MIN_SEGMENT_DURATION_SEC,
    MAX_BLOCK_CHARS,
    MAX_BLOCK_CHARS_SOFT,
)

# Hard and soft limits
HARD_CHAR_LIMIT = MAX_BLOCK_CHARS
SOFT_CHAR_LIMIT = MAX_BLOCK_CHARS_SOFT

__all__ = [
    "split_lines",
    "segment_words",
]


def split_lines(text: str) -> str:
    """Split *text* into one or two lines that meet readability constraints.

    Rules:
    1. Prefer a **balanced** break where both lines are <= ``MAX_LINE_CHARS``.
    2. Reject break positions that leave either line *very* short (\<25 % of
       ``MAX_LINE_CHARS`` **or** fewer than 10 characters). This avoids
       captions that end with a dangling word such as ``"The"``.
    3. Fall back to a greedy split just before the limit if no balanced break
       fulfils the minimum-length requirement.
    """

    if len(text) <= MAX_LINE_CHARS:
        return text

    # Minimum length any line should have to be considered acceptable.
    _MIN_LINE_LEN: int = max(10, int(MAX_LINE_CHARS * 0.25))

    best_split: tuple[str, str] | None = None
    best_delta = 10**9

    for idx, char in enumerate(text):
        if char != " ":
            continue
        line1, line2 = text[:idx].strip(), text[idx + 1 :].strip()
        # Hard limits
        if len(line1) > MAX_LINE_CHARS or len(line2) > MAX_LINE_CHARS:
            continue
        # Reject lines that are too short – avoids "orphan" second lines
        if len(line1) < _MIN_LINE_LEN or len(line2) < _MIN_LINE_LEN:
            continue
        delta = abs(len(line1) - len(line2))
        if delta < best_delta:
            best_delta, best_split = delta, (line1, line2)
            if delta == 0:
                break  # cannot get better than perfect balance

    if not best_split:
        # Greedy fallback: cut as late as possible while keeping first line
        # within the limit, ensuring the second line is not empty.
        first_break = text.rfind(" ", 0, MAX_LINE_CHARS)
        if first_break == -1 or first_break == len(text) - 1:
            # No space found or would create empty second line – force split.
            first_break = MAX_LINE_CHARS
        best_split = text[:first_break].strip(), text[first_break:].strip()

    return "\n".join(best_split)


def _respect_limits(words: List[Word], *, soft: bool = False) -> bool:
    """Return True if *words* obey character count, duration and CPS limits.

    If *soft* is True, the softer char limit is used to allow slight overflow
    when merging already-readable sentences.
    """
    text_plain = " ".join(w.word for w in words)
    chars = len(text_plain)
    dur = words[-1].end - words[0].start
    cps = chars / max(dur, 1e-3)
    char_limit = SOFT_CHAR_LIMIT if soft else HARD_CHAR_LIMIT
    return chars <= char_limit and dur <= MAX_SEGMENT_DURATION_SEC and cps <= MAX_CPS


def _sentence_chunks(words: List[Word]) -> List[List[Word]]:
    sent_acc: List[Word] = []
    sentences: List[List[Word]] = []
    for w in words:
        sent_acc.append(w)
        if w.word.endswith((".", "!", "?")):
            sentences.append(sent_acc)
            sent_acc = []
    if sent_acc:
        sentences.append(sent_acc)
    return sentences


def segment_words(words: List[Word]) -> List[Segment]:
    """Convert raw word list into a list of subtitle *Segment*s.

    The algorithm applies a *sentence-first, clause-aware* strategy:
    1. Split words into sentences using strong punctuation.
    2. Any sentence violating hard limits is further split by clause commas.
    3. Remaining violations trigger a greedy word grouping fallback.
    4. Adjacent sentences are merged while combined block still satisfies
       all limits.
    """

    if not words:
        return []

    # Sentence split and fix overly long sentences
    sentences_fixed: List[List[Word]] = []
    for sentence in _sentence_chunks(words):
        # Accept sentence immediately only if it isn't extremely short
        if _respect_limits(sentence) and not (
            len(" ".join(w.word for w in sentence)) < 15
            and (sentence[-1].end - sentence[0].start) < 1.0
        ):
            sentences_fixed.append(sentence)
            continue
        # clause-level split
        clause: List[Word] = []
        for w in sentence:
            clause.append(w)
            if w.word.endswith((",", ";", ":")) and _respect_limits(clause):
                sentences_fixed.append(clause)
                clause = []
        # greedy fallback on the remainder
        if clause:
            greedy: List[Word] = []
            for w in clause:
                if greedy and not _respect_limits(greedy + [w]):
                    sentences_fixed.append(greedy)
                    greedy = [w]
                else:
                    greedy.append(w)
            if greedy:
                sentences_fixed.append(greedy)

    # Merge consecutive sentences when possible
    captions: List[List[Word]] = []
    current: List[Word] = []
    for sent in sentences_fixed:
        if not current:
            current = sent
            continue
        if _respect_limits(current + sent, soft=True):
            current += sent
        else:
            captions.append(current)
            current = sent
    if current:
        captions.append(current)

    # Convert to Segment objects
    segments: List[Segment] = []
    for cap in captions:
        text_plain = " ".join(w.word for w in cap)
        start_time = cap[0].start
        natural_end = cap[-1].end + DISPLAY_BUFFER_SEC
        # Stretch caption to minimum display duration if it is too short
        end_time = max(natural_end, start_time + MIN_SEGMENT_DURATION_SEC)
        segments.append(
            Segment(
                text=split_lines(text_plain),
                words=cap,
                start=start_time,
                end=end_time,
            )
        )
    return segments
