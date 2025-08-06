"""Utilities for converting word-level timestamps into readability-compliant subtitle segments.

This module implements the main sentence/clause segmentation algorithm that was
previously embedded in ``timestamps/adapt.py``. It now lives in a dedicated
module so that the logic can be unit-tested in isolation and imported by both
the NeMo adaptor and formatting layers.
"""

from __future__ import annotations

from typing import List

from parakeet_nemo_asr_rocm.timestamps.models import Segment, Word
from parakeet_nemo_asr_rocm.utils.constant import (
    DISPLAY_BUFFER_SEC,
    MAX_BLOCK_CHARS,
    MAX_BLOCK_CHARS_SOFT,
    MAX_CPS,
    MAX_LINE_CHARS,
    MAX_SEGMENT_DURATION_SEC,
    MIN_SEGMENT_DURATION_SEC,
)

# Hard and soft limits


def _fix_overlaps(segments: List[Segment]) -> List[Segment]:
    """Trim or merge segments so that start times are monotonically increasing.

    If a segment *i* starts before previous segment *i-1* ends, we shorten
    *i-1*'s end to *i*.start - 0.04 s (40 ms gap).  If that would violate
    *i-1*'s minimum duration, we instead merge the two segments.
    """
    if not segments:
        return segments

    fixed: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = fixed[-1]
        if seg.start < prev.end:
            # Overlap – decide whether to trim prev or merge
            new_prev_end = max(prev.start + MIN_SEGMENT_DURATION_SEC, seg.start - 0.04)
            if (
                new_prev_end - prev.start >= MIN_SEGMENT_DURATION_SEC
                and new_prev_end < seg.start
            ):
                fixed[-1] = prev.copy(update={"end": new_prev_end})
            else:
                # Merge segments
                combined_words = prev.words + seg.words
                combined_text_plain = " ".join(w.word for w in combined_words)
                fixed[-1] = Segment(
                    text=split_lines(combined_text_plain),
                    words=combined_words,
                    start=prev.start,
                    end=max(prev.end, seg.end),
                )
                continue  # Skip adding seg separately
        fixed.append(seg)
    return fixed


def _merge_short_segments(segments: List[Segment]) -> List[Segment]:
    """Merge adjacent *Segment*s that are too short to stand alone.

    Rules for merging `cur` with the following `nxt` segment:
    1. If `cur` duration < MIN_SEGMENT_DURATION_SEC **or** its plain text length < 15 chars.
    2. Combined segment must still respect MAX_BLOCK_CHARS, MAX_SEGMENT_DURATION_SEC and MAX_CPS.
    3. Repeat until `cur` meets limits or no more segments left.
    """
    if not segments:
        return segments

    merged: List[Segment] = []
    i = 0
    while i < len(segments):
        cur = segments[i]

        # Work with plain text (no line breaks)
        def _plain_text(s: Segment) -> str:
            return s.text.replace("\n", " ")

        while (
            (cur.end - cur.start) < MIN_SEGMENT_DURATION_SEC
            or len(_plain_text(cur)) < 15
        ) and i + 1 < len(segments):
            nxt = segments[i + 1]
            combined_words = cur.words + nxt.words
            combined_text_plain = " ".join(w.word for w in combined_words)
            duration = combined_words[-1].end - combined_words[0].start
            cps = len(combined_text_plain) / max(duration, 1e-3)
            if (
                len(combined_text_plain) <= MAX_BLOCK_CHARS
                and duration <= MAX_SEGMENT_DURATION_SEC
                and cps <= MAX_CPS
            ):
                cur = Segment(
                    text=split_lines(combined_text_plain),
                    words=combined_words,
                    start=combined_words[0].start,
                    end=max(nxt.end, combined_words[-1].end),
                )
                i += 1  # Skip over the next segment as it's merged
            else:
                break
        merged.append(cur)
        i += 1
    return merged


HARD_CHAR_LIMIT = MAX_BLOCK_CHARS
SOFT_CHAR_LIMIT = MAX_BLOCK_CHARS_SOFT


def _split_at_clause_boundaries(sentence: List[Word]) -> List[List[Word]]:
    """Split a long sentence at clause boundaries using backtracking.

    This function intelligently splits sentences that exceed limits by:
    1. Identifying clause boundaries (commas, semicolons, colons)
    2. Backtracking to find the best split point that maintains readability
    3. Using fallback strategies when no good clause boundaries exist
    """
    if not sentence:
        return []

    # If already within limits, return as-is
    if _respect_limits(sentence):
        return [sentence]

    # Find all potential clause boundaries
    clause_boundaries = []
    for i, word in enumerate(sentence):
        # Check for clause-ending punctuation
        if word.word.rstrip().endswith((",", ";", ":", "--", "—")):
            # Look for natural break points around this punctuation
            left_context = sentence[: i + 1]
            right_context = sentence[i + 1 :]

            # Only consider if both sides have meaningful content
            left_text = " ".join(w.word for w in left_context).strip()
            right_text = " ".join(w.word for w in right_context).strip()

            if (
                len(left_text) >= 10
                and len(right_text) >= 10
                and _respect_limits(left_context)
            ):
                clause_boundaries.append(i + 1)

    # Try to split at clause boundaries
    if clause_boundaries:
        # Find the boundary closest to the middle for balanced splits
        target_split = len(sentence) // 2
        best_boundary = min(clause_boundaries, key=lambda x: abs(x - target_split))

        left_split = sentence[:best_boundary]
        right_split = sentence[best_boundary:]

        # Recursively split both parts if needed
        result = []
        if left_split:
            result.extend(_split_at_clause_boundaries(left_split))
        if right_split:
            result.extend(_split_at_clause_boundaries(right_split))
        return result

    # No clause boundaries found, use greedy fallback
    return _greedy_split_fallback(sentence)


def _greedy_split_fallback(sentence: List[Word]) -> List[List[Word]]:
    """Fallback splitting strategy when no clause boundaries exist.

    Uses a greedy approach to split at word boundaries while maintaining
    readability constraints.
    """
    if not sentence:
        return []

    splits = []
    current_chunk = [sentence[0]]

    for word in sentence[1:]:
        test_chunk = current_chunk + [word]
        if _respect_limits(test_chunk):
            current_chunk = test_chunk
        else:
            # Found a split point, but check if we can do better
            # by looking for the last space within limits
            if len(current_chunk) > 1:
                splits.append(current_chunk)
                current_chunk = [word]
            else:
                # Single word is too long, just add it
                splits.append(current_chunk)
                current_chunk = [word]

    if current_chunk:
        splits.append(current_chunk)

    return splits


def _eliminate_orphan_words(sentences: List[List[Word]]) -> List[List[Word]]:
    """Post-process sentences to eliminate orphan words.

    Prevents single words or very short phrases from appearing as separate
    segments by intelligently merging them with adjacent segments.
    """
    if len(sentences) <= 1:
        return sentences

    processed = []
    i = 0

    while i < len(sentences):
        current = sentences[i]
        text = " ".join(w.word for w in current).strip()

        # Check if this is an orphan (very short segment)
        is_orphan = len(text) < 15 or len(current) <= 2 or len(text.split()) <= 1

        if is_orphan and i > 0:
            # Try to merge with previous sentence
            previous = processed[-1]
            combined = previous + current

            if _respect_limits(combined):
                processed[-1] = combined
            else:
                processed.append(current)
        elif is_orphan and i == 0 and len(sentences) > 1:
            # First segment is orphan, try to merge with next
            next_segment = sentences[i + 1]
            combined = current + next_segment

            if _respect_limits(combined):
                processed.append(combined)
                i += 1  # Skip the next segment as it's merged
            else:
                processed.append(current)
        else:
            processed.append(current)

        i += 1

    return processed


__all__ = [
    "split_lines",
    "segment_words",
]


def split_lines(text: str) -> str:
    """Split *text* into one or two lines that meet readability constraints.

    Rules:
    1. Prefer a **balanced** break where both lines are <= ``MAX_LINE_CHARS``.
    2. Reject break positions that leave either line *very* short (<25 % of
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
    """Split words into sentences using strong punctuation, with clause boundary awareness.

    This function implements intelligent sentence boundary detection that:
    1. Identifies strong punctuation (., !, ?) as primary sentence boundaries
    2. Uses clause boundaries (commas, semicolons, colons) for backtracking
    3. Prevents orphan words by ensuring meaningful segments
    4. Handles edge cases like trailing fragments
    """
    if not words:
        return []

    sentences: List[List[Word]] = []
    current_sentence: List[Word] = []

    for i, word in enumerate(words):
        current_sentence.append(word)

        # Check for sentence-ending punctuation
        if word.word.rstrip().endswith((".", "!", "?")):
            # Only create a sentence if it's meaningful (not just punctuation)
            if len(current_sentence) > 1 or not word.word.rstrip().endswith(
                (".", "!", "?")
            ):
                sentences.append(current_sentence)
                current_sentence = []

    # Handle any remaining words as a final sentence
    if current_sentence:
        # If we have a very short trailing fragment, consider merging with previous
        # but only if it makes sense linguistically
        text = " ".join(w.word for w in current_sentence)
        if len(text.strip()) < 10 and sentences:
            # Check if we can merge without violating limits
            last_sentence = sentences[-1]
            combined_text = " ".join(w.word for w in last_sentence + current_sentence)
            combined_duration = current_sentence[-1].end - last_sentence[0].start
            combined_chars = len(combined_text)

            # Only merge if it doesn't violate basic constraints
            if (
                combined_chars <= MAX_BLOCK_CHARS
                and combined_duration <= MAX_SEGMENT_DURATION_SEC
            ):
                sentences[-1].extend(current_sentence)
            else:
                sentences.append(current_sentence)
        else:
            sentences.append(current_sentence)

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

    # Sentence split and fix overly long sentences with intelligent backtracking
    sentences_fixed: List[List[Word]] = []
    for sentence in _sentence_chunks(words):
        text = " ".join(w.word for w in sentence)
        duration = sentence[-1].end - sentence[0].start
        char_count = len(text)

        # Accept sentence immediately if it meets all constraints and isn't too short
        if _respect_limits(sentence) and char_count >= 15 and duration >= 1.0:
            sentences_fixed.append(sentence)
            continue

        # For overly long sentences, use intelligent clause boundary detection
        if char_count > HARD_CHAR_LIMIT or duration > MAX_SEGMENT_DURATION_SEC:
            # First, try to split at clause boundaries with backtracking
            clause_splits = _split_at_clause_boundaries(sentence)
            sentences_fixed.extend(clause_splits)
        else:
            # Sentence is short enough but might be too brief - keep as-is
            sentences_fixed.append(sentence)

    # Post-process to eliminate orphan words
    sentences_fixed = _eliminate_orphan_words(sentences_fixed)

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
    # Post-pass: merge any adjacent segments that are still too short
    # Merge tiny cues
    segments = _merge_short_segments(segments)

    # Ensure timestamps are strictly monotonic and non-overlapping
    segments = _fix_overlaps(segments)
    return segments
