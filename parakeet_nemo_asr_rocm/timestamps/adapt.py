"""
This module provides functions for adapting NeMo's timestamped ASR output
into a standardized format.

The goal is to create a common data structure (`AlignedResult`) that can be used
by various formatters (e.g., SRT, VTT, JSON) regardless of the specifics of the
ASR model's output.
"""

from __future__ import annotations

from typing import List

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult, Segment, Word
from parakeet_nemo_asr_rocm.timestamps.segmentation import segment_words, split_lines

from .word_timestamps import get_word_timestamps


def adapt_nemo_hypotheses(
    hypotheses: List[Hypothesis], model: ASRModel, time_stride: float | None = None
) -> AlignedResult:
    """Converts a list of NeMo Hypothesis objects into a standardized AlignedResult."""
    word_timestamps = get_word_timestamps(hypotheses, model, time_stride)

    if not word_timestamps:
        return AlignedResult(segments=[], word_segments=[])

        # Group words into segments for SRT/VTT. Strategy:
    # ---------------------------------------------
    # Readability-aware segmentation implementation
    # ---------------------------------------------
    from parakeet_nemo_asr_rocm.utils.constant import (
        DISPLAY_BUFFER_SEC,
        MAX_CPS,
        MAX_LINE_CHARS,
        MAX_LINES_PER_BLOCK,
        MAX_SEGMENT_DURATION_SEC,
        MIN_SEGMENT_DURATION_SEC,
    )

    MAX_BLOCK_CHARS = MAX_LINE_CHARS * MAX_LINES_PER_BLOCK

    # Use new sentence-aware segmentation implementation
    segments_raw = segment_words(word_timestamps)

    def _is_clause_boundary(word: Word) -> bool:
        """Return True if *word* ends with punctuation that signals a clause break."""
        return word.word.endswith((",", ";", ":", ".", "?", "!"))

    def _segment_words(words: list[Word]) -> list[Segment]:
        """(Deprecated) Wrapper forwarding to new segmentation module."""
        return segment_words(words)
        """LEGACY SEGMENTATION CODE BELOW (disabled)"""
        """Segment *words* into readability-compliant caption blocks following
        a sentence-first strategy.

        Steps:
        1. Split the word list into *sentences* ending with strong punctuation
           (., !, ?). If a sentence itself violates hard limits, back-off to
           clause boundaries (commas / semicolons). If still too long, fall
           back to greedy word grouping respecting the limits.
        2. Combine consecutive sentences into a caption while combined text
           still obeys all limits (chars, CPS, duration).
        """

        if not words:
            return []

        def _sentence_chunks(ws: list[Word]) -> list[list[Word]]:
            sent_acc: list[Word] = []
            sents: list[list[Word]] = []
            for w in ws:
                sent_acc.append(w)
                if w.word.endswith((".", "!", "?")):
                    sents.append(sent_acc)
                    sent_acc = []
            # leftover
            if sent_acc:
                sents.append(sent_acc)
            return sents

        def _respect_limits(wlist: list[Word]) -> bool:
            txt = " ".join(w.word for w in wlist)
            chars = len(txt)
            dur = wlist[-1].end - wlist[0].start
            cps = chars / max(dur, 1e-3)
            return (
                chars <= MAX_BLOCK_CHARS
                and dur <= MAX_SEGMENT_DURATION_SEC
                and cps <= MAX_CPS
            )

        # 1. initial sentence split
        sentences = _sentence_chunks(words)

        # 1b. back-off: fix sentences that break limits
        fixed_sentences: list[list[Word]] = []
        for s in sentences:
            if _respect_limits(s):
                fixed_sentences.append(s)
                continue
            # try clause split by ',' ';'
            clause: list[Word] = []
            for w in s:
                clause.append(w)
                if w.word.endswith((",", ";", ":")) and _respect_limits(clause):
                    fixed_sentences.append(clause)
                    clause = []
            # add remainder greedily
            if clause:
                # fallback: greedy chunk obeying limits
                greedy: list[Word] = []
                for w in clause:
                    if greedy and not _respect_limits(greedy + [w]):
                        fixed_sentences.append(greedy)
                        greedy = [w]
                    else:
                        greedy.append(w)
                if greedy:
                    fixed_sentences.append(greedy)
        sentences = fixed_sentences

        # 2. build captions by merging sentences while within limits
        captions: list[list[Word]] = []
        current: list[Word] = []
        for sent in sentences:
            if not current:
                current = sent
                continue
            if _respect_limits(current + sent):
                current += sent
            else:
                captions.append(current)
                current = sent
        if current:
            captions.append(current)

        # Convert to Segment dataclass list
        result: list[Segment] = []
        for cap in captions:
            text_plain = " ".join(w.word for w in cap)
            result.append(
                Segment(
                    text=split_lines(text_plain),
                    words=cap,
                    start=cap[0].start,
                    end=cap[-1].end + DISPLAY_BUFFER_SEC,
                )
            )
        return result

    # Post-processing adjustments
    # -----------------------------
    POST_GAP_SEC = 0.05  # minimal gap to keep between captions
    MIN_CHARS_FOR_STANDALONE = 15

    merged: list[Segment] = []
    i = 0
    while i < len(segments_raw):
        seg = segments_raw[i]
        chars = len(seg.text.replace("\n", " "))
        dur = seg.end - seg.start
        # Criteria for merging: too short & too few chars
        if i + 1 < len(segments_raw) and (
            dur < MIN_SEGMENT_DURATION_SEC or chars < MIN_CHARS_FOR_STANDALONE
        ):
            nxt = segments_raw[i + 1]
            # Merge seg + nxt
            merged_words = seg.words + nxt.words
            merged_text = split_lines(" ".join(w.word for w in merged_words))
            seg = Segment(
                text=merged_text,
                words=merged_words,
                start=seg.start,
                end=nxt.end,
            )
            i += 2  # skip next
        else:
            i += 1
        merged.append(seg)

    # Fix overlaps due to display buffer
    for j in range(len(merged) - 1):
        cur = merged[j]
        nxt = merged[j + 1]
        if cur.end + POST_GAP_SEC > nxt.start:
            cur_end_new = max(cur.start + 0.2, nxt.start - POST_GAP_SEC)
            merged[j] = cur.copy(update={"end": cur_end_new})

    # ---------------------------------
    # Forward-merge small leading words
    # ---------------------------------
    def _can_append(prev: Segment, word: Word) -> bool:
        new_text = prev.text.replace("\n", " ") + " " + word.word
        if len(new_text) > MAX_BLOCK_CHARS:
            return False
        duration = word.end - prev.start
        cps = len(new_text) / max(duration, 1e-3)
        return cps <= MAX_CPS and duration <= MAX_SEGMENT_DURATION_SEC

    k = 0
    while k < len(merged) - 1:
        prev = merged[k]
        nxt = merged[k + 1]
        # only attempt if next caption starts with 1 short word (<5 chars)
        first_word = nxt.words[0]
        # Only move small leading words if the previous caption does *not* already
        # end with sentence‐terminating punctuation. This prevents orphan words
        # starting a new sentence (e.g. "The", "Just") from being attached to
        # the preceding caption.
        if (
            len(first_word.word) <= 5
            and not prev.text.strip().endswith((".", "!", "?"))
            and _can_append(prev, first_word)
        ):
            # move word from nxt to prev
            updated_prev_words = prev.words + [first_word]
            updated_prev_text = split_lines(
                " ".join(w.word for w in updated_prev_words)
            )
            merged[k] = prev.copy(
                update={
                    "words": updated_prev_words,
                    "text": updated_prev_text,
                    "end": first_word.end,
                }
            )
            # trim next
            trimmed_words = nxt.words[1:]
            if not trimmed_words:
                merged.pop(k + 1)
                continue  # re-evaluate same k
            trimmed_text = split_lines(" ".join(w.word for w in trimmed_words))
            merged[k + 1] = nxt.copy(
                update={
                    "words": trimmed_words,
                    "text": trimmed_text,
                    "start": trimmed_words[0].start,
                }
            )
        k += 1

    # ---------------------------------
    # Merge tiny leading captions entirely when possible
    m = 0
    while m < len(merged) - 1:
        cur = merged[m]
        nxt = merged[m + 1]
        first_line = nxt.text.split("\n", 1)[0]
        if len(first_line) <= 12 or len(first_line.split()) <= 2:
            combined_words = cur.words + nxt.words
            combined_text_plain = " ".join(w.word for w in combined_words)
            duration = combined_words[-1].end - combined_words[0].start
            cps = len(combined_text_plain) / max(duration, 1e-3)
            if (
                len(combined_text_plain) <= MAX_BLOCK_CHARS
                and duration <= MAX_SEGMENT_DURATION_SEC
                and cps <= MAX_CPS
            ):
                cur = cur.copy(
                    update={
                        "words": combined_words,
                        "text": split_lines(combined_text_plain),
                        "end": nxt.end,
                    }
                )
                merged[m] = cur
                merged.pop(m + 1)
                continue
        m += 1

    # Ensure segments end with punctuation
    # ---------------------------------
    j = 0
    while j < len(merged) - 1:
        cur = merged[j]
        nxt = merged[j + 1]
        if not cur.text.strip().endswith((".", "!", "?")):
            combined_words = cur.words + nxt.words
            combined_text_plain = " ".join(w.word for w in combined_words)
            if (
                len(combined_text_plain) <= MAX_BLOCK_CHARS
                and (combined_words[-1].end - combined_words[0].start)
                <= MAX_SEGMENT_DURATION_SEC
                and (
                    len(combined_text_plain)
                    / max(combined_words[-1].end - combined_words[0].start, 1e-3)
                )
                <= MAX_CPS
            ):
                cur = cur.copy(
                    update={
                        "words": combined_words,
                        "text": split_lines(combined_text_plain),
                        "end": nxt.end,
                    }
                )
                merged[j] = cur
                merged.pop(j + 1)
                continue  # re-evaluate merged cur with following
        j += 1

    return AlignedResult(segments=merged, word_segments=word_timestamps)
