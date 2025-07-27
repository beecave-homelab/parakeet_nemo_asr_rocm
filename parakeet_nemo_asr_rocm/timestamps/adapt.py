"""
This module provides functions for adapting NeMo's timestamped ASR output
into a standardized format.

The goal is to create a common data structure (`AlignedResult`) that can be used
by various formatters (e.g., SRT, VTT, JSON) regardless of the specifics of the
ASR model's output.
"""

from typing import List, Optional

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from pydantic import BaseModel, Field


class Word(BaseModel):
    """Represents a single word with timing information."""

    word: str = Field(..., description="The transcribed word.")
    start: float = Field(..., description="Start time of the word in seconds.")
    end: float = Field(..., description="End time of the word in seconds.")
    score: Optional[float] = Field(None, description="Confidence score of the word.")


class Segment(BaseModel):
    """Represents a segment of transcription, containing multiple words."""

    text: str = Field(..., description="The full text of the segment.")
    words: List[Word] = Field(..., description="A list of timed words in the segment.")
    start: float = Field(..., description="Start time of the segment in seconds.")
    end: float = Field(..., description="End time of the segment in seconds.")


class AlignedResult(BaseModel):
    """Represents the full, timestamp-aligned transcription result."""

    segments: List[Segment] = Field(..., description="A list of transcribed segments.")
    word_segments: List[Word] = Field(
        ..., description="A flat list of all words across all segments."
    )


from .word_timestamps import get_word_timestamps

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from .word_timestamps import get_word_timestamps

def adapt_nemo_hypotheses(
    hypotheses: List[Hypothesis], model: ASRModel, time_stride: float | None = None
) -> AlignedResult:
    """Converts a list of NeMo Hypothesis objects into a standardized AlignedResult."""
    word_timestamps = get_word_timestamps(hypotheses, model, time_stride)

    if not word_timestamps:
        return AlignedResult(segments=[], word_segments=[])

        # Group words into segments for SRT/VTT. Strategy:
    # – start new segment if time gap ≥ 1.0s between consecutive words
    # – or segment duration exceeds 6 s
    # – or more than 40 words
    from parakeet_nemo_asr_rocm.utils.constant import (
    SEGMENT_MAX_GAP_SEC,
    SEGMENT_MAX_DURATION_SEC,
    SEGMENT_MAX_WORDS,
)

    def _group_words(words, max_gap=SEGMENT_MAX_GAP_SEC, max_duration=SEGMENT_MAX_DURATION_SEC, max_words=SEGMENT_MAX_WORDS):
        segments: list[Segment] = []
        cur: list[Word] = []
        for w in words:
            if not cur:
                cur.append(w)
                continue
            gap = w.start - cur[-1].end
            duration = w.end - cur[0].start
            if gap >= max_gap or duration >= max_duration or len(cur) >= max_words:
                seg_text = " ".join([cw.word for cw in cur])
                segments.append(
                    Segment(text=seg_text, words=cur, start=cur[0].start, end=cur[-1].end)
                )
                cur = [w]
            else:
                cur.append(w)
        if cur:
            seg_text = " ".join([cw.word for cw in cur])
            segments.append(
                Segment(text=seg_text, words=cur, start=cur[0].start, end=cur[-1].end)
            )
        return segments
    segments = _group_words(word_timestamps)
    return AlignedResult(segments=segments, word_segments=word_timestamps)
