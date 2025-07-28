"""Common data models for timestamped transcription results.

This module defines pydantic models that are shared across timestamp
processing, segmentation and formatting utilities.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "Word",
    "Segment",
    "AlignedResult",
]


class Word(BaseModel):
    """Represents a single word with timing information."""

    word: str = Field(..., description="The transcribed word.")
    start: float = Field(..., description="Start time of the word in seconds.")
    end: float = Field(..., description="End time of the word in seconds.")
    score: Optional[float] = Field(
        None, description="Optional confidence score of the word."
    )


class Segment(BaseModel):
    """A subtitle / caption segment consisting of multiple *Word*s."""

    text: str = Field(..., description="Rendered text (may contain line breaks).")
    words: List[Word] = Field(..., description="Ordered list of words in the segment.")
    start: float = Field(..., description="Segment start time (seconds).")
    end: float = Field(..., description="Segment end time (seconds).")


class AlignedResult(BaseModel):
    """Full timestamp-aligned transcription result."""

    segments: List[Segment] = Field(..., description="List of caption segments.")
    word_segments: List[Word] = Field(
        ..., description="Flat list of *all* words across *segments*."
    )
