"""Formatter for SubRip Subtitle format (.srt)."""

import math

from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,ms)."""
    assert seconds >= 0, "non-negative timestamp required"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(math.modf(s)[0] * 1000):03d}"


def to_srt(result: AlignedResult, highlight_words: bool = False) -> str:
    """Convert an ``AlignedResult`` to an SRT formatted string.

    Args:
        result: The AlignedResult object containing transcription segments.
        highlight_words: If ``True``, wrap each word in ``<b>`` tags for emphasis.

    Returns:
        A string in SRT format.

    """
    srt_lines = []
    for i, segment in enumerate(result.segments, start=1):
        start_time = _format_timestamp(segment.start)
        end_time = _format_timestamp(segment.end)
        srt_lines.append(str(i))
        srt_lines.append(f"{start_time} --> {end_time}")
        if highlight_words:
            # Bold each word for simple emphasis
            text = " ".join(f"<b>{w.word}</b>" for w in segment.words)
        else:
            text = segment.text.strip()
        srt_lines.append(text)
        srt_lines.append("")  # Add a blank line between entries
    return "\n".join(srt_lines)
