"""
Formatter for Web Video Text Tracks format (.vtt).
"""

import math

from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult


def _format_timestamp(seconds: float) -> str:
    """Converts seconds to VTT timestamp format (HH:MM:SS.ms)."""
    assert seconds >= 0, "non-negative timestamp required"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int(math.modf(s)[0] * 1000):03d}"


def to_vtt(result: AlignedResult, highlight_words: bool = False) -> str:
    """
    Converts AlignedResult to a VTT formatted string.

    Args:
        result: The AlignedResult object containing transcription segments.

    Returns:
        A string in VTT format.
    """
    vtt_lines = ["WEBVTT", ""]
    for segment in result.segments:
        start_time = _format_timestamp(segment.start)
        end_time = _format_timestamp(segment.end)
        vtt_lines.append(f"{start_time} --> {end_time}")
        if highlight_words:
            text = " ".join(f"<c.highlight>{w.word}</c.highlight>" for w in segment.words)
        else:
            text = segment.text.strip()
        vtt_lines.append(text)
        vtt_lines.append("")  # Add a blank line between entries
    return "\n".join(vtt_lines)
