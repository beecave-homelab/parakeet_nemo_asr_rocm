"""
Formatter for plain text (.txt) output.
"""

from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult


def to_txt(result: AlignedResult) -> str:
    """
    Converts AlignedResult to a plain text string.

    Args:
        result: The AlignedResult object containing transcription segments.

    Returns:
        A single string containing the concatenated text of all segments.
    """
    return " ".join([segment.text for segment in result.segments])
