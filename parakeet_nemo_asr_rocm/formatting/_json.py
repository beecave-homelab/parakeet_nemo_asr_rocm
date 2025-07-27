"""
Formatter for JSON (.json) output.
"""

from parakeet_nemo_asr_rocm.timestamps.adapt import AlignedResult


def to_json(result: AlignedResult) -> str:
    """
    Converts AlignedResult to a JSON string.

    Args:
        result: The AlignedResult object.

    Returns:
        A JSON string representation of the AlignedResult.
    """
    return result.model_dump_json(indent=2)
