"""
This module provides a registry of output formatters for transcriptions.

It allows for easy extension with new formats by adding a new formatter function
and registering it in the `FORMATTERS` dictionary.
"""

from typing import Callable, Dict

from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult

from ._json import to_json
from ._srt import to_srt
from ._txt import to_txt
from ._vtt import to_vtt

# A registry mapping format names to their respective formatter functions.
FORMATTERS: Dict[str, Callable[[AlignedResult], str]] = {
    "txt": to_txt,
    "json": to_json,
    "srt": to_srt,
    "vtt": to_vtt,
}


def get_formatter(format_name: str) -> Callable[[AlignedResult], str]:
    """
    Retrieves the formatter function for a given format name.

    Args:
        format_name: The name of the format (e.g., 'txt', 'json').

    Returns:
        The corresponding formatter function.

    Raises:
        ValueError: If the format_name is not supported.
    """
    formatter = FORMATTERS.get(format_name.lower())
    if not formatter:
        raise ValueError(
            f"Unsupported format: '{format_name}'. Supported formats are: {list(FORMATTERS.keys())}"
        )
    return formatter
