"""File utilities for handling output file naming, overwrite protection, and
helper functions for resolving audio input paths / patterns.

All functions are type-hinted and documented. The module exposes:
• `get_unique_filename` – unchanged
• `resolve_input_paths` – expand wildcard patterns / directories into concrete paths
• `AUDIO_EXTENSIONS` – set of allowed audio filename extensions
"""

from __future__ import annotations

import pathlib
from glob import glob
from typing import Iterable, List, Sequence, Union


PathLike = Union[str, pathlib.Path]

# Supported audio file extensions (lower-case, dot-prefixed)
# Extended set covering common audio and video containers/codecs that ffmpeg can
# usually decode. Keeping it modest in size avoids pathological glob scans while
# ensuring typical user files are accepted.
AUDIO_EXTENSIONS: set[str] = {
    # audio
    ".wav",
    ".mp3",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".m4a",
    ".wma",
    ".aiff",
    ".alac",
    ".amr",
    # video
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".flv",
    ".ts",
}

__all__ = [
    "AUDIO_EXTENSIONS",
    "get_unique_filename",
    "resolve_input_paths",
]


def get_unique_filename(
    base_path: PathLike,
    overwrite: bool = False,
    separator: str = "-",
) -> pathlib.Path:
    """Generate a unique filename to avoid overwriting existing files.

    If the file does not exist or overwrite is True, returns the original path.
    Otherwise, appends a numbered suffix like " -1", " -2", etc.

    Args:
        base_path: The desired file path.
        overwrite: If True, return the original path even if it exists.
        separator: The separator to use before the number suffix.

    Returns:
        A pathlib.Path that is guaranteed not to exist (unless overwrite=True).
    """
    path = pathlib.Path(base_path)

    if overwrite or not path.exists():
        return path

    # Find the next available number
    counter = 1
    while True:
        new_name = f"{path.stem}{separator}{counter}{path.suffix}"
        new_path = path.parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1

        # Safety check to prevent infinite loops
        if counter > 9999:
            raise RuntimeError(f"Cannot find unique filename for {base_path}")


def _is_audio_file(
    path: pathlib.Path, exts: Sequence[str] | set[str] | None = None
) -> bool:  # noqa: D401
    """Return *True* if *path* points to a supported audio file.

    Args:
        path: File path to test.
        exts: Optional iterable of file extensions to accept; defaults to
            :data:`AUDIO_EXTENSIONS`.
    """
    _exts = set(ext.lower() for ext in (exts or AUDIO_EXTENSIONS))
    return path.is_file() and path.suffix.lower() in _exts


def resolve_input_paths(
    patterns: Iterable[PathLike] | PathLike,
    *,
    audio_exts: Sequence[str] | set[str] | None = None,
    recursive: bool = True,
) -> List[pathlib.Path]:
    """Expand wildcard *patterns* / directories into concrete audio file paths.

    This helper centralises glob logic so that both the CLI's *audio_files*
    argument and the new ``--watch`` feature share identical expansion rules.

    Behaviour:
        • A *pattern* may be a file path, directory, or shell wildcard (`*.wav`).
        • Directories are scanned (recursively by default) for supported audio
          files.
        • Duplicates are removed while preserving order.
        • Non-existent paths produce an empty match (they are silently skipped).

    Args:
        patterns: String or Path (or iterable of) to resolve.
        audio_exts: Allowed extensions; defaults to :data:`AUDIO_EXTENSIONS`.
        recursive: If *True*, directory search is recursive.

    Returns:
        Sorted list (by insertion order, duplicates removed) of ``pathlib.Path``
        objects that exist and match the extension filter.
    """

    if isinstance(patterns, (str, pathlib.Path)):
        patterns = [patterns]

    _exts = set(ext.lower() for ext in (audio_exts or AUDIO_EXTENSIONS))

    resolved: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()

    def _add(p: pathlib.Path):
        if p not in seen and _is_audio_file(p, _exts):
            seen.add(p)
            resolved.append(p)

    for patt in patterns:
        p = pathlib.Path(patt).expanduser()
        if p.is_dir():
            # Walk directory
            walker = p.rglob("*") if recursive else p.glob("*")
            for child in walker:
                _add(child)
        else:
            # Use glob for wildcard expansion; if no wildcard, treat as literal
            matches = glob(str(p), recursive=True)
            for m in matches:
                _add(pathlib.Path(m))
    return resolved
