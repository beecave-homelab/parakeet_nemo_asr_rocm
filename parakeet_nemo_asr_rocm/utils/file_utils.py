"""File utilities for handling output file naming and overwrite protection."""

from __future__ import annotations

import pathlib
from typing import Union

PathLike = Union[str, pathlib.Path]


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
