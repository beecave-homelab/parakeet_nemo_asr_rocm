"""Project-wide constants for convenient reuse."""

from __future__ import annotations

import pathlib
from typing import Final


# Repository root resolved relative to this file (utils/constant.py → package → repo)
REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[2]

# Default path of the dotenv file containing runtime overrides
ENV_FILE: Final[pathlib.Path] = REPO_ROOT / ".env"
