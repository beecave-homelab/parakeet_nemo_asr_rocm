"""Project-wide constants for convenient reuse."""

from __future__ import annotations

import pathlib
import os

# Ensure .env is loaded exactly once at import time for the whole project
from .env_loader import load_project_env  # local import to avoid cycles
load_project_env()
from typing import Final


# Repository root resolved relative to this file (utils/constant.py → package → repo)
REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[2]

# Default path of the dotenv file containing runtime overrides
ENV_FILE: Final[pathlib.Path] = REPO_ROOT / ".env"

# Default audio chunk length (seconds) used for streaming/segmented inference.
# Can be overridden by setting CHUNK_LEN_SEC in the environment (e.g. in .env file).
DEFAULT_CHUNK_LEN_SEC: Final[int] = int(os.getenv("CHUNK_LEN_SEC", "20"))

# Default batch size for model inference
DEFAULT_BATCH_SIZE: Final[int] = int(os.getenv("BATCH_SIZE", "1"))
