"""Project-wide constants for convenient reuse."""

from __future__ import annotations

import os
import pathlib
from typing import Final

from parakeet_nemo_asr_rocm.utils.env_loader import load_project_env

# Ensure .env is loaded exactly once at import time for the whole project
load_project_env()


# Repository root resolved relative to this file (utils/constant.py → package → repo)
REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[2]

# Default path of the dotenv file containing runtime overrides
ENV_FILE: Final[pathlib.Path] = REPO_ROOT / ".env"

# Default audio chunk length (seconds) used for segmented inference.
# Can be overridden by CHUNK_LEN_SEC env var.
DEFAULT_CHUNK_LEN_SEC: Final[int] = int(os.getenv("CHUNK_LEN_SEC", "300"))

# Default low-latency chunk length for *pseudo-streaming* mode.
DEFAULT_STREAM_CHUNK_SEC: Final[int] = int(os.getenv("STREAM_CHUNK_SEC", "8"))

# Default batch size for model inference
DEFAULT_BATCH_SIZE: Final[int] = int(os.getenv("BATCH_SIZE", "12"))

# Prefer FFmpeg for audio decoding (1 = yes, 0 = try soundfile first)
FORCE_FFMPEG: Final[bool] = os.getenv("FORCE_FFMPEG", "1") == "1"

# Subtitle readability constraints (industry-standard defaults)
MAX_CPS: Final[float] = float(
    os.getenv("MAX_CPS", "17")
)  # characters per second upper bound
MIN_CPS: Final[float] = float(
    os.getenv("MIN_CPS", "12")
)  # lower bound (rarely enforced)
MAX_LINE_CHARS: Final[int] = int(os.getenv("MAX_LINE_CHARS", "42"))
MAX_LINES_PER_BLOCK: Final[int] = int(os.getenv("MAX_LINES_PER_BLOCK", "2"))
DISPLAY_BUFFER_SEC: Final[float] = float(
    os.getenv("DISPLAY_BUFFER_SEC", "0.2")
)  # trailing buffer after last word
MAX_SEGMENT_DURATION_SEC: Final[float] = float(
    os.getenv("MAX_SEGMENT_DURATION_SEC", "5.5")
)
MIN_SEGMENT_DURATION_SEC: Final[float] = float(
    os.getenv("MIN_SEGMENT_DURATION_SEC", "1.2")
)

# Subtitle punctuation boundaries
BOUNDARY_CHARS: Final[str] = os.getenv("BOUNDARY_CHARS", ".?!…")
CLAUSE_CHARS: Final[str] = os.getenv("CLAUSE_CHARS", ",;:")

# Soft boundary keywords (lowercase) treated as optional breakpoints
SOFT_BOUNDARY_WORDS: Final[tuple[str, ...]] = tuple(
    w.strip().lower()
    for w in os.getenv(
        "SOFT_BOUNDARY_WORDS", "and,but,that,which,who,where,when,while,so"
    ).split(",")
)

# Interjection whitelist allowing stand-alone short cues
INTERJECTION_WHITELIST: Final[tuple[str, ...]] = tuple(
    w.strip().lower()
    for w in os.getenv("INTERJECTION_WHITELIST", "whoa,wow,what,oh,hey,ah").split(",")
)

# Caption block character limits
# Hard limit (two full lines using MAX_LINE_CHARS) unless overridden
MAX_BLOCK_CHARS: Final[int] = int(
    os.getenv(
        "MAX_BLOCK_CHARS",
        str(MAX_LINE_CHARS * MAX_LINES_PER_BLOCK),
    )
)
# Softer limit used when evaluating potential merges; allows slight overflow
MAX_BLOCK_CHARS_SOFT: Final[int] = int(os.getenv("MAX_BLOCK_CHARS_SOFT", "90"))

# Legacy caption segmentation thresholds (kept for backward compatibility)
SEGMENT_MAX_GAP_SEC: Final[float] = float(os.getenv("SEGMENT_MAX_GAP_SEC", "1.0"))
SEGMENT_MAX_DURATION_SEC: Final[float] = MAX_SEGMENT_DURATION_SEC  # alias
SEGMENT_MAX_WORDS: Final[int] = int(os.getenv("SEGMENT_MAX_WORDS", "40"))

# Logging configuration
NEMO_LOG_LEVEL: Final[str] = os.getenv("NEMO_LOG_LEVEL", "ERROR")
TRANSFORMERS_VERBOSITY: Final[str] = os.getenv("TRANSFORMERS_VERBOSITY", "ERROR")
