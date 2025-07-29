"""Utility for loading project-level environment variables.

The file uses `python-dotenv` (if present) to load a `.env` file sitting at
repository root *early* in the application lifecycle so that downstream
imports (PyTorch / NeMo) pick up any relevant flags such as
`PYTORCH_HIP_ALLOC_CONF`.

Usage (call as soon as possible in your CLI / entry-point):

    from parakeet_nemo_asr_rocm.utils.env_loader import load_project_env
    load_project_env()

Re-invocation is a no-op, so callers can safely call multiple times.
"""

from __future__ import annotations

import functools
import os
import pathlib
from typing import Final

try:
    # `python-dotenv` provides load_dotenv helper. It is an optional dep – we
    # degrade gracefully if missing.
    from dotenv import load_dotenv as LOAD_DOTENV  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    LOAD_DOTENV = None  # type: ignore


_REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[2]
_ENV_FILE: Final[pathlib.Path] = _REPO_ROOT / ".env"


@functools.lru_cache(maxsize=1)
def load_project_env(force: bool = False) -> None:
    """Load the project-level `.env` file into the process environment.

    This function is decorated with `lru_cache` to ensure it runs only once.
    It attempts to load environment variables from a `.env` file at the
    repository root. If `python-dotenv` is installed, it is used; otherwise,
    a simple manual parser is used as a fallback.

    Args:
        force: If True, bypasses the cache and forces a reload of the
            environment file. Defaults to False.
    """
    if force:
        load_project_env.cache_clear()  # type: ignore[attr-defined]

    if not _ENV_FILE.exists():
        # Nothing to load – silently return.
        return

    if LOAD_DOTENV is not None:
        # `override=False` ensures we do **not** clobber env-vars already set
        # by the user / shell.
        LOAD_DOTENV(dotenv_path=_ENV_FILE, override=False)
    else:  # pragma: no cover
        # Manual fallback – parse simple KEY=VALUE lines.
        with _ENV_FILE.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip() or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                os.environ.setdefault(key, value)


__all__ = [
    "load_project_env",
]
