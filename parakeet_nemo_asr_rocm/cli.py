"""Console-script entry point (exposed as *parakeet-nemo-asr-rocm*).

This thin wrapper just forwards to :pyfunc:`parakeet_nemo_asr_rocm.app.main` so
that users can either run the module or the installed script.
"""
from __future__ import annotations

import sys
from typing import Sequence

from .app import main as _module_main


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    """Entry point used in *pyproject.toml* [project.scripts]."""
    _module_main(argv or sys.argv[1:])
