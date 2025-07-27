"""Console-script entry point (exposed as *parakeet-nemo-asr-rocm*).

This thin wrapper just forwards to :pyfunc:`parakeet_nemo_asr_rocm.app.main` so
that users can either run the module or the installed script.
"""

from __future__ import annotations

import sys
from typing import Sequence

from parakeet_nemo_asr_rocm.app import main as _module_main


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    """Entry point for the console script.

    This function serves as the entry point for the `parakeet-nemo-asr-rocm`
    console script defined in `pyproject.toml`. It forwards the command-line
    arguments to the main application logic.

    Args:
        argv: A sequence of strings representing the command-line arguments.
            If None, `sys.argv[1:]` is used.
    """
    _module_main(argv or sys.argv[1:])
