"""Parakeet NeMo ASR ROCm – Python package init."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parakeet-nemo-asr-rocm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.2.1"

__all__ = ["__version__"]
