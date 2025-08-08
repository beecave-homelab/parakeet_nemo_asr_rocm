"""Parakeet NeMo ASR ROCm – Python package init."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parakeet-rocm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.5.2"

__all__ = ["__version__"]
