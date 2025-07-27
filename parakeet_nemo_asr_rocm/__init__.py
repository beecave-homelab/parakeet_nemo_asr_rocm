"""Parakeet NeMo ASR ROCm – Python package init."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("parakeet-nemo-asr-rocm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.2.0"

__all__ = ["__version__"]
