"""Audio I/O helpers.

Currently provides a single helper to load audio into a float32 numpy array
at a desired sample rate, using *soundfile* and *librosa*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf  # type: ignore
import librosa  # type: ignore

__all__ = ["load_audio"]


DEFAULT_SAMPLE_RATE = 16000


def load_audio(path: Path | str, target_sr: int = DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to *target_sr*.

    Returns
    -------
    audio : np.ndarray
        1-D float32 waveform in range [-1, 1].
    sr : int
        Sample rate after resampling.
    """
    # First, read raw PCM data at native rate
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)  # convert to mono

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr, dtype=np.float32)
        sr = target_sr

    # Ensure float32 dtype and value range
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    return data, sr
