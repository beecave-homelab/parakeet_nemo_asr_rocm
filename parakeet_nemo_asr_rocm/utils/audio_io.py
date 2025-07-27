"""Audio I/O helpers.

Currently provides a single helper to load audio into a float32 numpy array
at a desired sample rate, using *soundfile* and *librosa*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa  # type: ignore
import numpy as np
import soundfile as sf  # type: ignore
from pydub import AudioSegment  # type: ignore

__all__ = ["load_audio"]


DEFAULT_SAMPLE_RATE = 16000


def _load_with_pydub(path: Path | str) -> Tuple[np.ndarray, int]:
    """Fallback loader using pydub/ffmpeg for formats unsupported by soundfile.

    Args:
        path: The path to the audio file.

    Returns:
        A tuple containing:
        - data: Mono float32 waveform in range [-1, 1].
        - sr: Native sample rate of the decoded audio.
    """
    # pydub loads into AudioSegment (16-bit PCM by default)
    seg: AudioSegment = AudioSegment.from_file(path)
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples())
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)
    # Convert from int16 range to [-1, 1] float32
    data = (samples.astype(np.float32) / (1 << 15)).clip(-1.0, 1.0)
    return data, sr


def load_audio(
    path: Path | str, target_sr: int = DEFAULT_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to a target sample rate.

    Args:
        path: The path to the audio file.
        target_sr: The target sample rate to resample the audio to. Defaults
            to `DEFAULT_SAMPLE_RATE`.

    Returns:
        A tuple containing:
        - audio: A 1-D float32 waveform in the range [-1, 1].
        - sr: The sample rate after resampling (equal to `target_sr`).
    """
    # Attempt fast path via soundfile (libsndfile). This covers WAV/FLAC/OGG…
    try:
        data, sr = sf.read(str(path), always_2d=False)
    except (RuntimeError, sf.LibsndfileError):  # unsupported format
        data, sr = _load_with_pydub(path)

    # Ensure mono
    if data.ndim > 1:
        data = np.mean(data, axis=-1)  # convert to mono

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr, dtype=np.float32)
        sr = target_sr

    # Ensure float32 dtype and value range
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    return data, sr
