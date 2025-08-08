"""Unit tests covering audio loading backends and fallbacks.

These tests validate ffmpeg, pydub, and soundfile code paths for
``load_audio`` and helpers, ensuring correct sampling rate handling and
dtype.
"""

import numpy as np
import pytest

from parakeet_nemo_asr_rocm.utils import audio_io


def test_load_with_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """ffmpeg path should decode to int16 PCM and cast to float32.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching modules.

    Returns:
        None: This is a pytest test function.
    """

    monkeypatch.setattr(audio_io.shutil, "which", lambda cmd: "/usr/bin/ffmpeg")

    class _Result:
        stdout = np.array([0, 32767], dtype=np.int16).tobytes()

    monkeypatch.setattr(
        audio_io.subprocess, "run", lambda cmd, capture_output, check: _Result()
    )
    data, sr = audio_io._load_with_ffmpeg("dummy", 16000)
    assert sr == 16000 and isinstance(data, np.ndarray)


def test_load_with_pydub(monkeypatch: pytest.MonkeyPatch) -> None:
    """pydub path should return float32 numpy array and sample rate.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching modules.

    Returns:
        None: This is a pytest test function.
    """

    class _Seg:
        frame_rate = 8000
        channels = 1

        def get_array_of_samples(self):
            return [0, 1]

    monkeypatch.setattr(audio_io.AudioSegment, "from_file", lambda p: _Seg())
    data, sr = audio_io._load_with_pydub("x")
    assert sr == 8000 and data.dtype == np.float32


def test_load_audio_soundfile(monkeypatch: pytest.MonkeyPatch) -> None:
    """soundfile path should resample to target SR using librosa.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching modules.

    Returns:
        None: This is a pytest test function.
    """

    monkeypatch.setattr(audio_io, "FORCE_FFMPEG", False)
    monkeypatch.setattr(
        audio_io.sf,
        "read",
        lambda p, always_2d=False: (np.array([0.0, 0.5], dtype=np.float32), 8000),
    )
    called = {}

    def _resample(data, orig_sr, target_sr, dtype):
        called["resample"] = True
        return data

    monkeypatch.setattr(audio_io.librosa, "resample", _resample)
    data, sr = audio_io.load_audio("f", target_sr=16000)
    assert sr == 16000 and called["resample"]


def test_load_audio_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """ffmpeg failure should fall back to pydub and resample to target SR.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching modules.

    Returns:
        None: This is a pytest test function.
    """

    monkeypatch.setattr(audio_io, "FORCE_FFMPEG", True)

    def _ffmpeg_fail(path, sr):
        raise RuntimeError("fail")

    monkeypatch.setattr(audio_io, "_load_with_ffmpeg", _ffmpeg_fail)
    monkeypatch.setattr(audio_io.sf, "read", lambda *a, **k: (_ffmpeg_fail("", 0)))

    def _pydub(path):
        return (np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32), 8000)

    monkeypatch.setattr(audio_io, "_load_with_pydub", _pydub)
    monkeypatch.setattr(
        audio_io.librosa, "resample", lambda d, orig_sr, target_sr, dtype: d
    )

    data, sr = audio_io.load_audio("f", target_sr=16000)
    assert data.shape == (2, 2) or data.shape == (2,) and sr == 16000
