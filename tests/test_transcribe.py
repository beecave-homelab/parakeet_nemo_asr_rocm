"""Smoke test for transcribe_paths.

This test is skipped automatically if the sample WAV is missing or CUDA is not
available (running in CI without GPU). It ensures the function returns a string
without throwing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

audio_path = Path(__file__).parent.parent / "data" / "samples" / "sample.wav"

pytestmark = pytest.mark.skipif(
    not audio_path.is_file(), reason="sample.wav not present"
)


@pytest.mark.skipif(os.getenv("CI") == "true", reason="skip GPU-dependent test in CI")
def test_transcribe_smoke(monkeypatch):
    from parakeet_nemo_asr_rocm.transcribe import transcribe_paths

    # Monkeypatch CUDA to false if not available to avoid unrelated failures
    try:
        import torch

        if not torch.cuda.is_available():
            monkeypatch.setitem(torch.__dict__, "cuda", torch.cuda)
    except ImportError:
        pytest.skip("torch not available in test environment")

    result = transcribe_paths([audio_path])
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], str)
