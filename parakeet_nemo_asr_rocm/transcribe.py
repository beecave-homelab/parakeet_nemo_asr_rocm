"""Batch transcription helper functions.

Designed to be imported *and* run as a script via ``python -m
parakeet_nemo_asr_rocm.transcribe <audio files>``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

from .models.parakeet import get_model
from .utils.audio_io import DEFAULT_SAMPLE_RATE, load_audio

__all__ = ["transcribe_paths", "main"]


def _chunks(seq: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def transcribe_paths(paths: Sequence[Path], batch_size: int = 1) -> List[str]:
    """Transcribe a list of audio file paths.

    Parameters
    ----------
    paths : Sequence[Path]
        Audio file paths.
    batch_size : int, default=1
        Batch size for inference. GPU memory usage scales roughly linearly with
        this value.

    Returns
    -------
    List[str]
        Transcribed text for each input path (same order).
    """
    # Eager-load model (cached)
    model = get_model()
    device = next(model.parameters()).device  # type: ignore

    results: List[str] = []

    # Pre-load audio into memory (mono float32). This guarantees the waveform
    # shape is always (time,) even if the source file is stereo, preventing the
    # output-shape mismatch we observed when passing file paths directly.
    waveforms = [load_audio(p, DEFAULT_SAMPLE_RATE)[0] for p in paths]

    # Feed numpy arrays directly to NeMo; each element is a 1-D waveform.
    for batch_wavs in _chunks(waveforms, batch_size):
        with torch.inference_mode():
            batch_out = model.transcribe(batch_wavs, batch_size=len(batch_wavs))
        results.extend(batch_out)

    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Batch transcribe audio files using Parakeet-TDT 0.6B v2 (NeMo).",
    )
    parser.add_argument("audio", nargs="+", type=Path, help="Audio file(s) to transcribe")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    transcripts = transcribe_paths(args.audio, batch_size=args.batch_size)
    for path, text in zip(args.audio, transcripts, strict=True):
        print(f"{path}: {text}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
