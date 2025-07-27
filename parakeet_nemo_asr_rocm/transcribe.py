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
from .utils.constant import DEFAULT_CHUNK_LEN_SEC

__all__ = ["transcribe_paths", "main"]


def _chunks(seq: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _segment_waveform(wav: "np.ndarray", sr: int, chunk_len_sec: int) -> List["np.ndarray"]:
    """Split mono waveform into equal-length chunks."""
    import numpy as np
    if chunk_len_sec <= 0:
        return [wav]
    max_samples = chunk_len_sec * sr
    segments = [wav[i : i + max_samples] for i in range(0, len(wav), max_samples)]
    return segments


def transcribe_paths(paths: Sequence[Path], batch_size: int = 1, *, chunk_len: int = DEFAULT_CHUNK_LEN_SEC) -> List[str]:
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

    import numpy as np
    # Load and, if necessary, split each audio into smaller chunks
    segmented_lists: List[List[np.ndarray]] = []
    for p in paths:
        wav, _sr = load_audio(p, DEFAULT_SAMPLE_RATE)
        segments = _segment_waveform(wav, _sr, chunk_len)
        segmented_lists.append(segments)

    # Flatten for batching
    flat_wavs: List[np.ndarray] = [seg for segments in segmented_lists for seg in segments]
    seg_counts = [len(segs) for segs in segmented_lists]

    transcribed_flat: List[str] = []
    for batch_wavs in _chunks(flat_wavs, batch_size):
        with torch.inference_mode():
            for _h in model.transcribe(batch_wavs, batch_size=len(batch_wavs)):
                text = _h if isinstance(_h, str) else getattr(_h, "text", str(_h))
                transcribed_flat.append(text)

    # Re-assemble per original file
    idx = 0
    for count in seg_counts:
        joined = " ".join(transcribed_flat[idx : idx + count])
        results.append(joined)
        idx += count

    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Batch transcribe audio files using Parakeet-TDT 0.6B v2 (NeMo).",
    )
    parser.add_argument("audio", nargs="+", type=Path, help="Audio file(s) to transcribe")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--chunk-len", type=int, default=20, help="Segment length in seconds before transcription")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    transcripts = transcribe_paths(
        args.audio,
        batch_size=args.batch_size,
        chunk_len=args.chunk_len,
    )
    for path, text in zip(args.audio, transcripts, strict=True):
        print(f"{path}: {text}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
