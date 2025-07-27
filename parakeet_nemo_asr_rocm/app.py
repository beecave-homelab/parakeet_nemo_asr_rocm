"""Package module entry point.

Allows running `python -m parakeet_nemo_asr_rocm.app <audio files>` to quickly
transcribe one or more WAV files. Internally delegates to
:pyfunc:`parakeet_nemo_asr_rocm.transcribe.transcribe_paths`.

This keeps runtime dependencies minimal (only argparse). CLI helpers are in
``cli.py`` which is exposed as a console_script via ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import sys

from .utils.constant import DEFAULT_CHUNK_LEN_SEC, DEFAULT_BATCH_SIZE
from pathlib import Path
from typing import Sequence

from .transcribe import transcribe_paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="parakeet_nemo_asr_rocm",
        description="Transcribe audio files with Parakeet-TDT 0.6B v2 (NeMo) on AMD ROCm GPUs.",
    )
    parser.add_argument(
        "audio",
        nargs="+",
        type=Path,
        help="Path(s) to audio files (wav/flac/ogg etc.) to transcribe.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=DEFAULT_CHUNK_LEN_SEC,
        help="Segment length in seconds for chunked inference (overridden by CHUNK_LEN_SEC env)",
    )
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
