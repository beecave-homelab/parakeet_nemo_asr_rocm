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
from pathlib import Path
from typing import Sequence

from parakeet_nemo_asr_rocm.transcribe import transcribe_paths
from parakeet_nemo_asr_rocm.utils.constant import (DEFAULT_BATCH_SIZE,
                                                   DEFAULT_CHUNK_LEN_SEC)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the transcription script.

    Args:
        argv: A sequence of strings representing the command-line arguments.
            Defaults to None, which makes argparse use `sys.argv`.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="parakeet_nemo_asr_rocm",
        description=(
            "Transcribe audio files with Parakeet-TDT 0.6B v2 (NeMo) on AMD ROCm GPUs."
        ),
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
        "--chunk-len-sec",
        dest="chunk_len_sec",
        type=int,
        default=DEFAULT_CHUNK_LEN_SEC,
        help="Segment length in seconds for chunked inference (overridden by CHUNK_LEN_SEC env)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    """Main entry point for the script.

    Parses command-line arguments and transcribes the specified audio files,
    printing the results to standard output.

    Args:
        argv: A sequence of strings representing the command-line arguments.
            Defaults to None, which makes argparse use `sys.argv`.
    """
    args = _parse_args(argv)
    transcripts = transcribe_paths(
        args.audio,
        batch_size=args.batch_size,
        chunk_len_sec=args.chunk_len_sec,
    )
    for path, text in zip(args.audio, transcripts, strict=True):
        print(f"{path}: {text}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
