"""
This module provides the command-line interface for the Parakeet-NEMO ASR application,
built using the Typer library.

It is designed to replace the older `argparse`-based CLI with a more robust and
user-friendly interface that supports subcommands, rich help messages, and better
argument handling.

Features:
- `transcribe` command for running ASR on audio files.
- Options for model selection, output formatting, and batch processing.
- Verbose mode for detailed logging.
"""

import pathlib
from typing import List

import typer
from typing_extensions import Annotated

from parakeet_nemo_asr_rocm import __version__
from parakeet_nemo_asr_rocm.utils.constant import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_LEN_SEC,
)


# Create the main Typer application instance
def version_callback(value: bool):
    if value:
        print(f"parakeet-rocm version: {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="parakeet-rocm",
    help="A CLI for transcribing audio files using NVIDIA Parakeet-TDT via NeMo on ROCm.",
    add_completion=False,
)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the application's version and exit.",
            callback=version_callback,
            is_eager=True,
            is_flag=True,
        ),
    ] = False,
):
    """
    Manage parakeet-rocm commands.
    """


@app.command()
def transcribe(
    audio_files: Annotated[
        List[pathlib.Path],
        typer.Argument(
            help="Path to one or more audio files to transcribe.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    model_name: Annotated[
        str,
        typer.Option(
            "--model", help="Hugging Face Hub or local path to the NeMo ASR model."
        ),
    ] = "nvidia/parakeet-tdt-0.6b-v2",
    output_dir: Annotated[
        pathlib.Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save the transcription outputs.",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = "./output",
    output_format: Annotated[
        str,
        typer.Option(help="Format for the output file(s) (e.g., txt, srt, vtt, json)."),
    ] = "txt",
    output_template: Annotated[
        str,
        typer.Option(
            help=(
                "Template for output filenames. "
                "Supports placeholders: {parent}, {filename}, {index}, {date}."
            ),
        ),
    ] = "{filename}",
    batch_size: Annotated[
        int, typer.Option(help="Batch size for transcription inference.")
    ] = DEFAULT_BATCH_SIZE,
    chunk_len_sec: Annotated[
        int,
        typer.Option(help="Segment length in seconds for chunked transcription."),
    ] = DEFAULT_CHUNK_LEN_SEC,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            help="Enable pseudo-streaming mode (low-latency small chunks)",
        ),
    ] = False,
    stream_chunk_sec: Annotated[
        int,
        typer.Option(
            "--stream-chunk-sec",
            help="Chunk length in seconds when --stream is enabled (overrides default).",
        ),
    ] = 0,
    overlap_duration: Annotated[
        int,
        typer.Option(
            "--overlap-duration",
            help="Overlap between consecutive chunks in seconds (for long audio).",
        ),
    ] = 15,
    highlight_words: Annotated[
        bool,
        typer.Option(
            "--highlight-words",
            help="Highlight each word in SRT/VTT outputs (e.g., bold).",
        ),
    ] = False,
    word_timestamps: Annotated[
        bool,
        typer.Option(
            "--word-timestamps",
            help="Enable word-level timestamp generation.",
        ),
    ] = False,
    merge_strategy: Annotated[
        str,
        typer.Option(
            "--merge-strategy",
            help="Strategy for merging overlapping chunks: 'none' (concatenate), 'contiguous' (fast merge), or 'lcs' (accurate, default)",
            case_sensitive=False,
        ),
    ] = "lcs",
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite existing output files instead of appending numbered suffixes.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Enable verbose output.",
        ),
    ] = False,
    no_progress: Annotated[
        bool,
        typer.Option(
            "--no-progress",
            help="Disable the Rich progress bar (silent mode).",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            help="Suppress console messages except the progress bar and final output.",
        ),
    ] = False,
    fp32: Annotated[
        bool,
        typer.Option(
            "--fp32",
            help="Force full-precision (FP32) inference. Default if no precision flag is provided.",
        ),
    ] = False,
    fp16: Annotated[
        bool,
        typer.Option(
            "--fp16",
            help="Enable half-precision (FP16) inference for faster processing on compatible hardware.",
        ),
    ] = False,
):
    """
    Transcribe one or more audio files using the specified NVIDIA NeMo Parakeet model.
    """
    # Delegation to heavy implementation (lazy import)
    from importlib import import_module  # pylint: disable=import-outside-toplevel

    _impl = import_module("parakeet_nemo_asr_rocm.transcribe").cli_transcribe

    return _impl(
        audio_files=audio_files,
        model_name=model_name,
        output_dir=output_dir,
        output_format=output_format,
        output_template=output_template,
        batch_size=batch_size,
        chunk_len_sec=chunk_len_sec,
        stream=stream,
        stream_chunk_sec=stream_chunk_sec,
        overlap_duration=overlap_duration,
        highlight_words=highlight_words,
        word_timestamps=word_timestamps,
        merge_strategy=merge_strategy,
        overwrite=overwrite,
        verbose=verbose,
        quiet=quiet,
        no_progress=no_progress,
        fp32=fp32,
        fp16=fp16,
    )
