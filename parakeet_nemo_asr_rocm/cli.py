"""Command-line interface for the Parakeet-NeMo ASR application using Typer.

Replaces the older `argparse`-based CLI with a more robust and user-friendly
interface that supports subcommands, rich help messages, and improved argument
handling.

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
# Placeholder for lazy import; enables monkeypatching in tests.
resolve_input_paths = None  # type: ignore[assignment]


# Create the main Typer application instance
def version_callback(value: bool):
    """Show the application's version and exit."""
    if value:
        print(f"parakeet-rocm version: {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="parakeet-rocm",
    help="A CLI for transcribing audio files using NVIDIA Parakeet-TDT via NeMo on ROCm.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
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
    watch: Annotated[
        List[str] | None,
        typer.Option(
            "--watch",
            help="Watch directory/pattern for new audio files and transcribe automatically.",
        ),
    ] = None,
):
    """Root command acts like `transcribe` when arguments are passed without subcommand."""
    # Root invoked without subcommand ⇒ show help fast without heavy imports.
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def transcribe(
    audio_files: Annotated[
        List[str] | None,
        typer.Argument(
            help="Path(s) or wildcard pattern(s) to audio files (e.g. '*.wav').",
            show_default=False,
        ),
    ] = None,
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
            help=(
                "Strategy for merging overlapping chunks: 'none' (concatenate), "
                "'contiguous' (fast merge), or 'lcs' (accurate, default)"
            ),
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
    watch: Annotated[
        List[str] | None,
        typer.Option(
            "--watch",
            help="Watch directory/pattern for new audio files and transcribe automatically.",
        ),
    ] = None,
    fp16: Annotated[
        bool,
        typer.Option(
            "--fp16",
            help=(
                "Enable half-precision (FP16) inference for faster processing on "
                "compatible hardware."
            ),
        ),
    ] = False,
):
    """Transcribe audio files using the specified NVIDIA NeMo Parakeet model."""
    # Delegation to heavy implementation (lazy import)
    from importlib import import_module  # pylint: disable=import-outside-toplevel

    # Normalise default
    if audio_files is None:
        audio_files = []

    # Validate input combinations
    if not audio_files and not watch:
        raise typer.BadParameter("Provide AUDIO_FILES or --watch pattern(s).")

    # Expand provided audio file patterns now (for immediate run or watcher seed)
    global resolve_input_paths  # pylint: disable=global-statement
    if resolve_input_paths is None:
        from parakeet_nemo_asr_rocm.utils.file_utils import (
            resolve_input_paths as _resolve_input_paths,
        )  # Lazy import to keep --help snappy.

        resolve_input_paths = _resolve_input_paths

    resolved_paths = resolve_input_paths(audio_files)

    if watch:
        # Lazy import watcher to avoid unnecessary deps if not used
        watcher = import_module(
            "parakeet_nemo_asr_rocm.utils.watch"
        ).watch_and_transcribe

        def _transcribe_fn(new_files):
            _impl = import_module("parakeet_nemo_asr_rocm.transcribe").cli_transcribe
            return _impl(
                audio_files=new_files,
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

        # Start watcher loop (blocking)
        return watcher(
            patterns=watch,
            transcribe_fn=_transcribe_fn,
            output_dir=output_dir,
            output_format=output_format,
            output_template=output_template,
            verbose=verbose,
        )

    # No watch mode: immediate transcription
    _impl = import_module("parakeet_nemo_asr_rocm.transcribe").cli_transcribe

    return _impl(
        audio_files=resolved_paths,
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
