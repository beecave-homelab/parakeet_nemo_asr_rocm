"""Command-line interface for the Parakeet-NeMo ASR application using Typer.

Replaces the older `argparse`-based CLI with a more robust and user-friendly
interface that supports subcommands, rich help messages, and improved argument
handling.

Features:
- `transcribe` command for running ASR on audio files.
- `webui` command for launching the Gradio web interface.
- Options for model selection, output formatting, and batch processing.
- Benchmark mode for capturing runtime and GPU telemetry metrics.
- Verbose mode for detailed logging.
"""

import pathlib
from typing import Annotated

import typer

from parakeet_rocm import __version__
from parakeet_rocm.utils.console import get_console
from parakeet_rocm.utils.constant import (
    ALLOW_UNSAFE_FILENAMES,
    API_SERVER_NAME,
    API_SERVER_PORT,
    BENCHMARK_OUTPUT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_LEN_SEC,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    PARAKEET_MODEL_NAME,
)

# Placeholder for lazy import; enables monkeypatching in tests.
RESOLVE_INPUT_PATHS = None  # type: ignore[assignment]


def version_callback(value: bool) -> None:
    """Show the application's version and exit.

    Args:
        value: When True, print the version and exit.

    Raises:
        typer.Exit: Always raised after printing when value is True.

    """
    if value:
        get_console().print(f"parakeet-rocm version: {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="parakeet-rocm",
    help=("A CLI for transcribing audio files using NVIDIA Parakeet-TDT via NeMo on ROCm."),
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
            is_eager=True,  # still OK
        ),
    ] = False,
) -> None:
    """Display the application's CLI help when no subcommand is invoked.

    The `--version` option triggers the version callback.

    Raises:
        typer.Exit: Terminate the CLI after displaying help or version.
    """
    if ctx.invoked_subcommand is None:
        get_console().print(ctx.get_help(), markup=False)
        raise typer.Exit()


def _setup_watch_mode(
    watch: list[str],
    model_name: str,
    output_dir: pathlib.Path,
    output_format: str,
    output_template: str,
    batch_size: int,
    chunk_len_sec: int,
    stream: bool,
    stream_chunk_sec: int,
    overlap_duration: int,
    highlight_words: bool,
    word_timestamps: bool,
    stabilize: bool,
    demucs: bool,
    vad: bool,
    vad_threshold: float,
    merge_strategy: str,
    overwrite: bool,
    verbose: bool,
    quiet: bool,
    no_progress: bool,
    fp32: bool,
    fp16: bool,
    allow_unsafe_filenames: bool = False,
) -> None:
    """Start a filesystem watcher that automatically transcribes newly detected audio files.

    This initializes and runs a watcher using the provided patterns; when new files appear
    the watcher will invoke the transcribe routine with the same CLI-style options. Directory
    entries from `watch` are used as base directories for mirroring subdirectories in outputs.
    This function blocks until the watcher stops.

    Parameters:
        watch (list[str]): Directory paths or glob patterns to monitor for new audio files.
        model_name (str): Model identifier or local path to use for transcription.
        output_dir (pathlib.Path): Directory where transcription outputs will be written.
        output_format (str): Output file format (e.g., "txt", "srt", "vtt", "json").
        output_template (str): Filename template for outputs (placeholders are supported).
        batch_size (int): Inference batch size.
        chunk_len_sec (int): Chunk length in seconds for chunked transcription.
        stream (bool): Enable pseudo-streaming (low-latency chunked) mode.
        stream_chunk_sec (int): Chunk size in seconds when `stream` is enabled.
        overlap_duration (int): Seconds of overlap between adjacent chunks.
        highlight_words (bool): Include word highlighting in formats that support it.
        word_timestamps (bool): Include word-level timestamps.
        stabilize (bool): Enable stabilization/refinement pass.
        demucs (bool): Enable Demucs denoising before transcription.
        vad (bool): Enable voice activity detection during stabilization.
        vad_threshold (float): VAD probability threshold (0.0-1.0) used when VAD is enabled.
        merge_strategy (str): Strategy for merging overlapping chunk transcriptions
                              (e.g., "none", "contiguous", "lcs").
        overwrite (bool): Overwrite existing output files when present.
        verbose (bool): Enable verbose logging.
        quiet (bool): Suppress non-essential output.
        no_progress (bool): Disable progress bars.
        fp32 (bool): Force FP32 precision.
        fp16 (bool): Enable FP16 precision.
        allow_unsafe_filenames (bool): Use relaxed filename validation.
    """
    # Validate output directory is writable before starting the watcher
    from parakeet_rocm.utils.file_utils import (  # pylint: disable=import-outside-toplevel
        ensure_dir_writable,
    )

    ensure_dir_writable(output_dir)

    # Lazy import watcher to avoid unnecessary deps if not used
    from importlib import import_module  # pylint: disable=import-outside-toplevel

    watcher = import_module("parakeet_rocm.utils.watch").watch_and_transcribe

    # Determine base directories for mirroring subdirectories under --watch
    # Only directory paths are considered watch bases. Glob patterns are ignored
    # for mirroring to avoid ambiguous roots.
    base_dirs = []
    for w in watch:
        try:
            p = pathlib.Path(w).resolve()
            if p.is_dir():
                base_dirs.append(p)
        except Exception:
            # Ignore invalid paths; watcher will handle patterns
            pass

    def _transcribe_fn(new_files: list[pathlib.Path]) -> None:
        """Trigger transcription for newly detected audio files using the CLI's configured options.

        Parameters:
            new_files (list[pathlib.Path]): Paths to audio files discovered by the watcher.
        """
        _impl = import_module("parakeet_rocm.transcribe").cli_transcribe
        _impl(
            audio_files=new_files,
            model_name=model_name,
            output_dir=output_dir,
            output_format=output_format,
            output_template=output_template,
            watch_base_dirs=base_dirs,
            batch_size=batch_size,
            chunk_len_sec=chunk_len_sec,
            stream=stream,
            stream_chunk_sec=stream_chunk_sec,
            overlap_duration=overlap_duration,
            highlight_words=highlight_words,
            word_timestamps=word_timestamps,
            stabilize=stabilize,
            demucs=demucs,
            vad=vad,
            vad_threshold=vad_threshold,
            merge_strategy=merge_strategy,
            overwrite=overwrite,
            verbose=verbose,
            quiet=quiet,
            no_progress=no_progress,
            fp32=fp32,
            fp16=fp16,
            allow_unsafe_filenames=allow_unsafe_filenames,
        )

    # Start watcher loop (blocking)
    return watcher(
        patterns=watch,
        transcribe_fn=_transcribe_fn,
        output_dir=output_dir,
        output_format=output_format,
        output_template=output_template,
        watch_base_dirs=base_dirs,
        verbose=verbose,
        quiet=quiet,
    )


@app.command()
def transcribe(
    audio_files: Annotated[
        list[str] | None,
        typer.Argument(
            help=("Path(s) or wildcard pattern(s) to audio files (e.g. '*.wav')."),
            show_default=False,
        ),
    ] = None,
    # Inputs
    watch: Annotated[
        list[str] | None,
        typer.Option(
            "--watch",
            help=("Watch directory/pattern for new audio files and transcribe automatically."),
        ),
    ] = None,
    # Model
    model_name: Annotated[
        str,
        typer.Option(
            "--model",
            help=("Hugging Face Hub model ID or local path to the NeMo ASR model."),
        ),
    ] = PARAKEET_MODEL_NAME,
    # Outputs
    output_dir: Annotated[
        pathlib.Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save the transcription outputs.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = "./output",
    output_format: Annotated[
        str,
        typer.Option(help=("Format for the output file(s) (e.g., txt, srt, vtt, json).")),
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
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help=("Overwrite existing output files instead of appending numbered suffixes."),
        ),
    ] = False,
    # Timestamps and subtitles
    word_timestamps: Annotated[
        bool,
        typer.Option(
            "--word-timestamps",
            help="Enable word-level timestamp generation.",
        ),
    ] = False,
    highlight_words: Annotated[
        bool,
        typer.Option(
            "--highlight-words",
            help="Highlight each word in SRT/VTT outputs (e.g., bold).",
        ),
    ] = False,
    stabilize: Annotated[
        bool,
        typer.Option(
            "--stabilize",
            help="Refine word timestamps using stable-ts.",
        ),
    ] = False,
    vad: Annotated[
        bool,
        typer.Option(
            "--vad",
            help="Enable voice activity detection during stabilization.",
        ),
    ] = False,
    demucs: Annotated[
        bool,
        typer.Option(
            "--demucs",
            help="Use Demucs denoiser during stabilization.",
        ),
    ] = False,
    vad_threshold: Annotated[
        float,
        typer.Option(
            "--vad-threshold",
            help="VAD probability threshold for suppression.",
        ),
    ] = 0.35,
    # Chunking and streaming
    chunk_len_sec: Annotated[
        int,
        typer.Option(help="Segment length in seconds for chunked transcription."),
    ] = DEFAULT_CHUNK_LEN_SEC,
    overlap_duration: Annotated[
        int,
        typer.Option(
            "--overlap-duration",
            help=("Overlap between consecutive chunks in seconds (for long audio)."),
        ),
    ] = 15,
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
            help=("Chunk length in seconds when --stream is enabled (overrides default)."),
        ),
    ] = 0,
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
    # Performance
    batch_size: Annotated[
        int, typer.Option(help="Batch size for transcription inference.")
    ] = DEFAULT_BATCH_SIZE,
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
    fp32: Annotated[
        bool,
        typer.Option(
            "--fp32",
            help=(
                "Force full-precision (FP32) inference. Default if no precision flag is provided."
            ),
        ),
    ] = False,
    # Benchmarking
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
            help=(
                "Enable benchmark mode: capture runtime, GPU telemetry, and quality "
                "metrics. Results are written to JSON files in the benchmark output directory."
            ),
        ),
    ] = False,
    benchmark_dir: Annotated[
        pathlib.Path,
        typer.Option(
            "--benchmark-dir",
            help="Directory for benchmark output JSON files.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = BENCHMARK_OUTPUT_DIR,
    # UX and logging
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
            help=("Suppress console messages except the progress bar and final output."),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Enable verbose output.",
        ),
    ] = False,
    allow_unsafe_filenames: Annotated[
        bool,
        typer.Option(
            "--allow-unsafe-filenames",
            help=(
                "Use relaxed filename validation. Allows spaces, brackets, quotes, "
                "and unicode in filenames. Security invariants (path traversal, "
                "directory separators, control characters) remain enforced. "
                "Cross-platform filesystem compatibility is NOT guaranteed."
            ),
        ),
    ] = ALLOW_UNSAFE_FILENAMES,
) -> list[pathlib.Path] | None:
    """Transcribe audio files using the specified NeMo Parakeet model.

    Args:
        audio_files: Explicit paths or patterns to transcribe.
        watch: Directory or glob(s) to monitor for new audio files.
        model_name: Hugging Face model ID or local path.
        output_dir: Directory to save transcription outputs.
        output_format: Output format, e.g. ``txt``, ``srt``, ``vtt`` or ``json``.
        output_template: Filename template supporting placeholders.
        overwrite: Overwrite existing outputs when True.
        word_timestamps: Enable word-level timestamps.
        highlight_words: Highlight words in subtitle outputs.
        stabilize: Refine timestamps using stable-ts.
        vad: Enable VAD during stabilization.
        demucs: Enable Demucs denoising during stabilization.
        vad_threshold: VAD probability threshold.
        chunk_len_sec: Chunk length in seconds.
        overlap_duration: Overlap between chunks in seconds.
        stream: Enable pseudo-streaming mode.
        stream_chunk_sec: Streaming chunk size in seconds.
        merge_strategy: Strategy for merging overlapping chunks.
        batch_size: Batch size for inference.
        fp16: Use half precision.
        fp32: Use full precision.
        benchmark: Enable benchmark mode for capturing metrics.
        benchmark_dir: Directory for benchmark JSON output files.
        no_progress: Disable progress bar output.
        quiet: Suppress non-error output.
        verbose: Enable verbose logging.
        allow_unsafe_filenames: Use relaxed filename validation.

    Returns:
        A list of created output paths, or ``None`` when running in watch mode.

    Raises:
        typer.BadParameter: When neither ``audio_files`` nor ``--watch`` is
            provided, or when both are supplied at the same time.

    """
    # Delegation to heavy implementation (lazy import)
    from importlib import import_module  # pylint: disable=import-outside-toplevel

    # Normalise default
    if audio_files is None:
        audio_files = []

    # Validate input combinations: exactly one of AUDIO_FILES or --watch
    has_files = bool(audio_files)
    has_watch = bool(watch)
    if not has_files and not has_watch:
        raise typer.BadParameter("Provide AUDIO_FILES or --watch pattern(s).")
    if has_files and has_watch:
        raise typer.BadParameter(
            "AUDIO_FILES and --watch cannot be used together; choose one input mode."
        )

    # Expand provided audio file patterns now (for immediate run or watcher seed)
    global RESOLVE_INPUT_PATHS  # pylint: disable=global-statement
    if RESOLVE_INPUT_PATHS is None:
        from parakeet_rocm.utils.file_utils import (  # pylint: disable=import-outside-toplevel
            resolve_input_paths as _resolve_input_paths,
        )

        RESOLVE_INPUT_PATHS = _resolve_input_paths

    resolved_paths = RESOLVE_INPUT_PATHS(audio_files)

    if watch:
        return _setup_watch_mode(
            watch=watch,
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
            stabilize=stabilize,
            demucs=demucs,
            vad=vad,
            vad_threshold=vad_threshold,
            merge_strategy=merge_strategy,
            overwrite=overwrite,
            verbose=verbose,
            quiet=quiet,
            no_progress=no_progress,
            fp32=fp32,
            fp16=fp16,
            allow_unsafe_filenames=allow_unsafe_filenames,
        )

    # No watch mode: immediate transcription
    _impl = import_module("parakeet_rocm.transcribe").cli_transcribe

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
        stabilize=stabilize,
        demucs=demucs,
        vad=vad,
        vad_threshold=vad_threshold,
        merge_strategy=merge_strategy,
        overwrite=overwrite,
        verbose=verbose,
        quiet=quiet,
        no_progress=no_progress,
        fp32=fp32,
        fp16=fp16,
        benchmark=benchmark,
        benchmark_dir=benchmark_dir,
        allow_unsafe_filenames=allow_unsafe_filenames,
    )


@app.command()
def webui(
    server_name: Annotated[
        str,
        typer.Option(
            "--host",
            help="Server hostname or IP address to bind to.",
        ),
    ] = GRADIO_SERVER_NAME,
    server_port: Annotated[
        int,
        typer.Option(
            "--port",
            help="Server port number.",
        ),
    ] = GRADIO_SERVER_PORT,
    share: Annotated[
        bool,
        typer.Option(
            "--share",
            help=(
                "Deprecated: public share links are not supported in FastAPI-mounted WebUI mode."
            ),
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode with verbose logging.",
        ),
    ] = False,
) -> None:
    """Launch the Gradio WebUI for interactive transcription.

    Starts a web server with a user-friendly interface for uploading
    audio files, configuring transcription options, and viewing results.

    Args:
        server_name: Server hostname or IP address to bind to.
        server_port: Server port number.
        share: Deprecated; public share links are not supported in mounted mode.
        debug: Enable debug mode with verbose logging.

    Examples:
        Launch on default settings (localhost:7860)::

            $ parakeet-rocm webui

        Show deprecation warning for public sharing::

            $ parakeet-rocm webui --share

        Launch on custom port with debug mode::

            $ parakeet-rocm webui --port 8080 --debug
    """
    from parakeet_rocm.utils.logging_config import configure_logging

    configure_logging(level="DEBUG" if debug else "INFO")

    from parakeet_rocm.api import create_app

    _run_uvicorn_app(
        app_instance=create_app(),
        server_name=server_name,
        server_port=server_port,
        debug=debug,
        share=share,
    )


@app.command()
def api(
    server_name: Annotated[
        str,
        typer.Option(
            "--host",
            help="Server hostname or IP address to bind to.",
        ),
    ] = API_SERVER_NAME,
    server_port: Annotated[
        int,
        typer.Option(
            "--port",
            help="Server port number.",
        ),
    ] = API_SERVER_PORT,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode with verbose logging.",
        ),
    ] = False,
) -> None:
    """Launch only the OpenAI-compatible REST API without the Gradio UI.

    Args:
        server_name: Server hostname or IP address to bind to.
        server_port: Server port number.
        debug: Enable debug mode with verbose logging.
    """
    from parakeet_rocm.utils.logging_config import configure_logging

    configure_logging(level="DEBUG" if debug else "INFO")

    from parakeet_rocm.api import create_api_app

    _run_uvicorn_app(
        app_instance=create_api_app(),
        server_name=server_name,
        server_port=server_port,
        debug=debug,
        share=False,
    )


def _run_uvicorn_app(
    *,
    app_instance: object,
    server_name: str,
    server_port: int,
    debug: bool,
    share: bool,
) -> None:
    """Run a FastAPI app instance through uvicorn with shared CLI behavior.

    Args:
        app_instance: ASGI application instance to run.
        server_name: Hostname or IP address to bind.
        server_port: Port to bind.
        debug: Whether to enable debug logging.
        share: Whether the caller requested Gradio-style public sharing.
    """
    from parakeet_rocm.utils.logging_config import get_logger

    logger = get_logger(__name__)

    if share:
        logger.warning(
            "--share is not supported with mounted Gradio on FastAPI. "
            "Use a tunnel (for example ngrok or cloudflared) to expose this server."
        )

    import uvicorn

    uvicorn.run(
        app_instance,
        host=server_name,
        port=server_port,
        log_level="debug" if debug else "info",
    )
