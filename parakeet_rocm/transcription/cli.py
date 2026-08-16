"""CLI-facing transcription orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parakeet_rocm.benchmarks.collector import BenchmarkCollector

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from parakeet_rocm.config import (
    OutputConfig,
    StabilizationConfig,
    TranscriptionConfig,
    UIConfig,
)
from parakeet_rocm.formatting import get_formatter
from parakeet_rocm.transcription.file_processor import (
    transcribe_file,
    validate_output_filenames,
)
from parakeet_rocm.transcription.utils import (
    compute_total_segments,
    configure_environment,
)
from parakeet_rocm.utils.console import get_console, print_error, print_status
from parakeet_rocm.utils.constant import (
    BENCHMARK_OUTPUT_DIR,
    DEFAULT_CHUNK_LEN_SEC,
    DEFAULT_STREAM_CHUNK_SEC,
    DISPLAY_BUFFER_SEC,
    MAX_CPS,
    MAX_LINE_CHARS,
    MAX_LINES_PER_BLOCK,
    MAX_SEGMENT_DURATION_SEC,
    MIN_SEGMENT_DURATION_SEC,
    NEMO_LOG_LEVEL,
    TRANSFORMERS_VERBOSITY,
)
from parakeet_rocm.utils.file_utils import ensure_dir_writable
from parakeet_rocm.utils.logging_config import get_logger

logger = get_logger(__name__)


def _display_settings(
    audio_files: Sequence[Path],
    model_name: str,
    output_dir: Path,
    output_format: str,
    output_template: str,
    batch_size: int,
    chunk_len_sec: int,
    stream: bool,
    stream_chunk_sec: int,
    overlap_duration: int,
    word_timestamps: bool,
    highlight_words: bool,
    merge_strategy: str,
    stabilize: bool,
    demucs: bool,
    vad: bool,
    vad_threshold: float,
    overwrite: bool,
    quiet: bool,
    no_progress: bool,
    fp16: bool,
    fp32: bool,
) -> None:
    """Render the current CLI configuration as a Rich-formatted table to the console.

    Parameters:
        audio_files (Sequence[Path]): Sequence of input audio file paths (used to show count).
        model_name (str): Model identifier.
        output_dir (Path): Directory where outputs will be written.
        output_format (str): Output serialization format (e.g., "txt", "srt").
        output_template (str): Template for output filenames.
        batch_size (int): Number of items processed in a batch.
        chunk_len_sec (int): Chunk length in seconds for segmentation.
        stream (bool): If True, include streaming-related settings in the table.
        stream_chunk_sec (int): Stream chunk length in seconds (displayed when > 0).
        overlap_duration (int): Overlap duration in seconds between chunks.
        word_timestamps (bool): Whether word-level timestamps are enabled.
        highlight_words (bool): Whether word highlighting is enabled in outputs.
        merge_strategy (str): Strategy used to merge partial/transcribed segments.
        stabilize (bool): If True, include stabilization-related settings in the table.
        demucs (bool): Whether Demucs separation is enabled (shown when stabilize is True).
        vad (bool): Whether voice activity detection is enabled (shown when stabilize is True).
        vad_threshold (float): VAD sensitivity threshold (shown when stabilize is True).
        overwrite (bool): Whether existing output files will be overwritten.
        quiet (bool): Whether CLI quiet mode is enabled (affects displayed settings).
        no_progress (bool): Whether progress reporting is suppressed.
        fp16 (bool): Whether FP16 precision is requested.
        fp32 (bool): Whether FP32 precision is requested.
    """
    if quiet:
        return
    console = get_console()
    table = Table(title="CLI Settings", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Setting", style="green")
    table.add_column("Value", style="yellow")

    table.add_row("Model", "Model Name", model_name)
    table.add_row("Model", "Output Directory", str(output_dir))
    table.add_row("Model", "Output Format", output_format)
    table.add_row("Model", "Output Template", output_template)

    table.add_row("Processing", "Batch Size", str(batch_size))
    table.add_row("Processing", "Chunk Length (s)", str(chunk_len_sec))

    if stream:
        table.add_row("Streaming", "Stream Mode", str(stream))
        if stream_chunk_sec > 0:
            table.add_row("Streaming", "Stream Chunk Length (s)", str(stream_chunk_sec))
        table.add_row("Streaming", "Overlap Duration (s)", str(overlap_duration))

    table.add_row("Features", "Word Timestamps", str(word_timestamps))
    table.add_row("Features", "Highlight Words", str(highlight_words))
    table.add_row("Features", "Merge Strategy", merge_strategy)
    table.add_row("Features", "Stabilize", str(stabilize))
    if stabilize:
        table.add_row("Features", "Demucs", str(demucs))
        table.add_row("Features", "VAD", str(vad))
        table.add_row("Features", "VAD Threshold", str(vad_threshold))

    table.add_row("Output", "Overwrite", str(overwrite))
    table.add_row("Output", "Quiet Mode", str(quiet))
    table.add_row("Output", "No Progress", str(no_progress))

    precision = "FP16" if fp16 else "FP32" if fp32 else "Default"
    table.add_row("Precision", "Mode", precision)
    table.add_row("Files", "Transcribing", f"{len(audio_files)} file(s)")

    console.print(table)


def cli_transcribe(
    *,
    audio_files: Sequence[Path],
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
    output_dir: Path = Path("./output"),
    output_format: str = "txt",
    output_template: str = "{filename}",
    watch_base_dirs: Sequence[Path] | None = None,
    batch_size: int = 1,
    chunk_len_sec: int = DEFAULT_CHUNK_LEN_SEC,
    stream: bool = False,
    stream_chunk_sec: int = 0,
    overlap_duration: int = 15,
    highlight_words: bool = False,
    word_timestamps: bool = False,
    merge_strategy: str = "lcs",
    stabilize: bool = False,
    demucs: bool = False,
    vad: bool = False,
    vad_threshold: float = 0.35,
    overwrite: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    no_progress: bool = False,
    fp32: bool = False,
    fp16: bool = False,
    benchmark: bool = False,
    benchmark_dir: Path = Path(BENCHMARK_OUTPUT_DIR),
    progress_callback: Callable[[int, int], None] | None = None,
    collector: BenchmarkCollector | None = None,
    allow_unsafe_filenames: bool = False,
) -> list[Path]:
    """Run batch transcription for the given audio files and return the created output file paths.

    Processes each file using the specified model and formatting options.
    Optionally enables streaming, stabilization (Demucs, VAD), and progress
    reporting, and writes outputs into ``output_dir`` according to
    ``output_template``.

    Parameters:
        audio_files (Sequence[Path]): Audio file paths to transcribe.
        model_name (str): Model identifier to load.
        output_dir (Path): Directory to write output files.
        output_format (str): Output format identifier (e.g., "txt", "srt").
        output_template (str): Filename template supporting `{filename}` and `{index}`.
        batch_size (int): Number of chunks processed per batch.
        chunk_len_sec (int): Segment length in seconds.
        stream (bool): Enable streaming mode.
        stream_chunk_sec (int): When >0, overrides chunk_len_sec for streaming.
        overlap_duration (int): Overlap between adjacent chunks in seconds.
        highlight_words (bool): Enable highlighting of words in supported outputs.
        word_timestamps (bool): Include word-level timestamps.
        merge_strategy (str): Strategy to merge overlapping word timestamps (e.g., "lcs").
        stabilize (bool): Enable post-processing to refine word timestamps.
        demucs (bool): Enable Demucs denoising when stabilization is enabled.
        vad (bool): Enable voice activity detection when stabilization is enabled.
        vad_threshold (float): Probability threshold used by VAD.
        overwrite (bool): Overwrite existing output files.
        no_progress (bool): Disable progress display.
        fp32 (bool): Force 32-bit model precision.
        fp16 (bool): Force 16-bit model precision.
        benchmark (bool): Enable benchmark mode for capturing runtime and GPU metrics.
        benchmark_dir (Path): Directory for benchmark JSON output files.
        progress_callback (Callable[[int, int], None] | None): Optional callback for
            progress updates. Called with (current_batch, total_batches) after each batch.
        collector (BenchmarkCollector | None): Optional external benchmark collector
            (used by WebUI for centralized metrics). When provided, benchmark mode
            is implicitly enabled.
        allow_unsafe_filenames (bool): Use relaxed filename validation when ``True``.

    Returns:
        list[Path]: Paths to the files created by the transcription run.

    Raises:
        typer.Exit: If both ``fp32`` and ``fp16`` are specified or if the
            requested ``output_format`` cannot be resolved.
    """
    configure_environment(verbose)

    if quiet:
        verbose = False

    console = get_console()

    if fp32 and fp16:
        print_error("Cannot specify both --fp32 and --fp16")
        raise typer.Exit(code=1)

    if stream:
        if stream_chunk_sec <= 0:
            chunk_len_sec = DEFAULT_STREAM_CHUNK_SEC
        else:
            chunk_len_sec = stream_chunk_sec
        if overlap_duration >= chunk_len_sec:
            overlap_duration = max(0, chunk_len_sec // 2)
        if verbose:
            print_status(
                "stream",
                f"Using chunk_len_sec={chunk_len_sec},\noverlap_duration={overlap_duration}",
                quiet=quiet,
            )

    if verbose and not quiet:
        # Show effective configuration resolved via utils.constant
        print_status(
            "env",
            f"NEMO_LOG_LEVEL={NEMO_LOG_LEVEL}, "
            f"TRANSFORMERS_VERBOSITY={TRANSFORMERS_VERBOSITY}",
            quiet=quiet,
        )
        print_status(
            "env",
            f"CHUNK_LEN_SEC={DEFAULT_CHUNK_LEN_SEC}, "
            f"STREAM_CHUNK_SEC={DEFAULT_STREAM_CHUNK_SEC}, "
            f"MAX_LINE_CHARS={MAX_LINE_CHARS}, "
            f"MAX_LINES_PER_BLOCK={MAX_LINES_PER_BLOCK}",
            quiet=quiet,
        )
        print_status(
            "env",
            f"MAX_SEGMENT_DURATION_SEC={MAX_SEGMENT_DURATION_SEC}, "
            f"MIN_SEGMENT_DURATION_SEC={MIN_SEGMENT_DURATION_SEC}, "
            f"MAX_CPS={MAX_CPS}, "
            f"DISPLAY_BUFFER_SEC={DISPLAY_BUFFER_SEC}",
            quiet=quiet,
        )

    if not quiet:
        _display_settings(
            audio_files,
            model_name,
            output_dir,
            output_format,
            output_template,
            batch_size,
            chunk_len_sec,
            stream,
            stream_chunk_sec,
            overlap_duration,
            word_timestamps,
            highlight_words,
            merge_strategy,
            stabilize,
            demucs,
            vad,
            vad_threshold,
            overwrite,
            quiet,
            no_progress,
            fp16,
            fp32,
        )
        console.print()

    ensure_dir_writable(output_dir)

    # Pre-flight filename validation — fail before expensive model loading
    validation_errors = validate_output_filenames(
        audio_files,
        output_template,
        allow_unsafe=allow_unsafe_filenames,
    )
    if validation_errors:
        for path, msg in validation_errors:
            typer.echo(f"Error: {path}: {msg}", err=True)
        if not allow_unsafe_filenames:
            typer.echo(
                "Hint: use --allow-unsafe-filenames to allow spaces and "
                "special characters in filenames.",
                err=True,
            )
        raise typer.Exit(code=1)

    # Initialize benchmark collector and GPU sampler if benchmark mode is enabled
    benchmark_collector = collector
    gpu_sampler = None
    benchmark_enabled = benchmark or collector is not None
    if benchmark_enabled and benchmark_collector is None:
        from parakeet_rocm.benchmarks import BenchmarkCollector, GpuUtilSampler

        benchmark_collector = BenchmarkCollector(
            output_dir=benchmark_dir,
            slug=f"cli_{Path(audio_files[0]).stem}" if audio_files else "cli_batch",
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "chunk_len_sec": chunk_len_sec,
                "overlap_duration": overlap_duration,
                "word_timestamps": word_timestamps,
                "merge_strategy": merge_strategy,
                "stabilize": stabilize,
                "demucs": demucs,
                "vad": vad,
                "vad_threshold": vad_threshold,
                "fp16": fp16,
                "fp32": fp32,
                "output_format": output_format,
            },
            audio_path=str(audio_files[0]) if audio_files else None,
            task="transcribe",
        )
    if benchmark_enabled and benchmark_collector is not None:
        from parakeet_rocm.benchmarks import GpuUtilSampler

        gpu_sampler = GpuUtilSampler(interval_sec=1.0)
        gpu_sampler.start()
        if not quiet:
            print_status(
                "benchmark",
                "Enabled - capturing runtime and GPU metrics",
                quiet=quiet,
            )

    t0 = time.perf_counter()
    from parakeet_rocm.models.parakeet import get_model

    model = get_model(model_name)
    model = model.half() if fp16 else model.float()
    if verbose and not quiet:
        try:
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            try:
                # Import internal cache accessor for diagnostics only
                from parakeet_rocm.models.parakeet import (  # type: ignore
                    _get_cached_model,
                )

                cache_info = _get_cached_model.cache_info()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - cache info optional
                cache_info = None
            print_status(
                "model",
                f"device={device}, dtype={dtype}, cache={cache_info}",
                quiet=quiet,
            )
        except Exception:  # pragma: no cover
            pass

    try:
        formatter = get_formatter(output_format)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    total_segments = compute_total_segments(audio_files, chunk_len_sec, overlap_duration)
    if verbose and not quiet:
        print_status("plan", f"total_segments={total_segments}", quiet=quiet)

    progress_cm = (
        nullcontext()
        if no_progress
        else Progress(
            SpinnerColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        )
    )

    created_files: list[Path] = []
    current_batch = 0
    total_batches = ceil(total_segments / batch_size) if total_segments else 0

    def _on_batch_processed() -> None:
        nonlocal current_batch
        current_batch += 1
        if progress_callback is not None:
            progress_callback(current_batch, total_batches)
            if verbose and not quiet:
                print_status(
                    "progress",
                    f"{current_batch}/{total_batches} batches",
                    quiet=quiet,
                    err=True,
                )

    with progress_cm as progress:
        main_task = (
            None if no_progress else progress.add_task("Transcribing...", total=total_segments)
        )
        # Build configuration objects
        transcription_config = TranscriptionConfig(
            batch_size=batch_size,
            chunk_len_sec=chunk_len_sec,
            overlap_duration=overlap_duration,
            word_timestamps=word_timestamps,
            merge_strategy=merge_strategy,
        )
        stabilization_config = StabilizationConfig(
            enabled=stabilize,
            demucs=demucs,
            vad=vad,
            vad_threshold=vad_threshold,
        )
        output_config = OutputConfig(
            output_dir=output_dir,
            output_format=output_format,
            output_template=output_template,
            overwrite=overwrite,
            highlight_words=highlight_words,
            allow_unsafe_filenames=allow_unsafe_filenames,
        )
        ui_config = UIConfig(
            verbose=verbose,
            quiet=quiet,
            no_progress=no_progress,
        )

        for file_idx, audio_path in enumerate(audio_files, start=1):
            output_path = transcribe_file(
                audio_path,
                model=model,
                formatter=formatter,
                file_idx=file_idx,
                transcription_config=transcription_config,
                stabilization_config=stabilization_config,
                output_config=output_config,
                ui_config=ui_config,
                watch_base_dirs=watch_base_dirs,
                progress=progress,
                main_task=main_task,
                batch_progress_callback=_on_batch_processed,
                allow_unsafe_filenames=allow_unsafe_filenames,
            )
            if output_path is not None:
                created_files.append(output_path)
    if not quiet:
        for p in created_files:
            print_status("output", f'Created "{p}"', quiet=quiet)

    elapsed = time.perf_counter() - t0

    # Finalize benchmark collection
    if benchmark_enabled and benchmark_collector is not None:
        if output_format == "srt":
            for output_path in created_files[:1]:
                try:
                    from parakeet_rocm.formatting.refine import SubtitleRefiner

                    srt_text = output_path.read_text(encoding="utf-8")
                    cues = SubtitleRefiner().load_srt(
                        output_path,
                        base_dir=output_dir,
                    )
                    segments = [
                        {"start": cue.start, "end": cue.end, "text": cue.text} for cue in cues
                    ]
                    benchmark_collector.add_quality_analysis(
                        segments=segments,
                        srt_text=srt_text,
                        output_format=output_format,
                    )
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "Benchmark quality analysis skipped.",
                        exc_info=exc,
                    )

        if gpu_sampler is not None:
            gpu_sampler.stop()
            benchmark_collector.metrics["gpu_stats"] = gpu_sampler.get_stats() or {}

        benchmark_collector.metrics["runtime_seconds"] = elapsed
        benchmark_collector.metrics["total_wall_seconds"] = elapsed

        # Add file metrics for each processed file
        for audio_path in audio_files:
            # Duration estimation (simplified - actual duration tracking would
            # require deeper integration)
            benchmark_collector.add_file_metrics(
                filename=audio_path.name,
                duration_sec=0.0,  # Would need actual audio duration
                segment_count=0,  # Would need actual segment count
                processing_time_sec=elapsed / len(audio_files),
            )

        if collector is None:
            benchmark_path = benchmark_collector.write_json()
            if not quiet:
                print_status(
                    "benchmark",
                    f"Metrics written to: {benchmark_path}",
                    quiet=quiet,
                )

    if verbose and not quiet:
        print_status("timing", f"total_wall={elapsed:.2f}s", quiet=quiet)
        print_status("done", "Done.", quiet=quiet)
    return created_files
