"""CLI-facing transcription orchestration."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import List, Sequence

from parakeet_nemo_asr_rocm.transcription.file_processor import transcribe_file
from parakeet_nemo_asr_rocm.transcription.utils import (
    compute_total_segments,
    configure_environment,
)
from parakeet_nemo_asr_rocm.utils.constant import (
    DEFAULT_CHUNK_LEN_SEC,
    DEFAULT_STREAM_CHUNK_SEC,
    NEMO_LOG_LEVEL,
    TRANSFORMERS_VERBOSITY,
    MAX_LINE_CHARS,
    MAX_LINES_PER_BLOCK,
    MAX_SEGMENT_DURATION_SEC,
    MIN_SEGMENT_DURATION_SEC,
    MAX_CPS,
    DISPLAY_BUFFER_SEC,
)


def _display_settings(  # pragma: no cover - formatting helper
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
    """Display CLI settings via rich table."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
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
) -> List[Path]:
    """Run batch transcription and return created output files.

    Args:
        audio_files: Iterable of audio file paths to transcribe.
        model_name: Name of the NeMo model to load.
        output_dir: Directory where output files are written.
        output_format: Desired output format (e.g. ``"txt"`` or ``"srt"``).
        output_template: Filename template supporting ``{filename}`` and
            ``{index}`` placeholders.
        batch_size: Number of audio chunks processed per batch.
        chunk_len_sec: Chunk length in seconds for segmentation.
        stream: Enable streaming mode when ``True``.
        stream_chunk_sec: Custom stream chunk length in seconds.
        overlap_duration: Overlap between consecutive chunks in seconds.
        highlight_words: Highlight words in subtitle outputs when supported.
        word_timestamps: Include word-level timestamps in processing.
        merge_strategy: Strategy for merging overlapping word timestamps.
        stabilize: Refine word timestamps using stable-ts when ``True``.
        demucs: Enable Demucs denoising during stabilization.
        vad: Enable voice activity detection during stabilization.
        vad_threshold: VAD probability threshold when ``vad`` is enabled.
        overwrite: Overwrite existing output files if ``True``.
        verbose: Enable verbose logging output.
        quiet: Suppress non-error output when ``True``.
        no_progress: Disable progress bar display.
        fp32: Force 32-bit floating point precision.
        fp16: Force 16-bit floating point precision.

    Returns:
        List of paths to created output files.
    """
    import typer
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    import time  # pylint: disable=import-outside-toplevel

    from parakeet_nemo_asr_rocm.formatting import get_formatter
    from parakeet_nemo_asr_rocm.models.parakeet import get_model

    configure_environment(verbose)

    if quiet:
        verbose = False

    if fp32 and fp16:
        typer.echo("Error: Cannot specify both --fp32 and --fp16", err=True)
        raise typer.Exit(code=1)

    if stream:
        if stream_chunk_sec <= 0:
            chunk_len_sec = DEFAULT_STREAM_CHUNK_SEC
        else:
            chunk_len_sec = stream_chunk_sec
        if overlap_duration >= chunk_len_sec:
            overlap_duration = max(0, chunk_len_sec // 2)
        if verbose:
            typer.echo(
                f"[stream] Using chunk_len_sec={chunk_len_sec}, overlap_duration={overlap_duration}"
            )

    if verbose and not quiet:
        # Show effective configuration resolved via utils.constant (loaded once from .env)
        typer.echo(
            f"[env] NEMO_LOG_LEVEL={NEMO_LOG_LEVEL}, TRANSFORMERS_VERBOSITY={TRANSFORMERS_VERBOSITY}"
        )
        typer.echo(
            f"[env] CHUNK_LEN_SEC={DEFAULT_CHUNK_LEN_SEC}, STREAM_CHUNK_SEC={DEFAULT_STREAM_CHUNK_SEC}, "
            f"MAX_LINE_CHARS={MAX_LINE_CHARS}, MAX_LINES_PER_BLOCK={MAX_LINES_PER_BLOCK}"
        )
        typer.echo(
            f"[env] MAX_SEGMENT_DURATION_SEC={MAX_SEGMENT_DURATION_SEC}, MIN_SEGMENT_DURATION_SEC={MIN_SEGMENT_DURATION_SEC}, "
            f"MAX_CPS={MAX_CPS}, DISPLAY_BUFFER_SEC={DISPLAY_BUFFER_SEC}"
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
        typer.echo()

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = get_model(model_name)
    model = model.half() if fp16 else model.float()
    if verbose and not quiet:
        try:
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            cache_info = get_model.cache_info()  # type: ignore[attr-defined]
            typer.echo(
                f"[model] device={device}, dtype={dtype}, cache={cache_info}"
            )
        except Exception:  # pragma: no cover
            pass

    try:
        formatter = get_formatter(output_format)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    total_segments = compute_total_segments(
        audio_files, chunk_len_sec, overlap_duration
    )
    if verbose and not quiet:
        typer.echo(f"[plan] total_segments={total_segments}")

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

    created_files: List[Path] = []
    with progress_cm as progress:
        main_task = (
            None
            if no_progress
            else progress.add_task("Transcribing...", total=total_segments)
        )
        for file_idx, audio_path in enumerate(audio_files, start=1):
            output_path = transcribe_file(
                audio_path,
                model=model,
                formatter=formatter,
                file_idx=file_idx,
                output_dir=output_dir,
                output_format=output_format,
                output_template=output_template,
                batch_size=batch_size,
                chunk_len_sec=chunk_len_sec,
                overlap_duration=overlap_duration,
                highlight_words=highlight_words,
                word_timestamps=word_timestamps,
                merge_strategy=merge_strategy,
                stabilize=stabilize,
                demucs=demucs,
                vad=vad,
                vad_threshold=vad_threshold,
                overwrite=overwrite,
                verbose=verbose,
                quiet=quiet,
                no_progress=no_progress,
                progress=progress,
                main_task=main_task,
            )
            if output_path is not None:
                created_files.append(output_path)
    if not quiet:
        for p in created_files:
            typer.echo(f'Created "{p}"')
    if verbose and not quiet:
        elapsed = time.perf_counter() - t0
        typer.echo(f"[timing] total_wall={elapsed:.2f}s")
        typer.echo("Done.")
    return created_files
