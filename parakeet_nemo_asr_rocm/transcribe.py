"""Batch transcription helper functions.

Designed to be imported *and* run as a script via ``python -m
parakeet_nemo_asr_rocm.transcribe <audio files>``.
"""

# pylint: disable=import-outside-toplevel, multiple-imports

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Literal, Sequence, Tuple, Union

import numpy as np
import torch

from parakeet_nemo_asr_rocm.chunking import (
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    segment_waveform,
)
from parakeet_nemo_asr_rocm.models.parakeet import get_model
from parakeet_nemo_asr_rocm.timestamps.models import Word
from parakeet_nemo_asr_rocm.timestamps.word_timestamps import get_word_timestamps
from parakeet_nemo_asr_rocm.utils.audio_io import DEFAULT_SAMPLE_RATE, load_audio
from parakeet_nemo_asr_rocm.utils.constant import DEFAULT_CHUNK_LEN_SEC

__all__ = ["transcribe_paths", "cli_transcribe"]


def _transcribe_chunks(
    model,
    chunks: List[Tuple[np.ndarray, float]],
    batch_size: int = 1,
    word_timestamps: bool = False,
) -> Tuple[List[str], List[List[Word]]]:
    """Transcribe audio chunks and return both text and word timestamps.

    Args:
        model: The ASR model to use for transcription.
        chunks: List of (audio_chunk, offset) tuples where offset is in seconds.
        batch_size: Number of chunks to process in each batch.
        word_timestamps: Whether to include word-level timestamps.

    Returns:
        A tuple of (texts, words_list) where:
        - texts: List of transcribed text strings
        - words_list: List of Word objects with timing information
    """
    # Separate chunks and their offsets
    chunk_audio = [chunk for chunk, _ in chunks]
    chunk_offsets = [offset for _, offset in chunks]

    # Helper for batching
    def _batch(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    texts = []
    words_list = []

    # Process in batches
    for batch_idx, batch_chunks in enumerate(_batch(chunk_audio, batch_size)):
        with torch.inference_mode():
            batch_results = model.transcribe(batch_chunks, batch_size=len(batch_chunks))

            for result in batch_results:
                # Get text and words
                text = result.text if hasattr(result, "text") else str(result)
                words = getattr(result, "words", None) if word_timestamps else None

                # Apply offset to word timings if available
                if words and batch_idx < len(chunk_offsets):
                    offset = chunk_offsets[batch_idx]
                    for word in words:
                        word.start += offset
                        word.end += offset

                texts.append(text)
                words_list.append(words or [])

                batch_idx += 1

    return texts, words_list


def _calc_time_stride(model, verbose: bool = False) -> float:
    """Return seconds-per-frame stride for timestamp conversion.

    This is copied from the previous `cli.py` implementation but moved here so that
    `cli.py` can remain a thin Typer entry-point with zero heavy runtime imports.
    The function performs a best-effort heuristic to determine the total encoder
    subsampling factor and therefore the seconds-per-frame stride used for
    timestamp conversion.
    """
    import warnings

    # 1. Determine feature hop (window) stride in seconds
    window_stride: float | None = getattr(model.cfg.preprocessor, "window_stride", None)
    if window_stride is None and hasattr(model.cfg.preprocessor, "features"):
        window_stride = getattr(model.cfg.preprocessor.features, "window_stride", None)
    if window_stride is None and hasattr(model.cfg.preprocessor, "hop_length"):
        hop = getattr(model.cfg.preprocessor, "hop_length")
        sr = getattr(model.cfg.preprocessor, "sample_rate", 16000)
        window_stride = hop / sr
    if window_stride is None:
        window_stride = 0.01  # sensible default (10 ms)

    # 2. Heuristically derive the encoder's total subsampling factor
    subsampling_factor = 1
    candidates = [
        (
            "conv_subsampling",
            lambda enc: (
                enc.conv_subsampling.get_stride()
                if hasattr(enc, "conv_subsampling")
                else None
            ),
        ),
        ("stride", lambda enc: getattr(enc, "stride", None)),
        ("subsampling_factor", lambda enc: getattr(enc, "subsampling_factor", None)),
        ("_stride", lambda enc: getattr(enc, "_stride", None)),
    ]
    enc = model.encoder
    for _name, getter in candidates:
        try:
            val = getter(enc)
        except Exception:
            val = None
        if val is not None:
            subsampling_factor = val
            break

    if subsampling_factor == 1:
        cfg_val = getattr(model.cfg.encoder, "stride", None)
        if cfg_val is not None:
            subsampling_factor = cfg_val

    if isinstance(subsampling_factor, (list, tuple)):
        from math import prod

        subsampling_factor = int(prod(subsampling_factor))
    try:
        subsampling_factor = int(subsampling_factor)
    except Exception:  # pragma: no cover
        subsampling_factor = 1
        warnings.warn("Could not determine subsampling factor; defaulting to 1.")

    return subsampling_factor * window_stride


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
    overwrite: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    no_progress: bool = False,
    fp32: bool = False,
    fp16: bool = False,
) -> None:
    """Heavy-weight implementation backing the Typer CLI.

    This function encapsulates all imports that significantly slow down process
    start-up (numpy, torch, NeMo, rich, etc.). Keeping them inside this function
    ensures that `python -m parakeet_nemo_asr_rocm.cli --help` returns almost
    instantly, while the full dependency graph is only initialised when the user
    actually runs the `transcribe` command.
    """

    # ---------------------------------------------------------------------
    # Early logging configuration based on --verbose flag
    # ---------------------------------------------------------------------
    import os

    if verbose:
        # Enable verbose logging
        os.environ["NEMO_LOG_LEVEL"] = "INFO"
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    else:
        # Suppress logs by default
        import logging
        import warnings

        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")
        os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            from functools import partial as _partial

            import tqdm as _tqdm

            _tqdm.tqdm = _partial(_tqdm.tqdm, disable=True)  # type: ignore[attr-defined]
        except ImportError:
            pass

    # Heavy imports – intentionally local to avoid slowing down `--help` calls
    import typer
    from nemo.collections.asr.parts.utils.rnnt_utils import (  # pylint: disable=import-error
        Hypothesis,
    )
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from parakeet_nemo_asr_rocm.formatting import get_formatter
    from parakeet_nemo_asr_rocm.timestamps.adapt import adapt_nemo_hypotheses
    from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult, Segment
    from parakeet_nemo_asr_rocm.utils.constant import (
        DEFAULT_STREAM_CHUNK_SEC,
        MAX_CPS,
        MAX_LINE_CHARS,
    )
    from parakeet_nemo_asr_rocm.utils.file_utils import get_unique_filename

    # ---------------------------------------------------------------------
    # Argument validation & pre-processing
    # ---------------------------------------------------------------------
    # If quiet, also disable verbose flag (after heavy imports)
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

    if not quiet:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="CLI Settings", show_header=True, header_style="bold magenta"
        )
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Setting", style="green")
        table.add_column("Value", style="yellow")

        # Model Settings
        table.add_row("Model", "Model Name", model_name)
        table.add_row("Model", "Output Directory", str(output_dir))
        table.add_row("Model", "Output Format", output_format)
        table.add_row("Model", "Output Template", output_template)

        # Processing Settings
        table.add_row("Processing", "Batch Size", str(batch_size))
        table.add_row("Processing", "Chunk Length (s)", str(chunk_len_sec))

        # Streaming Settings
        if stream:
            table.add_row("Streaming", "Stream Mode", str(stream))
            if stream_chunk_sec > 0:
                table.add_row(
                    "Streaming", "Stream Chunk Length (s)", str(stream_chunk_sec)
                )
            table.add_row("Streaming", "Overlap Duration (s)", str(overlap_duration))

        # Feature Settings
        table.add_row("Features", "Word Timestamps", str(word_timestamps))
        table.add_row("Features", "Highlight Words", str(highlight_words))
        table.add_row("Features", "Merge Strategy", merge_strategy)

        # Output Settings
        table.add_row("Output", "Overwrite", str(overwrite))
        table.add_row("Output", "Quiet Mode", str(quiet))
        table.add_row("Output", "No Progress", str(no_progress))

        # Precision Settings
        precision = "FP16" if fp16 else "FP32" if fp32 else "Default"
        table.add_row("Precision", "Mode", precision)

        # File Count
        table.add_row("Files", "Transcribing", f"{len(audio_files)} file(s)")

        console.print(table)
        typer.echo()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (cached inside get_model)
    model = get_model(model_name)
    model = model.half() if fp16 else model.float()

    try:
        formatter = get_formatter(output_format)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # ---------------------------------------------------------------------
    # Pre-compute total segments across all input files for a single progress bar
    # ---------------------------------------------------------------------
    total_segments: int = 0
    for _p in audio_files:
        _wav, _sr = load_audio(_p, DEFAULT_SAMPLE_RATE)
        total_segments += len(
            segment_waveform(_wav, _sr, chunk_len_sec, overlap_duration)
        )

    from contextlib import nullcontext

    if no_progress:
        progress_cm = nullcontext()
    else:
        progress_cm = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        )

    created_files: list[Path] = []

    with progress_cm as progress:
        if not no_progress:
            main_task = progress.add_task("Transcribing...", total=total_segments)
        else:
            main_task = None

        for file_idx, audio_path in enumerate(audio_files, start=1):
            if not no_progress:
                progress.update(main_task, description=f"Processing {audio_path.name}")

            # 1. Load and segment
            wav, _sr = load_audio(audio_path, DEFAULT_SAMPLE_RATE)
            segments = segment_waveform(wav, _sr, chunk_len_sec, overlap_duration)

            hypotheses: list[Hypothesis] = []
            transcribed_texts: list[str] = []

            def _chunks(seq, size):
                for i in range(0, len(seq), size):
                    yield seq[i : i + size]

            for batch in _chunks(segments, batch_size):
                batch_wavs = [seg for seg, _off in batch]
                batch_offsets = [_off for _seg, _off in batch]
                with torch.inference_mode():
                    results = model.transcribe(
                        audio=batch_wavs,
                        batch_size=len(batch_wavs),
                        return_hypotheses=word_timestamps,
                        verbose=False,  # suppress NeMo's internal tqdm bars
                    )
                    if not results:
                        continue

                    if word_timestamps:
                        for hyp, off in zip(results, batch_offsets):
                            setattr(hyp, "start_offset", off)
                        hypotheses.extend(results)
                    else:
                        texts = (
                            [hyp.text for hyp in results]
                            if hasattr(results[0], "text")
                            else list(results)
                        )
                        transcribed_texts.extend(texts)

                    # Advance unified progress bar by processed batch size (applies to all modes)
                    if not no_progress and main_task is not None:
                        progress.advance(main_task, len(batch_wavs))

            if word_timestamps:
                if not hypotheses:
                    typer.echo(
                        f"Warning: No transcription generated for {audio_path.name}",
                        err=True,
                    )
                    continue

                # First, let adapt_nemo_hypotheses process all hypotheses to get word timestamps
                time_stride = _calc_time_stride(model, verbose)
                if verbose:
                    typer.echo("\n" + "=" * 80)
                    typer.echo("CHUNK MERGING DIAGNOSTICS")
                    typer.echo("=" * 80)
                    typer.echo(f"• Processing {len(hypotheses)} audio chunks")
                    typer.echo(f"• Merge strategy: {merge_strategy.upper()}")
                    typer.echo(f"• Time stride: {time_stride:.6f} seconds")
                    typer.echo(f"• Overlap duration: {overlap_duration:.3f}s")

                aligned_result = adapt_nemo_hypotheses(hypotheses, model, time_stride)

                # If we have multiple chunks and merging is requested, apply the merge strategy
                if merge_strategy != "none" and len(hypotheses) > 1:
                    # --- Pairwise chunk merging (streaming path) ---
                    performed_merge = False
                    # Build per-chunk word lists from individual hypotheses
                    chunk_word_lists: list[list[Word]] = [
                        get_word_timestamps([h], model, time_stride) for h in hypotheses
                    ]

                    if verbose:
                        typer.echo("\n[MERGE] Pairwise merging of chunk word lists")
                        typer.echo(f"• Total chunks: {len(chunk_word_lists)}")

                    # Iteratively merge chunks
                    merged_words: list[Word] = chunk_word_lists[0]
                    for idx in range(1, len(chunk_word_lists)):
                        next_words = chunk_word_lists[idx]

                        if merge_strategy == "contiguous":
                            merged_words = merge_longest_contiguous(
                                merged_words,
                                next_words,
                                overlap_duration=overlap_duration,
                            )
                        else:  # "lcs" and any other value defaults to LCS
                            merged_words = merge_longest_common_subsequence(
                                merged_words,
                                next_words,
                                overlap_duration=overlap_duration,
                            )

                    performed_merge = True

                    # Post-process merged words: deduplicate and rebuild segments
                    from parakeet_nemo_asr_rocm.timestamps.segmentation import (
                        segment_words,
                    )

                    deduped: list[Word] = []
                    for w in merged_words:
                        # Skip immediate duplicates occurring within 200 ms
                        if (
                            deduped
                            and w.word == deduped[-1].word
                            and (w.start - deduped[-1].start) < 0.2
                        ):
                            continue
                        deduped.append(w)

                    merged_words = deduped

                    # Update aligned_result with cleaned word list and new segments
                    aligned_result.word_segments = merged_words
                    aligned_result.segments = segment_words(merged_words)

                    if verbose:
                        pre_merge_count = sum(len(c) for c in chunk_word_lists)
                        reduction = (
                            (pre_merge_count - len(merged_words))
                            / pre_merge_count
                            * 100.0
                            if pre_merge_count
                            else 0.0
                        )
                        typer.echo("\n[MERGE SUMMARY]")
                        typer.echo(f"• Words before: {pre_merge_count:,}")
                        typer.echo(f"• Words after:  {len(merged_words):,}")
                        typer.echo(f"• Reduction:    {reduction:.1f}%")
                    if hasattr(aligned_result, "word_segments") and not performed_merge:
                        words = aligned_result.word_segments

                        if verbose and words:
                            typer.echo("\n[PRE-MERGE ANALYSIS]")
                            typer.echo(f"• Total words: {len(words):,}")
                            typer.echo(
                                f"• Time range: {words[0].start:.2f}s - {words[-1].end:.2f}s (duration: {words[-1].end - words[0].start:.2f}s)"
                            )

                            # Calculate word density
                            duration = words[-1].end - words[0].start
                            words_per_sec = len(words) / duration if duration > 0 else 0
                            typer.echo(
                                f"• Word density: {words_per_sec:.1f} words/second"
                            )

                            # Show sample of words with timestamps
                            sample = " | ".join(
                                f"{w.word} ({w.start:.2f}-{w.end:.2f}s)"
                                for w in words[:3]
                            )
                            if len(words) > 3:
                                sample += " | ..."
                            typer.echo(f"• Sample: {sample}")

                        # Sort words by start time
                        words_sorted = sorted(words, key=lambda w: w.start)

                        # Apply merge strategy if we have words to merge
                        if len(words_sorted) > 1:
                            if verbose:
                                typer.echo("\n[APPLYING MERGE STRATEGY]")
                                typer.echo(f"• Strategy: {merge_strategy.upper()}")
                                start_time = time.time()

                            if merge_strategy == "contiguous":
                                merged_words = merge_longest_contiguous(
                                    words_sorted, [], overlap_duration=overlap_duration
                                )
                            else:  # Default to 'lcs' for any other value
                                merged_words = merge_longest_common_subsequence(
                                    words_sorted, [], overlap_duration=overlap_duration
                                )

                            if verbose:
                                merge_time = time.time() - start_time
                                typer.echo("\n[POST-MERGE RESULTS]")
                                typer.echo(
                                    f"• Merge completed in {merge_time * 1000:.1f}ms"
                                )

                                # Show before/after comparison
                                typer.echo(f"• Words before: {len(words_sorted):,}")
                                typer.echo(f"• Words after:  {len(merged_words):,}")

                                if merged_words:
                                    # Calculate reduction
                                    reduction = (
                                        (len(words_sorted) - len(merged_words))
                                        / len(words_sorted)
                                    ) * 100
                                    typer.echo(f"• Reduction:    {reduction:.1f}%")

                                    # Show time range info
                                    typer.echo(
                                        f"• Time range:   {merged_words[0].start:.2f}s - {merged_words[-1].end:.2f}s"
                                    )

                                    # Show sample of merged words with timing
                                    sample = []
                                    for i, w in enumerate(merged_words[:3]):
                                        sample.append(
                                            f"{w.word} ({w.start:.2f}-{w.end:.2f}s)"
                                        )
                                    if len(merged_words) > 3:
                                        sample.append("...")
                                    typer.echo(f"• Sample: {' | '.join(sample)}")

                                    # Check for potential issues
                                    if len(merged_words) == len(words_sorted):
                                        typer.echo(
                                            "\n[WARNING] No words were removed during merging"
                                        )
                                        typer.echo(
                                            "  • This could indicate that the merging strategy isn't working as expected"
                                        )
                                        typer.echo("  • Possible causes:")
                                        typer.echo(
                                            "    - Insufficient overlap between chunks"
                                        )
                                        typer.echo("    - Incorrect word timestamps")
                                        typer.echo(
                                            "    - Non-overlapping chunk boundaries"
                                        )

                            # Update the aligned result with merged words
                            aligned_result.word_segments = merged_words
            else:
                if output_format not in ["txt", "json"]:
                    typer.echo(
                        f"Error: Format '{output_format}' requires word timestamps. Please use --word-timestamps.",
                        err=True,
                    )
                    raise typer.Exit(code=1)

                full_text = " ".join(transcribed_texts)
                mock_segment = Segment(text=full_text, words=[], start=0, end=0)
                aligned_result = AlignedResult(
                    segments=[mock_segment], word_segments=[]
                )

            if verbose and word_timestamps:
                typer.echo("\n--- Subtitle Segments Debug ---")
                for i, seg in enumerate(aligned_result.segments[:10]):
                    chars = len(seg.text.replace("\n", " "))
                    dur = seg.end - seg.start
                    cps = chars / max(dur, 1e-3)
                    lines = seg.text.count("\n") + 1
                    flag = (
                        "⚠︎"
                        if cps > MAX_CPS
                        or any(
                            len(line) > MAX_LINE_CHARS for line in seg.text.split("\n")
                        )
                        else "OK"
                    )
                    typer.echo(
                        f"Seg {i}: {chars} chars, {dur:.2f}s, {cps:.1f} cps, {lines} lines [{flag}] -> '{seg.text.replace(chr(10), ' | ')}'"
                    )
                typer.echo("------------------------------\n")

            formatted_text = (
                formatter(aligned_result, highlight_words=highlight_words)
                if output_format.lower() in {"srt", "vtt"}
                else formatter(aligned_result)
            )

            # Build filename using template placeholders
            try:
                filename_part = output_template.format(
                    filename=audio_path.stem, index=file_idx
                )
            except KeyError as exc:  # pragma: no cover
                raise ValueError(
                    f"Unknown placeholder in --output-template: {exc}"
                ) from exc

            base_output_path = output_dir / f"{filename_part}.{output_format.lower()}"
            output_path = get_unique_filename(base_output_path, overwrite=overwrite)
            output_path.write_text(formatted_text, encoding="utf-8")

            created_files.append(output_path)

            # per-batch progress updates handled above

    # After progress bar closed, print created file paths
    if not quiet:
        for p in created_files:
            typer.echo(f'Created "{p}"')

    if verbose and not quiet:
        typer.echo("Done.")


def transcribe_paths(
    paths: Sequence[Path],
    batch_size: int = 1,
    *,
    chunk_len_sec: int = DEFAULT_CHUNK_LEN_SEC,
    overlap_duration: float = 0.0,
    merge_strategy: Literal["none", "contiguous", "lcs"] = "lcs",
    word_timestamps: bool = False,
) -> Union[List[str], List[List[Word]]]:
    """Transcribe a list of audio file paths with optional chunk merging.

    Args:
        paths: A sequence of `Path` objects pointing to audio files.
        batch_size: The batch size for model inference. GPU memory usage scales
            roughly linearly with this value. Defaults to 1.
        chunk_len_sec: The length of audio chunks in seconds to split the audio
            into before transcription. Defaults to `DEFAULT_CHUNK_LEN_SEC`.
        overlap_duration: Duration of overlap between chunks in seconds. Defaults to 0.0.
        merge_strategy: Strategy to merge overlapping chunks. One of:
            - 'none': No merging, concatenate results
            - 'contiguous': Use fast contiguous merging
            - 'lcs': Use LCS-based merging (most accurate, default)
        word_timestamps: If True, returns word-level timestamps instead of plain text.

    Returns:
        If word_timestamps is False (default): A list of transcribed text strings.
        If word_timestamps is True: A list of Word objects with timing information.
    """
    # Eager-load model (cached)
    model = get_model()
    results: Union[List[str], List[List[Word]]] = []

    for path in paths:
        # Load and segment audio with overlap
        wav, _sr = load_audio(path, DEFAULT_SAMPLE_RATE)
        segments_with_offsets = segment_waveform(
            wav, _sr, chunk_len_sec, overlap_duration=overlap_duration
        )

        # Transcribe all chunks with timing information
        texts, words_list = _transcribe_chunks(
            model=model,
            chunks=segments_with_offsets,
            batch_size=batch_size,
            word_timestamps=word_timestamps or merge_strategy != "none",
        )

        if word_timestamps:
            # Merge word timestamps according to strategy
            if len(words_list) == 0:
                results.append([])
                continue

            if merge_strategy == "none" or len(words_list) == 1:
                # Just concatenate all words
                merged_words = [word for words in words_list for word in words]
            else:
                # Merge overlapping chunks
                merged_words = words_list[0]
                for next_words in words_list[1:]:
                    # Debug: show edge words between chunks to verify textual overlap
                    if overlap_duration > 0 and merged_words and next_words:
                        tail = " | ".join(
                            f"{w.word}({w.start:.2f}s)" for w in merged_words[-3:]
                        )
                        head = " | ".join(
                            f"{w.word}({w.start:.2f}s)" for w in next_words[:3]
                        )
                        print("[DEBUG] Chunk boundary: tail ->", tail)
                        print("[DEBUG]                  head ->", head)
                    if merge_strategy == "contiguous":
                        merged_words = merge_longest_contiguous(
                            merged_words, next_words, overlap_duration=overlap_duration
                        )
                    else:  # 'lcs' or any other value defaults to LCS
                        merged_words = merge_longest_common_subsequence(
                            merged_words, next_words, overlap_duration=overlap_duration
                        )

            results.append(merged_words)
        else:
            # Simple text concatenation without merging
            results.append(" ".join(texts))

    return results
