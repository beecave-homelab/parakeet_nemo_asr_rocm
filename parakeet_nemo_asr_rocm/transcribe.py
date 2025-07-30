"""Batch transcription helper functions.

Designed to be imported *and* run as a script via ``python -m
parakeet_nemo_asr_rocm.transcribe <audio files>``.
"""

# pylint: disable=import-outside-toplevel, multiple-imports

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from parakeet_nemo_asr_rocm.chunking.chunker import segment_waveform
from parakeet_nemo_asr_rocm.models.parakeet import get_model
from parakeet_nemo_asr_rocm.utils.audio_io import DEFAULT_SAMPLE_RATE, load_audio
from parakeet_nemo_asr_rocm.utils.constant import DEFAULT_CHUNK_LEN_SEC

__all__ = ["transcribe_paths", "cli_transcribe"]


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
    # Early suppression if --quiet BEFORE importing heavy libraries
    # ---------------------------------------------------------------------
    if quiet:
        import logging
        import os
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

    if verbose:
        typer.echo("--- CLI Settings ---")
        typer.echo(f"Model: {model_name}")
        typer.echo(f"Output Directory: {output_dir}")
        typer.echo(f"Output Format: {output_format}")
        typer.echo(f"Output Template: {output_template}")
        typer.echo(f"Batch Size: {batch_size}")
        typer.echo(f"Chunk Length (s): {chunk_len_sec}")
        typer.echo("Precision: FP16" if fp16 else "Precision: FP32")
        typer.echo(f"Transcribing {len(audio_files)} file(s)...")
        typer.echo("--------------------\n")

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

                time_stride = _calc_time_stride(model, verbose)
                aligned_result = adapt_nemo_hypotheses(hypotheses, model, time_stride)
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
) -> List[str]:
    """Transcribe a list of audio file paths.

    Args:
        paths: A sequence of `Path` objects pointing to audio files.
        batch_size: The batch size for model inference. GPU memory usage scales
            roughly linearly with this value. Defaults to 1.
        chunk_len_sec: The length of audio chunks in seconds to split the audio
            into before transcription. Defaults to `DEFAULT_CHUNK_LEN_SEC`.

    Returns:
        A list of transcribed text strings, one for each input path.
    """
    # Eager-load model (cached)
    model = get_model()

    results: List[str] = []

    # Load and, if necessary, split each audio into smaller chunks
    segmented_lists: List[List[np.ndarray]] = []
    for p in paths:
        wav, _sr = load_audio(p, DEFAULT_SAMPLE_RATE)
        segments_with_offsets = segment_waveform(wav, _sr, chunk_len_sec)
        segments = [seg for seg, _offset in segments_with_offsets]
        segmented_lists.append(segments)

    # Flatten for batching
    flat_wavs: List[np.ndarray] = [
        seg for segments in segmented_lists for seg in segments
    ]
    seg_counts = [len(segs) for segs in segmented_lists]

    # Helper function for chunking sequences
    def _chunks(seq, size):
        """Yield successive n-sized chunks from a sequence."""
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

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
