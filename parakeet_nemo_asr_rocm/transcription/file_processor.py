"""Per-file transcription processing utilities."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

from parakeet_nemo_asr_rocm.chunking import (
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    segment_waveform,
)
from parakeet_nemo_asr_rocm.integrations.stable_ts import refine_word_timestamps
from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult, Segment, Word
from parakeet_nemo_asr_rocm.timestamps.segmentation import segment_words
from parakeet_nemo_asr_rocm.timestamps.word_timestamps import get_word_timestamps
from parakeet_nemo_asr_rocm.utils.audio_io import DEFAULT_SAMPLE_RATE, load_audio
from parakeet_nemo_asr_rocm.utils.constant import MAX_CPS, MAX_LINE_CHARS
from parakeet_nemo_asr_rocm.utils.file_utils import get_unique_filename

from .utils import calc_time_stride


def _chunks(seq: Sequence, size: int) -> Sequence:
    """Yield successive chunks from *seq* of size *size*."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _transcribe_batches(
    model,
    segments: Sequence[Tuple],
    batch_size: int,
    word_timestamps: bool,
    progress,
    main_task,
    no_progress: bool,
) -> Tuple[List, List[str]]:
    """Transcribe *segments* in batches and optionally track progress.

    Args:
        model: Loaded ASR model.
        segments: Sequence of ``(audio, offset)`` tuples.
        batch_size: Number of segments per batch.
        word_timestamps: Whether to request word-level timestamps.
        progress: Rich progress instance for updates.
        main_task: Task handle within the progress bar.
        no_progress: Disable progress updates when True.

    Returns:
        A tuple of ``(hypotheses, texts)`` where *hypotheses* is a list of
        model hypotheses and *texts* the plain transcription strings.
    """
    import torch  # pylint: disable=import-outside-toplevel

    hypotheses = []
    texts: List[str] = []
    for batch in _chunks(segments, batch_size):
        batch_wavs = [seg for seg, _off in batch]
        batch_offsets = [_off for _seg, _off in batch]
        with torch.inference_mode():
            results = model.transcribe(
                audio=batch_wavs,
                batch_size=len(batch_wavs),
                return_hypotheses=word_timestamps,
                verbose=False,
            )
        if not results:
            continue
        if word_timestamps:
            for hyp, off in zip(results, batch_offsets):
                setattr(hyp, "start_offset", off)
            hypotheses.extend(results)
        else:
            texts.extend(
                [hyp.text for hyp in results]
                if hasattr(results[0], "text")
                else list(results)
            )
        if not no_progress and main_task is not None:
            progress.advance(main_task, len(batch_wavs))
    return hypotheses, texts


def _merge_word_segments(
    hypotheses,
    model,
    merge_strategy: str,
    overlap_duration: int,
    verbose: bool,
) -> AlignedResult:
    """Merge word-level hypotheses from multiple chunks.

    Args:
        hypotheses: List of model hypotheses.
        model: Loaded ASR model.
        merge_strategy: Strategy identifier (``"lcs"`` or ``"contiguous"``).
        overlap_duration: Overlap duration between chunks in seconds.
        verbose: Whether to emit diagnostic messages.

    Returns:
        An ``AlignedResult`` containing merged word segments.
    """
    from parakeet_nemo_asr_rocm.timestamps.adapt import adapt_nemo_hypotheses

    time_stride = calc_time_stride(model, verbose)
    aligned_result = adapt_nemo_hypotheses(hypotheses, model, time_stride)
    if merge_strategy != "none" and len(hypotheses) > 1:
        chunk_word_lists: List[List[Word]] = [
            get_word_timestamps([h], model, time_stride) for h in hypotheses
        ]
        merged_words: List[Word] = chunk_word_lists[0]
        for next_words in chunk_word_lists[1:]:
            if merge_strategy == "contiguous":
                merged_words = merge_longest_contiguous(
                    merged_words, next_words, overlap_duration=overlap_duration
                )
            else:
                merged_words = merge_longest_common_subsequence(
                    merged_words, next_words, overlap_duration=overlap_duration
                )
        words_sorted = sorted(merged_words, key=lambda w: w.start)
        if merge_strategy == "contiguous":
            merged_words = merge_longest_contiguous(
                words_sorted, [], overlap_duration=overlap_duration
            )
        else:
            merged_words = merge_longest_common_subsequence(
                words_sorted, [], overlap_duration=overlap_duration
            )
        aligned_result.word_segments = merged_words
    return aligned_result


def transcribe_file(
    audio_path: Path,
    *,
    model,
    formatter,
    file_idx: int,
    output_dir: Path,
    output_format: str,
    output_template: str,
    batch_size: int,
    chunk_len_sec: int,
    overlap_duration: int,
    highlight_words: bool,
    word_timestamps: bool,
    merge_strategy: str,
    stabilize: bool,
    demucs: bool,
    vad: bool,
    vad_threshold: float,
    overwrite: bool,
    verbose: bool,
    quiet: bool,
    no_progress: bool,
    progress: Any,
    main_task: Any,
) -> Path | None:
    """Transcribe a single audio file and save formatted output.

    Args:
        audio_path: Path to the audio file.
        model: Loaded ASR model.
        formatter: Output formatter callable.
        file_idx: Index of the audio file for template substitution.
        output_dir: Directory to store output files.
        output_format: Desired output format extension.
        output_template: Filename template for outputs.
        batch_size: Number of segments processed per batch.
        chunk_len_sec: Length of each chunk in seconds.
        overlap_duration: Overlap between chunks in seconds.
        highlight_words: Highlight words in output when supported.
        word_timestamps: Request word-level timestamps from the model.
        merge_strategy: Strategy for merging timestamps (``"lcs"`` or ``"contiguous"``).
        stabilize: Refine word timestamps using stable-ts when ``True``.
        demucs: Enable Demucs denoising during stabilization.
        vad: Enable voice activity detection during stabilization.
        vad_threshold: VAD probability threshold when ``vad`` is enabled.
        overwrite: Overwrite existing files when ``True``.
        verbose: Enable verbose output.
        quiet: Suppress non-error output.
        no_progress: Disable progress bar when ``True``.
        progress: Rich progress instance for updates.
        main_task: Task handle within the progress bar.

    Returns:
        Path to the created file or ``None`` if processing failed.
    """
    import time  # pylint: disable=import-outside-toplevel

    import typer  # pylint: disable=import-outside-toplevel

    t_load = time.perf_counter()
    wav, _sr = load_audio(audio_path, DEFAULT_SAMPLE_RATE)
    load_elapsed = time.perf_counter() - t_load
    duration_sec = len(wav) / float(_sr) if _sr else 0.0
    segments = segment_waveform(wav, _sr, chunk_len_sec, overlap_duration)
    if verbose and not quiet:
        typer.echo(
            "[file] "
            f"{audio_path.name}: sr={_sr}, dur={duration_sec:.2f}s, "
            f"segments={len(segments)}, chunk={chunk_len_sec}s, "
            f"overlap={overlap_duration}s, t_load={load_elapsed:.2f}s"
        )
        # show first few segment ranges
        preview = 3
        for i, (_seg, off) in enumerate(segments[:preview]):
            start = off
            end = off + chunk_len_sec
            typer.echo(f"[plan] seg{i}: {start:.2f}s→{end:.2f}s")

    t_asr = time.perf_counter()
    hypotheses, texts = _transcribe_batches(
        model, segments, batch_size, word_timestamps, progress, main_task, no_progress
    )
    asr_elapsed = time.perf_counter() - t_asr
    if verbose and not quiet:
        n_hyps = len(hypotheses) if word_timestamps else 0
        n_txt = len(texts) if not word_timestamps else 0
        typer.echo(
            f"[asr] batches done: hyps={n_hyps}, texts={n_txt}, t_asr={asr_elapsed:.2f}s"
        )
    if word_timestamps:
        if not hypotheses:
            if not quiet:
                typer.echo(
                    f"Warning: No transcription generated for {audio_path.name}",
                    err=True,
                )
            return None
        aligned_result = _merge_word_segments(
            hypotheses, model, merge_strategy, overlap_duration, verbose
        )
        if stabilize:
            try:
                # Pre-stabilization diagnostics
                pre_words: List[Word] = list(aligned_result.word_segments or [])
                if verbose and not quiet:
                    # Detect package versions without importing heavy modules
                    try:  # Python 3.10+: importlib.metadata
                        from importlib.metadata import (  # type: ignore
                            PackageNotFoundError,
                            version,
                        )
                    except ImportError:  # pragma: no cover - extremely unlikely
                        version = None  # type: ignore
                        PackageNotFoundError = Exception  # type: ignore

                    sw_ver = None
                    demucs_ver = None
                    vad_ver = None
                    if version is not None:
                        try:
                            sw_ver = version("stable-ts")
                        except PackageNotFoundError:
                            sw_ver = None
                        if sw_ver is None:
                            try:
                                sw_ver = version("stable_whisper")
                            except PackageNotFoundError:
                                sw_ver = None
                        if demucs:
                            try:
                                demucs_ver = version("demucs")
                            except PackageNotFoundError:
                                demucs_ver = None
                        if vad:
                            try:
                                vad_ver = version("silero-vad")
                            except PackageNotFoundError:
                                vad_ver = None

                    # Echo options about to be used by stable-ts
                    typer.echo(
                        "[stable-ts] preparing: "
                        f"version={sw_ver or 'unknown'} "
                        f"options={{'demucs': {demucs}, 'vad': {vad}, "
                        f"'vad_threshold': {vad_threshold if vad else None}}}"
                    )
                    if demucs:
                        typer.echo(
                            "[demucs] enabled: "
                            f"package_version={demucs_ver or 'unknown'}"
                        )
                    if vad:
                        typer.echo(
                            f"[vad] enabled: threshold={vad_threshold:.2f} "
                            f"package_version={vad_ver or 'unknown'}"
                        )
                    if demucs or vad:
                        # We default to stronger silence-suppression-based realignment
                        # so that Demucs/VAD effects are observable during stabilization.
                        typer.echo(
                            "[stable-ts] realign: suppress_silence=True "
                            "suppress_word_ts=True q_levels=10 k_size=3 "
                            "min_word_dur=0.03 force_order=True"
                        )

                t_stab = time.perf_counter()
                refined = refine_word_timestamps(
                    aligned_result.word_segments,
                    audio_path,
                    demucs=demucs,
                    vad=vad,
                    vad_threshold=vad_threshold,
                    verbose=bool(verbose and not quiet),
                )
                stab_elapsed = time.perf_counter() - t_stab
                new_segments = segment_words(refined)
                aligned_result = AlignedResult(
                    segments=new_segments,
                    word_segments=refined,
                )
                if verbose and not quiet:
                    # Keep existing summary line
                    typer.echo(
                        "[stable-ts] api=transcribe_any "
                        f"demucs={demucs} vad={vad} "
                        f"thr={vad_threshold} t_stab={stab_elapsed:.2f}s"
                    )
                    # Post-stabilization stats to help verify VAD/Demucs effects
                    n_pre = len(pre_words)
                    n_post = len(refined)
                    common = min(n_pre, n_post)
                    changed = 0
                    for i in range(common):
                        ds = abs(refined[i].start - pre_words[i].start)
                        de = abs(refined[i].end - pre_words[i].end)
                        if ds > 0.02 or de > 0.02:  # consider >20ms as a change
                            changed += 1
                    pct_changed = (100.0 * changed / common) if common else 0.0
                    start_shift = (
                        (refined[0].start - pre_words[0].start)
                        if (n_pre and n_post)
                        else 0.0
                    )
                    end_shift = (
                        (refined[-1].end - pre_words[-1].end)
                        if (n_pre and n_post)
                        else 0.0
                    )
                    words_removed = max(0, (n_pre - n_post)) if n_pre and n_post else 0
                    typer.echo(
                        "[stable-ts] result: "
                        f"segments={len(new_segments)} "
                        f"words_pre={n_pre} words_post={n_post} "
                        f"changed≈{changed} ({pct_changed:.1f}%) "
                        f"start_shift={start_shift:+.2f}s "
                        f"end_shift={end_shift:+.2f}s"
                    )
                    if vad:
                        typer.echo(f"[vad] post-stab: words_removed={words_removed}")
            except RuntimeError as exc:
                if verbose and not quiet:
                    typer.echo(f"Stabilization skipped: {exc}", err=True)
    else:
        if output_format not in ["txt", "json"]:
            if not quiet:
                typer.echo(
                    f"Error: Format '{output_format}' requires word timestamps. Please use --word-timestamps.",
                    err=True,
                )
            return None
        full_text = " ".join(texts)
        mock_segment = Segment(text=full_text, words=[], start=0, end=0)
        aligned_result = AlignedResult(segments=[mock_segment], word_segments=[])

    if verbose and word_timestamps and not quiet:
        typer.echo("\n--- Subtitle Segments Debug ---")
        for i, seg in enumerate(aligned_result.segments[:10]):
            chars = len(seg.text.replace("\n", " "))
            dur = seg.end - seg.start
            cps = chars / max(dur, 1e-3)
            lines = seg.text.count("\n") + 1
            flag = (
                "⚠︎"
                if cps > MAX_CPS
                or any(len(line) > MAX_LINE_CHARS for line in seg.text.split("\n"))
                else "OK"
            )
            typer.echo(
                f"Seg {i}: {chars} chars, {dur:.2f}s, {cps:.1f} cps, "
                f"{lines} lines [{flag}] -> '"
                f"{seg.text.replace(chr(10), ' | ')}'"
            )
        typer.echo("------------------------------\n")

    formatted_text = (
        formatter(aligned_result, highlight_words=highlight_words)
        if output_format.lower() in {"srt", "vtt"}
        else formatter(aligned_result)
    )

    try:
        filename_part = output_template.format(filename=audio_path.stem, index=file_idx)
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown placeholder in --output-template: {exc}") from exc

    base_output_path = output_dir / f"{filename_part}.{output_format.lower()}"
    output_path = get_unique_filename(base_output_path, overwrite=overwrite)
    output_path.write_text(formatted_text, encoding="utf-8")
    if verbose and not quiet:
        # Report coverage window if segments are present
        if aligned_result.segments:
            first_ts = aligned_result.segments[0].start
            last_ts = aligned_result.segments[-1].end
            typer.echo(
                "[output] "
                f"path={output_path.name} overwrite={overwrite} "
                f"blocks={len(aligned_result.segments)} "
                f"range={first_ts:.2f}s→{last_ts:.2f}s"
            )
        else:
            typer.echo(
                f"[output] path={output_path.name} overwrite={overwrite} blocks=0"
            )
    return output_path
