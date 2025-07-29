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
import warnings
from typing import Iterable, List, Sequence

import numpy as np
import torch
import typer

# pylint: disable=import-error,unused-import
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from typing_extensions import Annotated

from parakeet_nemo_asr_rocm.formatting import get_formatter
from parakeet_nemo_asr_rocm.models.parakeet import get_model
from parakeet_nemo_asr_rocm.timestamps.adapt import adapt_nemo_hypotheses
from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult, Segment
from parakeet_nemo_asr_rocm.utils.audio_io import DEFAULT_SAMPLE_RATE, load_audio
from parakeet_nemo_asr_rocm.utils.constant import DEFAULT_CHUNK_LEN_SEC
from parakeet_nemo_asr_rocm.utils.file_utils import get_unique_filename

# Create the main Typer application instance
app = typer.Typer(
    name="parakeet-rocm",
    help="A CLI for transcribing audio files using NVIDIA Parakeet-TDT via NeMo on ROCm.",
    add_completion=False,
)


def _chunks(seq: Sequence, size: int) -> Iterable[Sequence]:
    """Yield successive n-sized chunks from a sequence."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# DEPRECATED: retained temporarily for API compatibility; use chunking.chunker.segment_waveform instead.
def _segment_waveform(
    wav: np.ndarray,
    sr: int,
    chunk_len_sec: int,
    overlap_sec: int = 0,
) -> List[tuple[np.ndarray, float]]:
    """Split a mono waveform into chunks and return (segment, offset_seconds) tuples."""
    if chunk_len_sec <= 0 or len(wav) == 0:
        return [(wav, 0.0)]

    step_samples = int(max(chunk_len_sec - overlap_sec, 1) * sr)
    window_samples = int(chunk_len_sec * sr)

    segments: list[tuple[np.ndarray, float]] = []
    for start in range(0, len(wav), step_samples):
        seg = wav[start : start + window_samples]
        if seg.size == 0:
            break
        offset_sec = start / sr
        segments.append((seg, offset_sec))
    return segments


def _calc_time_stride(model, verbose: bool = False) -> float:
    """Return seconds-per-frame stride for timestamp conversion.

    Many NeMo encoders expose different APIs to indicate their total
    temporal subsampling factor. We attempt the following fall-back
    chain:
    1. `model.encoder.conv_subsampling.get_stride()` – QuartzNet style.
    2. `model.encoder.stride` – ConformerEncoder exposes this attr.
    3. `model.cfg.encoder.stride` – value in config.
    If none are found we assume no subsampling (factor=1).
    The returned stride in **seconds** is `subsampling_factor * window_stride`.
    """
    # 1. Determine feature hop (window) stride in seconds
    window_stride: float | None = getattr(model.cfg.preprocessor, "window_stride", None)
    # Some NeMo configs specify `window_stride` under `features` instead
    if window_stride is None and hasattr(model.cfg.preprocessor, "features"):
        window_stride = getattr(model.cfg.preprocessor.features, "window_stride", None)
    # Fallback: compute from hop_length / sample_rate if available
    if window_stride is None and hasattr(model.cfg.preprocessor, "hop_length"):
        hop = getattr(model.cfg.preprocessor, "hop_length")
        sr = getattr(model.cfg.preprocessor, "sample_rate", 16000)
        window_stride = hop / sr
    # Last-chance default
    if window_stride is None:
        window_stride = 0.01  # 10 ms, common for ASR models

    # 2. Heuristically derive the encoder's total subsampling factor
    subsampling_factor = 1
    candidates = [
        (
            "conv_subsampling",
            lambda enc: enc.conv_subsampling.get_stride()
            if hasattr(enc, "conv_subsampling")
            else None,
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

    # Config fallback
    if subsampling_factor == 1:
        cfg_val = getattr(model.cfg.encoder, "stride", None)
        if cfg_val is not None:
            subsampling_factor = cfg_val

    # Normalise to int
    if isinstance(subsampling_factor, (list, tuple)):
        subsampling_factor = int(np.prod(subsampling_factor))
    try:
        subsampling_factor = int(subsampling_factor)
    except Exception:  # pragma: no cover
        subsampling_factor = 1
        warnings.warn("Could not determine subsampling factor; defaulting to 1.")

    time_stride = subsampling_factor * window_stride
    if verbose:
        typer.echo(
            f"[debug] window_stride={window_stride:.5f}s, subsampling_factor={subsampling_factor}, time_stride={time_stride:.5f}s"
        )
    return time_stride


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
            help="Template for the output filename, e.g., '{filename}_{date}'."
        ),
    ] = "{filename}",
    batch_size: Annotated[
        int, typer.Option(help="Batch size for transcription inference.")
    ] = 1,
    chunk_len_sec: Annotated[
        int,
        typer.Option(help="Segment length in seconds for chunked transcription."),
    ] = DEFAULT_CHUNK_LEN_SEC,
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
    # Validate precision flags
    if fp32 and fp16:
        typer.echo("Error: Cannot specify both --fp32 and --fp16", err=True)
        raise typer.Exit(code=1)

    if verbose:
        typer.echo("--- CLI Settings ---")
        typer.echo(f"Model: {model_name}")
        typer.echo(f"Output Directory: {output_dir}")
        typer.echo(f"Output Format: {output_format}")
        typer.echo(f"Output Template: {output_template}")
        typer.echo(f"Batch Size: {batch_size}")
        typer.echo(f"Chunk Length (s): {chunk_len_sec}")
        if fp16:
            typer.echo("Precision: FP16 (half-precision)")
        else:
            typer.echo("Precision: FP32 (full-precision)")
        typer.echo(f"Transcribing {len(audio_files)} file(s)...")
        typer.echo("--------------------\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Eager-load model
    model = get_model(model_name)

    # Apply precision conversion
    if fp16:
        model = model.half()
    else:
        model = model.float()

    # Get the appropriate formatter function
    try:
        formatter = get_formatter(output_format)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        main_task = progress.add_task("Transcribing files...", total=len(audio_files))

        for audio_path in audio_files:
            progress.update(main_task, description=f"Processing {audio_path.name}")

            # 1. Load and segment the audio file
            wav, _sr = load_audio(audio_path, DEFAULT_SAMPLE_RATE)
            from parakeet_nemo_asr_rocm.chunking.chunker import segment_waveform

            segments = segment_waveform(wav, _sr, chunk_len_sec, overlap_duration)

            # 2. Transcribe all segments
            hypotheses: List[Hypothesis] = []
            transcribed_texts: List[str] = []

            for batch in _chunks(segments, batch_size):
                batch_wavs = [seg for seg, _off in batch]
                batch_offsets = [_off for _seg, _off in batch]
                with torch.inference_mode():
                    results = model.transcribe(
                        audio=batch_wavs,
                        batch_size=len(batch_wavs),
                        return_hypotheses=word_timestamps,
                    )
                    if not results:
                        continue

                    if word_timestamps:
                        # Attach start_offset (monkey-patch) so downstream can restore global time
                        for hyp, off in zip(results, batch_offsets):
                            setattr(hyp, "start_offset", off)
                        hypotheses.extend(results)
                    else:
                        # Extract text from results - handle both string and Hypothesis cases
                        if results and hasattr(results[0], "text"):
                            # Results are Hypothesis objects
                            texts = [hyp.text for hyp in results]
                        else:
                            # Results are strings
                            texts = list(results)
                        transcribed_texts.extend(texts)

            # 3. Adapt and format the results
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
                # For non-timestamp formats, create a mock AlignedResult
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
                from parakeet_nemo_asr_rocm.utils.constant import (
                    MAX_CPS,
                    MAX_LINE_CHARS,
                )

                typer.echo("\n--- Subtitle Segments Debug ---")
                for i, seg in enumerate(
                    aligned_result.segments[:10]
                ):  # show first 10 for brevity
                    chars = len(seg.text.replace("\n", " "))
                    dur = seg.end - seg.start
                    cps = chars / max(dur, 1e-3)
                    lines = seg.text.count("\n") + 1
                    flag = "OK"
                    if cps > MAX_CPS or any(
                        len(line) > MAX_LINE_CHARS for line in seg.text.split("\n")
                    ):
                        flag = "⚠︎"
                    typer.echo(
                        f"Seg {i}: {chars} chars, {dur:.2f}s, {cps:.1f} cps, {lines} lines [{flag}] -> '{seg.text.replace(chr(10), ' | ')}'"
                    )
                typer.echo("------------------------------\n")

            if output_format.lower() in {"srt", "vtt"}:
                formatted_text = formatter(
                    aligned_result, highlight_words=highlight_words
                )
            else:
                formatted_text = formatter(aligned_result)

            # 4. Save the output
            base_output_path = output_dir / f"{audio_path.stem}.{output_format.lower()}"
            output_path = get_unique_filename(
                base_output_path, overwrite=overwrite, separator="-"
            )
            output_path.write_text(formatted_text)

            # Post-process subtitles for readability if applicable
            if output_format.lower() in {"srt", "vtt"} and word_timestamps:
                try:
                    from parakeet_nemo_asr_rocm.formatting.refine import SubtitleRefiner

                    refiner = SubtitleRefiner()
                    cues = refiner.load_srt(output_path)
                    refined = refiner.refine(cues)
                    refiner.save_srt(refined, output_path)
                    if verbose:
                        typer.echo(
                            f"  · Refined subtitle timing/formatting for '{output_path.name}'."
                        )
                except Exception as exc:  # pragma: no cover – keep CLI robust
                    typer.echo(f"Warning: subtitle refinement failed: {exc}", err=True)

            action = (
                "overwritten"
                if overwrite
                and output_path == base_output_path
                and base_output_path.exists()
                else "saved"
            )
            typer.echo(
                f"  - {action.capitalize()} transcription for '{audio_path.resolve()}' to '{output_path.resolve()}'"
            )

            progress.update(main_task, advance=1)

    typer.echo("\nTranscription complete.")


if __name__ == "__main__":
    app()
