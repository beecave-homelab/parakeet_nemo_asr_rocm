"""Stable-ts integration utilities.

This module refines word timestamps using the optional stable-ts library.
"""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

from pathlib import Path
from typing import List

from parakeet_nemo_asr_rocm.timestamps.models import Word


def refine_word_timestamps(
    words: List[Word],
    audio_path: Path,
    *,
    demucs: bool = False,
    vad: bool = False,
    vad_threshold: float = 0.35,
    verbose: bool = False,
) -> List[Word]:
    """Refine word timestamps using stable-ts when available.

    Args:
        words: Initial list of :class:`Word` objects.
        audio_path: Path to the original audio file.
        demucs: Enable Demucs denoising when ``True``.
        vad: Enable voice activity detection when ``True``.
        vad_threshold: Probability threshold for VAD suppression.

    Returns:
        A list of :class:`Word` objects with potentially adjusted timestamps.

    Raises:
        RuntimeError: If the ``stable_whisper`` library is not installed.
    """
    if not words:
        return []

    try:
        import importlib

        stable_whisper = importlib.import_module("stable_whisper")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable_whisper is required for --stabilize but is not installed."
        ) from exc

    segment = {
        "start": words[0].start,
        "end": words[-1].end,
        "text": " ".join(w.word for w in words),
        "words": [{"word": w.word, "start": w.start, "end": w.end} for w in words],
    }

    options = {}
    if demucs:
        options["demucs"] = True
        # Provide legacy key for older stable-ts variants
        options["denoiser"] = "demucs"
    if vad:
        options["vad"] = True
        options["vad_threshold"] = vad_threshold
    # When either Demucs or VAD is enabled, default to stronger
    # silence-suppression based realignment so effects are observable.
    if demucs or vad:
        # Ensure suppression runs and adjusts word timestamps.
        options["suppress_silence"] = True
        options["suppress_word_ts"] = True
        # Use slightly more aggressive settings to better surface changes.
        # Fewer q_levels => higher silence threshold; smaller k_size => more detection.
        options.setdefault("q_levels", 10)
        options.setdefault("k_size", 3)
        # Allow shorter words after suppression to move boundaries more.
        options.setdefault("min_word_dur", 0.03)
        # Be strict about ordering if boundaries cross.
        options.setdefault("force_order", True)

    def _log(msg: str) -> None:  # pragma: no cover - diagnostic logging
        if verbose:
            print(f"[stable-ts] {msg}")

    # Diagnostics for environment and VAD availability
    if verbose:
        try:
            sw_ver = getattr(stable_whisper, "__version__", "unknown")
            print(f"[stable-ts] stable_whisper version: {sw_ver}")
        except Exception:
            pass
        try:
            import torch

            print(f"[stable-ts] torch version: {torch.__version__}")
        except Exception as e:
            print(f"[stable-ts] torch import error: {e}")
        try:
            import torchaudio

            try:
                backend = torchaudio.get_audio_backend()
            except Exception:
                backend = None
            print(
                f"[stable-ts] torchaudio version: {getattr(torchaudio, '__version__', 'unknown')}, backend: {backend}"
            )
            try:
                info = torchaudio.info(str(audio_path))
                sr = getattr(info, "sample_rate", "?")
                ch = getattr(info, "num_channels", "?")
                print(f"[stable-ts] torchaudio.info: sr={sr}, ch={ch}")
            except Exception as e:
                print(f"[stable-ts] torchaudio.info error: {e}")
        except Exception as e:
            print(f"[stable-ts] torchaudio import error: {e}")
        if vad:
            try:
                import importlib as _il  # type: ignore

                _il.import_module("silero_vad")
                print("[stable-ts] silero_vad import: OK")
            except Exception as e:
                print(f"[stable-ts] silero_vad import error: {e}")
        _log(
            f"options: demucs={demucs}, vad={vad}, vad_threshold={vad_threshold}, audio='{audio_path}'"
        )

    def _infer(
        _audio=None, audio=None, input=None, **_kwargs
    ):  # pragma: no cover - simple passthrough
        # We ignore [audio] since we already have the model output to refine.
        # The presence of this signature allows stable-ts to run preprocessing
        # (Demucs/VAD) and call us with the expected parameters.
        return {"segments": [segment]}

    try:
        _log("calling stable_whisper.transcribe_any(...)")
        result = stable_whisper.transcribe_any(
            _infer,
            str(audio_path),
            audio_type="str",
            check_sorted=False,
            verbose=True if verbose else False,
            regroup=True,
            **options,
        )
        segments_out = result.get("segments", []) if isinstance(result, dict) else []
        _log(f"transcribe_any returned segments={len(segments_out)}")
        # Fallback: some stable-ts versions may return 0 segments with our shim.
        # If so, try legacy postprocess_word_timestamps when available.
        if not segments_out and hasattr(stable_whisper, "postprocess_word_timestamps"):
            _log(
                "transcribe_any yielded 0 segments; falling back to postprocess_word_timestamps(...)"
            )
            processed = stable_whisper.postprocess_word_timestamps(
                {"segments": [segment]},
                audio=str(audio_path),
                **options,
            )
            segments_out = (
                processed.get("segments", []) if isinstance(processed, dict) else []
            )
    except Exception as e:  # pragma: no cover - fallback path
        if verbose:
            print(f"[stable-ts] transcribe_any error: {e}")
            import traceback as _tb

            _tb.print_exc()
        # Stable-ts 2.7.0+ recommends using transcribe_any; some legacy helper
        # functions (e.g. postprocess_word_timestamps) may not exist anymore.
        # If the legacy function is present, use it; otherwise, gracefully
        # fall back to returning the original words.
        if hasattr(stable_whisper, "postprocess_word_timestamps"):
            processed = stable_whisper.postprocess_word_timestamps(
                {"segments": [segment]},
                audio=str(audio_path),
                **options,
            )
            segments_out = (
                processed.get("segments", []) if isinstance(processed, dict) else []
            )
        else:
            segments_out = []

    refined: List[Word] = []
    for seg in segments_out:
        for w in seg.get("words", []):
            refined.append(
                Word(word=w["word"], start=w["start"], end=w["end"], score=None)
            )
    return refined or words
