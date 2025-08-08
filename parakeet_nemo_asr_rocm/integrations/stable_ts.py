"""Stable-ts integration utilities.

This module refines word timestamps using the optional stable-ts library.
"""

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
        "words": [
            {"word": w.word, "start": w.start, "end": w.end} for w in words
        ],
    }

    options = {}
    if demucs:
        options["denoiser"] = "demucs"
    if vad:
        options["vad"] = True
        options["vad_threshold"] = vad_threshold

    def _infer(_):  # pragma: no cover - simple passthrough
        return {"segments": [segment]}

    try:
        result = stable_whisper.transcribe_any(
            _infer,
            audio=audio_path,
            **options,
        )
        segments_out = result.get("segments", []) if isinstance(result, dict) else []
    except Exception:  # pragma: no cover - fallback path
        processed = stable_whisper.postprocess_word_timestamps(
            {"segments": [segment]},
            audio=audio_path,
            **options,
        )
        segments_out = (
            processed.get("segments", []) if isinstance(processed, dict) else []
        )

    refined: List[Word] = []
    for seg in segments_out:
        for w in seg.get("words", []):
            refined.append(
                Word(word=w["word"], start=w["start"], end=w["end"], score=None)
            )
    return refined or words
