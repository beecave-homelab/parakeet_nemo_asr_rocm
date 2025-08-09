import sys
import types

from parakeet_nemo_asr_rocm.integrations.stable_ts import refine_word_timestamps
from parakeet_nemo_asr_rocm.timestamps.models import Word


def test_refine_word_timestamps(monkeypatch, tmp_path):
    """Refinement uses fallback when transcribe_any fails."""
    dummy = types.SimpleNamespace()

    def _transcribe_any(fn, audio, **kwargs):  # noqa: ARG001
        raise RuntimeError("boom")

    def _postprocess(data, audio, **kwargs):  # noqa: ARG001
        return {
            "segments": [
                {
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5},
                        {"word": "world", "start": 0.5, "end": 1.0},
                    ]
                }
            ]
        }

    dummy.transcribe_any = _transcribe_any
    dummy.postprocess_word_timestamps = _postprocess
    monkeypatch.setitem(sys.modules, "stable_whisper", dummy)

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")

    words = [Word(word="test", start=0.0, end=1.0, score=None)]
    refined = refine_word_timestamps(words, audio)
    assert [w.word for w in refined] == ["hello", "world"]
    assert refined[0].start == 0.0
    assert refined[-1].end == 1.0


def test_refine_word_timestamps_no_legacy(monkeypatch, tmp_path):
    """Gracefully degrades when legacy postprocess function is absent.

    Simulates stable_whisper exposing only transcribe_any (which raises),
    but lacking postprocess_word_timestamps. Should return original words.
    """
    dummy = types.SimpleNamespace()

    def _transcribe_any(fn, audio, **kwargs):  # noqa: ARG001
        raise RuntimeError("boom")

    dummy.transcribe_any = _transcribe_any
    # Note: intentionally DO NOT set postprocess_word_timestamps
    monkeypatch.setitem(sys.modules, "stable_whisper", dummy)

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")

    words = [Word(word="test", start=0.0, end=1.0, score=None)]
    refined = refine_word_timestamps(words, audio)
    # Without legacy API, we should get the original words back
    assert refined == words
