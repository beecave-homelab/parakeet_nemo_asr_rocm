"""Unit tests for timestamps adapt core (lightweight, no real NeMo needed).

We monkeypatch `ASRModel`, `Hypothesis`, and `get_word_timestamps` in
`parakeet_nemo_asr_rocm.timestamps.adapt` to avoid heavy dependencies while
still satisfying type checks (Typeguard).
"""

from __future__ import annotations

from typing import List

import pytest

from parakeet_nemo_asr_rocm.timestamps import adapt as adapt_mod
from parakeet_nemo_asr_rocm.timestamps.models import AlignedResult, Word


class _DummyModel:
    pass


class _DummyHyp:
    pass


def _w(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end)


def test_adapt_returns_empty_when_no_words(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch NeMo classes in module to dummy ones for typeguard satisfaction
    monkeypatch.setattr(adapt_mod, "ASRModel", _DummyModel)
    monkeypatch.setattr(adapt_mod, "Hypothesis", _DummyHyp)

    # get_word_timestamps -> empty list
    monkeypatch.setattr(adapt_mod, "get_word_timestamps", lambda *_a, **_k: [])

    res = adapt_mod.adapt_nemo_hypotheses([], _DummyModel())
    assert isinstance(res, AlignedResult)
    assert res.segments == [] and res.word_segments == []


def test_adapt_basic_merge_and_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapt_mod, "ASRModel", _DummyModel)
    monkeypatch.setattr(adapt_mod, "Hypothesis", _DummyHyp)

    # Construct words: a too-short first phrase without punctuation, followed by another
    words: List[Word] = [
        _w("Short", 0.00, 0.20),
        _w("bit", 0.21, 0.40),
        _w("then", 0.60, 0.90),
        _w("continues.", 0.91, 1.40),  # punctuation at the end
    ]

    monkeypatch.setattr(adapt_mod, "get_word_timestamps", lambda *_a, **_k: words)

    res = adapt_mod.adapt_nemo_hypotheses([_DummyHyp()], _DummyModel())

    # Expect at least one segment; short leading should merge into following
    assert len(res.segments) >= 1

    # Final segment text should include the whole sentence and end with punctuation
    text = res.segments[0].text.replace("\n", " ").strip()
    assert text.endswith(".")
    assert "Short bit" in text and "continues" in text
