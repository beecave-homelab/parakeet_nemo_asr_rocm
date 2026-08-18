"""Tests for merge handling inside the transcription file processor."""

from __future__ import annotations

import pytest

from parakeet_rocm.timestamps.models import AlignedResult, Segment, Word
from parakeet_rocm.transcription import file_processor as fp
from parakeet_rocm.utils.constant import DISPLAY_BUFFER_SEC


def _make_segment(words: list[Word]) -> Segment:
    return Segment(
        text=" ".join(w.word for w in words),
        words=words,
        start=words[0].start,
        end=words[-1].end + DISPLAY_BUFFER_SEC,
    )


def test_merge_word_segments_updates_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Merged words should also refresh segment timing.

    The LCS merge shifts later chunks earlier; segments must be regenerated from
    the shifted words to avoid cumulative drift in long files.
    """
    # Chunk A is on time; chunk B is 0.2 s late and contains one new word.
    chunk_a = [
        Word(word="foo", start=0.0, end=0.4),
        Word(word="bar", start=0.4, end=0.8),
    ]
    chunk_b = [
        Word(word="bar", start=0.6, end=1.0),
        Word(word="baz", start=1.0, end=1.4),
    ]

    # The adaptor returns an aligned result that still contains the unshifted
    # timestamps. _merge_word_segments() is expected to correct them.
    initial_words = chunk_a + [chunk_b[-1]]
    aligned = AlignedResult(
        segments=[_make_segment(initial_words)],
        word_segments=initial_words,
    )

    # Pin the segmentation constants this test depends on: they are read from
    # the environment at import time, so an ambient `.env` (e.g.
    # MIN_SEGMENT_DURATION_SEC=1.5) would otherwise change the expected timing.
    import parakeet_rocm.timestamps.segmentation as segmentation_mod

    monkeypatch.setattr(segmentation_mod, "MIN_SEGMENT_DURATION_SEC", 0.5)
    monkeypatch.setattr(segmentation_mod, "DISPLAY_BUFFER_SEC", 0.2)

    # Patch helpers used by _merge_word_segments
    monkeypatch.setattr(fp, "calc_time_stride", lambda _m, verbose=False: 1.0)

    import parakeet_rocm.timestamps.adapt as adapt_mod

    monkeypatch.setattr(adapt_mod, "adapt_nemo_hypotheses", lambda *_args, **_kwargs: aligned)

    word_lists = [chunk_a, chunk_b]
    monkeypatch.setattr(fp, "get_word_timestamps", lambda _h, _m, _ts: word_lists.pop(0))

    merged = fp._merge_word_segments(
        hypotheses=[object(), object()],
        model=object(),
        merge_strategy="lcs",
        overlap_duration=1,
        verbose=False,
    )

    # LCS merge shifts chunk B by -0.2s, so baz should start at 0.8 and end at 1.2
    assert merged.word_segments[-1].start == pytest.approx(0.8, rel=1e-3)
    assert merged.word_segments[-1].end == pytest.approx(1.2, rel=1e-3)

    # Segments should be regenerated from the shifted words, not the stale ones
    expected_end = merged.word_segments[-1].end + DISPLAY_BUFFER_SEC
    assert merged.segments[0].end == pytest.approx(expected_end, rel=1e-3)
    assert merged.segments[0].end < aligned.segments[0].end


def test_merge_text_segments_removes_overlap() -> None:
    left = "hello world this is"
    right = "world this is a test"

    merged = fp._merge_text_pair(left, right)
    assert merged == "hello world this is a test"


def test_merge_text_segments_no_overlap_falls_back_to_concat() -> None:
    left = "hello world"
    right = "different start"

    merged = fp._merge_text_pair(left, right)
    assert merged == "hello world different start"


def test_merge_text_pair_collapses_adjacent_repeats() -> None:
    left = "in de zorg"
    right = "in de zorg volgens mij"

    merged = fp._merge_text_pair(left, right)
    assert merged == "in de zorg volgens mij"


def test_merge_text_pair_removes_fuzzy_overlap_with_prefix_skip() -> None:
    left = "weer niet in mijn eentje ik doe het lekker samen"
    right = "ze weer niet in mijn eentje ik doe het lekker samen met martijn"

    merged = fp._merge_text_pair(left, right)
    assert merged == "weer niet in mijn eentje ik doe het lekker samen met martijn"


def test_merge_text_pair_collapses_near_duplicate_adjacent_repeats() -> None:
    left = "leuk dat jullie er zijn heel goed kun je alvast introduceren"
    right = "leuk dat jullie er zijn heel goed kun je even introduceren"

    merged = fp._merge_text_pair(left, right)
    assert merged == "leuk dat jullie er zijn heel goed kun je alvast introduceren"


def test_merge_text_pair_removes_fuzzy_overlap_with_single_token_substitution() -> None:
    left = "weer niet in mijn eentje ik doe het lekker samen"
    right = "ze weer niet in mijn eentje je doe het lekker samen met martijn"

    merged = fp._merge_text_pair(left, right)
    assert merged == "weer niet in mijn eentje ik doe het lekker samen met martijn"


def test_merge_text_pair_removes_nearby_repeated_short_sentence() -> None:
    left = "ik doe het lekker samen met Martijn. Voor de uitnodiging Arnoud."
    right = "Met Martijn. Dank voor de uitnodiging."

    merged = fp._merge_text_pair(left, right)
    assert merged.count("met Martijn.") == 1
