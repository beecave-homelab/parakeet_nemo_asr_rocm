"""Unit tests for SRT quality metrics."""

from __future__ import annotations

import pytest

from parakeet_rocm.formatting.srt_quality import compute_srt_quality


def test_compute_srt_quality_empty_segments() -> None:
    """compute_srt_quality should return zeros when no segments provided."""
    metrics = compute_srt_quality([], "")

    assert metrics["total_segments"] == 0
    assert metrics["avg_duration_sec"] == 0.0
    assert metrics["avg_cps"] == 0.0
    assert metrics["duration_ok_rate"] == 0.0
    assert metrics["cps_ok_rate"] == 0.0
    assert metrics["line_ok_rate"] == 0.0
    assert metrics["readability_score"] == 0.0
    assert metrics["score"] == 0.0
    assert metrics["srt_length"] == 0


def test_compute_srt_quality_basic_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """compute_srt_quality should compute averages and readability rates."""
    # Pin the thresholds this test asserts against: they are read from the
    # environment at import time, so an ambient `.env` (e.g.
    # MIN_SEGMENT_DURATION_SEC=1.5, MAX_CPS=14) would otherwise flip the rates.
    import parakeet_rocm.formatting.srt_quality as srt_quality_mod

    monkeypatch.setattr(srt_quality_mod, "MIN_SEGMENT_DURATION_SEC", 0.5)
    monkeypatch.setattr(srt_quality_mod, "MAX_SEGMENT_DURATION_SEC", 7.0)
    monkeypatch.setattr(srt_quality_mod, "MIN_CPS", 10.0)
    monkeypatch.setattr(srt_quality_mod, "MAX_CPS", 22.0)
    monkeypatch.setattr(srt_quality_mod, "MAX_LINE_CHARS", 42)
    monkeypatch.setattr(srt_quality_mod, "MAX_LINES_PER_BLOCK", 2)

    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello world"},
        {"start": 1.0, "end": 2.0, "text": "this is a longer line"},
    ]
    srt_text = (
        "1\n00:00:00,000 --> 00:00:01,000\nhello world\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nthis is a longer line\n"
    )

    metrics = compute_srt_quality(segments, srt_text)

    assert metrics["total_segments"] == 2
    assert metrics["avg_duration_sec"] == pytest.approx(1.0)
    assert metrics["avg_cps"] == pytest.approx(16.0)
    assert metrics["duration_ok_rate"] == 1.0
    assert metrics["cps_ok_rate"] == 1.0
    assert metrics["line_ok_rate"] == 1.0
    assert metrics["readability_score"] == 100.0
    assert metrics["score"] == 1.0
    assert metrics["srt_length"] == len(srt_text)
