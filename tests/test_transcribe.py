"""Placeholder tests after removing `transcribe_paths`.

This file keeps pytest happy after deprecating the old function.
"""

import pytest


@pytest.mark.skip(reason="transcribe_paths removed")
def test_transcribe_placeholder() -> None:
    """Always-pass placeholder test to satisfy pytest collection"""
    assert True
