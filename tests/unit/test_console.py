"""Unit tests for shared Rich console helpers."""

from __future__ import annotations

import pytest
from _pytest.capture import CaptureFixture

from parakeet_rocm.utils.console import (
    get_console,
    print_error,
    print_status,
    print_success,
    print_warning,
)


def test_semantic_helpers_emit_expected_labels_and_text(
    capsys: CaptureFixture[str],
) -> None:
    """Semantic helpers should print human-readable messages through Rich."""
    print_success("done")
    print_warning("careful")
    print_error("broken")
    print_status("watch", "ready")

    captured = capsys.readouterr()

    assert "done" in captured.out
    assert "Warning: careful" in captured.out
    assert "[watch] ready" in captured.out
    assert "Error: broken" in captured.err


def test_quiet_suppresses_semantic_output(capsys: CaptureFixture[str]) -> None:
    """Quiet mode should suppress helper output consistently."""
    print_success("done", quiet=True)
    print_warning("careful", quiet=True)
    print_error("broken", quiet=True)
    print_status("watch", "ready", quiet=True)

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""


def test_status_can_emit_to_stderr(capsys: CaptureFixture[str]) -> None:
    """Status helper should preserve stderr diagnostics when requested."""
    print_status("progress", "1/2 batches", err=True)

    captured = capsys.readouterr()

    assert captured.out == ""
    assert "[progress] 1/2 batches" in captured.err


def test_console_respects_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared console construction should respect Rich's NO_COLOR handling."""
    monkeypatch.setenv("NO_COLOR", "1")
    get_console.cache_clear()

    try:
        console = get_console()
        assert console.no_color is True
        assert console.color_system is None
    finally:
        get_console.cache_clear()


def test_console_respects_dumb_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared console construction should disable color for dumb terminals."""
    monkeypatch.setenv("TERM", "dumb")
    get_console.cache_clear()

    try:
        console = get_console()
        assert console.color_system is None
    finally:
        get_console.cache_clear()
