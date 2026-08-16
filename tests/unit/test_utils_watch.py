"""Unit tests for parakeet_rocm.utils.watch module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from parakeet_rocm.utils.watch import (
    _needs_transcription,
    watch_and_transcribe,
)


def test_needs_transcription_with_watch_base_dirs_exception() -> None:
    """Tests _needs_transcription when relative_to raises exception."""
    # Create test paths
    audio_path = Path("/fake/path/audio.wav")
    output_dir = Path("/fake/output")
    base_dirs = [Path("/fake/base")]

    # Mock relative_to to raise exception
    with patch.object(Path, "relative_to", side_effect=ValueError("not relative")):
        result = _needs_transcription(
            audio_path, output_dir, "{filename}", "txt", watch_base_dirs=base_dirs
        )

    # Should continue to next base dir and use default output_dir
    assert result is True  # No output files exist


def test_needs_transcription_with_subdirectory_mirroring(tmp_path: Path) -> None:
    """Tests _needs_transcription with subdirectory mirroring."""
    # Create test paths with subdirectory structure
    base_dir = tmp_path / "base"
    audio_path = base_dir / "subdir" / "audio.wav"
    output_dir = tmp_path / "output"
    base_dirs = [base_dir]

    # Create output subdirectory with existing file
    output_subdir = output_dir / "subdir"
    output_subdir.mkdir(parents=True, exist_ok=True)
    existing_file = output_subdir / "audio.txt"
    existing_file.write_text("test")

    result = _needs_transcription(
        audio_path,
        output_dir,
        "{filename}",
        "txt",
        watch_base_dirs=base_dirs,
    )

    # Should find existing file in mirrored subdirectory
    assert result is False


def test_needs_transcription_target_dir_not_exists() -> None:
    """Tests _needs_transcription when target directory doesn't exist."""
    audio_path = Path("/fake/audio.wav")
    output_dir = Path("/fake/nonexistent")

    result = _needs_transcription(audio_path, output_dir, "{filename}", "txt")

    # Should return True when target dir doesn't exist
    assert result is True


@patch("parakeet_rocm.utils.watch.time.monotonic")
@patch("parakeet_rocm.utils.watch.resolve_input_paths")
@patch("parakeet_rocm.utils.watch.unload_model_to_cpu")
@patch("parakeet_rocm.utils.watch.clear_model_cache")
def test_watch_and_transcribe_idle_handling(
    mock_clear_cache: MagicMock,
    mock_unload: MagicMock,
    mock_resolve: MagicMock,
    mock_monotonic: MagicMock,
    tmp_path: Path,
) -> None:
    """Tests idle model unloading and cache clearing in watch_and_transcribe."""
    from parakeet_rocm.utils.constant import IDLE_CLEAR_TIMEOUT_SEC, IDLE_UNLOAD_TIMEOUT_SEC

    # Setup mock time progression
    time_values = [0.0, IDLE_UNLOAD_TIMEOUT_SEC + 1.0, IDLE_CLEAR_TIMEOUT_SEC + 1.0]
    mock_monotonic.side_effect = time_values

    # Mock no files found
    mock_resolve.return_value = []

    # Mock transcribe function
    transcribe_mock = MagicMock()

    # Start watch but break after idle handling for unload and clear
    call_count = 0

    def mock_sleep(*_args: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt()

    with patch("time.sleep", side_effect=mock_sleep):
        try:
            watch_and_transcribe(
                patterns=[tmp_path],
                transcribe_fn=transcribe_mock,
                poll_interval=0.1,
                output_dir=tmp_path,
                output_format="txt",
                output_template="{filename}",
                verbose=False,
            )
        except KeyboardInterrupt:
            pass

    # Should have called unload and clear due to idle timeout
    mock_unload.assert_called_once()
    mock_clear_cache.assert_called_once()


@patch("parakeet_rocm.utils.watch.time.monotonic")
@patch("parakeet_rocm.utils.watch.resolve_input_paths")
def test_watch_and_transcribe_verbose_logging(
    mock_resolve: MagicMock,
    mock_monotonic: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tests verbose logging output in watch_and_transcribe."""
    # Setup mock time
    mock_monotonic.return_value = 0.0

    # Create test audio file
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    # Mock file resolution
    mock_resolve.return_value = [audio_file]

    # Mock transcribe function
    transcribe_mock = MagicMock()

    # Start watch but break after first iteration
    with patch("time.sleep", side_effect=KeyboardInterrupt()):
        try:
            watch_and_transcribe(
                patterns=[tmp_path],
                transcribe_fn=transcribe_mock,
                poll_interval=0.1,
                output_dir=tmp_path,
                output_format="txt",
                output_template="{filename}",
                verbose=True,
            )
        except KeyboardInterrupt:
            pass

    # Check verbose output
    captured = capsys.readouterr()
    assert "[watch] Monitoring" in captured.out
    assert "[watch] Scan found 1 candidate file(s)" in captured.out
    assert "[watch] Found 1 new file(s):" in captured.out
    assert f"- {audio_file}" in captured.out
    # Status helper adds Rich markup; verify it is rendered rather than leaked.
    assert "[cyan bold]" not in captured.out


@patch("parakeet_rocm.utils.watch.time.monotonic")
@patch("parakeet_rocm.utils.watch.resolve_input_paths")
def test_watch_and_transcribe_verbose_already_processed(
    mock_resolve: MagicMock,
    mock_monotonic: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tests verbose logging for already processed files."""
    # Setup mock time
    mock_monotonic.return_value = 0.0

    # Create test audio file
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    # Mock file resolution returning same file multiple times
    mock_resolve.return_value = [audio_file]

    # Mock transcribe function
    transcribe_mock = MagicMock()

    # Track seen files by simulating multiple iterations
    call_count = 0

    def mock_sleep(*_args: object) -> None:
        """Simulate controlled sleep in tests, stopping after a fixed call count.

        Raises:
            KeyboardInterrupt: When calls reach 2, breaking the watch loop.
        """
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt()

    # Start watch
    with patch("time.sleep", side_effect=mock_sleep):
        try:
            watch_and_transcribe(
                patterns=[tmp_path],
                transcribe_fn=transcribe_mock,
                poll_interval=0.1,
                output_dir=tmp_path,
                output_format="txt",
                output_template="{filename}",
                verbose=True,
            )
        except KeyboardInterrupt:
            pass

    # Check verbose output for already processed file
    captured = capsys.readouterr()
    assert "[watch] ✗ Already processed:" in captured.out


@patch("parakeet_rocm.utils.watch.time.monotonic")
@patch("parakeet_rocm.utils.watch.resolve_input_paths")
def test_watch_and_transcribe_verbose_output_exists(
    mock_resolve: MagicMock,
    mock_monotonic: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tests verbose logging when output already exists."""
    # Setup mock time
    mock_monotonic.return_value = 0.0

    # Create test audio file and existing output
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")
    output_file = tmp_path / "test.txt"
    output_file.write_text("existing output")

    # Mock file resolution
    mock_resolve.return_value = [audio_file]

    # Mock transcribe function
    transcribe_mock = MagicMock()

    # Start watch but break after first iteration
    with patch("time.sleep", side_effect=KeyboardInterrupt()):
        try:
            watch_and_transcribe(
                patterns=[tmp_path],
                transcribe_fn=transcribe_mock,
                poll_interval=0.1,
                output_dir=tmp_path,
                output_format="txt",
                output_template="{filename}",
                verbose=True,
            )
        except KeyboardInterrupt:
            pass

    # Check verbose output for skipped file
    captured = capsys.readouterr()
    assert "[watch] ✗ Output exists, skipping:" in captured.out


@patch("parakeet_rocm.utils.watch.time.monotonic")
@patch("parakeet_rocm.utils.watch.resolve_input_paths")
def test_watch_and_transcribe_verbose_no_new_files(
    mock_resolve: MagicMock,
    mock_monotonic: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tests verbose logging when no new files are found."""
    # Setup mock time
    mock_monotonic.return_value = 0.0

    # Mock no files found
    mock_resolve.return_value = []

    # Mock transcribe function
    transcribe_mock = MagicMock()

    # Start watch but break after first iteration
    with patch("time.sleep", side_effect=KeyboardInterrupt()):
        try:
            watch_and_transcribe(
                patterns=[tmp_path],
                transcribe_fn=transcribe_mock,
                poll_interval=0.1,
                output_dir=tmp_path,
                output_format="txt",
                output_template="{filename}",
                verbose=True,
            )
        except KeyboardInterrupt:
            pass

    # Check verbose output for no new files
    captured = capsys.readouterr()
    assert "[watch] No new files - waiting…" in captured.out


@patch("parakeet_rocm.utils.watch.time.monotonic")
@patch("parakeet_rocm.utils.watch.resolve_input_paths")
@patch("parakeet_rocm.utils.watch.unload_model_to_cpu")
def test_watch_and_transcribe_activity_resets_idle_state(
    mock_unload: MagicMock,
    mock_resolve: MagicMock,
    mock_monotonic: MagicMock,
    tmp_path: Path,
) -> None:
    """Tests that new activity resets idle state flags."""
    from parakeet_rocm.utils.constant import IDLE_UNLOAD_TIMEOUT_SEC

    # Setup mock time to trigger idle, then reset
    time_values = [
        0.0,  # Initial
        IDLE_UNLOAD_TIMEOUT_SEC + 1.0,  # Trigger idle
        IDLE_UNLOAD_TIMEOUT_SEC + 2.0,  # New activity
        IDLE_UNLOAD_TIMEOUT_SEC + 3.0,  # Check idle again
    ]
    mock_monotonic.side_effect = time_values

    # Create test audio file
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    # Mock file resolution - no files first, then files
    mock_resolve.side_effect = [[], [audio_file]]

    # Mock transcribe function
    transcribe_mock = MagicMock()

    # Track iterations
    call_count = 0

    def mock_sleep(*args: object) -> None:
        """Local helper defining a short loop-controlled sleep in tests.

        Raises:
            KeyboardInterrupt: When the loop count reaches 2.
        """
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt()

    # Start watch
    with patch("time.sleep", side_effect=mock_sleep):
        try:
            watch_and_transcribe(
                patterns=[tmp_path],
                transcribe_fn=transcribe_mock,
                poll_interval=0.1,
                output_dir=tmp_path,
                output_format="txt",
                output_template="{filename}",
                verbose=False,
            )
        except KeyboardInterrupt:
            pass

    # Should have called unload once, but new activity should have reset state
    assert mock_unload.call_count == 1
    assert transcribe_mock.call_count == 1


@patch("parakeet_rocm.utils.watch.resolve_input_paths")
def test_watch_cooperative_sigint_shutdown(
    mock_resolve: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SIGINT sets the stop event; watch_and_transcribe exits cleanly without sys.exit."""
    import signal

    from parakeet_rocm.utils.watch import _default_sig_handler, _stop_event

    # Ensure the stop event is clear before the test
    _stop_event.clear()

    # Mock no files found so the loop stays idle
    mock_resolve.return_value = []

    transcribe_mock = MagicMock()

    # Simulate SIGINT after one sleep cycle
    call_count = 0

    def mock_sleep(*_args: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            # Simulate what the signal handler does
            _default_sig_handler(signal.SIGINT, None)

    original_sig = signal.getsignal(signal.SIGINT)
    with patch("time.sleep", side_effect=mock_sleep):
        watch_and_transcribe(
            patterns=[tmp_path],
            transcribe_fn=transcribe_mock,
            poll_interval=0.1,
            output_dir=tmp_path,
            output_format="txt",
            output_template="{filename}",
            verbose=False,
        )

    # Function should have returned normally (no SystemExit / KeyboardInterrupt)
    captured = capsys.readouterr()
    assert "[watch] Stopping…" in captured.out

    # The original SIGINT handler must be restored
    assert signal.getsignal(signal.SIGINT) is original_sig

    # The stop event must be cleared after exit (ready for reuse)
    assert not _stop_event.is_set()
