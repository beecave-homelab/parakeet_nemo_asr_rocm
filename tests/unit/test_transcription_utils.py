"""Unit tests for transcription utility functions."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from parakeet_rocm.transcription.utils import (
    calc_time_stride,
    compute_total_segments,
    configure_environment,
)


class TestConfigureEnvironment:
    """Tests for environment configuration utility."""

    def test_configure_environment_verbose_enabled(self) -> None:
        """Test verbose mode enables INFO logging."""
        # Arrange & Act
        configure_environment(verbose=True)

        # Assert
        assert os.environ.get("NEMO_LOG_LEVEL") == "INFO"
        assert os.environ.get("TRANSFORMERS_VERBOSITY") == "info"

    def test_configure_environment_verbose_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test non-verbose mode sets ERROR logging."""
        # Arrange
        # Pin the project defaults: they are read from the environment at
        # import time, so an ambient `.env` (e.g. TRANSFORMERS_VERBOSITY=error)
        # would otherwise change what non-verbose mode resets to.
        import parakeet_rocm.utils.constant as constant_mod

        monkeypatch.setattr(constant_mod, "NEMO_LOG_LEVEL", "ERROR")
        monkeypatch.setattr(constant_mod, "TRANSFORMERS_VERBOSITY", "ERROR")

        # Act
        configure_environment(verbose=False)

        # Assert
        # Should deterministically reset to project defaults
        assert os.environ.get("NEMO_LOG_LEVEL") == "ERROR"
        assert os.environ.get("TRANSFORMERS_VERBOSITY") == "ERROR"

    def test_configure_environment_disables_tqdm(self) -> None:
        """Test non-verbose mode attempts to disable tqdm progress bars."""
        # Arrange & Act - Just ensure it doesn't raise when tqdm is available
        try:
            configure_environment(verbose=False)
            # If tqdm is installed, it should be disabled
        except Exception:
            pytest.fail("configure_environment should not raise")

    def test_configure_environment__does_not_disable_global_logging(self) -> None:
        """Test non-verbose mode keeps centralized Python logging enabled."""
        original_disable_level = logging.root.manager.disable
        try:
            logging.disable(logging.NOTSET)
            configure_environment(verbose=False)

            assert logging.root.manager.disable == logging.NOTSET
        finally:
            logging.disable(original_disable_level)


class TestComputeTotalSegments:
    """Tests for total segment computation."""

    def test_compute_total_segments_single_file(self, tmp_path: Path) -> None:
        """Test segment counting for a single audio file."""
        # Arrange - create a test audio file
        audio_file = tmp_path / "test.wav"
        # Create a simple 2-second mono audio at 16kHz
        sample_rate = 16000
        duration_sec = 2
        audio_data = np.random.randn(duration_sec * sample_rate).astype(np.float32)

        # Save as WAV using scipy
        from scipy.io import wavfile

        wavfile.write(audio_file, sample_rate, audio_data)

        # Act
        total = compute_total_segments(
            audio_files=[audio_file],
            chunk_len_sec=1,
            overlap_duration=0,
        )

        # Assert
        # 2 seconds with 1-second chunks = 2 segments
        assert total == 2

    def test_compute_total_segments_multiple_files(self, tmp_path: Path) -> None:
        """Test segment counting for multiple audio files."""
        # Arrange - create two test audio files
        from scipy.io import wavfile

        sample_rate = 16000
        files = []

        for i in range(2):
            audio_file = tmp_path / f"test_{i}.wav"
            audio_data = np.random.randn(sample_rate).astype(np.float32)  # 1 second
            wavfile.write(audio_file, sample_rate, audio_data)
            files.append(audio_file)

        # Act
        total = compute_total_segments(
            audio_files=files,
            chunk_len_sec=1,
            overlap_duration=0,
        )

        # Assert
        # 2 files × 1 segment each = 2 segments
        assert total == 2

    def test_compute_total_segments_with_overlap(self, tmp_path: Path) -> None:
        """Test segment counting with overlap."""
        # Arrange
        from scipy.io import wavfile

        audio_file = tmp_path / "test.wav"
        sample_rate = 16000
        # 5 seconds of audio
        audio_data = np.random.randn(5 * sample_rate).astype(np.float32)
        wavfile.write(audio_file, sample_rate, audio_data)

        # Act
        total = compute_total_segments(
            audio_files=[audio_file],
            chunk_len_sec=2,
            overlap_duration=1,
        )

        # Assert
        # With 2-sec chunks and 1-sec overlap (1-sec step),
        # 5 seconds yields multiple chunks
        assert total >= 4


class TestCalcTimeStride:
    """Tests for time stride calculation."""

    def test_calc_time_stride_with_window_stride(self) -> None:
        """Test stride calculation from preprocessor.window_stride."""
        # Arrange
        mock_model = Mock()
        mock_model.cfg.preprocessor.window_stride = 0.01
        mock_model.encoder = Mock()

        # Act
        stride = calc_time_stride(mock_model, verbose=False)

        # Assert
        # With window_stride=0.01 and no subsampling, should be 0.01
        assert stride >= 0.01
