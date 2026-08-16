"""Unit tests for file_processor helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from parakeet_rocm.config import (
    OutputConfig,
    StabilizationConfig,
    UIConfig,
)
from parakeet_rocm.timestamps.models import AlignedResult, Segment, Word
from parakeet_rocm.transcription.file_processor import (
    _apply_stabilization,
    _format_and_save_output,
    _load_and_prepare_audio,
)

pytestmark = pytest.mark.integration


class TestLoadAndPrepareAudio:
    """Tests for _load_and_prepare_audio() helper function."""

    @patch("parakeet_rocm.transcription.file_processor.load_audio")
    @patch("parakeet_rocm.transcription.file_processor.segment_waveform")
    def test_load_and_prepare_audio_basic(self, mock_segment: Mock, mock_load: Mock) -> None:
        """Test basic audio loading and segmentation."""
        # Arrange
        audio_path = Path("/fake/audio.wav")
        mock_wav = np.array([0.1, 0.2, 0.3] * 16000)  # 3 seconds at 16kHz
        mock_load.return_value = (mock_wav, 16000)
        mock_segment.return_value = [
            (mock_wav[:16000], 0),
            (mock_wav[16000:], 1),
        ]

        # Act
        wav, sample_rate, segments, load_elapsed, duration_sec = _load_and_prepare_audio(
            audio_path=audio_path,
            chunk_len_sec=300,
            overlap_duration=15,
            verbose=False,
            quiet=False,
        )

        # Assert
        assert sample_rate == 16000
        assert len(segments) == 2
        assert duration_sec == pytest.approx(3.0, rel=0.01)
        assert load_elapsed >= 0
        mock_load.assert_called_once_with(audio_path, 16000)
        mock_segment.assert_called_once_with(mock_wav, 16000, 300, 15)

    @patch("parakeet_rocm.transcription.file_processor.load_audio")
    @patch("parakeet_rocm.transcription.file_processor.segment_waveform")
    @patch("parakeet_rocm.transcription.file_processor.print_status")
    def test_load_and_prepare_audio_verbose(
        self, mock_print_status: Mock, mock_segment: Mock, mock_load: Mock
    ) -> None:
        """Test verbose logging during audio loading."""
        # Arrange
        audio_path = Path("/fake/audio.wav")
        mock_wav = np.array([0.1] * 48000)  # 3 seconds at 16kHz
        mock_load.return_value = (mock_wav, 16000)
        mock_segment.return_value = [(mock_wav, 0)]

        # Act
        _load_and_prepare_audio(
            audio_path=audio_path,
            chunk_len_sec=300,
            overlap_duration=15,
            verbose=True,
            quiet=False,
        )

        # Assert - verbose should trigger Rich-backed status output.
        assert mock_print_status.call_count >= 1
        # Check that file info was logged
        call_args = str(mock_print_status.call_args_list)
        assert "audio.wav" in call_args


class TestApplyStabilization:
    """Tests for _apply_stabilization() helper function."""

    def test_apply_stabilization_disabled(self) -> None:
        """Test that stabilization is skipped when disabled."""
        # Arrange
        words = [Word(word="test", start=0.0, end=1.0, score=0.9)]
        aligned_result = AlignedResult(
            segments=[Segment(text="test", words=words, start=0.0, end=1.0)],
            word_segments=words,
        )
        config = StabilizationConfig(enabled=False)
        ui_config = UIConfig(verbose=False, quiet=False)

        # Act
        result = _apply_stabilization(
            aligned_result=aligned_result,
            audio_path=Path("/fake/audio.wav"),
            stabilization_config=config,
            ui_config=ui_config,
        )

        # Assert - should return original unchanged
        assert result is aligned_result
        assert result.word_segments == words

    @patch("parakeet_rocm.transcription.file_processor.refine_word_timestamps")
    @patch("parakeet_rocm.transcription.file_processor.segment_words")
    def test_apply_stabilization_enabled(self, mock_segment: Mock, mock_refine: Mock) -> None:
        """Test stabilization when enabled."""
        # Arrange
        original_words = [Word(word="test", start=0.0, end=1.0, score=0.9)]
        refined_words = [Word(word="test", start=0.05, end=0.95, score=0.95)]
        aligned_result = AlignedResult(
            segments=[Segment(text="test", words=original_words, start=0.0, end=1.0)],
            word_segments=original_words,
        )
        mock_refine.return_value = refined_words
        mock_segment.return_value = [
            Segment(text="test", words=refined_words, start=0.05, end=0.95)
        ]

        config = StabilizationConfig(enabled=True, demucs=False, vad=True, vad_threshold=0.35)
        ui_config = UIConfig(verbose=False, quiet=False)

        # Act
        result = _apply_stabilization(
            aligned_result=aligned_result,
            audio_path=Path("/fake/audio.wav"),
            stabilization_config=config,
            ui_config=ui_config,
        )

        # Assert
        assert result.word_segments == refined_words
        assert len(result.segments) == 1
        mock_refine.assert_called_once_with(
            original_words,
            Path("/fake/audio.wav"),
            demucs=False,
            vad=True,
            vad_threshold=0.35,
            verbose=False,
        )

    @patch("parakeet_rocm.transcription.file_processor.refine_word_timestamps")
    def test_apply_stabilization_runtime_error(self, mock_refine: Mock) -> None:
        """Test graceful handling of stabilization errors."""
        # Arrange
        words = [Word(word="test", start=0.0, end=1.0, score=0.9)]
        aligned_result = AlignedResult(
            segments=[Segment(text="test", words=words, start=0.0, end=1.0)],
            word_segments=words,
        )
        mock_refine.side_effect = RuntimeError("Stabilization failed")

        config = StabilizationConfig(enabled=True)
        ui_config = UIConfig(verbose=True, quiet=False)

        # Act - should not raise, just return original
        result = _apply_stabilization(
            aligned_result=aligned_result,
            audio_path=Path("/fake/audio.wav"),
            stabilization_config=config,
            ui_config=ui_config,
        )

        # Assert - should return original on error
        assert result is aligned_result


class TestFormatAndSaveOutput:
    """Tests for _format_and_save_output() helper function."""

    def test_format_and_save_output_basic(self, tmp_path: Path) -> None:
        """Test basic output formatting and saving."""
        # Arrange
        words = [Word(word="hello", start=0.0, end=1.0, score=0.9)]
        aligned_result = AlignedResult(
            segments=[Segment(text="hello", words=words, start=0.0, end=1.0)],
            word_segments=words,
        )
        formatter = MagicMock(return_value="formatted output")
        output_config = OutputConfig(
            output_dir=tmp_path,
            output_format="txt",
            output_template="{filename}",
            overwrite=False,
            highlight_words=False,
        )
        ui_config = UIConfig(verbose=False, quiet=False)

        # Act
        output_path = _format_and_save_output(
            aligned_result=aligned_result,
            formatter=formatter,
            output_config=output_config,
            audio_path=Path("/fake/audio.wav"),
            file_idx=1,
            watch_base_dirs=None,
            ui_config=ui_config,
        )

        # Assert
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "formatted output"
        assert output_path.name == "audio.txt"
        formatter.assert_called_once_with(aligned_result)

    def test_format_and_save_output_with_highlight(self, tmp_path: Path) -> None:
        """Test output formatting with highlight_words for SRT/VTT."""
        # Arrange
        words = [Word(word="hello", start=0.0, end=1.0, score=0.9)]
        aligned_result = AlignedResult(
            segments=[Segment(text="hello", words=words, start=0.0, end=1.0)],
            word_segments=words,
        )
        formatter = MagicMock(return_value="<u>hello</u>")
        output_config = OutputConfig(
            output_dir=tmp_path,
            output_format="srt",
            output_template="{filename}",
            overwrite=False,
            highlight_words=True,
        )
        ui_config = UIConfig(verbose=False, quiet=False)

        # Act
        output_path = _format_and_save_output(
            aligned_result=aligned_result,
            formatter=formatter,
            output_config=output_config,
            audio_path=Path("/fake/audio.wav"),
            file_idx=1,
            watch_base_dirs=None,
            ui_config=ui_config,
        )

        # Assert
        assert output_path.exists()
        formatter.assert_called_once_with(aligned_result, highlight_words=True)

    def test_format_and_save_output_with_template(self, tmp_path: Path) -> None:
        """Test output template substitution."""
        # Arrange
        words = [Word(word="test", start=0.0, end=1.0, score=0.9)]
        aligned_result = AlignedResult(
            segments=[Segment(text="test", words=words, start=0.0, end=1.0)],
            word_segments=words,
        )
        formatter = MagicMock(return_value="output")
        output_config = OutputConfig(
            output_dir=tmp_path,
            output_format="txt",
            output_template="{filename}_output_{index}",
            overwrite=False,
            highlight_words=False,
        )
        ui_config = UIConfig(verbose=False, quiet=False)

        # Act
        output_path = _format_and_save_output(
            aligned_result=aligned_result,
            formatter=formatter,
            output_config=output_config,
            audio_path=Path("/fake/audio.wav"),
            file_idx=42,
            watch_base_dirs=None,
            ui_config=ui_config,
        )

        # Assert
        assert output_path.name == "audio_output_42.txt"

    def test_format_and_save_output_subdirectory_mirroring(self, tmp_path: Path) -> None:
        """Test subdirectory structure mirroring for watch mode."""
        # Arrange
        base_dir = tmp_path / "watch"
        base_dir.mkdir()
        subdir = base_dir / "subdir"
        subdir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        words = [Word(word="test", start=0.0, end=1.0, score=0.9)]
        aligned_result = AlignedResult(
            segments=[Segment(text="test", words=words, start=0.0, end=1.0)],
            word_segments=words,
        )
        formatter = MagicMock(return_value="output")
        output_config = OutputConfig(
            output_dir=output_dir,
            output_format="txt",
            output_template="{filename}",
            overwrite=False,
            highlight_words=False,
        )
        ui_config = UIConfig(verbose=False, quiet=False)

        # Act
        output_path = _format_and_save_output(
            aligned_result=aligned_result,
            formatter=formatter,
            output_config=output_config,
            audio_path=subdir / "audio.wav",
            file_idx=1,
            watch_base_dirs=[base_dir],
            ui_config=ui_config,
        )

        # Assert - should mirror subdirectory structure
        assert output_path.parent == output_dir / "subdir"
        assert output_path.name == "audio.txt"
        assert output_path.exists()

    def test_format_and_save_output_overwrite_disabled(self, tmp_path: Path) -> None:
        """Test unique filename generation when overwrite is disabled."""
        # Arrange
        existing_file = tmp_path / "audio.txt"
        existing_file.write_text("existing", encoding="utf-8")

        words = [Word(word="test", start=0.0, end=1.0, score=0.9)]
        aligned_result = AlignedResult(
            segments=[Segment(text="test", words=words, start=0.0, end=1.0)],
            word_segments=words,
        )
        formatter = MagicMock(return_value="new output")
        output_config = OutputConfig(
            output_dir=tmp_path,
            output_format="txt",
            output_template="{filename}",
            overwrite=False,
            highlight_words=False,
        )
        ui_config = UIConfig(verbose=False, quiet=False)

        # Act
        output_path = _format_and_save_output(
            aligned_result=aligned_result,
            formatter=formatter,
            output_config=output_config,
            audio_path=Path("/fake/audio.wav"),
            file_idx=1,
            watch_base_dirs=None,
            ui_config=ui_config,
        )

        # Assert - should create unique filename
        assert output_path != existing_file
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "new output"
        # Original file should be unchanged
        assert existing_file.read_text(encoding="utf-8") == "existing"
