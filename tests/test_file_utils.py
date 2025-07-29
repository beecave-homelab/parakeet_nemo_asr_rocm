"""Tests for file utilities."""

import pathlib
import tempfile
from typing import Generator

import pytest

from parakeet_nemo_asr_rocm.utils.file_utils import get_unique_filename


@pytest.fixture
def temp_dir() -> Generator[pathlib.Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield pathlib.Path(tmp_dir)


def test_get_unique_filename_no_conflict(temp_dir: pathlib.Path) -> None:
    """Test that original filename is returned when no conflict exists."""
    test_path = temp_dir / "test.txt"
    result = get_unique_filename(test_path)
    assert result == test_path
    assert not result.exists()


def test_get_unique_filename_with_overwrite(temp_dir: pathlib.Path) -> None:
    """Test that original filename is returned when overwrite=True."""
    test_path = temp_dir / "test.txt"
    test_path.write_text("existing content")

    result = get_unique_filename(test_path, overwrite=True)
    assert result == test_path
    assert result.exists()


def test_get_unique_filename_with_conflict(temp_dir: pathlib.Path) -> None:
    """Test that numbered suffix is appended when file exists."""
    test_path = temp_dir / "test.txt"
    test_path.write_text("existing content")

    result = get_unique_filename(test_path, separator="-")
    expected = temp_dir / "test-1.txt"
    assert result == expected
    assert not result.exists()


def test_get_unique_filename_multiple_conflicts(temp_dir: pathlib.Path) -> None:
    """Test that correct number is chosen for multiple conflicts."""
    test_path = temp_dir / "test.txt"
    test_path.write_text("existing content")

    # Create existing numbered files
    (temp_dir / "test-1.txt").write_text("content 1")
    (temp_dir / "test-2.txt").write_text("content 2")

    result = get_unique_filename(test_path, separator="-")
    expected = temp_dir / "test-3.txt"
    assert result == expected
    assert not result.exists()


def test_get_unique_filename_custom_separator(temp_dir: pathlib.Path) -> None:
    """Test that custom separator works correctly."""
    test_path = temp_dir / "test.txt"
    test_path.write_text("existing content")

    result = get_unique_filename(test_path, separator="_")
    expected = temp_dir / "test_1.txt"
    assert result == expected


def test_get_unique_filename_with_extension(temp_dir: pathlib.Path) -> None:
    """Test that file extensions are preserved correctly."""
    test_path = temp_dir / "document.srt"
    test_path.write_text("existing content")

    result = get_unique_filename(test_path, separator="-")
    expected = temp_dir / "document-1.srt"
    assert result == expected
