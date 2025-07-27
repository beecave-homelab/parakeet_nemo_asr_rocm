from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from parakeet_nemo_asr_rocm.cli_ty import app as cli_app

# Path to sample audio for tests
AUDIO_PATH = Path(__file__).parents[2] / "data" / "samples" / "sample.wav"

pytestmark = pytest.mark.skipif(
    not AUDIO_PATH.is_file(), reason="sample.wav not present for CLI test"
)


def _invoke_cli(*args: str):
    """Utility to invoke the Typer CLI and return result."""
    runner = CliRunner(mix_stderr=False)
    return runner.invoke(cli_app, list(args))


@pytest.mark.skipif(os.getenv("CI") == "true", reason="GPU-heavy test skipped in CI")
def test_cli_txt(tmp_path):
    """Smoke-test CLI transcribe to TXT output without word timestamps."""
    outdir = tmp_path / "out"
    result = _invoke_cli(
        "transcribe",
        str(AUDIO_PATH),
        "--output-dir",
        str(outdir),
        "--output-format",
        "txt",
    )
    # Typer returns 0 on success
    assert result.exit_code == 0, result.stderr
    txt_files = list(outdir.glob("*.txt"))
    assert txt_files, "No TXT file produced"
    assert txt_files[0].read_text().strip(), "TXT file is empty"


@pytest.mark.skipif(os.getenv("CI") == "true", reason="GPU-heavy test skipped in CI")
def test_cli_srt_word_timestamps(tmp_path):
    """CLI should produce SRT when word timestamps enabled."""
    outdir = tmp_path / "out"
    result = _invoke_cli(
        "transcribe",
        str(AUDIO_PATH),
        "--output-dir",
        str(outdir),
        "--output-format",
        "srt",
        "--word-timestamps",
    )
    assert result.exit_code == 0, result.stderr
    srt_files = list(outdir.glob("*.srt"))
    assert srt_files, "No SRT file produced"
    first_lines = srt_files[0].read_text().splitlines()[:4]
    # Basic SRT structure check
    assert first_lines and first_lines[0].strip() == "1", "Invalid SRT format"

