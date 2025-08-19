# tests/test_transcribe_and_diff.py
"""Test suite for `transcribe_and_diff.py`.

This module provides full coverage tests for the Typer-based CLI and helper
functions. It includes environment fakes for runner resolution, command
invocation recording, file system isolation, and CLI execution via Typer's
`CliRunner`.

The tests adhere to the project's coding standards:
- Google-style docstrings for all functions, classes, and methods.
- Explicit type hints for all callables.
- PEP8-compliant formatting (≤ 88 chars).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import pytest
from typer.testing import CliRunner

import scripts.transcribe_and_diff as mod


class RunRecorder:
    """Record `subprocess.run` calls and optionally fail at a chosen call.

    Attributes:
        calls: Captured command lists in call order.
        fail_on_call: 1-based index of the call that should raise an error.
        returncode: Return code to use when raising the error.
    """

    def __init__(
        self,
        fail_on_call: Optional[int] = None,
        returncode: int = 1,
    ) -> None:
        """Initialize the recorder.

        Args:
            fail_on_call: 1-based call index to raise on; None to never fail.
            returncode: Return code to use for the raised error.
        """
        self.calls: List[List[str]] = []
        self.fail_on_call = fail_on_call
        self.returncode = returncode

    def __call__(self, cmd: Sequence[str], check: bool = True) -> None:
        """Record a command invocation, optionally raising an error.

        Args:
            cmd: Command sequence requested to be executed.
            check: Unused; included to match `subprocess.run` signature.

        Raises:
            subprocess.CalledProcessError: When configured to fail on this call.
        """
        self.calls.append(list(cmd))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise mod.subprocess.CalledProcessError(
                returncode=self.returncode,
                cmd=list(cmd),
            )


def fake_which_factory(present: Tuple[str, ...]) -> Callable[[str], Optional[str]]:
    """Create a fake `shutil.which` that only finds specified commands.

    Args:
        present: Command names to be considered available.

    Returns:
        A function emulating `shutil.which`.
    """

    def _fake_which(cmd: str) -> Optional[str]:
        """Return a fake absolute path if the command is present.

        Args:
            cmd: Command to check.

        Returns:
            A fake path if present; otherwise None.
        """
        return f"/usr/bin/{cmd}" if cmd in present else None

    return _fake_which


def test_resolve_runners_pdm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure `resolve_runners` prefers PDM when available."""
    monkeypatch.setattr(mod.shutil, "which", fake_which_factory(("pdm",)))
    runners = mod.resolve_runners()
    assert runners.transcribe[:3] == ("pdm", "run", "parakeet-rocm")
    assert runners.diff_report[:4] == ("pdm", "run", "python", "-m")


def test_resolve_runners_parakeet_and_srt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure `parakeet-rocm` and `srt-diff-report` are used when available."""
    monkeypatch.setattr(
        mod.shutil,
        "which",
        fake_which_factory(("parakeet-rocm", "srt-diff-report")),
    )
    runners = mod.resolve_runners()
    assert runners.transcribe == ("parakeet-rocm",)
    assert runners.diff_report == ("srt-diff-report",)


def test_resolve_runners_python_fallback_with_srt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure Python module transcriber is used with `srt-diff-report` fallback."""
    monkeypatch.setattr(mod.shutil, "which", fake_which_factory(("srt-diff-report",)))
    runners = mod.resolve_runners()
    assert runners.transcribe[:3] == ("python", "-m", "parakeet_nemo_asr_rocm.cli")
    assert runners.diff_report == ("srt-diff-report",)


def test_find_srt_exact_and_newest(tmp_path: Path) -> None:
    """Validate exact match and newest pattern matching for SRT discovery."""
    directory = tmp_path
    exact = directory / "clip.srt"
    older = directory / "clip_2020.srt"
    newer = directory / "clip_2024.srt"

    older.write_text("old", encoding="utf-8")
    os.utime(older, (1, 1))

    newer.write_text("new", encoding="utf-8")
    os.utime(newer, (2, 2))

    # Newest pattern when no exact file exists.
    found = mod.find_srt(directory, "clip")
    assert found == newer

    # Exact match wins when present.
    exact.write_text("exact", encoding="utf-8")
    os.utime(exact, (3, 3))
    found2 = mod.find_srt(directory, "clip")
    assert found2 == exact


def test_transcribe_three_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify three transcriptions are invoked with correct flags."""
    recorder = RunRecorder()
    monkeypatch.setattr(mod, "run", recorder)
    input_file = tmp_path / "audio.wav"
    input_file.write_bytes(b"RIFF....")

    runners = mod.Runners(
        transcribe=("parakeet-rocm",),
        diff_report=("srt-diff-report",),
    )

    mod.transcribe_three(runners, input_file)

    assert len(recorder.calls) == 3
    call1, call2, call3 = recorder.calls
    for call in (call1, call2, call3):
        assert "transcribe" in call
        assert "--word-timestamps" in call
        assert "--output-format" in call and "srt" in call

    assert call1[call1.index("--output-dir") + 1] == str(mod.D_DEFAULT)
    assert "--stabilize" in call2 and str(input_file) in call2
    assert all(flag in call3 for flag in ("--stabilize", "--vad", "--demucs"))


def _seed_srts(tmp_path: Path, stem: str = "audio") -> None:
    """Create minimal SRT files for testing report generation.

    Args:
        tmp_path: Base temporary path for the test run.
        stem: Base name used for SRT files.
    """
    mod.D_DEFAULT.mkdir(parents=True, exist_ok=True)
    mod.D_STABILIZE.mkdir(parents=True, exist_ok=True)
    mod.D_SVD.mkdir(parents=True, exist_ok=True)
    (mod.D_DEFAULT / f"{stem}.srt").write_text("default", encoding="utf-8")
    (mod.D_STABILIZE / f"{stem}.srt").write_text("stab", encoding="utf-8")
    (mod.D_SVD / f"{stem}.srt").write_text("svd", encoding="utf-8")


@pytest.fixture(autouse=True)
def isolate_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect module constants to isolated temp directories for each test.

    This fixture is autouse to ensure filesystem side-effects are contained.
    It rewires `D_DEFAULT`, `D_STABILIZE`, `D_SVD`, and `DEFAULT_OUT_DIR`.
    """
    monkeypatch.setattr(mod, "D_DEFAULT", tmp_path / "default")
    monkeypatch.setattr(mod, "D_STABILIZE", tmp_path / "stabilize")
    monkeypatch.setattr(mod, "D_SVD", tmp_path / "svd")
    monkeypatch.setattr(mod, "DEFAULT_OUT_DIR", tmp_path / "out")


def test_report_diffs_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generate reports for all pairs; verify invocations and outputs."""
    _seed_srts(tmp_path)
    recorder = RunRecorder()
    monkeypatch.setattr(mod, "run", recorder)

    runners = mod.Runners(
        transcribe=("parakeet-rocm",),
        diff_report=("srt-diff-report",),
    )
    out_dir = tmp_path / "reports"

    mod.report_diffs(
        runners=runners,
        stem="audio",
        out_dir=out_dir,
        show_violations=5,
    )

    # 3 pairs × 2 formats = 6 calls
    assert len(recorder.calls) == 6
    for call in recorder.calls:
        assert "--output-format" in call
        assert "-o" in call
        assert "--show-violations" in call
        out_index = call.index("-o") + 1
        assert Path(call[out_index]).parent == out_dir


def test_report_diffs_missing_files_raises(tmp_path: Path) -> None:
    """Ensure missing SRTs lead to a helpful FileNotFoundError."""
    runners = mod.Runners(("parakeet-rocm",), ("srt-diff-report",))
    with pytest.raises(FileNotFoundError) as exc:
        mod.report_diffs(
            runners=runners,
            stem="nope",
            out_dir=tmp_path / "out",
            show_violations=0,
        )
    assert "Missing SRT(s)" in str(exc.value) and "nope" in str(exc.value)


def test_cli_transcribe_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run CLI in transcribe-only mode; expect three invocations."""
    input_file = tmp_path / "a.wav"
    input_file.write_bytes(b"\x00\x01")

    recorder = RunRecorder()
    monkeypatch.setattr(mod, "run", recorder)
    monkeypatch.setattr(mod, "resolve_runners", lambda: mod.Runners(("x",), ("y",)))

    runner = CliRunner()
    result = runner.invoke(mod.app, ["run", str(input_file), "--transcribe"])
    assert result.exit_code == 0
    assert len(recorder.calls) == 3


def test_cli_report_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run CLI in report-only mode; expect six invocations."""
    input_file = tmp_path / "b.wav"
    input_file.write_bytes(b"\x00\x01")
    _seed_srts(tmp_path, stem="b")

    recorder = RunRecorder()
    monkeypatch.setattr(mod, "run", recorder)
    monkeypatch.setattr(mod, "resolve_runners", lambda: mod.Runners(("x",), ("y",)))

    runner = CliRunner()
    result = runner.invoke(
        mod.app, ["run", str(input_file), "--report", "--show-violations", "2"]
    )
    assert result.exit_code == 0
    assert len(recorder.calls) == 6


def test_cli_both_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run CLI with no flags; expect transcribe + report (3 + 6 calls)."""
    input_file = tmp_path / "c.wav"
    input_file.write_bytes(b"\x00\x01")
    _seed_srts(tmp_path, stem="c")

    recorder = RunRecorder()
    monkeypatch.setattr(mod, "run", recorder)
    monkeypatch.setattr(mod, "resolve_runners", lambda: mod.Runners(("x",), ("y",)))

    runner = CliRunner()
    result = runner.invoke(mod.app, ["run", str(input_file)])
    assert result.exit_code == 0
    assert len(recorder.calls) == 9


def test_cli_mutually_exclusive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure CLI rejects simultaneous --transcribe and --report flags."""
    input_file = tmp_path / "d.wav"
    input_file.write_bytes(b"\x00\x01")

    runner = CliRunner()
    result = runner.invoke(
        mod.app, ["run", str(input_file), "--transcribe", "--report"]
    )
    assert result.exit_code != 0
    assert "ONLY one of --transcribe or --report" in result.stdout


def test_cli_subprocess_failure_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Propagate subprocess return code on failure."""
    input_file = tmp_path / "e.wav"
    input_file.write_bytes(b"\x00\x01")

    failing = RunRecorder(fail_on_call=1, returncode=7)
    monkeypatch.setattr(mod, "run", failing)
    monkeypatch.setattr(mod, "resolve_runners", lambda: mod.Runners(("x",), ("y",)))

    runner = CliRunner()
    result = runner.invoke(mod.app, ["run", str(input_file), "--transcribe"])
    assert result.exit_code == 7


def test_cli_missing_file_validation(tmp_path: Path) -> None:
    """Validate Typer's file existence handling for the required argument."""
    runner = CliRunner()
    result = runner.invoke(mod.app, ["run", str(tmp_path / "nope.wav")])
    assert result.exit_code != 0
    assert "Invalid value for 'AUDIO_FILE'" in result.stdout
