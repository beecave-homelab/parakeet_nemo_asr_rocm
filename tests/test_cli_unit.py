"""Unit tests for the top-level CLI entry points.

These tests validate help output, version callback behavior, and the
transcribe command wiring without loading heavy dependencies.
"""

import importlib
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from parakeet_nemo_asr_rocm import cli


def test_version_callback() -> None:
    """Ensure ``--version`` callback exits the process cleanly.

    Returns:
        None: This is a pytest test function.
    """

    with pytest.raises(typer.Exit):
        cli.version_callback(True)


def test_main_help() -> None:
    """Invoking the app without args should print usage and exit 0.

    Returns:
        None: This is a pytest test function.
    """

    runner = CliRunner()
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_transcribe_basic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Basic transcribe call should resolve inputs and return paths.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: This is a pytest test function.
    """

    audio = tmp_path / "a.wav"
    audio.write_text("x")
    monkeypatch.setattr(cli, "RESOLVE_INPUT_PATHS", lambda files: [audio])

    class DummyModule:
        @staticmethod
        def cli_transcribe(**kwargs):
            DummyModule.called = kwargs.get("audio_files")
            return [Path("out.txt")]

    def fake_import_module(name):
        return DummyModule

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    result = cli.transcribe(
        audio_files=[str(audio)], output_dir=tmp_path, output_format="txt"
    )
    assert DummyModule.called == [audio]
    assert result == [Path("out.txt")]


def test_transcribe_watch_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When --watch is used, the watcher module should be invoked.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: This is a pytest test function.
    """

    def fake_import_module(_name):
        if _name.endswith("utils.watch"):

            class Watch:
                @staticmethod
                def watch_and_transcribe(**kwargs):
                    kwargs["transcribe_fn"]([Path("file.wav")])
                    return []

            return Watch

        class Trans:
            @staticmethod
            def cli_transcribe(**_kwargs):
                Trans.called = True
                return []

        return Trans

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(cli, "RESOLVE_INPUT_PATHS", lambda files: [])
    result = cli.transcribe(
        audio_files=None, watch=["*.wav"], output_dir=tmp_path, output_format="txt"
    )
    assert result == []


def test_transcribe_requires_input() -> None:
    """CLI should require at least one input source (files or --watch).

    Returns:
        None: This is a pytest test function.
    """

    with pytest.raises(cli.typer.BadParameter):
        cli.transcribe(audio_files=None, watch=None)
