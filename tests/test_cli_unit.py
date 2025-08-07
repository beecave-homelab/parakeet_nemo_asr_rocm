import importlib
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from parakeet_nemo_asr_rocm import cli


def test_version_callback():
    with pytest.raises(typer.Exit):
        cli.version_callback(True)


def test_main_help():
    runner = CliRunner()
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_transcribe_basic(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_text("x")
    monkeypatch.setattr(cli, "resolve_input_paths", lambda files: [audio])

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


def test_transcribe_watch_mode(monkeypatch, tmp_path):
    def fake_import_module(name):
        if name.endswith("utils.watch"):

            class Watch:
                @staticmethod
                def watch_and_transcribe(**kwargs):
                    kwargs["transcribe_fn"]([Path("file.wav")])
                    return []

            return Watch

        class Trans:
            @staticmethod
            def cli_transcribe(**kwargs):
                Trans.called = True
                return []

        return Trans

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(cli, "resolve_input_paths", lambda files: [])
    result = cli.transcribe(
        audio_files=None, watch=["*.wav"], output_dir=tmp_path, output_format="txt"
    )
    assert result == []


def test_transcribe_requires_input():
    with pytest.raises(cli.typer.BadParameter):
        cli.transcribe(audio_files=None, watch=None)
