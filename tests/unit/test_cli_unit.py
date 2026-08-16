"""Unit tests for the top-level CLI entry points.

These tests validate help output, version callback behavior, and the
transcribe command wiring without loading heavy dependencies.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from parakeet_rocm import cli


def test_version_callback(capsys: pytest.CaptureFixture[str]) -> None:
    """Ensure ``--version`` callback prints via the shared console and exits."""
    with pytest.raises(typer.Exit):
        cli.version_callback(True)
    captured = capsys.readouterr()
    assert "parakeet-rocm version:" in captured.out


def test_main_help() -> None:
    """Invoking the app without args should print usage and exit 0."""
    runner = CliRunner()
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0
    assert "Usage" in result.stdout
    assert "api" in result.stdout


def test_transcribe_basic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify basic transcribe path resolution and delegation.

    Ensures input paths are resolved, the CLI delegates to the transcribe
    implementation, and the returned output paths are propagated.
    """
    audio = tmp_path / "a.wav"
    audio.write_text("x")
    monkeypatch.setattr(cli, "RESOLVE_INPUT_PATHS", lambda files: [audio])

    class DummyModule:
        @staticmethod
        def cli_transcribe(**kwargs: object) -> list[Path]:
            DummyModule.called = kwargs.get("audio_files")
            return [Path("out.txt")]

    def fake_import_module(name: str) -> type[DummyModule]:
        return DummyModule

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    result = cli.transcribe(audio_files=[str(audio)], output_dir=tmp_path, output_format="txt")
    assert DummyModule.called == [audio]
    assert result == [Path("out.txt")]


def test_transcribe_watch_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Invoke watcher mode and verify no immediate outputs are returned.

    When ``--watch`` is provided, the watcher module should be imported and
    used, and the watch-based transcribe flow should return no immediate
    outputs.

    This test stubs ``importlib.import_module`` and ``RESOLVE_INPUT_PATHS``
    to simulate watcher behavior and asserts that ``cli.transcribe`` returns
    an empty list.

    Parameters:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to override
            imports and functions.
        tmp_path (Path): Temporary directory used as the output directory.
    """

    def fake_import_module(_name: str) -> type[object]:
        if _name.endswith("utils.watch"):

            class Watch:
                @staticmethod
                def watch_and_transcribe(**kwargs: object) -> list[Path]:
                    kwargs["transcribe_fn"]([Path("file.wav")])
                    return []

            return Watch

        class Trans:
            @staticmethod
            def cli_transcribe(**_kwargs: object) -> list[Path]:
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
    """CLI should require at least one input source (files or --watch)."""
    with pytest.raises(cli.typer.BadParameter):
        cli.transcribe(audio_files=None, watch=None)


def test_api_command_starts_api_only_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """API command should launch uvicorn with the API-only application factory."""
    called: dict[str, object] = {}

    fake_api = types.ModuleType("parakeet_rocm.api")

    def create_api_app() -> object:
        called["created"] = True
        return object()

    fake_api.create_api_app = create_api_app
    monkeypatch.setitem(sys.modules, "parakeet_rocm.api", fake_api)

    def fake_run_uvicorn_app(**kwargs: object) -> None:
        called.update(kwargs)

    monkeypatch.setattr(cli, "_run_uvicorn_app", fake_run_uvicorn_app)

    cli.api(server_name="127.0.0.1", server_port=9000, debug=True)

    assert called["created"] is True
    assert called["server_name"] == "127.0.0.1"
    assert called["server_port"] == 9000
    assert called["debug"] is True
    assert called["share"] is False
