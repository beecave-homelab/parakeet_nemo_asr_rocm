"""Unit tests for transcription CLI helpers and entry points."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest
import typer

from parakeet_rocm.transcription import cli as transcription_cli


class _DummyModel:
    """Minimal model stub supporting precision methods."""

    def half(self) -> _DummyModel:
        """Return self for FP16 selection."""
        return self

    def float(self) -> _DummyModel:
        """Return self for FP32 selection."""
        return self


def test_cli_transcribe_basic_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run a minimal CLI transcription flow with stream settings."""
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    audio_a.write_text("x")
    audio_b.write_text("y")

    called: dict[str, object] = {"configs": [], "progress": []}

    def fake_configure_environment(verbose: bool) -> None:
        called["verbose"] = verbose

    def fake_compute_total_segments(
        audio_files: list[Path], chunk_len_sec: int, overlap_duration: int
    ) -> int:
        called["segments_args"] = (chunk_len_sec, overlap_duration, tuple(audio_files))
        return 2

    def fake_transcribe_file(
        audio_path: Path,
        *,
        model: object,
        formatter: object,
        file_idx: int,
        transcription_config: transcription_cli.TranscriptionConfig,
        stabilization_config: transcription_cli.StabilizationConfig,
        output_config: transcription_cli.OutputConfig,
        ui_config: transcription_cli.UIConfig,
        watch_base_dirs: list[Path] | None,
        progress: object,
        main_task: object,
        batch_progress_callback: callable | None,
        allow_unsafe_filenames: bool = False,
    ) -> Path:
        called["configs"].append((
            transcription_config.chunk_len_sec,
            transcription_config.overlap_duration,
            ui_config.quiet,
            output_config.output_format,
        ))
        if batch_progress_callback is not None:
            batch_progress_callback()
        return output_config.output_dir / f"{audio_path.stem}.txt"

    def progress_callback(current: int, total: int) -> None:
        called["progress"].append((current, total))

    monkeypatch.setattr(transcription_cli, "configure_environment", fake_configure_environment)
    monkeypatch.setattr(transcription_cli, "compute_total_segments", fake_compute_total_segments)
    monkeypatch.setattr(transcription_cli, "get_formatter", lambda _fmt: object())
    monkeypatch.setattr(transcription_cli, "transcribe_file", fake_transcribe_file)

    fake_model_module = types.ModuleType("parakeet_rocm.models.parakeet")
    fake_model_module.get_model = lambda _name: _DummyModel()
    monkeypatch.setitem(sys.modules, "parakeet_rocm.models.parakeet", fake_model_module)

    output_dir = tmp_path / "out"
    results = transcription_cli.cli_transcribe(
        audio_files=[audio_a, audio_b],
        output_dir=output_dir,
        output_format="txt",
        stream=True,
        stream_chunk_sec=0,
        overlap_duration=999,
        quiet=True,
        no_progress=True,
        progress_callback=progress_callback,
    )

    stream_chunk = transcription_cli.DEFAULT_STREAM_CHUNK_SEC
    assert output_dir.exists()
    assert results == [output_dir / "a.txt", output_dir / "b.txt"]
    assert called["segments_args"][:2] == (
        stream_chunk,
        max(0, stream_chunk // 2),
    )
    assert called["progress"] == [(1, 2), (2, 2)]
    assert called["configs"] == [
        (stream_chunk, max(0, stream_chunk // 2), True, "txt"),
        (stream_chunk, max(0, stream_chunk // 2), True, "txt"),
    ]


def test_cli_transcribe_rejects_dual_precision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure conflicting precision flags raise a Typer exit."""
    audio = tmp_path / "a.wav"
    audio.write_text("x")

    monkeypatch.setattr(transcription_cli, "configure_environment", lambda _v: None)

    with pytest.raises(typer.Exit):
        transcription_cli.cli_transcribe(
            audio_files=[audio],
            output_dir=tmp_path,
            output_format="txt",
            fp16=True,
            fp32=True,
            quiet=True,
            no_progress=True,
        )


def test_cli_transcribe_invalid_format_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure invalid output formats raise a Typer exit."""
    audio = tmp_path / "a.wav"
    audio.write_text("x")

    monkeypatch.setattr(transcription_cli, "configure_environment", lambda _v: None)
    monkeypatch.setattr(
        transcription_cli,
        "get_formatter",
        lambda _fmt: (_ for _ in ()).throw(ValueError("bad")),
    )

    fake_model_module = types.ModuleType("parakeet_rocm.models.parakeet")
    fake_model_module.get_model = lambda _name: _DummyModel()
    monkeypatch.setitem(sys.modules, "parakeet_rocm.models.parakeet", fake_model_module)

    with pytest.raises(typer.Exit):
        transcription_cli.cli_transcribe(
            audio_files=[audio],
            output_dir=tmp_path,
            output_format="bad",
            quiet=True,
            no_progress=True,
        )


def test_package_main_invokes_cli_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing `python -m parakeet_rocm` should call the CLI app."""
    called: dict[str, bool] = {}
    fake_cli = types.ModuleType("parakeet_rocm.cli")

    def app() -> None:
        called["app"] = True

    fake_cli.app = app
    monkeypatch.setitem(sys.modules, "parakeet_rocm.cli", fake_cli)
    sys.modules.pop("parakeet_rocm.__main__", None)

    runpy.run_module("parakeet_rocm.__main__", run_name="__main__")

    assert called["app"] is True


def test_webui_main_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing `python -m parakeet_rocm.webui` should call webui CLI."""
    called: dict[str, bool] = {}
    fake_cli = types.ModuleType("parakeet_rocm.webui.cli")

    def main() -> None:
        called["main"] = True

    fake_cli.main = main
    monkeypatch.setitem(sys.modules, "parakeet_rocm.webui.cli", fake_cli)
    sys.modules.pop("parakeet_rocm.webui.__main__", None)

    runpy.run_module("parakeet_rocm.webui.__main__", run_name="__main__")

    assert called["main"] is True


def test_transcribe_module_exports_cli_transcribe() -> None:
    """The compatibility wrapper should re-export cli_transcribe."""
    from parakeet_rocm import transcribe

    assert transcribe.cli_transcribe is transcription_cli.cli_transcribe


# -----------------------------------------------------------------------
# Fail-fast: filename validation exits before model loading
# -----------------------------------------------------------------------
class TestFilenameValidationFailFast:
    """Ensure invalid filenames cause early exit without loading the model."""

    def test_cli_transcribe__invalid_filename_exits_before_model_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Invalid input filename in strict mode causes exit before model load."""
        bad = tmp_path / "bad file.wav"

        model_loaded = {"called": False}

        def _assert_never_called(_name: str) -> None:
            model_loaded["called"] = True
            raise AssertionError("get_model should not be called on invalid filenames")

        monkeypatch.setattr(transcription_cli, "configure_environment", lambda _v: None)
        fake_model_module = types.ModuleType("parakeet_rocm.models.parakeet")
        fake_model_module.get_model = _assert_never_called
        monkeypatch.setitem(sys.modules, "parakeet_rocm.models.parakeet", fake_model_module)

        with pytest.raises(typer.Exit):
            transcription_cli.cli_transcribe(
                audio_files=[bad],
                output_dir=tmp_path,
                output_format="txt",
                quiet=True,
                no_progress=True,
            )

        assert not model_loaded["called"], "get_model was called despite invalid filename"

    def test_cli_transcribe__invalid_template_placeholder_exits_before_model_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Invalid output template placeholder causes exit before model load."""
        audio = tmp_path / "good.wav"
        audio.write_text("x")

        model_loaded = {"called": False}

        def _assert_never_called(_name: str) -> None:
            model_loaded["called"] = True
            raise AssertionError("get_model should not be called on invalid template")

        monkeypatch.setattr(transcription_cli, "configure_environment", lambda _v: None)
        fake_model_module = types.ModuleType("parakeet_rocm.models.parakeet")
        fake_model_module.get_model = _assert_never_called
        monkeypatch.setitem(sys.modules, "parakeet_rocm.models.parakeet", fake_model_module)

        with pytest.raises(typer.Exit):
            transcription_cli.cli_transcribe(
                audio_files=[audio],
                output_dir=tmp_path,
                output_format="txt",
                output_template="{unknown}",
                quiet=True,
                no_progress=True,
            )

        assert not model_loaded["called"], "get_model was called despite invalid template"

    def test_cli_transcribe__valid_filenames_proceed_to_model_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Valid filenames allow execution to proceed to model loading."""
        audio = tmp_path / "good.wav"
        audio.write_text("x")

        model_loaded = {"called": False}

        def _record_model_load(_name: str) -> _DummyModel:
            model_loaded["called"] = True
            return _DummyModel()

        monkeypatch.setattr(transcription_cli, "configure_environment", lambda _v: None)
        monkeypatch.setattr(transcription_cli, "compute_total_segments", lambda *_: 0)
        monkeypatch.setattr(transcription_cli, "get_formatter", lambda _fmt: object())
        monkeypatch.setattr(transcription_cli, "transcribe_file", lambda *a, **kw: None)

        fake_model_module = types.ModuleType("parakeet_rocm.models.parakeet")
        fake_model_module.get_model = _record_model_load
        monkeypatch.setitem(sys.modules, "parakeet_rocm.models.parakeet", fake_model_module)

        transcription_cli.cli_transcribe(
            audio_files=[audio],
            output_dir=tmp_path,
            output_format="txt",
            quiet=True,
            no_progress=True,
        )

        assert model_loaded["called"], "get_model was not called for valid filenames"
