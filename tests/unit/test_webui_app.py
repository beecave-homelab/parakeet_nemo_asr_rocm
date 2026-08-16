"""Unit tests for the Gradio WebUI app module.

These tests use lightweight fake modules (gradio/torch/scipy) so that
`parakeet_rocm.webui.app` can be imported and exercised without requiring
GPU or UI dependencies.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest


class _FakeProgress:
    """Callable progress tracker compatible with `gr.Progress()` usage."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, str]] = []

    def __call__(self, value: float, *, desc: str = "") -> None:
        self.calls.append((value, desc))


class _FakeComponent:
    """Base class for fake Gradio components."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self._change_fn: Callable[..., object] | None = None
        self._click_fn: Callable[..., object] | None = None

    def change(
        self,
        *,
        fn: Callable[..., object],
        inputs: list[object],
        outputs: list[object],
    ) -> None:
        self._change_fn = fn

    def click(
        self,
        *,
        fn: Callable[..., object],
        inputs: list[object],
        outputs: list[object],
    ) -> None:
        self._click_fn = fn


class _FakeContext:
    """Context manager used by layout primitives (Row/Group/etc.)."""

    def __enter__(self) -> _FakeContext:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object | None,
    ) -> None:
        return None


class _FakeBlocks(_FakeContext):
    """Fake `gr.Blocks` root with a `.launch()` method."""

    def __init__(self, *_args: object, **kwargs: object) -> None:
        super().__init__()
        self.init_kwargs: dict[str, object] = dict(kwargs)
        self.launch_calls: list[dict[str, object]] = []

    def launch(self, **kwargs: object) -> None:
        self.launch_calls.append(dict(kwargs))


def _install_fake_gradio(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake `gradio` module into `sys.modules`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The fake `gradio` module.
    """
    gr = types.ModuleType("gradio")
    gr._created: list[_FakeComponent] = []

    def _register(obj: _FakeComponent) -> _FakeComponent:
        gr._created.append(obj)
        return obj

    def update(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    class _Themes:
        class Color:  # noqa: D106
            pass

        class Soft:  # noqa: D106
            def __init__(self, **_kwargs: object) -> None:
                return None

            def set(self, **_kwargs: object) -> _Themes.Soft:
                return cast(_Themes.Soft, self)

    gr.themes = _Themes

    gr.Blocks = _FakeBlocks
    gr.Group = _FakeContext
    gr.Row = _FakeContext
    gr.Column = _FakeContext
    gr.Tabs = _FakeContext
    gr.TabItem = lambda *_args, **_kwargs: _FakeContext()  # type: ignore[assignment]
    gr.Accordion = lambda *_args, **_kwargs: _FakeContext()  # type: ignore[assignment]

    gr.State = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.Markdown = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.File = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.Dropdown = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.Slider = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.Checkbox = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.Radio = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.Button = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.Textbox = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.DownloadButton = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]
    gr.JSON = lambda *_args, **_kwargs: _register(_FakeComponent(*_args, **_kwargs))  # type: ignore[assignment]

    gr.Progress = _FakeProgress
    gr.update = update

    monkeypatch.setitem(sys.modules, "gradio", gr)
    return gr


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake `torch` module into `sys.modules`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

    torch.cuda = _Cuda
    monkeypatch.setitem(sys.modules, "torch", torch)


def _install_fake_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake `scipy` module (with `scipy.linalg`) into `sys.modules`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    scipy = types.ModuleType("scipy")
    scipy_linalg = types.ModuleType("scipy.linalg")
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.linalg", scipy_linalg)


def _install_fake_model_accessors(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake `parakeet_rocm.models.parakeet` module.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The fake module.
    """
    mod = types.ModuleType("parakeet_rocm.models.parakeet")
    mod.unload_model_to_cpu_called = False
    mod.clear_model_cache_called = False

    def unload_model_to_cpu() -> None:
        mod.unload_model_to_cpu_called = True

    def clear_model_cache() -> None:
        mod.clear_model_cache_called = True

    mod.unload_model_to_cpu = unload_model_to_cpu
    mod.clear_model_cache = clear_model_cache
    monkeypatch.setitem(sys.modules, "parakeet_rocm.models.parakeet", mod)
    return mod


def _install_fake_webui_job_manager(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake `parakeet_rocm.webui.core.job_manager` module.

    This prevents importing the real job manager, which pulls in the
    transcription stack and NeMo (requiring a full torch install).

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The fake module.
    """
    mod = types.ModuleType("parakeet_rocm.webui.core.job_manager")

    class JobStatus:  # noqa: D106
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"

    class TranscriptionJob:  # noqa: D106
        def __init__(self) -> None:
            self.job_id = "job-00000000"
            self.status = JobStatus.PENDING
            self.outputs: list[Path] = []
            self.error: str | None = None

        @property
        def metrics(self) -> dict[str, object] | None:
            return None

    class JobManager:  # noqa: D106
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_current_job(self) -> None:
            return None

        def get_last_completed_job(self) -> None:
            return None

    mod.JobStatus = JobStatus
    mod.TranscriptionJob = TranscriptionJob
    mod.JobManager = JobManager
    monkeypatch.setitem(sys.modules, "parakeet_rocm.webui.core.job_manager", mod)
    return mod


class _UploadedFile:
    """Simple uploaded-file stub compatible with Gradio's file objects."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeJob:
    """Minimal job object compatible with `webui.app` expectations."""

    def __init__(self, job_id: str, *, status: object, outputs: list[Path]) -> None:
        self.job_id = job_id
        self.status = status
        self.outputs = outputs
        self.error: str | None = None

    @property
    def metrics(self) -> dict[str, object] | None:
        return None


class _FakeJobManager:
    """JobManager stub used to drive the `transcribe_files` handler."""

    def __init__(self, *, outputs: list[Path]) -> None:
        self._outputs = outputs
        self._job: _FakeJob | None = None

    def submit_job(self, files: list[Path], config: object) -> _FakeJob:
        self._job = _FakeJob("job-12345678", status=None, outputs=[])
        self._job.files = files  # type: ignore[attr-defined]
        self._job.config = config  # type: ignore[attr-defined]
        return self._job

    def run_job(
        self,
        job_id: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> _FakeJob:
        assert self._job is not None
        self._job.outputs = self._outputs
        self._job.status = "completed"
        if progress_callback is not None:
            progress_callback(1, 2)
            progress_callback(2, 2)
        return self._job

    def get_current_job(self) -> None:
        return None

    def get_last_completed_job(self) -> None:
        return None


def test_webui_app_build_and_handlers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`build_app()` should wire handlers and core logic should be runnable."""
    gr = _install_fake_gradio(monkeypatch)
    _install_fake_torch(monkeypatch)
    _install_fake_scipy(monkeypatch)
    _install_fake_model_accessors(monkeypatch)
    _install_fake_webui_job_manager(monkeypatch)

    sys.modules.pop("parakeet_rocm.webui.app", None)
    app_mod = importlib.import_module("parakeet_rocm.webui.app")

    # Avoid filesystem validation in this test.
    monkeypatch.setattr(app_mod, "validate_audio_files", lambda _paths: _paths)

    # Build app with job manager stub.
    out1 = tmp_path / "out1.srt"
    out2 = tmp_path / "out2.srt"
    out1.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    out2.write_text("1\n00:00:00,000 --> 00:00:01,000\nworld\n")

    fake_jm = _FakeJobManager(outputs=[out1, out2])
    blocks = app_mod.build_app(job_manager=fake_jm, analytics_enabled=False)
    assert isinstance(blocks, _FakeBlocks)

    # Blocks retain visual styling. Install metadata is applied only by the
    # FastAPI composition, which owns the referenced static routes.
    assert blocks.init_kwargs["title"] == "Parakeet-ROCm WebUI"
    assert blocks.init_kwargs["analytics_enabled"] is False
    assert blocks.init_kwargs["theme"] is not None
    assert blocks.init_kwargs["css"] == app_mod.WEBUI_CONTAINER_CSS
    assert "head" not in blocks.init_kwargs
    markdown_values = [
        component.args[0]
        for component in gr._created
        if component.args and isinstance(component.args[0], str)
    ]
    assert app_mod.WEBUI_TITLE_HTML in markdown_values
    assert "./parakeet-assets/parakeet-rocm-icon.svg" in app_mod.WEBUI_TITLE_HTML
    assert "🎤" not in app_mod.WEBUI_TITLE_HTML

    # Find the registered click handlers.
    click_fns = [c._click_fn for c in gr._created if getattr(c, "_click_fn", None) is not None]
    assert click_fns

    # Extract handler functions by signature behavior.
    transcribe_fn = None
    refresh_fn = None
    clear_fn = None
    for fn in click_fns:
        if fn is None:
            continue
        if fn.__name__ == "transcribe_files":
            transcribe_fn = fn
        elif fn.__name__ == "refresh_benchmarks":
            refresh_fn = fn
        elif fn.__name__ == "clear_all":
            clear_fn = fn

    assert transcribe_fn is not None
    assert refresh_fn is not None
    assert clear_fn is not None

    # No files -> early return.
    status, output_files_val = transcribe_fn(
        [],
        "model",
        1,
        30,
        0,
        False,
        0,
        True,
        "lcs",
        False,
        False,
        False,
        False,
        0.35,
        False,
        "fp16",
        "srt",
        progress=_FakeProgress(),
    )
    assert "Please upload" in status
    assert output_files_val is None

    # Files -> zip path branch (2 outputs).
    uploaded = [_UploadedFile(str(tmp_path / "in1.wav"))]
    (tmp_path / "in1.wav").write_text("x")

    status, file_list, download = transcribe_fn(
        uploaded,
        "model",
        1,
        30,
        0,
        False,
        0,
        True,
        "lcs",
        False,
        False,
        False,
        False,
        0.35,
        False,
        "fp16",
        "srt",
        progress=_FakeProgress(),
    )
    assert "Transcription completed" in status
    assert file_list.get("visible") is False
    assert download.get("visible") is True

    # Single-output branch.
    out3 = tmp_path / "out3.srt"
    out3.write_text("1\n00:00:00,000 --> 00:00:01,000\nsolo\n")
    fake_jm_single = _FakeJobManager(outputs=[out3])
    blocks = app_mod.build_app(job_manager=fake_jm_single, analytics_enabled=False)
    transcribe_fn = None
    for c in gr._created:
        if c._click_fn is not None and c._click_fn.__name__ == "transcribe_files":
            transcribe_fn = c._click_fn
    assert transcribe_fn is not None

    status, file_list, download = transcribe_fn(
        uploaded,
        "model",
        1,
        30,
        0,
        False,
        0,
        True,
        "lcs",
        False,
        False,
        False,
        False,
        0.35,
        False,
        "fp16",
        "srt",
        progress=_FakeProgress(),
    )
    assert file_list.get("visible") is True
    assert download.get("visible") is False

    # Benchmark refresh: no data.
    monkeypatch.setattr(app_mod, "BENCHMARK_OUTPUT_DIR", tmp_path)
    status, runtime_md, gpu_md, json_data = refresh_fn()
    assert "No metrics" in status
    assert "Runtime" in runtime_md
    assert json_data.get("visible") is False

    # Benchmark refresh: load from disk.
    metrics = {
        "runtime_seconds": 10.0,
        "total_wall_seconds": 12.0,
        "gpu_stats": {
            "utilization_percent": {"avg": 50.0},
            "vram_used_mb": {"avg": 1024.0},
        },
        "files": ["x"],
    }
    p = tmp_path / "20250101_000000_job_abcdef12.json"
    p.write_text(json.dumps(metrics))

    status, runtime_md, gpu_md, json_data = refresh_fn()
    assert "Job ID" in status
    assert "Runtime" in runtime_md
    assert "GPU" in gpu_md
    assert json_data.get("visible") is True

    cleared = clear_fn()
    assert isinstance(cleared, dict)


def test_launch_app__uses_root_mounted_fastapi_composition_and_cleans_models(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`launch_app()` should launch the root-mounted FastAPI composition."""
    _install_fake_gradio(monkeypatch)
    _install_fake_torch(monkeypatch)
    _install_fake_scipy(monkeypatch)
    fake_models = _install_fake_model_accessors(monkeypatch)
    _install_fake_webui_job_manager(monkeypatch)

    sys.modules.pop("parakeet_rocm.webui.app", None)
    app_mod = importlib.import_module("parakeet_rocm.webui.app")

    monkeypatch.setattr(app_mod, "configure_logging", lambda **_kwargs: None)

    fake_api = types.ModuleType("parakeet_rocm.api")
    created: dict[str, object] = {}

    def create_app(**kwargs: object) -> object:
        created.update(kwargs)
        return "root-mounted-app"

    fake_api.create_app = create_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "parakeet_rocm.api", fake_api)
    fake_uvicorn = types.ModuleType("uvicorn")
    runs: list[tuple[object, dict[str, object]]] = []

    def run(app: object, **kwargs: object) -> None:
        runs.append((app, kwargs))

    fake_uvicorn.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    caplog.set_level("WARNING", logger=app_mod.logger.name)
    app_mod.launch_app(server_name="127.0.0.1", server_port=9999, share=True, debug=True)
    assert created == {"ui_path": ""}
    assert "share links are not supported" in caplog.text
    assert runs == [
        (
            "root-mounted-app",
            {"host": "127.0.0.1", "port": 9999, "log_level": "debug"},
        )
    ]

    app_mod._cleanup_models()
    assert fake_models.unload_model_to_cpu_called
    assert fake_models.clear_model_cache_called


@pytest.mark.parametrize("reserved_key", ["host", "port", "log_level"])
def test_launch_app__rejects_reserved_uvicorn_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    reserved_key: str,
) -> None:
    """launch_app should reject kwargs already controlled by explicit arguments."""
    _install_fake_gradio(monkeypatch)
    _install_fake_torch(monkeypatch)
    _install_fake_scipy(monkeypatch)
    _install_fake_model_accessors(monkeypatch)
    _install_fake_webui_job_manager(monkeypatch)
    sys.modules.pop("parakeet_rocm.webui.app", None)
    app_mod = importlib.import_module("parakeet_rocm.webui.app")
    monkeypatch.setattr(app_mod, "configure_logging", lambda **_kwargs: None)

    with pytest.raises(ValueError, match=reserved_key):
        app_mod.launch_app(**{reserved_key: "conflict"})
