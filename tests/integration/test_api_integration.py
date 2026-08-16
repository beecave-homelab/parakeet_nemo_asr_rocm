"""Integration tests for the unified FastAPI + mounted Gradio application."""

from __future__ import annotations

import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.fixture(autouse=True)
def _stub_model_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid importing optional NeMo dependencies in API integration tests."""
    fake_models = types.ModuleType("parakeet_rocm.models.parakeet")
    fake_models.get_model = lambda *_args, **_kwargs: object()
    monkeypatch.setitem(sys.modules, "parakeet_rocm.models.parakeet", fake_models)
    fake_transcription = types.ModuleType("parakeet_rocm.transcription")
    fake_transcription.cli_transcribe = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "parakeet_rocm.transcription", fake_transcription)


def _install_fake_webui_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install lightweight fakes for WebUI modules used by API app factory.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    fake_webui_app = types.ModuleType("parakeet_rocm.webui.app")

    def build_app(*, job_manager: object | None = None, analytics_enabled: bool = False) -> object:
        del job_manager, analytics_enabled
        return object()

    def _start_idle_offload_thread(_job_manager: object) -> None:
        return None

    def _cleanup_models() -> None:
        return None

    fake_webui_app.build_app = build_app
    fake_webui_app._start_idle_offload_thread = _start_idle_offload_thread
    fake_webui_app._cleanup_models = _cleanup_models
    monkeypatch.setitem(sys.modules, "parakeet_rocm.webui.app", fake_webui_app)

    fake_job_manager = types.ModuleType("parakeet_rocm.webui.core.job_manager")

    class JobManager:
        pass

    fake_job_manager.JobManager = JobManager
    monkeypatch.setitem(sys.modules, "parakeet_rocm.webui.core.job_manager", fake_job_manager)

    fake_gradio = types.ModuleType("gradio")

    class _Themes:
        class Color:
            pass

        class Soft:
            def __init__(self, **_kwargs: object) -> None:
                return None

            def set(self, **_kwargs: object) -> _Themes.Soft:
                return self

    fake_gradio.themes = _Themes

    def mount_gradio_app(
        app: FastAPI,
        _gradio_app: object,
        *,
        path: str,
    ) -> FastAPI:
        assert path == "/ui"
        return app

    fake_gradio.mount_gradio_app = mount_gradio_app
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)


def test_combined_app_exposes_docs_and_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Combined app should expose OpenAPI docs, API routes, and UI redirect."""
    from parakeet_rocm.api import app as api_app

    _install_fake_webui_runtime(monkeypatch)
    monkeypatch.setattr(api_app, "API_ENABLED", True)
    monkeypatch.setattr(api_app, "API_CORS_ORIGINS", "")

    app = api_app.create_app()
    client = TestClient(app)

    docs = client.get("/docs")
    assert docs.status_code == 200

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    paths = openapi.json().get("paths", {})
    assert "/v1/audio/transcriptions" in paths

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/ui"
