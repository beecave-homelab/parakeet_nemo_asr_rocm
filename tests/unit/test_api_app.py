"""Unit tests for unified FastAPI + Gradio application factory."""

from __future__ import annotations

import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _stub_model_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid importing optional NeMo dependencies in API factory tests."""
    fake_models = types.ModuleType("parakeet_rocm.models.parakeet")
    fake_models.get_model = lambda *_args, **_kwargs: object()
    monkeypatch.setitem(sys.modules, "parakeet_rocm.models.parakeet", fake_models)
    fake_transcription = types.ModuleType("parakeet_rocm.transcription")
    fake_transcription.cli_transcribe = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "parakeet_rocm.transcription", fake_transcription)


def _install_fake_webui_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install fake WebUI modules consumed by ``create_app``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Mutable state dict tracking startup/shutdown side effects.
    """
    state: dict[str, object] = {
        "idle_thread_started": False,
        "cleanup_called": False,
        "mount_kwargs": {},
        "mount_path": None,
    }

    fake_webui_app = types.ModuleType("parakeet_rocm.webui.app")

    def build_app(*, job_manager: object | None = None, analytics_enabled: bool = False) -> object:
        del job_manager, analytics_enabled
        return object()

    def _start_idle_offload_thread(_job_manager: object) -> None:
        state["idle_thread_started"] = True

    def _cleanup_models() -> None:
        state["cleanup_called"] = True

    fake_webui_app.build_app = build_app
    fake_webui_app._start_idle_offload_thread = _start_idle_offload_thread
    fake_webui_app._cleanup_models = _cleanup_models
    fake_webui_app.WEBUI_CONTAINER_CSS = ".gradio-container { max-width: 1200px; margin: auto; }"
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
        **kwargs: object,
    ) -> FastAPI:
        state["mount_path"] = path
        state["mount_kwargs"] = kwargs
        gradio_routes = FastAPI()

        @gradio_routes.get("/assets/gradio-frontend.js")
        async def gradio_asset() -> dict[str, str]:
            """Represent Gradio's reserved frontend-asset route.

            Returns:
                Marker payload from the simulated Gradio route.
            """
            return {"source": "gradio"}

        app.mount(path, gradio_routes)
        return app

    fake_gradio.mount_gradio_app = mount_gradio_app
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)

    return state


def test_create_app_root_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_app should expose health endpoint and root redirect."""
    from parakeet_rocm.api import app as api_app

    state = _install_fake_webui_modules(monkeypatch)
    monkeypatch.setattr(api_app, "API_ENABLED", True)
    monkeypatch.setattr(api_app, "API_CORS_ORIGINS", "")
    monkeypatch.setattr(api_app, "API_BEARER_TOKEN", "sk-test")
    monkeypatch.setattr(api_app, "API_MODEL_WARMUP_ON_START", False)
    monkeypatch.setattr(api_app, "_start_api_idle_offload_thread", lambda: None)

    app = api_app.create_app()
    client = TestClient(app)

    assert state["mount_path"] == "/ui"
    mount_kwargs = state["mount_kwargs"]
    assert isinstance(mount_kwargs, dict)
    assert mount_kwargs["theme"] is not None
    assert mount_kwargs["css"] == ".gradio-container { max-width: 1200px; margin: auto; }"
    assert "apple-mobile-web-app-title" in str(mount_kwargs["head"])
    assert mount_kwargs["favicon_path"] is None
    assert mount_kwargs["pwa"] is False

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/ui"

    with TestClient(app):
        pass

    assert state["idle_thread_started"] is True
    assert state["cleanup_called"] is True


def test_create_app__serves_route_relative_webui_icons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_app should expose icons without masking Gradio's asset route."""
    from parakeet_rocm.api import app as api_app

    _install_fake_webui_modules(monkeypatch)
    monkeypatch.setattr(api_app, "API_ENABLED", False)
    monkeypatch.setattr(api_app, "API_CORS_ORIGINS", "")
    monkeypatch.setattr(api_app, "API_MODEL_WARMUP_ON_START", False)

    app = api_app.create_app()
    client = TestClient(app)

    for asset_name in (
        "favicon.ico",
        "icon-192.png",
        "icon-512.png",
        "apple-touch-icon.png",
        "manifest.webmanifest",
    ):
        response = client.get(f"/ui/parakeet-assets/{asset_name}")
        assert response.status_code == 200

    gradio_asset = client.get("/ui/assets/gradio-frontend.js")
    assert gradio_asset.status_code == 200
    assert gradio_asset.json() == {"source": "gradio"}


def test_create_app__serves_root_relative_webui_icons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root-mounted UI should expose icons without masking Gradio assets."""
    from parakeet_rocm.api import app as api_app

    state = _install_fake_webui_modules(monkeypatch)
    monkeypatch.setattr(api_app, "API_ENABLED", False)
    monkeypatch.setattr(api_app, "API_CORS_ORIGINS", "")
    monkeypatch.setattr(api_app, "API_MODEL_WARMUP_ON_START", False)

    app = api_app.create_app(ui_path="")
    client = TestClient(app)

    assert state["mount_path"] == "/"
    root_mount_kwargs = state["mount_kwargs"]
    assert isinstance(root_mount_kwargs, dict)
    assert str(root_mount_kwargs["favicon_path"]).endswith("/favicon.ico")
    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200
    assert favicon.content == client.get("/parakeet-assets/favicon.ico").content
    for asset_name in ("favicon.ico", "apple-touch-icon.png", "manifest.webmanifest"):
        response = client.get(f"/parakeet-assets/{asset_name}")
        assert response.status_code == 200

    gradio_asset = client.get("/assets/gradio-frontend.js")
    assert gradio_asset.status_code == 200
    assert gradio_asset.json() == {"source": "gradio"}


def test_create_app__normalizes_root_and_trailing_ui_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_app should normalize root and trailing-slash UI paths before mounting."""
    from parakeet_rocm.api import app as api_app

    monkeypatch.setattr(api_app, "API_ENABLED", False)
    monkeypatch.setattr(api_app, "API_CORS_ORIGINS", "")
    monkeypatch.setattr(api_app, "API_MODEL_WARMUP_ON_START", False)

    root_state = _install_fake_webui_modules(monkeypatch)
    root_app = api_app.create_app(ui_path="/")
    root_client = TestClient(root_app)
    assert root_state["mount_path"] == "/"
    assert root_client.get("/", follow_redirects=False).status_code != 307
    assert root_client.get("/parakeet-assets/favicon.ico").status_code == 200

    trailing_state = _install_fake_webui_modules(monkeypatch)
    trailing_app = api_app.create_app(ui_path="/ui/")
    trailing_client = TestClient(trailing_app)
    assert trailing_state["mount_path"] == "/ui"
    assert trailing_client.get("/ui/parakeet-assets/favicon.ico").status_code == 200
    assert trailing_client.get("/ui//parakeet-assets/favicon.ico").status_code == 404


@pytest.mark.parametrize("ui_path", ["ui", "https://example.test/ui"])
def test_create_app__rejects_non_root_relative_ui_paths(
    monkeypatch: pytest.MonkeyPatch,
    ui_path: str,
) -> None:
    """create_app should reject UI paths that cannot be mounted safely."""
    from parakeet_rocm.api import app as api_app

    with pytest.raises(ValueError, match="root-relative"):
        api_app.create_app(ui_path=ui_path)


def test_create_api_app__warms_model_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_api_app should schedule model warmup on startup when enabled."""
    from parakeet_rocm.api import app as api_app

    state = {"warmup_thread_started": False}

    def _start_warmup_thread() -> None:
        state["warmup_thread_started"] = True

    monkeypatch.setattr(api_app, "API_ENABLED", True)
    monkeypatch.setattr(api_app, "API_CORS_ORIGINS", "")
    monkeypatch.setattr(api_app, "API_BEARER_TOKEN", "sk-test")
    monkeypatch.setattr(api_app, "API_MODEL_WARMUP_ON_START", True)
    monkeypatch.setattr(api_app, "_start_api_warmup_thread", _start_warmup_thread)
    monkeypatch.setattr(api_app, "_start_api_idle_offload_thread", lambda: None)

    app = api_app.create_api_app()

    with TestClient(app):
        pass

    assert state["warmup_thread_started"] is True


def test_create_api_app__logs_warning_when_auth_token_unset(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """create_api_app should warn operators when API auth is disabled."""
    from parakeet_rocm.api import app as api_app

    monkeypatch.setattr(api_app, "API_ENABLED", True)
    monkeypatch.setattr(api_app, "API_CORS_ORIGINS", "")
    monkeypatch.setattr(api_app, "API_BEARER_TOKEN", None)
    monkeypatch.setattr(api_app, "API_MODEL_WARMUP_ON_START", False)
    monkeypatch.setattr(api_app, "_start_api_idle_offload_thread", lambda: None)
    caplog.set_level("WARNING", logger=api_app.logger.name)

    api_app.create_api_app()

    assert "API_BEARER_TOKEN is not set" in caplog.text


def test_create_api_app_root_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_api_app should expose API metadata at root without UI redirect."""
    from parakeet_rocm.api import app as api_app

    monkeypatch.setattr(api_app, "API_ENABLED", True)
    monkeypatch.setattr(api_app, "API_CORS_ORIGINS", "")
    monkeypatch.setattr(api_app, "API_BEARER_TOKEN", "sk-test")

    app = api_app.create_api_app()
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert root.json() == {
        "service": "parakeet-rocm-api",
        "docs": "/docs",
        "health": "/health",
    }

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
