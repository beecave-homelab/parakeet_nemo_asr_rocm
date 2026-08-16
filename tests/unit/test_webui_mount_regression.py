"""Regression coverage for the real Gradio mounted-app contract."""

from __future__ import annotations

import json
import re
import sys
import types
from urllib.parse import urljoin

import gradio as gr
import pytest
from fastapi.testclient import TestClient


def test_mounted_webui__uses_parakeet_styling_and_install_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify mounted styling, metadata, and assets.

    Args:
        monkeypatch: Fixture used to isolate model and WebUI modules.
    """
    fake_models = types.ModuleType("parakeet_rocm.models.parakeet")
    setattr(fake_models, "get_model", lambda *_args, **_kwargs: object())
    monkeypatch.setitem(sys.modules, "parakeet_rocm.models.parakeet", fake_models)
    fake_transcription = types.ModuleType("parakeet_rocm.transcription")
    setattr(fake_transcription, "cli_transcribe", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "parakeet_rocm.transcription", fake_transcription)

    fake_webui_app = types.ModuleType("parakeet_rocm.webui.app")

    def build_app(**_kwargs: object) -> gr.Blocks:
        with gr.Blocks(title="Parakeet-ROCm WebUI") as blocks:
            gr.Markdown("# Parakeet")
        return blocks

    setattr(fake_webui_app, "build_app", build_app)
    setattr(fake_webui_app, "_start_idle_offload_thread", lambda _manager: None)
    setattr(fake_webui_app, "_cleanup_models", lambda: None)
    setattr(
        fake_webui_app,
        "WEBUI_CONTAINER_CSS",
        ".gradio-container { max-width: 1200px; margin: auto; }",
    )
    monkeypatch.setitem(sys.modules, "parakeet_rocm.webui.app", fake_webui_app)
    fake_job_manager = types.ModuleType("parakeet_rocm.webui.core.job_manager")
    setattr(fake_job_manager, "JobManager", type("JobManager", (), {}))
    monkeypatch.setitem(sys.modules, "parakeet_rocm.webui.core.job_manager", fake_job_manager)
    monkeypatch.delitem(sys.modules, "parakeet_rocm.webui.ui.theme", raising=False)

    from parakeet_rocm.api import app as api_app

    monkeypatch.setattr(api_app, "API_ENABLED", False)
    monkeypatch.setattr(api_app, "API_CORS_ORIGINS", "")
    monkeypatch.setattr(api_app, "API_MODEL_WARMUP_ON_START", False)

    client = TestClient(api_app.create_app())
    config = client.get("/ui/config").json()
    assert config["css"] == ".gradio-container { max-width: 1200px; margin: auto; }"
    assert config["theme_hash"]
    assert config["title"] == "Parakeet-ROCm WebUI"
    assert "<title>Parakeet-ROCm WebUI</title>" in config["head"]

    document = client.get("/ui/").text
    for marker in (
        "application-name",
        "apple-mobile-web-app-title",
        "apple-mobile-web-app-capable",
        "mobile-web-app-capable",
        "apple-touch-icon",
        "./parakeet-assets/manifest.webmanifest",
        "./parakeet-assets/favicon.ico",
        "theme-color",
    ):
        assert marker in document
    assert document.count("manifest.webmanifest") == 1
    assert "/assets/logo.svg" not in document

    for asset_name in (
        "favicon.ico",
        "icon-192.png",
        "icon-512.png",
        "apple-touch-icon.png",
        "manifest.webmanifest",
    ):
        assert client.get(f"/ui/parakeet-assets/{asset_name}").status_code == 200
    favicon = client.get("/ui/favicon.ico")
    assert favicon.content == client.get("/ui/parakeet-assets/favicon.ico").content

    manifest = json.loads(client.get("/ui/parakeet-assets/manifest.webmanifest").text)
    base = "http://testserver/ui/parakeet-assets/manifest.webmanifest"
    assert manifest["name"] == "Parakeet-ROCm WebUI"
    assert manifest["short_name"] == "Parakeet"
    for key in ("id", "start_url", "scope"):
        assert urljoin(base, manifest[key]) == "http://testserver/ui/"
    assert {icon["purpose"] for icon in manifest["icons"]} == {"any", "maskable"}

    frontend_asset_path = re.search(r'src="(\./assets/[^"]+\.js)"', document)
    assert frontend_asset_path is not None
    frontend_asset = client.get(urljoin("http://testserver/ui/", frontend_asset_path.group(1)))
    assert frontend_asset.status_code == 200
