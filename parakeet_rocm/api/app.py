"""FastAPI application factory for combined REST API and Gradio UI serving."""

from __future__ import annotations

import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from parakeet_rocm.api import routes as api_routes
from parakeet_rocm.api.routes import router as api_router
from parakeet_rocm.utils.constant import (
    API_BEARER_TOKEN,
    API_CORS_ORIGINS,
    API_ENABLED,
    API_MODEL_NAME,
    API_MODEL_WARMUP_ON_START,
    IDLE_CLEAR_TIMEOUT_SEC,
    IDLE_UNLOAD_TIMEOUT_SEC,
)
from parakeet_rocm.utils.logging_config import get_logger

logger = get_logger(__name__)


def _warmup_api_model_cache() -> None:
    """Warm API model cache on startup when configured.

    This is a best-effort operation and must not fail server startup.
    """
    try:
        from parakeet_rocm.models.parakeet import get_model

        logger.info("API model warmup started for model=%s", API_MODEL_NAME)
        get_model(API_MODEL_NAME)
        logger.info("API model warmup completed for model=%s", API_MODEL_NAME)
    except Exception:
        logger.exception("API model warmup failed; continuing without warm cache")


def _start_api_warmup_thread() -> None:
    """Start non-blocking warmup so API startup readiness is not delayed."""
    logger.info("Scheduling API model warmup thread for model=%s", API_MODEL_NAME)
    thread = threading.Thread(
        name="api-model-warmup",
        target=_warmup_api_model_cache,
        daemon=True,
    )
    thread.start()


def _start_api_idle_offload_thread() -> None:
    """Start API idle offload loop to keep VRAM usage low between requests."""

    def _worker() -> None:
        from parakeet_rocm.models.parakeet import clear_model_cache, unload_model_to_cpu

        unloaded = False
        cleared = False
        while True:
            try:
                if api_routes.has_active_api_requests():
                    unloaded = False
                    cleared = False
                else:
                    idle_for = time.monotonic() - api_routes.get_last_api_activity_monotonic()
                    if not unloaded and idle_for >= IDLE_UNLOAD_TIMEOUT_SEC:
                        unload_model_to_cpu(API_MODEL_NAME)
                        unloaded = True
                    if not cleared and idle_for >= IDLE_CLEAR_TIMEOUT_SEC:
                        clear_model_cache()
                        cleared = True
            except Exception:
                logger.debug("API idle offload loop iteration failed", exc_info=True)
            time.sleep(5.0)

    thread = threading.Thread(name="api-idle-offloader", target=_worker, daemon=True)
    thread.start()


def create_app(*, include_ui: bool = True, ui_path: str = "/ui") -> FastAPI:
    """Create a FastAPI application with optional mounted Gradio UI.

    Args:
        include_ui: Whether to mount the Gradio UI.
        ui_path: Root-relative URL path at which to mount the Gradio UI. Use an
            empty string or ``/`` to serve the UI at the server root. Trailing
            slashes are removed from non-root paths.

    Returns:
        Configured FastAPI application instance.

    Raises:
        ValueError: If ``ui_path`` is not empty or root-relative.
    """
    if not ui_path or ui_path == "/":
        ui_path = ""
    elif not ui_path.startswith("/"):
        raise ValueError("ui_path must be empty or root-relative")
    else:
        ui_path = ui_path.rstrip("/")

    app = FastAPI(
        title="Parakeet-ROCm API",
        docs_url="/docs",
        openapi_url="/openapi.json",
        redoc_url="/redoc",
    )

    if API_ENABLED:
        if not API_BEARER_TOKEN:
            logger.warning(
                "API_BEARER_TOKEN is not set; OpenAI-compatible API authentication is disabled."
            )
        app.include_router(api_router)

    origins = [origin.strip() for origin in API_CORS_ORIGINS.split(",") if origin.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Return a minimal health status payload."""
        return {"status": "ok"}

    if not include_ui:

        @app.on_event("startup")
        async def _on_api_startup() -> None:
            """Start API runtime background tasks for model warmup and idle offload."""
            if API_MODEL_WARMUP_ON_START:
                _start_api_warmup_thread()
            _start_api_idle_offload_thread()

        @app.get("/")
        async def root() -> dict[str, str]:
            """Return API-mode service metadata for root requests.

            Returns:
                API-mode metadata containing docs and health endpoint paths.
            """
            return {
                "service": "parakeet-rocm-api",
                "docs": "/docs",
                "health": "/health",
            }

        return app

    from parakeet_rocm.webui.app import (
        WEBUI_CONTAINER_CSS,
        _cleanup_models,
        _start_idle_offload_thread,
        build_app,
    )
    from parakeet_rocm.webui.assets import WEBUI_ASSET_DIR, WEBUI_HEAD_HTML
    from parakeet_rocm.webui.core.job_manager import JobManager
    from parakeet_rocm.webui.ui.theme import configure_theme

    # Single-process architecture: both API and Gradio share the same model cache.
    job_manager = JobManager()
    gradio_app = build_app(job_manager=job_manager)

    # ``/assets`` belongs to Gradio's frontend. Register Parakeet's icons at a
    # distinct route before mounting Gradio, without exposing package files.
    app.mount(
        f"{ui_path}/parakeet-assets",
        StaticFiles(directory=WEBUI_ASSET_DIR),
        name="parakeet-assets",
    )

    @app.get(f"{ui_path}/favicon.ico", include_in_schema=False)
    async def webui_favicon() -> FileResponse:
        """Serve the Parakeet favicon at the mounted browser-default URL.

        Returns:
            The packaged multi-size Parakeet favicon.
        """
        return FileResponse(WEBUI_ASSET_DIR / "favicon.ico", media_type="image/x-icon")

    @app.on_event("startup")
    async def _on_startup() -> None:
        """Start background model idle offload worker."""
        if API_MODEL_WARMUP_ON_START:
            _start_api_warmup_thread()
        _start_idle_offload_thread(job_manager)
        _start_api_idle_offload_thread()

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        """Run best-effort model cleanup during server shutdown."""
        _cleanup_models()

    if ui_path:

        @app.get("/")
        async def root() -> Response:
            """Redirect root traffic to the mounted Gradio UI.

            Returns:
                Redirect response to the mounted UI path.
            """
            return RedirectResponse(url=ui_path, status_code=307)

    import gradio as gr

    return gr.mount_gradio_app(
        app,
        gradio_app,
        path=ui_path or "/",
        theme=configure_theme(),
        css=WEBUI_CONTAINER_CSS,
        head=WEBUI_HEAD_HTML,
        favicon_path=(str(WEBUI_ASSET_DIR / "favicon.ico") if not ui_path else None),
        pwa=False,
    )


def create_api_app() -> FastAPI:
    """Create API-only FastAPI app without mounted Gradio UI.

    Returns:
        Configured FastAPI application instance exposing REST endpoints only.
    """
    return create_app(include_ui=False)
