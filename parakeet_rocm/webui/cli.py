"""Typer CLI for launching the Parakeet-ROCm WebUI."""

from __future__ import annotations

import typer

from parakeet_rocm.api import create_app
from parakeet_rocm.utils.console import print_warning
from parakeet_rocm.utils.constant import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT
from parakeet_rocm.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

app = typer.Typer(add_completion=False)


@app.command()
def webui(
    host: str = typer.Option(
        GRADIO_SERVER_NAME,
        "--host",
        help="Server hostname or IP address to bind to.",
    ),
    port: int = typer.Option(
        GRADIO_SERVER_PORT,
        "--port",
        help="Server port number.",
    ),
    share: bool = typer.Option(
        False,
        "--share",
        help="Create a public Gradio share link.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode with verbose logging.",
    ),
) -> None:
    """Launch the unified FastAPI + Gradio application."""
    configure_logging(level="DEBUG" if debug else "INFO")

    if share:
        print_warning(
            "--share is not supported with mounted Gradio on FastAPI. "
            "Use a tunnel (for example ngrok or cloudflared) to expose this server."
        )

    app_instance = create_app()

    import uvicorn

    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
    )


def main() -> None:
    """Run the WebUI CLI."""
    app()
