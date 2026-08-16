"""Gradio WebUI sub-module for Parakeet-ROCm ASR.

This module provides a production-ready web interface for the
Parakeet-NEMO ASR transcription engine, built with Gradio.

The WebUI follows modern architectural patterns with clean separation
of concerns, protocol-oriented design for testability, and comprehensive
input validation.

Examples:
    Launch the supported standalone WebUI composition from code:

    >>> from parakeet_rocm.webui import launch_app
    >>> launch_app()

    Build Blocks for embedding in the FastAPI composition:

    >>> from parakeet_rocm.webui import build_app
    >>> blocks = build_app()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy imports to avoid loading Gradio unless needed
__all__ = ["build_app", "launch_app"]

if TYPE_CHECKING:
    import gradio as gr


def build_app() -> gr.Blocks:
    """Build the Gradio WebUI application.

    Returns:
        Configured Gradio Blocks application.
    """
    from parakeet_rocm.webui.app import build_app as _build_app

    return _build_app()


def launch_app(**kwargs: object) -> None:
    """Launch the Gradio WebUI application.

    Args:
        **kwargs: Keyword arguments passed to launch_app function.
    """
    from parakeet_rocm.webui.app import launch_app as _launch_app

    _launch_app(**kwargs)
