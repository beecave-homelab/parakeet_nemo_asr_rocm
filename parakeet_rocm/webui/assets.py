"""WebUI icon asset locations and head markup."""

from __future__ import annotations

from pathlib import Path

WEBUI_ASSET_DIR = Path(__file__).with_name("assets")
"""Directory containing public WebUI icon assets."""

WEBUI_HEAD_HTML = """\
<title>Parakeet-ROCm WebUI</title>
<meta name="application-name" content="Parakeet-ROCm WebUI">
<meta name="apple-mobile-web-app-title" content="Parakeet">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="icon" href="./parakeet-assets/favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="192x192" href="./parakeet-assets/icon-192.png">
<link rel="apple-touch-icon" sizes="180x180" href="./parakeet-assets/apple-touch-icon.png">
<link rel="manifest" href="./parakeet-assets/manifest.webmanifest">
<meta name="theme-color" content="#071827">
"""
"""Route-relative icon metadata injected into the mounted Gradio page."""
