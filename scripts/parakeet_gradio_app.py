"""
Enhanced Gradio WebUI for Parakeet‑NEMO ASR
------------------------------------------------

This module provides a ready‑to‑run Web UI built with the
[`gradio`](https://gradio.app) framework. It exposes the
transcription capabilities of the `parakeet_nemo_asr_rocm` package
through a polished interface with light/dark mode support,
collapsible configuration sections and quick presets for common
use cases.

The underlying logic is adapted from the original project
and uses the same `cli_transcribe` function under the hood. All
configuration options are surfaced via intuitive widgets. A user
can upload one or more audio/video files, adjust the model and
precision settings, choose output formats and more.

To run the app simply execute this file with Python:

```
python parakeet_gradio_app.py
```

It will start a local Gradio server and open the UI in your
default web browser.
"""

import pathlib

import gradio as gr

from parakeet_nemo_asr_rocm.models.parakeet import DEFAULT_MODEL_NAME
from parakeet_nemo_asr_rocm.transcribe import cli_transcribe
from parakeet_nemo_asr_rocm.utils import constant


def enforce_precision(fp16: bool, fp32: bool) -> tuple[bool, bool]:
    """Enforce that only one of fp16/fp32 flags is true.

    If both are selected the function prefers fp16 by returning
    ``(True, False)``; otherwise it returns the original values.

    Args:
        fp16: whether FP16 precision is requested
        fp32: whether FP32 precision is requested

    Returns:
        A tuple of booleans (fp16, fp32) with at most one True.
    """
    if fp16 and fp32:
        return True, False
    return fp16, fp32


def transcribe_webui(
    files: list[str],
    model_name: str,
    output_dir: str,
    output_format: str,
    output_template: str,
    batch_size: int,
    chunk_len_sec: int,
    stream: bool,
    stream_chunk_sec: int,
    overlap_duration: int,
    highlight_words: bool,
    word_timestamps: bool,
    merge_strategy: str,
    overwrite: bool,
    verbose: bool,
    no_progress: bool,
    quiet: bool,
    fp16: bool,
    fp32: bool,
) -> list[str]:
    """Wrapper around ``cli_transcribe`` that accepts Gradio inputs.

    Parameters mirror the command‑line options of the Parakeet CLI. The
    returned list contains paths of generated transcription files.
    """
    fp16, fp32 = enforce_precision(fp16, fp32)
    path_files = [pathlib.Path(f) for f in files]
    outputs = cli_transcribe(
        audio_files=path_files,
        model_name=model_name,
        output_dir=pathlib.Path(output_dir),
        output_format=output_format,
        output_template=output_template,
        batch_size=batch_size,
        chunk_len_sec=chunk_len_sec,
        stream=stream,
        stream_chunk_sec=stream_chunk_sec,
        overlap_duration=overlap_duration,
        highlight_words=highlight_words,
        word_timestamps=word_timestamps,
        merge_strategy=merge_strategy,
        overwrite=overwrite,
        verbose=verbose,
        quiet=quiet,
        no_progress=no_progress,
        fp16=fp16,
        fp32=fp32,
    )
    # Convert all Path objects to strings for Gradio compatibility
    return [str(path) for path in outputs]


# --- Custom styling for Gradio ---
CUSTOM_CSS = """
:root {
  --bg:#f5f7fa;
  --card:#ffffff;
  --fg:#1f2937;
  --muted:#6b7280;
  --radius:10px;
  --shadow:0 12px 30px -4px rgba(31,41,55,0.15);
  --border:1px solid rgba(0,0,0,0.05);
}
[data-theme="dark"] {
  --bg:#0f111a;
  --card:#1f1f2c;
  --fg:#e1e5ee;
  --muted:#9ca3af;
  --shadow:0 12px 30px -4px rgba(0,0,0,0.5);
  --border:1px solid rgba(255,255,255,0.08);
}
body {
  background: var(--bg) !important;
  color: var(--fg) !important;
  transition: background .25s ease, color .25s ease;
  font-family: system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;
}
.gradio-container, .gradio-box {
  background: transparent !important;
}
.group-box {
  background: var(--card) !important;
  padding: 1rem 1rem 1.25rem;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  border: var(--border);
  margin-bottom: 1rem;
}
h1 { font-size:1.75rem; margin-bottom:0.25rem; font-weight:700; }
h2 { font-size:1.25rem; margin:0.4rem 0 0.6rem; font-weight:600; }
h3 { font-size:1rem; margin:0.3rem 0 0.4rem; font-weight:600; }
.small-label {
  font-size: 0.65rem;
  color: var(--muted);
  margin-top: 3px;
}
.warning {
  color: #f59e0b;
  background: #fffbeb;
  border-left: 3px solid #f59e0b;
  padding: 10px;
  border-radius: 4px;
  margin: 10px 0;
}
.output-files {
  margin-top: 1em;
}
.output-files .wrap {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid var(--border-color-primary);
  border-radius: 4px;
  padding: 10px;
  margin-top: 10px;
  background: var(--card);
}
.download-btn {
  margin-left: 10px;
}
.gradio-button.primary {
  background: #ff6f00 !important;
  border: none !important;
  color: white !important;
}
.gradio-button.primary:hover {
  filter: brightness(1.05);
}
#theme-toggle-btn {
  border-radius: 6px;
  padding: 6px 12px;
  min-width: 100px;
}
* {
  user-select: auto !important;
}
::selection {
  background: rgba(255,111,0,0.2);
  color: inherit;
}
"""

# JavaScript to persist and toggle theme across sessions
CUSTOM_JS = """
(function(){
    const stored = localStorage.getItem("parakeet-theme") || "light";
    function setTheme(t) {
        document.documentElement.setAttribute("data-theme", t);
        localStorage.setItem("parakeet-theme", t);
        const btn = document.getElementById("theme-toggle-btn");
        if (btn) btn.textContent = t === "dark" ? "Light Mode" : "Dark Mode";
    }
    setTheme(stored);
    const btn = document.getElementById("theme-toggle-btn");
    if (btn) {
        btn.addEventListener("click", () => {
            const current = document.documentElement.getAttribute("data-theme");
            setTheme(current === "dark" ? "light" : "dark");
        });
    }
})();
"""


def build_ui() -> gr.Blocks:
    """Assemble and return the Gradio Blocks UI."""
    with gr.Blocks(
        title="Parakeet‑NEMO ASR WebUI", css=CUSTOM_CSS, analytics_enabled=False
    ) as demo:
        # Header row
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## Parakeet‑NEMO ASR WebUI")
                gr.Markdown(
                    "**Upload audio/video and configure transcription settings.**  \n"
                    "Fast presets, streaming, precision control, and customizable outputs."
                )
            with gr.Column(scale=1, min_width=150):
                with gr.Row():
                    gr.Button(
                        "Toggle",
                        elem_id="theme-toggle-btn",
                        variant="secondary",
                        size="sm",
                    )
                gr.Markdown(
                    '<div class="small-label">Switch Light / Dark. Preference saved.</div>'
                )

        # Always visible inputs
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Input & Model")
                with gr.Column(elem_classes=["group-box"]):
                    files = gr.File(
                        label="Upload Audio / Video",
                        file_count="multiple",
                        type="filepath",
                    )
                    gr.Markdown(
                        '<div class="small-label">Supports common audio/video containers.</div>'
                    )
                    model_name = gr.Textbox(
                        label="Model Name or Path",
                        value=DEFAULT_MODEL_NAME,
                    )
                    gr.Markdown(
                        '<div class="small-label">Local path or pretrained model identifier.</div>'
                    )

        # Collapsible: output settings
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Output Settings", open=False):
                    with gr.Column(elem_classes=["group-box"]):
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            value="./output",
                        )
                        gr.Markdown(
                            '<div class="small-label">Where transcription files will be written.</div>'
                        )
                        output_format = gr.Dropdown(
                            ["txt", "srt", "vtt", "json"],
                            label="Output Format",
                            value="txt",
                        )
                        output_template = gr.Textbox(
                            label="Output Filename Template",
                            value="{filename}",
                        )
                        gr.Markdown(
                            '<div class="small-label">Use placeholders like {filename}, {lang}.</div>'
                        )
            with gr.Column(scale=3):
                with gr.Accordion("Transcription Controls", open=False):
                    with gr.Column(elem_classes=["group-box"]):
                        with gr.Row():
                            batch_size = gr.Slider(
                                1,
                                128,
                                step=1,
                                value=constant.DEFAULT_BATCH_SIZE,
                                label="Batch Size",
                            )
                            chunk_len_sec = gr.Slider(
                                1,
                                600,
                                step=1,
                                value=constant.DEFAULT_CHUNK_LEN_SEC,
                                label="Chunk Length (sec)",
                            )
                        with gr.Row():
                            stream = gr.Checkbox(label="Stream Mode")
                            stream_chunk_sec = gr.Slider(
                                0,
                                60,
                                step=1,
                                value=0,
                                label="Stream Chunk Length (sec)",
                            )
                            overlap_duration = gr.Slider(
                                0,
                                60,
                                step=1,
                                value=15,
                                label="Overlap Duration (sec)",
                            )
                        with gr.Row():
                            highlight_words = gr.Checkbox(label="Highlight Words")
                            word_timestamps = gr.Checkbox(label="Word Timestamps")
                            merge_strategy = gr.Dropdown(
                                ["none", "contiguous", "lcs"],
                                label="Merge Strategy",
                                value="lcs",
                            )
                with gr.Accordion("Execution Flags", open=False):
                    with gr.Column(elem_classes=["group-box"]):
                        with gr.Row():
                            overwrite = gr.Checkbox(label="Overwrite Outputs")
                            verbose = gr.Checkbox(label="Verbose Output")
                            no_progress = gr.Checkbox(label="Disable Progress Bar")
                            quiet = gr.Checkbox(label="Quiet Mode")
                        with gr.Row():
                            fp16 = gr.Checkbox(label="Use FP16 Precision")
                            fp32 = gr.Checkbox(label="Use FP32 Precision")
                        # Placeholder for precision warning
                        fp_warning_output = gr.HTML()

        # Presets
        with gr.Accordion("Presets / Quick Actions", open=False):
            with gr.Row():
                default_preset = gr.Button("Default Fast")
                high_quality = gr.Button("High Quality")
                stream_preset = gr.Button("Streaming Mode")

        # Main action and output
        with gr.Row():
            transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
            _status = gr.Markdown("", visible=False)
        output_files = gr.Files(label="Transcription Outputs", interactive=False)

        # Inject custom JS for theme persistence
        gr.HTML(
            f"""
        <script>
        {CUSTOM_JS}
        </script>
        """
        )

        # Precision enforcement callback: displays a warning and adjusts toggles
        def enforce_and_warn(fp16_val: bool, fp32_val: bool):
            if fp16_val and fp32_val:
                return (
                    "<div class='warning'>FP16 and FP32 both selected; FP16 will take precedence.</div>",
                    True,
                    False,
                )
            return ("", fp16_val, fp32_val)

        fp16.change(
            fn=enforce_and_warn,
            inputs=[fp16, fp32],
            outputs=[fp_warning_output, fp16, fp32],
        )
        fp32.change(
            fn=enforce_and_warn,
            inputs=[fp16, fp32],
            outputs=[fp_warning_output, fp16, fp32],
        )

        # Preset callbacks
        def apply_default():
            return (
                DEFAULT_MODEL_NAME,
                "./output",
                "txt",
                "{filename}",
                constant.DEFAULT_BATCH_SIZE,
                constant.DEFAULT_CHUNK_LEN_SEC,
                False,
                0,
                15,
                False,
                False,
                "lcs",
                False,
                False,
                False,
                False,
                True,
                False,
            )

        def apply_high_quality():
            return (
                DEFAULT_MODEL_NAME,
                "./output",
                "txt",
                "{filename}",
                4,
                300,
                False,
                0,
                30,
                True,
                True,
                "lcs",
                True,
                True,
                False,
                False,
                False,
                False,
            )

        def apply_streaming():
            return (
                DEFAULT_MODEL_NAME,
                "./output",
                "txt",
                "{filename}",
                8,
                60,
                True,
                5,
                10,
                False,
                False,
                "contiguous",
                False,
                False,
                False,
                False,
                True,
                False,
            )

        default_preset.click(
            fn=apply_default,
            inputs=[],
            outputs=[
                model_name,
                output_dir,
                output_format,
                output_template,
                batch_size,
                chunk_len_sec,
                stream,
                stream_chunk_sec,
                overlap_duration,
                highlight_words,
                word_timestamps,
                merge_strategy,
                overwrite,
                verbose,
                no_progress,
                quiet,
                fp16,
                fp32,
            ],
        )
        high_quality.click(
            fn=apply_high_quality,
            inputs=[],
            outputs=[
                model_name,
                output_dir,
                output_format,
                output_template,
                batch_size,
                chunk_len_sec,
                stream,
                stream_chunk_sec,
                overlap_duration,
                highlight_words,
                word_timestamps,
                merge_strategy,
                overwrite,
                verbose,
                no_progress,
                quiet,
                fp16,
                fp32,
            ],
        )
        stream_preset.click(
            fn=apply_streaming,
            inputs=[],
            outputs=[
                model_name,
                output_dir,
                output_format,
                output_template,
                batch_size,
                chunk_len_sec,
                stream,
                stream_chunk_sec,
                overlap_duration,
                highlight_words,
                word_timestamps,
                merge_strategy,
                overwrite,
                verbose,
                no_progress,
                quiet,
                fp16,
                fp32,
            ],
        )

        # Transcribe button callback
        transcribe_btn.click(
            fn=transcribe_webui,
            inputs=[
                files,
                model_name,
                output_dir,
                output_format,
                output_template,
                batch_size,
                chunk_len_sec,
                stream,
                stream_chunk_sec,
                overlap_duration,
                highlight_words,
                word_timestamps,
                merge_strategy,
                overwrite,
                verbose,
                no_progress,
                quiet,
                fp16,
                fp32,
            ],
            outputs=output_files,
        )

    return demo


if __name__ == "__main__":
    # Build and launch the Gradio demo
    demo_app = build_ui()
    # Launch with configured server settings
    demo_app.launch(
        server_name=constant.GRADIO_SERVER_NAME,
        server_port=constant.GRADIO_SERVER_PORT,
        share=False,  # Keep share=False by default for security
        show_error=True,
        favicon_path=None,
        auth=None,
        auth_message=None,
        prevent_thread_lock=False,
        show_api=True,
        debug=False,
        max_threads=40,
    )
