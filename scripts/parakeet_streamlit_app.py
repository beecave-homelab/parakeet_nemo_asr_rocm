"""
Streamlit WebUI for Parakeet‑NEMO ASR
-------------------------------------

This script implements a fully functional proof‑of‑concept Web UI for
speech transcription using the `parakeet_nemo_asr_rocm` package. It is
built with [Streamlit](https://streamlit.io) and aims to provide a clean
user experience with sensible defaults, collapsible configuration
panels, a light/dark mode toggle and quick presets for common
transcription scenarios.

The app exposes most of the command‑line options of the `cli_transcribe`
function as interactive widgets. When the **Transcribe** button is
pressed the selected audio/video files are sent to `cli_transcribe`
and the generated output files are listed for download.

To run the app, install streamlit and execute:

```
pip install streamlit
python parakeet_streamlit_app.py
```

Then open the provided local URL in your browser.
"""

from __future__ import annotations

import base64
import pathlib
import tempfile
from typing import List, Tuple

import streamlit as st

from parakeet_nemo_asr_rocm.utils import constant
from parakeet_nemo_asr_rocm.transcribe import cli_transcribe
from parakeet_nemo_asr_rocm.models.parakeet import DEFAULT_MODEL_NAME


def enforce_precision(fp16: bool, fp32: bool) -> Tuple[bool, bool]:
    """Ensure only one of FP16 or FP32 precision flags is active.

    Streamlit checkboxes can both be ticked simultaneously. When this
    happens we favour FP16 and disable FP32. Otherwise, the original
    values are returned unchanged.
    """
    if fp16 and fp32:
        return True, False
    return fp16, fp32


def save_uploaded_files(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> List[pathlib.Path]:
    """Persist uploaded files to a temporary directory and return their paths.

    Streamlit returns file‑like objects for uploads. In order to pass
    them to `cli_transcribe` we need to write them to disk. Files are
    saved into a temporary directory that is cleaned up when the
    application shuts down.
    """
    temp_dir = tempfile.mkdtemp(prefix="parakeet_uploads_")
    paths: List[pathlib.Path] = []
    for file_obj in uploaded_files:
        # Use the original filename if available
        filename = pathlib.Path(file_obj.name).name
        dest = pathlib.Path(temp_dir) / filename
        with open(dest, "wb") as f:
            f.write(file_obj.getbuffer())
        paths.append(dest)
    return paths


def transcribe_action(
    files: List[pathlib.Path],
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
) -> List[str]:
    """Call the Parakeet transcription function with the supplied arguments.

    Returns a list of file paths pointing to the generated transcription files.
    """
    fp16, fp32 = enforce_precision(fp16, fp32)
    outputs = cli_transcribe(
        audio_files=files,
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
    return [str(p) for p in outputs]


def apply_default() -> Tuple[
    str,
    str,
    str,
    str,
    int,
    int,
    bool,
    int,
    int,
    bool,
    bool,
    str,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
]:
    """Return default preset values as a tuple matching state variables."""
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


def apply_high_quality() -> Tuple[
    str,
    str,
    str,
    str,
    int,
    int,
    bool,
    int,
    int,
    bool,
    bool,
    str,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
]:
    """Return high quality preset values."""
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


def apply_streaming() -> Tuple[
    str,
    str,
    str,
    str,
    int,
    int,
    bool,
    int,
    int,
    bool,
    bool,
    str,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
]:
    """Return streaming preset values."""
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


def set_theme(mode: str) -> None:
    """Inject CSS to switch between light and dark themes.

    Streamlit does not provide a native theme toggle, but you can
    override default styles by injecting a custom `<style>` tag. This
    function defines two simple palettes and applies the selected one
    to the body and form elements.
    """
    if mode == "dark":
        bg = "#121212"
        fg = "#e0e0e0"
        card = "#1e1e1e"
        border = "#333333"
        primary = "#ff6f00"
        secondary = "#ff9e40"
    else:
        bg = "#f8f9fa"
        fg = "#212529"
        card = "#ffffff"
        border = "#dee2e6"
        primary = "#ff6f00"
        secondary = "#e65100"

    css = f"""
    <style>
    /* Base styles */
    .stApp {{
        background-color: {bg};
        color: {fg};
    }}
    
    /* Text elements */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp label, .stApp p, .stApp div, .stApp span {{
        color: {fg} !important;
    }}
    
    /* Form elements */
    .stApp input, .stApp textarea, .stApp select, .stApp .stTextInput input {{
        color: {fg} !important;
        background-color: {card} !important;
        border-color: {border} !important;
    }}
    
    /* Checkboxes and radio buttons */
    .stCheckbox > label, .stRadio > label, .stToggle > label {{
        color: {fg} !important;
    }}
    
    /* Sliders */
    .stApp .stSlider > div[data-baseweb="slider"] {{
        color: {fg};
    }}
    
    /* Buttons */
    .stApp .stButton > button {{
        background-color: {primary};
        color: white;
        border: none;
        font-weight: 500;
    }}
    .stApp .stButton > button:hover {{
        background-color: {secondary};
        color: white;
    }}
    
    /* Expanders and containers */
    .stApp .streamlit-expander {{
        border: 1px solid {border};
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }}
    .stApp .streamlit-expanderHeader {{
        font-weight: 600;
        color: {fg};
    }}
    
    /* File uploader - Main container */
    .stApp .stFileUploader > div {{
        border: 2px dashed {border} !important;
        border-radius: 0.5rem !important;
        background-color: {card} !important;
        color: {fg} !important;
        padding: 1.5rem !important;
    }}
    
    /* Hover state */
    .stApp .stFileUploader > div:hover {{
        border-color: {primary} !important;
        background-color: {card} !important;
    }}
    
    /* Text elements */
    .stApp .stFileUploader > div > div > div,
    .stApp .stFileUploader > div > div > div > div > div,
    .stApp .stFileUploader > div > div > div > div > div > div,
    .stApp .stFileUploader > div > div > div > div > div > div > div,
    .stApp .stFileUploader > div > div > div > div > div > div > div > span,
    .stApp .stFileUploader > div > div > div > div > div > div > div > small {{
        color: {fg} !important;
    }}
    
    /* Upload icon */
    .stApp .stFileUploader svg {{
        fill: {fg} !important;
    }}
    
    /* Browse files button */
    .stApp .stFileUploader button[data-testid="stBaseButton-secondary"] {{
        background-color: {primary} !important;
        color: white !important;
        border: none !important;
        margin-top: 1rem !important;
    }}
    
    .stApp .stFileUploader button[data-testid="stBaseButton-secondary"]:hover {{
        background-color: {secondary} !important;
    }}
    
    /* Links */
    .stApp a {{
        color: {primary} !important;
    }}
    .stApp a:hover {{
        color: {secondary} !important;
    }}
    
    /* Code blocks */
    .stApp code {{
        background-color: {card};
        color: {fg};
        border: 1px solid {border};
        padding: 0.2em 0.4em;
        border-radius: 0.2em;
    }}
    
    /* Merge strategy dropdown */
    .stApp div[data-baseweb="select"] > div:first-child {{
        background-color: {card} !important;
        color: {fg} !important;
    }}
    .stApp div[data-baseweb="select"] > div:first-child:hover {{
        border-color: {primary} !important;
    }}
    .stApp div[data-baseweb="select"] div[role="listbox"] > div {{
        background-color: {card} !important;
        color: {fg} !important;
    }}
    .stApp div[data-baseweb="select"] div[role="listbox"] > div:hover {{
        background-color: {border} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def main() -> None:
    """Entry point for the Streamlit UI."""
    st.set_page_config(
        page_title="Parakeet‑NEMO ASR WebUI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Initialise theme state - default to dark mode
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    set_theme(st.session_state.theme)

    # Page header
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown("## Parakeet‑NEMO ASR WebUI")
        st.markdown(
            "**Upload audio/video and configure transcription settings.**\n"
            "Fast presets, streaming, precision control, and customizable outputs."
        )
    with col_right:
        if st.button(
            "Dark Mode" if st.session_state.theme == "light" else "Light Mode",
            key="theme_toggle",
        ):
            # Toggle theme state and apply new styles
            st.session_state.theme = (
                "dark" if st.session_state.theme == "light" else "light"
            )
            set_theme(st.session_state.theme)
        st.caption("Switch Light / Dark. Preference saved in session.")

    # Input and model selection
    st.markdown("### Input & Model")
    with st.container():
        uploaded_files = st.file_uploader(
            "Upload Audio / Video", accept_multiple_files=True, type=None
        )
        st.caption("Supports common audio/video containers.")
        model_name = st.text_input(
            "Model Name or Path", value=DEFAULT_MODEL_NAME, key="model_name"
        )
        st.caption("Local path or pretrained model identifier.")

    # Collapsible output settings
    with st.expander("Output Settings", expanded=False):
        output_dir = st.text_input(
            "Output Directory", value="./output", key="output_dir"
        )
        st.caption("Where transcription files will be written.")
        output_format = st.selectbox(
            "Output Format", ["txt", "srt", "vtt", "json"], index=0, key="output_format"
        )
        output_template = st.text_input(
            "Output Filename Template", value="{filename}", key="output_template"
        )
        st.caption("Use placeholders like {filename}, {lang}.")

    # Transcription controls
    with st.expander("Transcription Controls", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            batch_size = st.slider(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=constant.DEFAULT_BATCH_SIZE,
                step=1,
            )
            stream = st.checkbox("Stream Mode", key="stream")
            highlight_words = st.checkbox("Highlight Words", key="highlight_words")
            merge_strategy = st.selectbox(
                "Merge Strategy",
                ["none", "contiguous", "lcs"],
                index=2,
                key="merge_strategy",
            )
        with c2:
            chunk_len_sec = st.slider(
                "Chunk Length (sec)",
                min_value=1,
                max_value=600,
                value=constant.DEFAULT_CHUNK_LEN_SEC,
                step=1,
            )
            stream_chunk_sec = st.slider(
                "Stream Chunk Length (sec)", min_value=0, max_value=60, value=0, step=1
            )
            overlap_duration = st.slider(
                "Overlap Duration (sec)", min_value=0, max_value=60, value=15, step=1
            )
            word_timestamps = st.checkbox("Word Timestamps", key="word_timestamps")

    # Execution flags
    with st.expander("Execution Flags", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            overwrite = st.checkbox("Overwrite Outputs", key="overwrite")
            verbose = st.checkbox("Verbose Output", key="verbose")
            no_progress = st.checkbox("Disable Progress Bar", key="no_progress")
            quiet = st.checkbox("Quiet Mode", key="quiet")
        with c2:
            fp16 = st.checkbox("Use FP16 Precision", key="fp16")
            fp32 = st.checkbox("Use FP32 Precision", key="fp32")
        # Show a warning if both precisions are selected
        warn = ""
        if fp16 and fp32:
            warn = "FP16 and FP32 both selected; FP16 will take precedence."
            st.warning(warn)

    # Presets
    with st.expander("Presets / Quick Actions", expanded=False):
        preset_cols = st.columns(3)
        if preset_cols[0].button("Default Fast", key="preset_default"):
            (
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
            ) = apply_default()
            # Update session state values accordingly
            st.session_state.update(
                {
                    "model_name": model_name,
                    "output_dir": output_dir,
                    "output_format": output_format,
                    "output_template": output_template,
                    "stream": stream,
                    "stream_chunk_sec": stream_chunk_sec,
                    "highlight_words": highlight_words,
                    "word_timestamps": word_timestamps,
                    "merge_strategy": merge_strategy,
                    "overwrite": overwrite,
                    "verbose": verbose,
                    "no_progress": no_progress,
                    "quiet": quiet,
                    "fp16": fp16,
                    "fp32": fp32,
                }
            )
        if preset_cols[1].button("High Quality", key="preset_hq"):
            (
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
            ) = apply_high_quality()
            st.session_state.update(
                {
                    "model_name": model_name,
                    "output_dir": output_dir,
                    "output_format": output_format,
                    "output_template": output_template,
                    "stream": stream,
                    "stream_chunk_sec": stream_chunk_sec,
                    "highlight_words": highlight_words,
                    "word_timestamps": word_timestamps,
                    "merge_strategy": merge_strategy,
                    "overwrite": overwrite,
                    "verbose": verbose,
                    "no_progress": no_progress,
                    "quiet": quiet,
                    "fp16": fp16,
                    "fp32": fp32,
                }
            )
        if preset_cols[2].button("Streaming Mode", key="preset_streaming"):
            (
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
            ) = apply_streaming()
            st.session_state.update(
                {
                    "model_name": model_name,
                    "output_dir": output_dir,
                    "output_format": output_format,
                    "output_template": output_template,
                    "stream": stream,
                    "stream_chunk_sec": stream_chunk_sec,
                    "highlight_words": highlight_words,
                    "word_timestamps": word_timestamps,
                    "merge_strategy": merge_strategy,
                    "overwrite": overwrite,
                    "verbose": verbose,
                    "no_progress": no_progress,
                    "quiet": quiet,
                    "fp16": fp16,
                    "fp32": fp32,
                }
            )

    # Transcription action
    if st.button("Transcribe", key="transcribe_btn"):
        if not uploaded_files:
            st.error("Please upload at least one audio or video file.")
        else:
            with st.spinner(
                "Transcribing... this may take a while depending on file size and model."
            ):
                file_paths = save_uploaded_files(uploaded_files)
                outputs = transcribe_action(
                    files=file_paths,
                    model_name=model_name,
                    output_dir=output_dir,
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
                    no_progress=no_progress,
                    quiet=quiet,
                    fp16=fp16,
                    fp32=fp32,
                )
            st.success("🎉 Transcription complete!")
            if outputs:
                with st.expander("📝 Transcription Results", expanded=True):
                    st.markdown("### 📂 Generated Files")
                    st.markdown("Click on a file to download it.")

                    # Create a container for the files
                    file_container = st.container()

                    for out_file in outputs:
                        file_path = pathlib.Path(out_file)
                        try:
                            # Get file size in a human-readable format
                            size_bytes = file_path.stat().st_size
                            if size_bytes < 1024:
                                size_str = f"{size_bytes} B"
                            elif size_bytes < 1024 * 1024:
                                size_str = f"{size_bytes / 1024:.1f} KB"
                            else:
                                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

                            # Read file data for download
                            with open(file_path, "rb") as f:
                                data = f.read()
                            b64 = base64.b64encode(data).decode()

                            # Create a download button
                            file_name = file_path.name
                            file_ext = file_path.suffix.lower()

                            # Get appropriate icon based on file extension
                            if file_ext in [".txt", ".srt", ".vtt", ".json"]:
                                icon = "📄"
                            else:
                                icon = "📁"

                            # Create a nice card for each file
                            with file_container:
                                col1, col2 = st.columns([1, 4])
                                with col1:
                                    st.markdown(f"### {icon}", unsafe_allow_html=True)
                                with col2:
                                    file_info = f"**{file_name}**  \n*{size_str} • {file_ext.upper().replace('.', '')}*"
                                    st.markdown(file_info)

                                    # Create download button
                                    st.markdown(
                                        f'<a href="data:file/octet-stream;base64,{b64}" '
                                        f'download="{file_name}" '
                                        f'class="download-button">⬇️ Download</a>',
                                        unsafe_allow_html=True,
                                    )

                        except Exception as e:
                            st.error(f"Error processing {file_path.name}: {str(e)}")

                    # Add some CSS for the download button
                    st.markdown(
                        """
                    <style>
                    .download-button {
                        display: inline-block;
                        padding: 0.25rem 0.75rem;
                        background-color: #4CAF50;
                        color: white !important;
                        text-align: center;
                        text-decoration: none;
                        border-radius: 0.25rem;
                        font-size: 0.9rem;
                        margin-top: 0.5rem;
                    }
                    .download-button:hover {
                        background-color: #45a049;
                        color: white !important;
                        text-decoration: none;
                    }
                    </style>
                    """,
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
