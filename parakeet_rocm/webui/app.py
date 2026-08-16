"""Gradio WebUI application factory.

Builds and launches the complete web application using a layered
architecture with dependency injection for testability.
"""

from __future__ import annotations

import atexit
import gc
import json
import os
import pathlib
import signal
import sys
import threading
import time

try:
    import gradio as gr
except ImportError:  # pragma: no cover - optional dependency
    gr = None

# Pre-import scipy.linalg to avoid Cython fused_type errors when NeMo imports it later
# via lightning.pytorch -> torchmetrics -> scipy.signal -> scipy.linalg
import scipy.linalg  # noqa: F401
import torch

from parakeet_rocm.models.parakeet import clear_model_cache, unload_model_to_cpu
from parakeet_rocm.utils.constant import (
    BENCHMARK_OUTPUT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_LEN_SEC,
    DEFAULT_DEMUCS,
    DEFAULT_STABILIZE,
    DEFAULT_VAD,
    DEFAULT_WORD_TIMESTAMPS,
    GRADIO_ANALYTICS_ENABLED,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    IDLE_CLEAR_TIMEOUT_SEC,
    IDLE_UNLOAD_TIMEOUT_SEC,
    PARAKEET_MODEL_NAME,
    SUPPORTED_EXTENSIONS,
)
from parakeet_rocm.utils.logging_config import configure_logging, get_logger
from parakeet_rocm.webui.core.job_manager import JobManager, JobStatus
from parakeet_rocm.webui.core.session import (
    SessionManager,
    set_global_job_manager,
)
from parakeet_rocm.webui.ui.theme import configure_theme
from parakeet_rocm.webui.utils.metrics_formatter import (
    format_gpu_stats_section,
    format_runtime_section,
)
from parakeet_rocm.webui.utils.presets import PRESETS, get_preset
from parakeet_rocm.webui.utils.zip_creator import ZipCreator
from parakeet_rocm.webui.validation.file_validator import (
    FileValidationError,
    validate_audio_files,
)

# Module logger
logger = get_logger(__name__)

WEBUI_CONTAINER_CSS = """
.gradio-container { max-width: 1200px; margin: auto; }
.parakeet-title { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
.parakeet-title img { width: 2.5rem; height: 2.5rem; flex: 0 0 auto; }
.parakeet-title h1 { margin: 0; }
"""

WEBUI_TITLE_HTML = """\
<div class="parakeet-title">
  <img src="./parakeet-assets/parakeet-rocm-icon.svg" alt="">
  <h1>Parakeet-ROCm WebUI</h1>
</div>
"""


def _require_gradio() -> None:
    """Ensure Gradio is available for the WebUI.

    Raises:
        ImportError: If Gradio is not installed.
    """
    if gr is None:
        raise ImportError(
            "Gradio is required for the WebUI. Install the optional webui "
            "dependencies (for example: pdm add -G webui or pip install "
            "'parakeet-rocm[webui]')."
        )


def _cleanup_models() -> None:
    """Best-effort model cleanup to free GPU VRAM and host memory."""
    try:
        unload_model_to_cpu()
    finally:
        try:
            clear_model_cache()
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass


def _register_shutdown_handlers() -> None:
    """Register atexit and signal handlers for cleanup on shutdown."""
    atexit.register(_cleanup_models)

    def _sig_handler(_signum: int, _frame) -> None:  # noqa: ANN001, D401
        """Handle termination signals by cleaning up models and exiting."""
        try:
            _cleanup_models()
        finally:
            # Force immediate process termination; os._exit bypasses
            # Python cleanup (atexit, finally blocks) but ensures the
            # process actually dies on CTRL+C.
            print("\nShutting down...", file=sys.stderr)
            os._exit(128 + _signum)

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        # Some environments may not support signal registration (e.g., Windows)
        pass


def _start_idle_offload_thread(job_manager: JobManager) -> None:
    """Start a daemon thread to offload/clear model when idle in WebUI.

    The thread checks periodically whether a job is running. If the system
    remains idle for IDLE_UNLOAD_TIMEOUT_SEC, it moves the model to CPU.
    If idle for IDLE_CLEAR_TIMEOUT_SEC, it clears the model cache entirely.
    """

    def _worker() -> None:
        last_activity = time.monotonic()
        unloaded = False
        cleared = False
        while True:
            try:
                current = job_manager.get_current_job()
                if current is not None:
                    last_activity = time.monotonic()
                    if unloaded or cleared:
                        unloaded = False
                        cleared = False
                else:
                    now = time.monotonic()
                    if not unloaded and (now - last_activity) >= IDLE_UNLOAD_TIMEOUT_SEC:
                        try:
                            logger.info("[webui] Idle threshold reached - offloading model to CPU")
                            unload_model_to_cpu()
                        except Exception as e:
                            logger.warning(f"[webui] Failed to unload model: {e}")
                        finally:
                            unloaded = True
                    if not cleared and (now - last_activity) >= IDLE_CLEAR_TIMEOUT_SEC:
                        try:
                            logger.info("[webui] Extended idle - clearing model cache")
                            clear_model_cache()
                        except Exception as e:
                            logger.warning(f"[webui] Failed to clear model cache: {e}")
                        finally:
                            cleared = True
            except Exception as e:
                logger.warning(f"[webui] Idle offload thread error: {e}")
            time.sleep(5.0)

    t = threading.Thread(target=_worker, name="webui-idle-offloader", daemon=True)
    t.start()


def build_app(
    *,
    job_manager: JobManager | None = None,
    analytics_enabled: bool = GRADIO_ANALYTICS_ENABLED,
) -> gr.Blocks:
    """Build the Gradio WebUI application.

    Creates a complete web interface for the Parakeet-NEMO ASR system
    with file upload, configuration, and transcription capabilities.

    Args:
        job_manager: Job manager instance (creates default if None).
        analytics_enabled: Enable Gradio analytics tracking.

    Returns:
        Configured Gradio Blocks application ready for the FastAPI composition.

    Examples:
        >>> app = build_app()

        >>> # With custom job manager
        >>> manager = JobManager()
        >>> app = build_app(job_manager=manager)
    """
    _require_gradio()
    # Initialize dependencies
    if job_manager is None:
        job_manager = JobManager()

    # Wire job manager for session helpers
    set_global_job_manager(job_manager)

    session_manager = SessionManager()

    # Keep visual styling on Blocks for embedding and previews. Install metadata
    # is supplied by ``create_app()``, where the referenced asset routes exist.
    # ``launch_app()`` is the supported standalone server entry point.
    with gr.Blocks(
        title="Parakeet-ROCm WebUI",
        analytics_enabled=analytics_enabled,
        theme=configure_theme(),
        css=WEBUI_CONTAINER_CSS,
    ) as app:
        # Session state (for future use)
        # Reserved for future session features
        _session_state = gr.State(session_manager.create_session())

        # Header
        gr.Markdown(WEBUI_TITLE_HTML)
        gr.Markdown(
            "Upload audio or video files and transcribe them using NVIDIA's Parakeet-NEMO models."
        )

        # File upload section
        with gr.Group():
            gr.Markdown("### 📁 Upload Files")
            file_upload = gr.File(
                label="Audio/Video Files",
                file_count="multiple",
                file_types=list(SUPPORTED_EXTENSIONS),
            )

        # Configuration section
        with gr.Group():
            gr.Markdown("### ⚙️ Configuration")

            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value="default",
                    label="Quick Presets",
                    info="Select a preset or customize settings below",
                )

            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=[
                        PARAKEET_MODEL_NAME,
                        "nvidia/parakeet-tdt-0.6b-v3",
                        "nvidia/parakeet-tdt-0.6b-v2",
                    ],
                    value=PARAKEET_MODEL_NAME,
                    label="Model Selection",
                    info="v3=multilingual, v2=English only, unified=better WER",
                    allow_custom_value=True,
                )

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=32,
                        value=DEFAULT_BATCH_SIZE,
                        step=1,
                        label="Batch Size",
                    )
                    chunk_len_sec = gr.Slider(
                        minimum=30,
                        maximum=600,
                        value=DEFAULT_CHUNK_LEN_SEC,
                        step=30,
                        label="Chunk Length (seconds)",
                    )

                with gr.Row():
                    overlap_duration = gr.Slider(
                        minimum=0,
                        maximum=60,
                        value=15,
                        step=5,
                        label="Overlap Duration (seconds)",
                        info="Overlap between consecutive chunks for better continuity",
                    )
                    merge_strategy = gr.Dropdown(
                        choices=["lcs", "contiguous", "none"],
                        value="lcs",
                        label="Merge Strategy",
                        info="lcs=accurate, contiguous=fast, none=concatenate",
                        interactive=DEFAULT_WORD_TIMESTAMPS,
                    )

                with gr.Row():
                    word_timestamps = gr.Checkbox(
                        label="Word-level Timestamps",
                        value=DEFAULT_WORD_TIMESTAMPS,
                    )
                    stabilize = gr.Checkbox(
                        label="Stabilize Timestamps (stable-ts)",
                        value=DEFAULT_STABILIZE,
                    )
                    vad = gr.Checkbox(
                        label="Voice Activity Detection",
                        value=DEFAULT_VAD,
                    )
                    demucs = gr.Checkbox(
                        label="Audio Enhancement (Demucs)",
                        value=DEFAULT_DEMUCS,
                    )

                vad_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.35,
                    step=0.05,
                    label="VAD Threshold",
                    info="Voice activity detection sensitivity (higher = stricter)",
                    visible=False,  # Show only when VAD is enabled
                )

                with gr.Row():
                    stream_mode = gr.Checkbox(
                        label="Streaming Mode",
                        value=False,
                        info="Enable low-latency pseudo-streaming (smaller chunks)",
                    )
                    stream_chunk_sec = gr.Slider(
                        minimum=5,
                        maximum=30,
                        value=8,
                        step=1,
                        label="Stream Chunk Size (seconds)",
                        visible=False,  # Show only when streaming is enabled
                    )

                with gr.Row():
                    highlight_words = gr.Checkbox(
                        label="Highlight Words (SRT/VTT)",
                        value=False,
                        info="Bold each word in subtitle outputs",
                    )
                    overwrite_files = gr.Checkbox(
                        label="Overwrite Existing Files",
                        value=False,
                        info="Replace existing outputs instead of numbered copies",
                    )

                with gr.Row():
                    precision = gr.Radio(
                        choices=["fp16", "fp32"],
                        value="fp16",
                        label="Inference Precision",
                        info="fp16=faster (default), fp32=more accurate",
                    )

                output_format = gr.Dropdown(
                    choices=["txt", "srt", "vtt", "json"],
                    value="srt",
                    label="Output Format",
                )

        # Action buttons
        with gr.Row():
            transcribe_btn = gr.Button(
                "🚀 Start Transcription",
                variant="primary",
                size="lg",
            )
            clear_btn = gr.Button("Clear", variant="secondary")

        # Output section with tabs
        with gr.Group():
            gr.Markdown("### 📊 Results")

            with gr.Tabs():
                # Results tab
                with gr.TabItem("Results"):
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        show_label=True,
                    )
                    with gr.Row():
                        output_files = gr.File(
                            label="Output Files",
                            file_count="multiple",
                            interactive=False,
                            visible=True,
                        )
                        download_button = gr.DownloadButton(
                            label="📦 Download ZIP",
                            visible=False,
                            variant="primary",
                            size="lg",
                        )

                # Benchmarks tab
                with gr.TabItem("Benchmarks"):
                    gr.Markdown(
                        "Real-time performance metrics for the current or "
                        "last completed transcription job."
                    )

                    benchmark_status = gr.Markdown(
                        "*No metrics available yet. Start a transcription to see benchmarks.*",
                        elem_id="benchmark-status",
                    )

                    with gr.Row():
                        with gr.Column():
                            runtime_display = gr.Markdown(
                                "### Runtime\n*N/A*",
                                elem_id="runtime-metrics",
                            )
                        with gr.Column():
                            gpu_display = gr.Markdown(
                                "### GPU Statistics\n*N/A*",
                                elem_id="gpu-metrics",
                            )

                    benchmark_json_display = gr.JSON(
                        label="Raw Metrics (JSON)",
                        visible=False,
                    )

                    refresh_metrics_btn = gr.Button(
                        "🔄 Refresh Metrics",
                        variant="secondary",
                        size="sm",
                    )

        # Event handlers
        def apply_preset(preset_name: str) -> dict:  # type: ignore[type-arg]
            """Apply preset configuration.

            Returns:
                Dictionary of updated component values.
            """
            try:
                preset = get_preset(preset_name)
                config = preset.config
                return {
                    model_selector: config.model_name,
                    batch_size: config.batch_size,
                    chunk_len_sec: config.chunk_len_sec,
                    overlap_duration: config.overlap_duration,
                    stream_mode: config.stream,
                    stream_chunk_sec: config.stream_chunk_sec,
                    word_timestamps: gr.update(value=config.word_timestamps),
                    merge_strategy: gr.update(
                        value=config.merge_strategy,
                        interactive=config.word_timestamps,
                    ),
                    highlight_words: config.highlight_words,
                    stabilize: config.stabilize,
                    vad: config.vad,
                    demucs: config.demucs,
                    vad_threshold: config.vad_threshold,
                    overwrite_files: config.overwrite,
                    precision: ("fp16" if config.fp16 else "fp32"),
                    output_format: config.output_format,
                }
            except KeyError:
                return {}

        def transcribe_files(
            files: list[str],
            model_name_val: str,
            batch_size_val: int,
            chunk_len_val: int,
            overlap_dur: float,
            stream_val: bool,
            stream_chunk_val: int,
            word_ts: bool,
            merge_strat: str,
            highlight_val: bool,
            stab: bool,
            vad_val: bool,
            demucs_val: bool,
            vad_thresh: float,
            overwrite_val: bool,
            precision_val: str,
            out_format: str,
            progress: gr.Progress | None = None,
        ) -> tuple[str, gr.components.Component | None, gr.components.Component | None]:
            """Handle transcription request with progress tracking.

            Args:
                files: Uploaded audio/video files.
                model_name_val: Model name or path for transcription.
                batch_size_val: Batch size for inference.
                chunk_len_val: Chunk length in seconds.
                overlap_dur: Overlap between chunks in seconds.
                stream_val: Enable streaming mode.
                stream_chunk_val: Stream chunk size in seconds.
                word_ts: Enable word timestamps.
                merge_strat: Strategy for merging overlapping chunks.
                highlight_val: Highlight words in SRT/VTT outputs.
                stab: Enable stabilization.
                vad_val: Enable VAD.
                demucs_val: Enable Demucs.
                vad_thresh: VAD threshold (0.0-1.0).
                overwrite_val: Overwrite existing output files.
                precision_val: Inference precision (fp16/fp32).
                out_format: Output format (txt/srt/vtt/json).
                progress: Gradio progress tracker.

            Returns:
                Tuple of (status_message, output_files).
            """
            if progress is None:
                progress = gr.Progress()
            if not files:
                logger.warning("Transcription requested with no files uploaded")
                return "❌ Please upload files first.", None

            try:
                # Step 1: Validate files
                progress(0.0, desc="🔍 Validating uploaded files...")
                logger.info(f"Starting transcription for {len(files)} file(s)")
                file_paths = [pathlib.Path(f.name) for f in files]
                logger.debug(f"File paths: {[str(p) for p in file_paths]}")
                validate_audio_files(file_paths)
                logger.info("File validation successful")

                # Step 2: Create configuration
                progress(0.1, desc="⚙️ Configuring transcription...")
                from parakeet_rocm.webui.validation.schemas import TranscriptionConfig

                config = TranscriptionConfig(
                    model_name=model_name_val,
                    batch_size=batch_size_val,
                    chunk_len_sec=chunk_len_val,
                    overlap_duration=overlap_dur,
                    stream=stream_val,
                    stream_chunk_sec=stream_chunk_val,
                    word_timestamps=word_ts,
                    merge_strategy=merge_strat,
                    highlight_words=highlight_val,
                    stabilize=stab,
                    vad=vad_val,
                    demucs=demucs_val,
                    vad_threshold=vad_thresh,
                    overwrite=overwrite_val,
                    fp16=(precision_val == "fp16"),
                    fp32=(precision_val == "fp32"),
                    output_format=out_format,
                )
                logger.info(
                    f"Config: batch={batch_size_val}, chunk={chunk_len_val}s, "
                    f"format={out_format}, stabilize={stab}"
                )

                # Step 3: Submit job
                progress(0.2, desc="📝 Submitting transcription job...")
                job = job_manager.submit_job(file_paths, config)
                logger.info(f"Job submitted with ID: {job.job_id}")

                # Step 4: Run transcription with real-time progress tracking
                logger.info(
                    f"Starting transcription for {len(files)} file(s): "
                    f"{[p.name for p in file_paths]}"
                )

                # Create callback to update Gradio progress from batch progress
                def update_gradio_progress(current: int, total: int) -> None:
                    """Update Gradio progress bar from transcription batches."""
                    # Map batch progress (0-total) to Gradio progress (0.3-0.95)
                    batch_fraction = current / total if total > 0 else 0
                    gradio_progress = 0.3 + (batch_fraction * 0.65)
                    progress(
                        gradio_progress,
                        desc=f"🎙️ Transcribing batch {current}/{total}...",
                    )
                    logger.debug(f"Progress: {current}/{total} batches ({gradio_progress:.1%})")

                # Run transcription with progress callback
                result = job_manager.run_job(job.job_id, progress_callback=update_gradio_progress)

                # Step 5: Finalize
                progress(0.95, desc="✨ Finalizing results...")

                if result.status == JobStatus.COMPLETED:
                    logger.info(
                        f"Transcription completed! Generated {len(result.outputs)} output file(s)"
                    )

                    # Handle bulk download for multiple files
                    if len(result.outputs) > 1:
                        logger.info("Multiple files detected, creating ZIP archive")
                        zip_creator = ZipCreator()
                        zip_path = zip_creator.create_temporary_zip(
                            result.outputs,
                            prefix="transcriptions_",
                        )
                        logger.info(f"Created ZIP archive: {zip_path}")
                        status_msg = (
                            f"✅ Transcription completed! "
                            f"Processed {len(files)} file(s). "
                            f"Generated {len(result.outputs)} output(s). "
                            f"Click the download button below."
                        )
                        logger.debug(f"ZIP archive created: {zip_path}")
                        progress(1.0, desc="✅ Done!")
                        # Return: status, file_list (hidden),
                        # download_button (visible + file)
                        return (
                            status_msg,
                            gr.update(visible=False),
                            gr.update(visible=True, value=str(zip_path)),
                        )
                    else:
                        # Single file - return as-is
                        status_msg = (
                            f"✅ Transcription completed! "
                            f"Processed {len(files)} file(s). "
                            f"Generated {len(result.outputs)} output(s)."
                        )
                        output_paths = [str(p) for p in result.outputs]
                        logger.debug(f"Output paths: {output_paths}")
                        progress(1.0, desc="✅ Done!")
                        # Return: status, file_list (visible), download_button (hidden)
                        return (
                            status_msg,
                            gr.update(visible=True, value=output_paths),
                            gr.update(visible=False),
                        )
                else:
                    error_msg = f"❌ Transcription failed: {result.error}"
                    logger.error(f"Transcription failed: {result.error}")
                    # Return: error status, hide both file list and button
                    return (
                        error_msg,
                        gr.update(visible=False, value=None),
                        gr.update(visible=False),
                    )

            except FileValidationError as e:
                logger.warning(f"File validation error: {e}")
                return (
                    f"❌ Validation error: {e}",
                    gr.update(visible=False, value=None),
                    gr.update(visible=False),
                )
            except Exception as e:
                logger.exception(f"Unexpected error during transcription: {e}")
                return (
                    f"❌ Error: {e}",
                    gr.update(visible=False, value=None),
                    gr.update(visible=False),
                )

        def refresh_benchmarks() -> tuple[str, str, str, dict]:  # type: ignore[type-arg]
            """Refresh benchmark metrics from current job or persisted files.

            Tries to load metrics from:
            1. Current running job
            2. Last completed job in memory
            3. Most recent benchmark JSON file on disk

            Returns:
                Tuple of (status, runtime_md, gpu_md, json_data).
            """
            metrics = None
            job_id_short = None
            job_status = None
            num_outputs = 0

            # Try current job first, then last completed
            job = job_manager.get_current_job()
            if job is None:
                job = job_manager.get_last_completed_job()

            # If we have a job in memory with metrics, use it
            if job and job.metrics:
                metrics = job.metrics
                job_id_short = job.job_id[:8]
                job_status = job.status.value
                num_outputs = len(job.outputs)
                logger.info(f"Loaded benchmarks from in-memory job: {job_id_short}")

            # Otherwise, load from most recent JSON file on disk
            if metrics is None:
                logger.info(f"No in-memory job with metrics, scanning {BENCHMARK_OUTPUT_DIR}")
                try:
                    benchmark_dir = pathlib.Path(BENCHMARK_OUTPUT_DIR)
                    if benchmark_dir.exists():
                        # Find all JSON files
                        json_files = sorted(
                            benchmark_dir.glob("*.json"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,  # Most recent first
                        )

                        if json_files:
                            latest_file = json_files[0]
                            logger.info(f"Loading benchmark from: {latest_file.name}")

                            with latest_file.open("r") as f:
                                metrics = json.load(f)

                            # Extract job ID from filename
                            # (format: YYYYMMDD_HHMMSS_job_XXXXXXXX.json)
                            job_id_short = latest_file.stem.split("_job_")[-1][:8]
                            job_status = "completed (from file)"
                            num_outputs = len(metrics.get("files", []))

                            logger.info(
                                f"Loaded benchmark for job {job_id_short} from {latest_file.name}"
                            )
                        else:
                            logger.warning(f"No benchmark JSON files found in {benchmark_dir}")
                    else:
                        logger.warning(f"Benchmark directory does not exist: {benchmark_dir}")
                except Exception as e:
                    logger.exception(f"Error loading benchmark from disk: {e}")

            # If still no metrics, return empty state
            if metrics is None:
                logger.info("No benchmarks available (neither in-memory nor on disk)")
                return (
                    "*No metrics available yet. Start a transcription to see benchmarks.*",
                    "### Runtime\n*N/A*",
                    "### GPU Statistics\n*N/A*",
                    gr.update(visible=False, value=None),
                )

            # Format the metrics for display
            runtime_md = format_runtime_section(
                metrics.get("runtime_seconds"),
                metrics.get("total_wall_seconds"),
            )
            gpu_md = format_gpu_stats_section(metrics.get("gpu_stats"))

            status = (
                f"**Job ID**: `{job_id_short}`\n"
                f"**Status**: {job_status}\n"
                f"**Files**: {num_outputs} output(s)"
            )

            return (
                status,
                runtime_md,
                gpu_md,
                gr.update(visible=True, value=metrics),
            )

        def clear_all() -> dict:  # type: ignore[type-arg]
            """Clear all inputs and outputs.

            Returns:
                Dictionary of cleared component values.
            """
            return {
                file_upload: None,
                status_output: "",
                output_files: None,
                download_button: gr.update(visible=False, value=None),
            }

        # Conditional visibility for VAD threshold
        vad.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[vad],
            outputs=[vad_threshold],
        )

        # Conditional visibility for stream chunk size
        stream_mode.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[stream_mode],
            outputs=[stream_chunk_sec],
        )

        # Auto-enable word timestamps for subtitle formats
        output_format.change(
            fn=lambda fmt: (
                gr.update(value=(fmt in ("srt", "vtt"))),
                gr.update(interactive=(fmt in ("srt", "vtt"))),
            ),
            inputs=[output_format],
            outputs=[word_timestamps, merge_strategy],
        )

        word_timestamps.change(
            fn=lambda enabled: gr.update(interactive=bool(enabled)),
            inputs=[word_timestamps],
            outputs=[merge_strategy],
        )

        # Preset dropdown handlers
        preset_dropdown.change(
            fn=apply_preset,
            inputs=[preset_dropdown],
            outputs=[
                model_selector,
                batch_size,
                chunk_len_sec,
                overlap_duration,
                stream_mode,
                stream_chunk_sec,
                word_timestamps,
                merge_strategy,
                highlight_words,
                stabilize,
                vad,
                demucs,
                vad_threshold,
                overwrite_files,
                precision,
                output_format,
            ],
        )

        transcribe_btn.click(
            fn=transcribe_files,
            inputs=[
                file_upload,
                model_selector,
                batch_size,
                chunk_len_sec,
                overlap_duration,
                stream_mode,
                stream_chunk_sec,
                word_timestamps,
                merge_strategy,
                highlight_words,
                stabilize,
                vad,
                demucs,
                vad_threshold,
                overwrite_files,
                precision,
                output_format,
            ],
            outputs=[status_output, output_files, download_button],
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[file_upload, status_output, output_files, download_button],
        )

        # Benchmark metrics refresh
        refresh_metrics_btn.click(
            fn=refresh_benchmarks,
            inputs=[],
            outputs=[
                benchmark_status,
                runtime_display,
                gpu_display,
                benchmark_json_display,
            ],
        )

    return app


def launch_app(
    *,
    server_name: str = GRADIO_SERVER_NAME,
    server_port: int = GRADIO_SERVER_PORT,
    share: bool = False,
    debug: bool = False,
    **kwargs: object,
) -> None:
    """Launch the Gradio WebUI application.

    Builds and starts the web server with specified configuration.

    Args:
        server_name: Server hostname or IP address.
        server_port: Server port number.
        share: Request a public Gradio share link. Mounted mode does not support
            share links and logs a warning instead.
        debug: Enable debug mode with verbose logging.
        **kwargs: Additional Uvicorn arguments. ``host``, ``port``, and
            ``log_level`` are reserved because the explicit launch arguments
            control them.

    Raises:
        ValueError: If ``kwargs`` contains a reserved Uvicorn argument.

    Examples:
        >>> # Launch on localhost
        >>> launch_app()

        >>> # Share links are not supported in mounted mode (logs a warning)
        >>> launch_app(share=True)

        >>> # Custom port and debug mode
        >>> launch_app(server_port=8080, debug=True)
    """
    # Configure centralized logging
    configure_logging(level="DEBUG" if debug else "INFO")

    logger.info("Building Gradio WebUI application")
    print(f"🚀 Launching Parakeet-NEMO WebUI on http://{server_name}:{server_port}")
    if debug:
        print("📊 Debug mode enabled - check console for detailed logs")

    if share:
        logger.warning("Gradio share links are not supported in mounted WebUI mode.")

    reserved_kwargs = {"host", "port", "log_level"}.intersection(kwargs)
    if reserved_kwargs:
        names = ", ".join(sorted(reserved_kwargs))
        raise ValueError(f"launch_app() does not accept reserved Uvicorn kwargs: {names}")

    # Use the same FastAPI composition as the production API, but mount the UI
    # at root. Parakeet icons use ``/parakeet-assets`` so Gradio retains its
    # reserved ``/assets`` frontend bundle route.
    import uvicorn

    from parakeet_rocm.api import create_app

    logger.info(f"Starting server on {server_name}:{server_port}")
    uvicorn.run(
        create_app(ui_path=""),
        host=server_name,
        port=server_port,
        log_level="debug" if debug else "info",
        **kwargs,
    )
