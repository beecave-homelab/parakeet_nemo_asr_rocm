"""OpenAI-compatible REST routes for audio transcription."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path
from time import monotonic, perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

import torch
from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import ValidationError

from parakeet_rocm.api.auth import require_api_bearer_token
from parakeet_rocm.api.mapping import (
    convert_aligned_result_to_verbose,
    get_audio_duration,
    infer_language_for_model,
    map_model_name,
    map_response_format,
)
from parakeet_rocm.api.schemas import (
    ErrorObject,
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponseJson,
    TranscriptionResponseVerbose,
)
from parakeet_rocm.config import OutputConfig, StabilizationConfig, TranscriptionConfig
from parakeet_rocm.formatting import UnsupportedFormatError
from parakeet_rocm.models.parakeet import clear_model_cache, get_model, unload_model_to_cpu
from parakeet_rocm.timestamps.models import AlignedResult
from parakeet_rocm.transcription import cli_transcribe
from parakeet_rocm.utils.constant import API_DEFAULT_BATCH_SIZE, API_DEFAULT_CHUNK_LEN_SEC
from parakeet_rocm.utils.logging_config import get_logger
from parakeet_rocm.webui.validation.file_validator import FileValidationError, validate_audio_file

if TYPE_CHECKING:
    from nemo.collections.asr.models import ASRModel

logger = get_logger(__name__)

router = APIRouter()

_activity_lock = threading.RLock()
_last_api_activity_monotonic = monotonic()
_active_api_requests = 0
_api_model_lock = threading.Lock()
_active_api_model_name: str | None = None
_api_model_generation = 0


def mark_api_activity() -> None:
    """Record the current monotonic timestamp for API activity tracking."""
    global _last_api_activity_monotonic
    with _activity_lock:
        _last_api_activity_monotonic = monotonic()


def get_last_api_activity_monotonic() -> float:
    """Return the most recent API activity timestamp.

    Returns:
        Most recent monotonic timestamp representing API activity.
    """
    with _activity_lock:
        return _last_api_activity_monotonic


def start_api_request() -> None:
    """Mark a request as active and update last activity time."""
    global _active_api_requests
    with _activity_lock:
        _active_api_requests += 1
        mark_api_activity()


def finish_api_request() -> None:
    """Mark request completion for API activity tracking."""
    global _active_api_requests
    with _activity_lock:
        _active_api_requests = max(0, _active_api_requests - 1)
        mark_api_activity()


def has_active_api_requests() -> bool:
    """Return whether one or more API requests are currently in-flight.

    Returns:
        ``True`` when at least one API request is active, else ``False``.
    """
    with _activity_lock:
        return _active_api_requests > 0


def get_api_model(model_name: str) -> ASRModel:
    """Return an API model after offloading a previously active model.

    Only the active-model bookkeeping and the previous-model offload are
    performed under ``_api_model_lock``.  The expensive ``get_model()``
    call (which may trigger GPU device promotion) runs outside the lock
    so concurrent requests and the idle-offload worker are not blocked
    behind slow GPU operations.

    Args:
        model_name: Identifier of the model selected by the API request.

    Returns:
        Requested model placed on the preferred device.
    """
    global _active_api_model_name, _api_model_generation
    with _api_model_lock:
        previous_model_name = _active_api_model_name
        if previous_model_name is not None and previous_model_name != model_name:
            logger.info(
                "API model changed: offloading model=%s before loading model=%s",
                previous_model_name,
                model_name,
            )
            unload_model_to_cpu(previous_model_name)
        _active_api_model_name = model_name
        generation = _api_model_generation

    try:
        return get_model(model_name)
    except Exception:
        with _api_model_lock:
            if _api_model_generation == generation:
                _active_api_model_name = previous_model_name
        raise


def unload_active_api_model() -> None:
    """Offload the active API model without loading a cache miss.

    Acquires ``_api_model_lock`` and, when a model is currently active,
    moves it to CPU via ``unload_model_to_cpu``.  The active-model name is
    left unchanged so a subsequent ``get_api_model`` call can reuse it.
    """
    with _api_model_lock:
        if _active_api_model_name is not None:
            unload_model_to_cpu(_active_api_model_name)


def clear_api_model_cache() -> None:
    """Clear API model state and all cached model instances.

    Acquires ``_api_model_lock``, calls ``clear_model_cache`` to drop every
    cached ``ASRModel`` instance, resets ``_active_api_model_name`` to
    ``None``, and bumps ``_api_model_generation`` so in-flight
    ``get_api_model`` rollbacks do not clobber the cleared state.
    """
    global _active_api_model_name, _api_model_generation
    with _api_model_lock:
        clear_model_cache()
        _active_api_model_name = None
        _api_model_generation += 1


def _safe_cleanup(path: Path) -> None:
    """Delete temporary file or directory if it exists.

    Args:
        path: Path to remove.
    """
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Temporary path cleanup failed", exc_info=True)


def _build_error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
    code: str,
) -> JSONResponse:
    """Create an OpenAI-style error response.

    Args:
        status_code: HTTP status code.
        message: Error message for clients.
        error_type: OpenAI-compatible error type.
        code: Short machine-readable error code.

    Returns:
        JSON response containing an OpenAI-style ``error`` object.
    """
    payload = ErrorResponse(
        error=ErrorObject(
            message=message,
            type=error_type,
            code=code,
        )
    ).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


def _nemo_error_status(message: str) -> tuple[int, str, str]:
    """Map NeMo model exceptions to OpenAI-compatible status and error metadata.

    Args:
        message: Exception message.

    Returns:
        Tuple of ``(status_code, error_type, error_code)``.
    """
    lowered = message.lower()
    if "model" in lowered and (
        "unknown" in lowered
        or "not found" in lowered
        or "invalid" in lowered
        or "unsupported" in lowered
    ):
        return 400, "invalid_request_error", "invalid_model"
    return 503, "server_error", "model_unavailable"


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: list[str] | None = Form(default=None),
) -> Response:
    """Handle OpenAI-compatible transcription requests.

    Args:
        request: Incoming HTTP request metadata.
        background_tasks: FastAPI background task manager.
        file: Uploaded input file.
        model: Model identifier.
        language: Optional ISO-639-1 language code (accepted, ignored).
        prompt: Optional prompt (accepted, ignored).
        temperature: Optional temperature (accepted, ignored).
        response_format: Response format selector.
        timestamp_granularities: Optional verbose timestamp granularity list.

    Returns:
        OpenAI-compatible transcription output in requested format.
    """
    del language, prompt, temperature

    request_id = uuid4().hex[:8]
    started_at = perf_counter()
    temp_audio_path: Path | None = None
    temp_output_dir: Path | None = None
    output_file: Path | None = None
    success = False
    validation_elapsed_ms = 0.0
    model_acquire_elapsed_ms = 0.0
    transcribe_elapsed_ms = 0.0
    model_cache_hit: bool | None = None
    client_host = request.client.host if request.client is not None else "unknown"
    client_port = request.client.port if request.client is not None else "unknown"
    client_origin = f"{client_host}:{client_port}"
    request_started = False

    logger.debug(
        "API transcription request received: id=%s origin=%s method=%s path=%s "
        "file=%s model=%s format=%s granularities=%s",
        request_id,
        client_origin,
        request.method,
        request.url.path,
        file.filename,
        model,
        response_format,
        timestamp_granularities,
    )

    try:
        auth_error = require_api_bearer_token(request)
        if auth_error is not None:
            return auth_error

        start_api_request()
        request_started = True

        transcription_request = TranscriptionRequest(
            file=file,
            model=model,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
        )

        logger.info(
            "API transcription received: id=%s model=%s format=%s",
            request_id,
            transcription_request.model,
            transcription_request.response_format,
        )

        model_name = map_model_name(transcription_request.model)
        if model_name is None:
            logger.debug(
                "API request rejected: id=%s reason=invalid_model input=%s",
                request_id,
                model,
            )
            return _build_error_response(
                status_code=400,
                message="Unrecognized model name.",
                error_type="invalid_request_error",
                code="invalid_model",
            )

        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename or "upload.wav").suffix or ".wav",
            delete=False,
        ) as tmp_audio:
            content = await file.read()
            tmp_audio.write(content)
            temp_audio_path = Path(tmp_audio.name)

        file_size_bytes = temp_audio_path.stat().st_size

        validated_audio = validate_audio_file(temp_audio_path)

        try:
            mapped_format = map_response_format(transcription_request.response_format)
        except ValueError as exc:
            logger.debug(
                "API request rejected: id=%s reason=unsupported_format input=%s",
                request_id,
                transcription_request.response_format,
            )
            return _build_error_response(
                status_code=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="unsupported_format",
            )
        effective_output_format = "txt" if mapped_format == "text_only" else mapped_format
        temp_output_dir = Path(tempfile.mkdtemp(prefix="parakeet-api-"))

        granularities = set(transcription_request.timestamp_granularities or [])
        word_timestamps = (
            transcription_request.response_format == "verbose_json"
            or "word" in granularities
            or effective_output_format in {"srt", "vtt"}
        )
        logger.debug(
            "API request prepared: id=%s origin=%s mapped_model=%s output_format=%s "
            "word_timestamps=%s file_size_bytes=%d",
            request_id,
            client_origin,
            model_name,
            effective_output_format,
            word_timestamps,
            file_size_bytes,
        )

        validation_elapsed_ms = (perf_counter() - started_at) * 1000

        model_acquire_started_at = perf_counter()
        cache_hits_before: int | None = None
        cache_hits_after: int | None = None
        try:
            from parakeet_rocm.models.parakeet import _get_cached_model  # type: ignore

            cache_hits_before = _get_cached_model.cache_info().hits  # type: ignore[attr-defined]
        except Exception:
            cache_hits_before = None

        get_api_model(model_name)

        try:
            from parakeet_rocm.models.parakeet import _get_cached_model  # type: ignore

            cache_hits_after = _get_cached_model.cache_info().hits  # type: ignore[attr-defined]
        except Exception:
            cache_hits_after = None

        if cache_hits_before is not None and cache_hits_after is not None:
            model_cache_hit = cache_hits_after > cache_hits_before
        model_acquire_elapsed_ms = (perf_counter() - model_acquire_started_at) * 1000

        transcription_config = TranscriptionConfig(
            batch_size=API_DEFAULT_BATCH_SIZE,
            chunk_len_sec=API_DEFAULT_CHUNK_LEN_SEC,
            word_timestamps=word_timestamps,
            merge_strategy="lcs",
        )
        stabilization_config = StabilizationConfig(
            enabled=False,
            demucs=False,
            vad=False,
            vad_threshold=0.35,
        )
        output_config = OutputConfig(
            output_dir=temp_output_dir,
            output_format=effective_output_format,
            output_template="{filename}",
            overwrite=True,
            highlight_words=False,
            allow_unsafe_filenames=False,
        )
        logger.debug(
            "API transcription settings: id=%s origin=%s batch_size=%d chunk_len_sec=%d "
            "overlap_duration=%d merge_strategy=%s stabilize=%s demucs=%s vad=%s "
            "vad_threshold=%.2f overwrite=%s highlight_words=%s allow_unsafe_filenames=%s",
            request_id,
            client_origin,
            transcription_config.batch_size,
            transcription_config.chunk_len_sec,
            transcription_config.overlap_duration,
            transcription_config.merge_strategy,
            stabilization_config.enabled,
            stabilization_config.demucs,
            stabilization_config.vad,
            stabilization_config.vad_threshold,
            output_config.overwrite,
            output_config.highlight_words,
            output_config.allow_unsafe_filenames,
        )

        logger.info(
            "API transcription running: id=%s model=%s format=%s file_size_bytes=%d",
            request_id,
            model_name,
            transcription_request.response_format,
            file_size_bytes,
        )

        transcribe_started_at = perf_counter()
        created_files = cli_transcribe(
            audio_files=[validated_audio],
            model_name=model_name,
            output_dir=output_config.output_dir,
            output_format=output_config.output_format,
            output_template=output_config.output_template,
            batch_size=transcription_config.batch_size,
            chunk_len_sec=transcription_config.chunk_len_sec,
            overlap_duration=transcription_config.overlap_duration,
            word_timestamps=transcription_config.word_timestamps,
            merge_strategy=transcription_config.merge_strategy,
            stabilize=stabilization_config.enabled,
            demucs=stabilization_config.demucs,
            vad=stabilization_config.vad,
            vad_threshold=stabilization_config.vad_threshold,
            overwrite=output_config.overwrite,
            no_progress=True,
            quiet=True,
            allow_unsafe_filenames=output_config.allow_unsafe_filenames,
        )
        transcribe_elapsed_ms = (perf_counter() - transcribe_started_at) * 1000

        if not created_files:
            return _build_error_response(
                status_code=500,
                message="No transcription output was generated.",
                error_type="server_error",
                code="runtime_error",
            )

        output_file = created_files[0]
        output_text = output_file.read_text(encoding="utf-8")
        logger.info(
            "API transcription completed: id=%s chars=%d transcribe_ms=%.1f",
            request_id,
            len(output_text),
            transcribe_elapsed_ms,
        )
        logger.debug(
            "API transcription completed: id=%s output_file=%s chars=%d",
            request_id,
            output_file,
            len(output_text),
        )

        if transcription_request.response_format == "text":
            success = True
            background_tasks.add_task(_safe_cleanup, temp_audio_path)
            background_tasks.add_task(_safe_cleanup, temp_output_dir)
            return PlainTextResponse(content=output_text, media_type="text/plain")

        if transcription_request.response_format == "json":
            success = True
            background_tasks.add_task(_safe_cleanup, temp_audio_path)
            background_tasks.add_task(_safe_cleanup, temp_output_dir)
            payload = TranscriptionResponseJson(text=output_text).model_dump()
            return JSONResponse(content=payload)

        if transcription_request.response_format in {"srt", "vtt"}:
            success = True
            background_tasks.add_task(_safe_cleanup, temp_audio_path)
            background_tasks.add_task(_safe_cleanup, temp_output_dir)
            return PlainTextResponse(content=output_text, media_type="text/plain")

        try:
            parsed_output = json.loads(output_text)
        except json.JSONDecodeError:
            output_snippet = output_text[:500]
            logger.exception(
                "Invalid verbose_json output generated: id=%s snippet=%r",
                request_id,
                output_snippet,
            )
            return _build_error_response(
                status_code=500,
                message="Server produced invalid JSON for verbose response.",
                error_type="server_error",
                code="invalid_json_output",
            )

        try:
            aligned_result = AlignedResult.model_validate(parsed_output)
            verbose_data = convert_aligned_result_to_verbose(
                aligned_result,
                transcription_request.timestamp_granularities,
            )
            payload = TranscriptionResponseVerbose(
                task="transcribe",
                language=infer_language_for_model(model_name),
                duration=get_audio_duration(validated_audio),
                text=verbose_data["text"],
                segments=verbose_data["segments"],
                words=verbose_data["words"],
            ).model_dump()
        except ValidationError:
            logger.exception(
                "Server produced invalid structured verbose JSON: id=%s snippet=%r",
                request_id,
                output_text[:500],
            )
            return _build_error_response(
                status_code=500,
                message="Server produced invalid structured verbose JSON.",
                error_type="server_error",
                code="invalid_json_output",
            )

        success = True
        background_tasks.add_task(_safe_cleanup, temp_audio_path)
        background_tasks.add_task(_safe_cleanup, temp_output_dir)
        return JSONResponse(content=payload)

    except ValidationError as exc:
        fields_with_errors = {
            str(part)
            for err in exc.errors()
            for part in err.get("loc", ())
            if isinstance(part, str)
        }
        if "model" in fields_with_errors:
            logger.debug("API request validation failed: id=%s field=model", request_id)
            return _build_error_response(
                status_code=400,
                message="Model must be 'whisper-1' or start with 'nvidia/'.",
                error_type="invalid_request_error",
                code="invalid_model",
            )
        if "response_format" in fields_with_errors:
            logger.debug("API request validation failed: id=%s field=response_format", request_id)
            return _build_error_response(
                status_code=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="unsupported_format",
            )
        logger.debug(
            "API request validation failed: id=%s fields=%s",
            request_id,
            sorted(fields_with_errors),
        )
        return _build_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="validation_error",
        )
    except FileValidationError as exc:
        logger.warning("API request rejected due to invalid file: id=%s err=%s", request_id, exc)
        return _build_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_file",
        )
    except UnsupportedFormatError as exc:
        logger.warning(
            "API request rejected due to unsupported format: id=%s err=%s",
            request_id,
            exc,
        )
        return _build_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="unsupported_format",
        )
    except ValueError as exc:
        logger.warning("API request rejected due to invalid input: id=%s err=%s", request_id, exc)
        return _build_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request",
        )
    except RuntimeError as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            logger.exception(
                "API transcription failed due to GPU memory exhaustion: id=%s",
                request_id,
            )
            clear_api_model_cache()
            return _build_error_response(
                status_code=503,
                message="GPU is out of memory. Please retry after freeing GPU memory.",
                error_type="server_error",
                code="gpu_oom",
            )

        logger.exception("API transcription runtime failure: id=%s", request_id)
        lowered_error = str(exc).lower()
        is_audio_format_error = (
            ("ffmpeg" in lowered_error and "format" in lowered_error)
            or "invalid audio format" in lowered_error
            or "unknown format" in lowered_error
            or "could not find codec" in lowered_error
        )
        if is_audio_format_error:
            return _build_error_response(
                status_code=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_audio_format",
            )
        return _build_error_response(
            status_code=500,
            message="Transcription failed due to an internal runtime error.",
            error_type="server_error",
            code="runtime_error",
        )
    except Exception as exc:
        exception_module = getattr(exc.__class__, "__module__", "") or ""
        if "nemo" in exception_module.lower():
            status_code, error_type, error_code = _nemo_error_status(str(exc))
            if status_code == 503 or error_code == "model_unavailable":
                logger.exception("NeMo model unavailable error: %s", exc)
                return _build_error_response(
                    status_code=status_code,
                    message="Model temporarily unavailable, please try again later.",
                    error_type=error_type,
                    code=error_code,
                )
            return _build_error_response(
                status_code=status_code,
                message=str(exc),
                error_type=error_type,
                code=error_code,
            )

        logger.exception("Unhandled exception while processing API transcription request")
        return _build_error_response(
            status_code=500,
            message="Internal server error.",
            error_type="server_error",
            code="internal_error",
        )
    finally:
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.debug(
            "API transcription request finished: id=%s success=%s elapsed_ms=%.1f",
            request_id,
            success,
            elapsed_ms,
        )
        logger.debug(
            "API transcription timing: id=%s validation_ms=%.1f model_acquire_ms=%.1f "
            "transcribe_ms=%.1f total_ms=%.1f model_cache_hit=%s",
            request_id,
            validation_elapsed_ms,
            model_acquire_elapsed_ms,
            transcribe_elapsed_ms,
            elapsed_ms,
            model_cache_hit,
        )
        if not success:
            if output_file is not None:
                _safe_cleanup(output_file)
            if temp_audio_path is not None:
                _safe_cleanup(temp_audio_path)
            if temp_output_dir is not None:
                _safe_cleanup(temp_output_dir)
        if request_started:
            finish_api_request()
