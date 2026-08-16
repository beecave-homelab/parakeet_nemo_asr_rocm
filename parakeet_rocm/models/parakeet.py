"""Model accessors for NVIDIA Parakeet (NeMo) with lazy cache and idle unload.

This module exposes helpers to lazily load the Parakeet-TDT model with an LRU
cache and to offload the model from GPU VRAM when the application is idle.

Functions ensure the model is always placed on the best available device when
requested, and allow moving the cached instance back to CPU to free VRAM.

``unload_model_to_cpu`` is a no-load offload operation: it uses
``_peek_cached_model`` to check a parallel key-tracking set
(``_cached_keys``) before calling ``_get_cached_model``, so it never
triggers a model load or device promotion on cache miss.  A module-level
lock (``_cache_lock``) serialises the full offload (model move + VRAM
release) and cache-clear operations to prevent race conditions between
idle-offload threads and ``clear_model_cache``.
"""

from __future__ import annotations

import gc
import threading
from functools import lru_cache

import nemo.collections.asr as nemo_asr
import torch
from nemo.collections.asr.models import ASRModel

from parakeet_rocm.utils.constant import PARAKEET_MODEL_NAME
from parakeet_rocm.utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "get_model",
    "unload_model_to_cpu",
    "clear_model_cache",
]

# Non-reentrant by design: must not be held when calling into code
# that may reacquire.
_cache_lock = threading.Lock()
_cached_keys: set[str] = set()


def _best_device() -> str:
    """Return the preferred device string.

    Returns:
        str: ``"cuda"`` when CUDA/ROCm is available, else ``"cpu"``.

    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_device(
    model: ASRModel,
    *,
    device: str | None = None,
) -> None:
    """Ensure a model is placed on the target or best-available device.

    If ``device`` is ``None``, selects the preferred device (GPU if
    available, otherwise CPU). If the model's current device cannot be
    determined, it is assumed to be ``"cpu"``. If the model is already on
    the target device, no action is taken.

    Parameters:
        model (nemo_asr.models.ASRModel): NeMo ASR model instance to move.
        device (str | None): Target device (``"cuda"`` or ``"cpu"``). If
            ``None``, the preferred device is used.
    """
    target = device or _best_device()
    try:
        current = next(model.parameters()).device.type  # type: ignore[attr-defined]
    except Exception:
        current = "cpu"
    if current != target:
        model.to(target)


def _load_model(model_name: str) -> ASRModel:
    """Load and initialize a Parakeet ASR model by its identifier.

    The returned model is set to evaluation mode and placed on the best
    available device (GPU if available, otherwise CPU).

    Parameters:
        model_name (str): Identifier of the pretrained Parakeet model to
            load.

    Returns:
        ASRModel: Initialised ASR model instance prepared for inference.
    """
    model = nemo_asr.models.ASRModel.from_pretrained(model_name).eval()
    _ensure_device(model)
    return model


@lru_cache(maxsize=4)
def _get_cached_model(model_name: str = PARAKEET_MODEL_NAME) -> ASRModel:
    """Retrieve a cached Parakeet ASR model instance.

    Parameters:
        model_name (str): Model name or local path identifying the pretrained Parakeet model.

    Returns:
        ASRModel: Cached ASR model instance. This function does not modify
            the model's device placement.
    """
    return _load_model(model_name)


def _peek_cached_model(model_name: str = PARAKEET_MODEL_NAME) -> ASRModel | None:
    """Return the cached model for *model_name* without triggering a load.

    Checks the parallel ``_cached_keys`` set first.  When the key is
    present, calling ``_get_cached_model`` is guaranteed to be a cache
    hit (no load) because ``_cache_lock`` prevents concurrent
    ``clear_model_cache`` between the set check and the LRU lookup.

    Returns ``None`` when the key is not tracked, avoiding the
    cache-miss load that ``_get_cached_model`` would trigger.

    Parameters:
        model_name (str): Model name to look up in the cache.

    Returns:
        ASRModel | None: The cached model, or ``None`` if not present.
    """
    if model_name not in _cached_keys:
        return None
    try:
        return _get_cached_model(model_name)
    except Exception:
        return None


def get_model(model_name: str = PARAKEET_MODEL_NAME) -> ASRModel:
    """Retrieve the cached Parakeet ASR model and ensure it is on the best available device.

    Parameters:
        model_name (str): Model identifier or path to load if not already cached.

    Returns:
        model (ASRModel): The cached ASRModel instance placed on the appropriate device.

    Note:
        This function holds ``_cache_lock`` only for the LRU insertion +
        ``_cached_keys`` update, then releases it before calling
        ``_ensure_device``.  A concurrent ``unload_model_to_cpu`` call
        may move the model to CPU after the lock is released while this
        function re-promotes it to GPU.  This is acceptable by design:
        idle-offload only runs when no requests are in flight, so the
        race is self-correcting (next idle cycle offloads again).
        Holding the lock through ``model.to("cuda")`` would serialise
        every transcription request behind a slow GPU operation, adding
        unacceptable latency on the hot path.
    """
    with _cache_lock:
        model = _get_cached_model(model_name)
        _cached_keys.add(model_name)
    _ensure_device(model)
    return model


def unload_model_to_cpu(model_name: str = PARAKEET_MODEL_NAME) -> None:
    """Move the cached model to CPU to free GPU VRAM.

    This is a no-op when no model is cached or the model is already on CPU.
    The function never triggers a model load or device promotion on cache
    miss.

    Parameters:
        model_name (str): Name or key of the cached Parakeet model to unload
            from GPU.
    """
    with _cache_lock:
        try:
            model = _peek_cached_model(model_name)
        except Exception:
            logger.debug("failed to peek cached model %s", model_name, exc_info=True)
            return
        if model is None:
            return
        try:
            current = next(model.parameters()).device.type  # type: ignore[attr-defined]
        except Exception:
            logger.debug(
                "failed to read model parameters for %s, assuming CPU",
                model_name,
                exc_info=True,
            )
            current = "cpu"
        if current != "cpu":
            model.to("cpu")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.debug("torch.cuda.empty_cache() failed", exc_info=True)


def clear_model_cache(release_allocator: bool = True) -> None:
    """Clear the internal LRU cache of loaded model instances.

    After this call, cached models are discarded and will be recreated when
    next requested.

    Parameters:
        release_allocator (bool): When ``True`` (default), also run a garbage
            collection pass and release PyTorch's CUDA/ROCm caching-allocator
            memory after the cached references are dropped. This reclaims
            allocator blocks that outlive the discarded model objects, so
            VRAM is actually returned instead of remaining parked in the
            allocator's free pool.
    """
    with _cache_lock:
        try:
            _get_cached_model.cache_clear()  # type: ignore[attr-defined]
            _cached_keys.clear()
        except Exception:
            logger.debug("cache_clear() failed", exc_info=True)
            return
    if release_allocator:
        _release_gpu_allocator_memory()


def _release_gpu_allocator_memory() -> None:
    """Release PyTorch CUDA/ROCm allocator memory held by freed blocks.

    Best-effort: every step is guarded, so a failure never propagates to the
    idle-cleanup caller. Garbage collection runs first so tensor objects
    whose only reference was the cleared LRU cache become collectable before
    the allocator release; ``torch.cuda.empty_cache()`` then returns the
    freed blocks to the driver.
    """
    try:
        gc.collect()
    except Exception:
        logger.debug("gc.collect() failed", exc_info=True)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("torch.cuda.empty_cache() failed", exc_info=True)
