"""Unit tests for parakeet_rocm.models.parakeet module."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from parakeet_rocm.models.parakeet import (
    _best_device,
    _cache_lock,
    _ensure_device,
    _get_cached_model,
    _load_model,
    clear_model_cache,
    get_model,
    unload_model_to_cpu,
)


def _make_mock_model(device_type: str = "cuda") -> MagicMock:
    """Create a mock model with a configurable device type on its parameters.

    Returns:
        MagicMock: Mock model whose ``parameters()`` iterator yields a
            parameter with ``device.type`` set to *device_type*.
    """
    mock_model = MagicMock()
    mock_param = MagicMock()
    mock_param.device.type = device_type
    mock_model.parameters.return_value = iter([mock_param])
    return mock_model


def test_best_device() -> None:
    """Returns cuda when available, cpu otherwise."""
    with patch("torch.cuda.is_available", return_value=True):
        assert _best_device() == "cuda"

    with patch("torch.cuda.is_available", return_value=False):
        assert _best_device() == "cpu"


def test_ensure_device_with_exception_and_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests device detection exception handling and device move."""
    mock_model = MagicMock()
    mock_model.parameters.side_effect = RuntimeError("Cannot access parameters")
    monkeypatch.setattr("parakeet_rocm.models.parakeet._best_device", lambda: "cuda")
    _ensure_device(mock_model)
    mock_model.to.assert_called_with("cuda")


def test_ensure_device_already_on_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests no action when model is already on target device."""
    mock_model = _make_mock_model("cpu")
    monkeypatch.setattr("parakeet_rocm.models.parakeet._best_device", lambda: "cpu")
    _ensure_device(mock_model)
    mock_model.to.assert_not_called()


def _patch_peek_cached_model(
    return_value: object = None,
    **kwargs: object,
) -> patch:
    """Return a patch that replaces ``_peek_cached_model`` with a mock.

    Parameters:
        return_value (object): Value for the mock's ``return_value``.
            Defaults to ``None`` (cache miss / empty).
        **kwargs: Forwarded to the ``MagicMock`` constructor (e.g.
            ``side_effect``).

    Returns:
        patch: A ``patch`` context manager for
            ``parakeet_rocm.models.parakeet._peek_cached_model``.
    """
    mock_fn = MagicMock(return_value=return_value, **kwargs)
    return patch("parakeet_rocm.models.parakeet._peek_cached_model", mock_fn)


def test_unload_model_to_cpu_exceptions() -> None:
    """Tests exception handling in unload_model_to_cpu."""
    # Test _peek_cached_model exception (simulates concurrent cache clear)
    with _patch_peek_cached_model(side_effect=RuntimeError("Cache miss")):
        unload_model_to_cpu()

    # Test torch.cuda.empty_cache exception
    mock_model = _make_mock_model("cuda")
    with (
        _patch_peek_cached_model(return_value=mock_model),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache", side_effect=RuntimeError("Cache clear failed")),
    ):
        unload_model_to_cpu()
        mock_model.to.assert_called_with("cpu")


def test_unload_model_to_cpu_no_gpu() -> None:
    """Tests unload when GPU is not available."""
    mock_model = _make_mock_model("cuda")
    with (
        _patch_peek_cached_model(return_value=mock_model),
        patch("torch.cuda.is_available", return_value=False),
    ):
        unload_model_to_cpu()
        mock_model.to.assert_called_with("cpu")


def test_clear_model_cache__returns_false_when_eviction_fails() -> None:
    """Cache eviction failure is reported as False, not swallowed silently."""
    mock_fn = MagicMock()
    mock_fn.cache_clear.side_effect = RuntimeError("fail")
    with patch("parakeet_rocm.models.parakeet._get_cached_model", mock_fn):
        assert clear_model_cache() is False


def test_clear_model_cache__returns_true_on_success() -> None:
    """Successful eviction (with best-effort allocator release) returns True."""
    mock_fn = MagicMock()
    with (
        patch("parakeet_rocm.models.parakeet._get_cached_model", mock_fn),
        patch("parakeet_rocm.models.parakeet._release_gpu_allocator_memory"),
    ):
        assert clear_model_cache() is True
    mock_fn.cache_clear.assert_called_once()


@patch("parakeet_rocm.models.parakeet._load_model")
def test_get_model_caching(mock_load: MagicMock) -> None:
    """Tests that get_model uses cached model and ensures device placement."""
    clear_model_cache()
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    result1 = get_model("test_model")
    assert result1 is mock_model
    mock_load.assert_called_once_with("test_model")

    mock_load.reset_mock()

    result2 = get_model("test_model")
    assert result2 is mock_model
    mock_load.assert_not_called()


@patch("parakeet_rocm.models.parakeet._load_model")
def test_get_model_device_ensure(mock_load: MagicMock) -> None:
    """Tests that get_model calls _ensure_device on cached model."""
    clear_model_cache()
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    with patch("parakeet_rocm.models.parakeet._ensure_device") as mock_ensure:
        get_model("test_model")
        mock_ensure.assert_called_once()


# --- Tests for AC1-AC10 ---


def test_unload_no_op_when_cache_empty() -> None:
    """AC2: _peek_cached_model returns None → function returns immediately."""
    with _patch_peek_cached_model(return_value=None) as mock_peek:
        unload_model_to_cpu()
        mock_peek.assert_called_once()


def test_unload_uses_cached_model_directly() -> None:
    """AC1: _peek_cached_model is called, get_model / _ensure_device are NOT called."""
    mock_model = _make_mock_model("cuda")
    with (
        _patch_peek_cached_model(return_value=mock_model),
        patch("parakeet_rocm.models.parakeet.get_model") as mock_get,
        patch("parakeet_rocm.models.parakeet._ensure_device") as mock_ensure,
        patch("torch.cuda.is_available", return_value=False),
    ):
        unload_model_to_cpu()
        mock_get.assert_not_called()
        mock_ensure.assert_not_called()


def test_unload_skips_move_when_already_cpu() -> None:
    """AC3: Model on CPU → model.to is NOT called."""
    mock_model = _make_mock_model("cpu")
    with (
        _patch_peek_cached_model(return_value=mock_model),
        patch("torch.cuda.is_available", return_value=False),
    ):
        unload_model_to_cpu()
        mock_model.to.assert_not_called()


def test_unload_moves_when_on_gpu() -> None:
    """AC3: Model on GPU → model.to('cpu') called exactly once."""
    mock_model = _make_mock_model("cuda")
    with (
        _patch_peek_cached_model(return_value=mock_model),
        patch("torch.cuda.is_available", return_value=False),
    ):
        unload_model_to_cpu()
        mock_model.to.assert_called_once_with("cpu")


def test_unload_empty_cache_exception_swallowed() -> None:
    """AC6: torch.cuda.empty_cache() raises → no propagation."""
    mock_model = _make_mock_model("cuda")
    with (
        _patch_peek_cached_model(return_value=mock_model),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache", side_effect=RuntimeError("CUDA error")),
    ):
        unload_model_to_cpu()
        mock_model.to.assert_called_once_with("cpu")


def test_unload_no_load_on_concurrent_clear() -> None:
    """AC4: concurrent clear_model_cache → no _load_model triggered."""
    with (
        _patch_peek_cached_model(side_effect=RuntimeError("Cache cleared")),
        patch("parakeet_rocm.models.parakeet._load_model") as mock_load,
    ):
        unload_model_to_cpu()
        mock_load.assert_not_called()


def test_clear_cache_acquires_lock() -> None:
    """AC5: clear_model_cache holds _cache_lock during cache_clear."""
    lock_held_during_clear = False
    mock_fn = MagicMock()

    def check_lock_held() -> None:
        nonlocal lock_held_during_clear
        # threading.Lock is not reentrant: non-blocking acquire
        # fails if the current thread already holds it.
        # We must NOT release on success — that would break the
        # enclosing ``with _cache_lock:`` in clear_model_cache.
        acquired = _cache_lock.acquire(blocking=False)
        lock_held_during_clear = not acquired
        if acquired:
            _cache_lock.release()

    mock_fn.cache_clear.side_effect = check_lock_held
    with patch("parakeet_rocm.models.parakeet._get_cached_model", mock_fn):
        clear_model_cache()

    assert lock_held_during_clear, "_cache_lock was not held during cache_clear"


def test_get_model_unchanged() -> None:
    """AC7: get_model still calls _ensure_device and loads on miss."""
    clear_model_cache()
    mock_model = MagicMock()
    with (
        patch("parakeet_rocm.models.parakeet._load_model", return_value=mock_model),
        patch("parakeet_rocm.models.parakeet._ensure_device") as mock_ensure,
    ):
        get_model("test_model")
        mock_ensure.assert_called_once()


def test_existing_callers_unchanged() -> None:
    """AC8, AC9: public API signatures unchanged, @lru_cache preserved."""
    from parakeet_rocm.models.parakeet import (  # noqa: F401
        clear_model_cache,
        get_model,
        unload_model_to_cpu,
    )

    assert hasattr(_get_cached_model, "cache_info")
    assert hasattr(_get_cached_model, "cache_clear")
    cache_params = _get_cached_model.cache_parameters()  # type: ignore[attr-defined]
    assert cache_params["maxsize"] == 4


def test_no_private_import_in_callers() -> None:
    """AC10: No _get_cached_model imports outside parakeet.py and diagnostic sites."""
    import parakeet_rocm.api.app
    import parakeet_rocm.utils.watch
    import parakeet_rocm.webui.app as webui_app

    for mod in (parakeet_rocm.api.app, parakeet_rocm.utils.watch, webui_app):
        src = pathlib.Path(mod.__file__).read_text()  # type: ignore[arg-type]
        assert "_get_cached_model" not in src, f"{mod.__name__} imports private _get_cached_model"


@patch("nemo.collections.asr.models.ASRModel.from_pretrained")
def test_load_model(mock_from_pretrained: MagicMock) -> None:
    """Tests _load_model initialization and device placement."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_from_pretrained.return_value = mock_model

    with patch("parakeet_rocm.models.parakeet._ensure_device") as mock_ensure:
        result = _load_model("test_model")
        assert result is mock_model
        mock_from_pretrained.assert_called_once_with("test_model")
        mock_model.eval.assert_called_once()
        mock_ensure.assert_called_once_with(mock_model)


@patch("parakeet_rocm.models.parakeet._load_model")
def test_peek_cached_model_real_path(mock_load: MagicMock) -> None:
    """SF-2: Exercise the real _peek_cached_model code path (not mocked).

    Populates the LRU cache via get_model, then verifies that
    _peek_cached_model returns the cached model without triggering a
    second load, and returns None for an uncached key.
    """
    from parakeet_rocm.models.parakeet import _cached_keys, _peek_cached_model

    clear_model_cache()
    mock_model = _make_mock_model("cuda")
    mock_load.return_value = mock_model

    # Populate cache via get_model (exercises _get_cached_model → _cached_keys.add)
    model = get_model("test_model_peek")
    assert model is mock_model
    assert "test_model_peek" in _cached_keys

    # _peek_cached_model should return the same model without another load
    mock_load.reset_mock()
    peeked = _peek_cached_model("test_model_peek")
    assert peeked is mock_model
    mock_load.assert_not_called()

    # _peek_cached_model should return None for an uncached key
    assert _peek_cached_model("uncached_model") is None

    # After clear_model_cache, _peek_cached_model should return None
    clear_model_cache()
    assert _peek_cached_model("test_model_peek") is None
    assert "test_model_peek" not in _cached_keys


def test_clear_model_cache__releases_gpu_allocator() -> None:
    """Clearing the cache must also release the CUDA/ROCm allocator.

    Regression test: ``clear_model_cache()`` used to evict the LRU
    references without a GC pass or ``torch.cuda.empty_cache()``, so VRAM
    stayed parked in PyTorch's caching allocator even after the "cache
    clear".
    """
    mock_fn = MagicMock()
    with (
        patch("parakeet_rocm.models.parakeet._get_cached_model", mock_fn),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache") as mock_empty,
        patch("gc.collect") as mock_gc,
    ):
        clear_model_cache()
    mock_fn.cache_clear.assert_called_once()
    mock_gc.assert_called_once()
    mock_empty.assert_called_once()


def test_clear_model_cache__gc_only_when_no_gpu() -> None:
    """Without CUDA available the release degrades to a GC-only pass."""
    mock_fn = MagicMock()
    with (
        patch("parakeet_rocm.models.parakeet._get_cached_model", mock_fn),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.empty_cache") as mock_empty,
        patch("gc.collect") as mock_gc,
    ):
        clear_model_cache()
    mock_gc.assert_called_once()
    mock_empty.assert_not_called()


def test_clear_model_cache__allocator_errors_swallowed() -> None:
    """gc/empty_cache failures never propagate to the idle-cleanup caller."""
    mock_fn = MagicMock()
    with (
        patch("parakeet_rocm.models.parakeet._get_cached_model", mock_fn),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache", side_effect=RuntimeError("cuda gone")),
        patch("gc.collect", side_effect=RuntimeError("gc gone")),
    ):
        clear_model_cache()  # must not raise
    mock_fn.cache_clear.assert_called_once()


def test_clear_model_cache__skips_release_when_cache_clear_fails() -> None:
    """When the LRU eviction itself fails, no allocator release is attempted."""
    mock_fn = MagicMock()
    mock_fn.cache_clear.side_effect = RuntimeError("fail")
    with (
        patch("parakeet_rocm.models.parakeet._get_cached_model", mock_fn),
        patch("parakeet_rocm.models.parakeet._release_gpu_allocator_memory") as mock_release,
    ):
        result = clear_model_cache()
    assert result is False
    mock_release.assert_not_called()


def test_clear_model_cache__release_allocator_flag() -> None:
    """release_allocator=False keeps the pure LRU-eviction behavior."""
    mock_fn = MagicMock()
    with (
        patch("parakeet_rocm.models.parakeet._get_cached_model", mock_fn),
        patch("parakeet_rocm.models.parakeet._release_gpu_allocator_memory") as mock_release,
    ):
        clear_model_cache(release_allocator=False)
    mock_fn.cache_clear.assert_called_once()
    mock_release.assert_not_called()
