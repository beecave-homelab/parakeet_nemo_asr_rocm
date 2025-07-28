"""Wrapper to lazily load and cache the Parakeet-TDT 0.6B v2 model."""

from __future__ import annotations

from functools import lru_cache

import nemo.collections.asr as nemo_asr  # type: ignore
import torch

__all__ = ["get_model"]


DEFAULT_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


def _load_model(model_name: str) -> nemo_asr.models.ASRModel:
    """Loads and initializes the Parakeet ASR model.

    This function downloads the pre-trained model from NVIDIA's NGC, sets it
    to evaluation mode, and moves it to the appropriate device (GPU if
    available, otherwise CPU).

    Returns:
        The initialized ASRModel instance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.from_pretrained(model_name).eval().to(device)
    return model


@lru_cache(maxsize=4)
def get_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> nemo_asr.models.ASRModel:  # pragma: no cover
    """Lazily loads and returns a cached instance of the Parakeet ASR model.

    This function is decorated with `lru_cache` to ensure the model is loaded
    only once. On the first call, it invokes `_load_model` to get the model
    and then caches it. Subsequent calls return the cached instance.

    Returns:
        The cached ASRModel instance, moved to GPU if available.
    """
    return _load_model(model_name)
