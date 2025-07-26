"""Wrapper to lazily load and cache the Parakeet-TDT 0.6B v2 model."""
from __future__ import annotations

from functools import lru_cache
from typing import List

import torch
import nemo.collections.asr as nemo_asr  # type: ignore

__all__ = ["get_model"]


MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


def _load_model() -> nemo_asr.models.ASRModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME).eval().to(device)
    return model


@lru_cache(maxsize=1)
def get_model() -> nemo_asr.models.ASRModel:  # pragma: no cover
    """Return a cached instance of the model on GPU (if available)."""
    return _load_model()
