"""Validated state values carried between stateful VAD calls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VadStreamState:
    """Causal CNN and GRU state for the supported VAD graph."""

    cnn_context: np.ndarray
    gru_hidden: np.ndarray

    def __post_init__(self) -> None:
        _require_tensor("cnn_context", self.cnn_context, (1, 128, 4))
        _require_tensor("gru_hidden", self.gru_hidden, (1, 1, 40))


def _require_tensor(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite float32 {list(shape)}")
