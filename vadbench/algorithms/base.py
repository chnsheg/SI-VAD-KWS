from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from vadbench.frame_prediction import FramePrediction
from vadbench.manifest import ManifestRecord


class VADAlgorithm:
    name = "base"
    requires_training = False

    def __init__(self, frame_hop_ms: float = 10.0, **_: object) -> None:
        self.frame_hop_ms = float(frame_hop_ms)
        self.threshold: float | None = None

    def fit(
        self,
        train_manifest: Sequence[ManifestRecord],
        val_manifest: Sequence[ManifestRecord],
        base_dir: str | Path,
        **kwargs: object,
    ) -> dict[str, float]:
        return {}

    def predict(self, waveform: np.ndarray, sample_rate: int, source_id: str | None = None) -> FramePrediction:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)


def load_pickled_algorithm(path: str | Path) -> VADAlgorithm:
    with Path(path).open("rb") as handle:
        algorithm = pickle.load(handle)
    if not isinstance(algorithm, VADAlgorithm):
        raise TypeError(f"Pickle does not contain a VADAlgorithm: {path}")
    return algorithm

