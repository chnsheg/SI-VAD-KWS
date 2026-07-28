from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FramePrediction:
    scores: np.ndarray
    frame_hop_ms: float = 10.0
    source_id: str | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        self.scores = np.clip(scores, 0.0, 1.0)
        if self.frame_hop_ms <= 0:
            raise ValueError("frame_hop_ms must be positive")

    def to_segments(
        self,
        threshold: float = 0.5,
        min_speech_ms: float = 60.0,
        min_silence_ms: float = 100.0,
    ) -> list[dict[str, float]]:
        speech = self.scores >= float(threshold)
        speech = _fill_short_silences(speech, self.frame_hop_ms, min_silence_ms)
        speech = _remove_short_speech(speech, self.frame_hop_ms, min_speech_ms)
        return _mask_to_segments(speech, self.scores, self.frame_hop_ms)


def _runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    if len(mask) == 0:
        return []
    runs: list[tuple[bool, int, int]] = []
    start = 0
    current = bool(mask[0])
    for idx in range(1, len(mask)):
        value = bool(mask[idx])
        if value != current:
            runs.append((current, start, idx))
            start = idx
            current = value
    runs.append((current, start, len(mask)))
    return runs


def _fill_short_silences(mask: np.ndarray, hop_ms: float, min_silence_ms: float) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for value, start, end in _runs(out):
        if value:
            continue
        if start == 0 or end == len(out):
            continue
        if (end - start) * hop_ms < min_silence_ms:
            out[start:end] = True
    return out


def _remove_short_speech(mask: np.ndarray, hop_ms: float, min_speech_ms: float) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for value, start, end in _runs(out):
        if value and (end - start) * hop_ms < min_speech_ms:
            out[start:end] = False
    return out


def _mask_to_segments(mask: np.ndarray, scores: np.ndarray, hop_ms: float) -> list[dict[str, float]]:
    segments: list[dict[str, float]] = []
    for value, start, end in _runs(mask):
        if not value:
            continue
        score = float(np.mean(scores[start:end])) if end > start else 0.0
        segments.append(
            {
                "start_sec": start * hop_ms / 1000.0,
                "end_sec": end * hop_ms / 1000.0,
                "score": score,
            }
        )
    return segments

