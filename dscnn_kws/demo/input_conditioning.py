"""Causal mono input conditioning for realtime VAD/KWS inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_DB_EPSILON = 1e-8


@dataclass(frozen=True)
class InputConditioningMetrics:
    """Measurements recorded for one raw and conditioned PCM block."""

    raw_rms_dbfs: float
    raw_peak_dbfs: float
    conditioned_rms_dbfs: float
    conditioned_peak_dbfs: float
    applied_gain_db: float
    noise_floor_dbfs: float
    source_clipped: bool
    control_rms_dbfs: float | None = None
    noise_floor_calibrated: bool = True
    speech_detected: bool = False


class AdaptiveInputConditioner:
    """Apply bounded gain without amplifying blocks classified as background noise."""

    def __init__(
        self,
        *,
        target_rms_dbfs: float = -25.0,
        min_gain_db: float = -12.0,
        max_gain_db: float = 35.0,
        peak_ceiling_dbfs: float = -3.0,
        speech_margin_db: float = 12.0,
        rise_db_per_100ms: float = 1.0,
        noise_calibration_ms: int = 0,
        release_gain_on_background: bool = False,
    ) -> None:
        for name, value in (
            ("target_rms_dbfs", target_rms_dbfs),
            ("min_gain_db", min_gain_db),
            ("max_gain_db", max_gain_db),
            ("peak_ceiling_dbfs", peak_ceiling_dbfs),
            ("speech_margin_db", speech_margin_db),
            ("rise_db_per_100ms", rise_db_per_100ms),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if min_gain_db > max_gain_db:
            raise ValueError("min_gain_db must not exceed max_gain_db")
        if rise_db_per_100ms <= 0.0:
            raise ValueError("rise_db_per_100ms must be positive")
        if isinstance(noise_calibration_ms, bool) or not isinstance(noise_calibration_ms, int):
            raise ValueError("noise_calibration_ms must be an integer")
        if noise_calibration_ms < 0:
            raise ValueError("noise_calibration_ms must be nonnegative")
        if not isinstance(release_gain_on_background, bool):
            raise ValueError("release_gain_on_background must be a boolean")
        self.target_rms_dbfs = float(target_rms_dbfs)
        self.min_gain_db = float(min_gain_db)
        self.max_gain_db = float(max_gain_db)
        self.peak_ceiling_dbfs = float(peak_ceiling_dbfs)
        self.speech_margin_db = float(speech_margin_db)
        self.rise_db_per_100ms = float(rise_db_per_100ms)
        self.noise_calibration_ms = noise_calibration_ms
        self.release_gain_on_background = release_gain_on_background
        self._noise_calibration_samples = noise_calibration_ms * 16
        self.reset()

    def reset(self) -> None:
        """Forget input level state at an audio timeline discontinuity."""

        self._gain_db = 0.0
        self._noise_floor_dbfs = -60.0
        self._noise_floor_observed = False
        self._calibration_remaining_samples = self._noise_calibration_samples

    def process(
        self, pcm: np.ndarray, *, control_rms_dbfs: float | None = None
    ) -> tuple[np.ndarray, InputConditioningMetrics]:
        """Return peak-limited float32 PCM and its causal gain diagnostics."""

        values = np.asarray(pcm)
        if values.dtype != np.float32 or values.ndim != 1 or values.size == 0:
            raise ValueError("PCM must be a nonempty float32 mono array")
        if not np.all(np.isfinite(values)):
            raise ValueError("PCM must contain only finite values")

        raw_rms_dbfs = _rms_dbfs(values)
        raw_peak_dbfs = _peak_dbfs(values)
        if control_rms_dbfs is None:
            effective_rms_dbfs = raw_rms_dbfs
        elif isinstance(control_rms_dbfs, (int, float)) and math.isfinite(float(control_rms_dbfs)):
            effective_rms_dbfs = float(control_rms_dbfs)
        else:
            raise ValueError("control_rms_dbfs must be finite or None")
        source_clipped = bool(np.any(np.abs(values) >= 0.999))
        is_calibrating = self._calibration_remaining_samples > 0
        is_speech = False
        if is_calibrating:
            self._observe_noise_floor(effective_rms_dbfs)
            self._calibration_remaining_samples = max(
                0, self._calibration_remaining_samples - values.size
            )
            self._gain_db = self.min_gain_db
        else:
            is_speech = effective_rms_dbfs >= self._noise_floor_dbfs + self.speech_margin_db
        if not is_calibrating and not is_speech:
            self._observe_noise_floor(effective_rms_dbfs)
            if self.release_gain_on_background:
                self._gain_db = self.min_gain_db
        elif not is_calibrating:
            requested_gain_db = float(
                np.clip(
                    self.target_rms_dbfs - effective_rms_dbfs,
                    self.min_gain_db,
                    self.max_gain_db,
                )
            )
            peak_limited_gain_db = self.peak_ceiling_dbfs - raw_peak_dbfs
            requested_gain_db = min(requested_gain_db, peak_limited_gain_db)
            if requested_gain_db > self._gain_db:
                block_ms = values.size * 1000.0 / 16000.0
                requested_gain_db = min(
                    requested_gain_db,
                    self._gain_db + self.rise_db_per_100ms * block_ms / 100.0,
                )
            self._gain_db = float(np.clip(requested_gain_db, self.min_gain_db, self.max_gain_db))

        gain = np.float32(10.0 ** (self._gain_db / 20.0))
        ceiling = np.float32(10.0 ** (self.peak_ceiling_dbfs / 20.0))
        conditioned = np.clip(values * gain, -ceiling, ceiling).astype(np.float32, copy=False)
        metrics = InputConditioningMetrics(
            raw_rms_dbfs=raw_rms_dbfs,
            raw_peak_dbfs=raw_peak_dbfs,
            conditioned_rms_dbfs=_rms_dbfs(conditioned),
            conditioned_peak_dbfs=_peak_dbfs(conditioned),
            applied_gain_db=self._gain_db,
            noise_floor_dbfs=self._noise_floor_dbfs,
            source_clipped=source_clipped,
            control_rms_dbfs=effective_rms_dbfs,
            noise_floor_calibrated=self._calibration_remaining_samples == 0,
            speech_detected=is_speech,
        )
        return conditioned, metrics

    def _observe_noise_floor(self, observed_dbfs: float) -> None:
        if not self._noise_floor_observed:
            self._noise_floor_dbfs = observed_dbfs
            self._noise_floor_observed = True
            return
        if self._calibration_remaining_samples > 0:
            self._noise_floor_dbfs = min(self._noise_floor_dbfs, observed_dbfs)
            return
        self._noise_floor_dbfs = _update_noise_floor(self._noise_floor_dbfs, observed_dbfs)


def _rms_dbfs(values: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(np.square(values, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, _DB_EPSILON))


def _peak_dbfs(values: np.ndarray) -> float:
    peak = float(np.max(np.abs(values)))
    return 20.0 * math.log10(max(peak, _DB_EPSILON))


def _update_noise_floor(current_dbfs: float, observed_dbfs: float) -> float:
    """Follow quieter background quickly enough to remain useful, never voice bursts."""

    if observed_dbfs < current_dbfs:
        return 0.8 * current_dbfs + 0.2 * observed_dbfs
    return 0.98 * current_dbfs + 0.02 * observed_dbfs
