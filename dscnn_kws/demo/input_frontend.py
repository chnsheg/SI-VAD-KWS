"""Causal microphone frontends evaluated by the VAD-KWS tuning workflow."""

from __future__ import annotations

import math
from dataclasses import replace
from enum import StrEnum
from typing import Any

import numpy as np

from .input_conditioning import AdaptiveInputConditioner, InputConditioningMetrics


_VECTOR_LFILTER: Any | None = None


def _vector_lfilter() -> Any:
    """Load SciPy only for the offline, training-only vectorized replay path."""

    global _VECTOR_LFILTER
    if _VECTOR_LFILTER is None:
        try:
            from scipy.signal import lfilter
        except ImportError as error:
            raise RuntimeError("vectorized frontend replay requires scipy") from error
        _VECTOR_LFILTER = lfilter
    return _VECTOR_LFILTER


class FrontendProfile(StrEnum):
    """Input transformations that preserve the deployed PCM model contract."""

    RAW = "raw"
    PEAK_AGC = "peak_agc"
    HIGHPASS_AGC = "highpass_agc"
    SPEECH_BAND_AGC = "speech_band_agc"
    SPEECH_BAND_UPWARD_AGC = "speech_band_upward_agc"


class CausalHighPass:
    """One-pole high-pass filter whose state survives microphone chunks."""

    def __init__(self, *, cutoff_hz: float = 70.0, sample_rate_hz: int = 16_000) -> None:
        if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0 or cutoff_hz >= sample_rate_hz / 2.0:
            raise ValueError("cutoff_hz must be within the valid one-pole range")
        if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be a positive integer")
        self._alpha = sample_rate_hz / (sample_rate_hz + 2.0 * math.pi * cutoff_hz)
        self.reset()

    def reset(self) -> None:
        self._previous_input = 0.0
        self._previous_output = 0.0

    def process(self, pcm: np.ndarray) -> np.ndarray:
        values = _validate_pcm(pcm)
        result = np.empty_like(values)
        for index, sample in enumerate(values):
            output = self._alpha * (self._previous_output + float(sample) - self._previous_input)
            result[index] = output
            self._previous_input = float(sample)
            self._previous_output = output
        return result

    def process_vectorized(self, pcm: np.ndarray) -> np.ndarray:
        """Replay the same recurrence with SciPy for offline training data loading."""

        values = _validate_pcm(pcm)
        result, _ = _vector_lfilter()(
            np.asarray((self._alpha, -self._alpha), dtype=np.float64),
            np.asarray((1.0, -self._alpha), dtype=np.float64),
            values.astype(np.float64, copy=False),
            zi=np.asarray(
                [self._alpha * (self._previous_output - self._previous_input)],
                dtype=np.float64,
            ),
        )
        self._previous_input = float(values[-1])
        self._previous_output = float(result[-1])
        return result.astype(np.float32)


class _CausalLowPass:
    def __init__(self, *, cutoff_hz: float, sample_rate_hz: int = 16_000) -> None:
        self._retention = math.exp(-2.0 * math.pi * cutoff_hz / sample_rate_hz)
        self.reset()

    def reset(self) -> None:
        self._previous_output = 0.0

    def process(self, pcm: np.ndarray) -> np.ndarray:
        values = _validate_pcm(pcm)
        result = np.empty_like(values)
        for index, sample in enumerate(values):
            output = (1.0 - self._retention) * float(sample) + self._retention * self._previous_output
            result[index] = output
            self._previous_output = output
        return result

    def process_vectorized(self, pcm: np.ndarray) -> np.ndarray:
        """Replay the same recurrence with SciPy for offline training data loading."""

        values = _validate_pcm(pcm)
        result, _ = _vector_lfilter()(
            np.asarray((1.0 - self._retention,), dtype=np.float64),
            np.asarray((1.0, -self._retention), dtype=np.float64),
            values.astype(np.float64, copy=False),
            zi=np.asarray([self._retention * self._previous_output], dtype=np.float64),
        )
        self._previous_output = float(result[-1])
        return result.astype(np.float32)


class _SpeechBandMeter:
    """Measure 300-3400 Hz RMS without altering the waveform passed to a model."""

    def __init__(self) -> None:
        self._high_pass = CausalHighPass(cutoff_hz=300.0)
        self._low_pass = _CausalLowPass(cutoff_hz=3400.0)

    def reset(self) -> None:
        self._high_pass.reset()
        self._low_pass.reset()

    def rms_dbfs(self, pcm: np.ndarray) -> float:
        band = self._low_pass.process(self._high_pass.process(pcm))
        rms = math.sqrt(float(np.mean(np.square(band, dtype=np.float64))))
        return 20.0 * math.log10(max(rms, 1e-8))

    def rms_dbfs_vectorized(self, pcm: np.ndarray) -> float:
        band = self._low_pass.process_vectorized(self._high_pass.process_vectorized(pcm))
        rms = math.sqrt(float(np.mean(np.square(band, dtype=np.float64))))
        return 20.0 * math.log10(max(rms, 1e-8))


class InputFrontend:
    """Apply one selected causal candidate before VAD and KWS inference."""

    def __init__(
        self,
        profile: FrontendProfile | str,
        *,
        target_rms_dbfs: float,
        rise_db_per_100ms: float = 1.0,
        highpass_cutoff_hz: float = 70.0,
    ) -> None:
        try:
            self.profile = FrontendProfile(profile)
        except ValueError as error:
            raise ValueError("unsupported input frontend profile") from error
        if not math.isfinite(highpass_cutoff_hz):
            raise ValueError("highpass_cutoff_hz must be finite")
        upward_only = self.profile is FrontendProfile.SPEECH_BAND_UPWARD_AGC
        self._conditioner = AdaptiveInputConditioner(
            target_rms_dbfs=target_rms_dbfs,
            min_gain_db=0.0 if upward_only else -12.0,
            rise_db_per_100ms=rise_db_per_100ms,
            noise_calibration_ms=320 if upward_only else 0,
            release_gain_on_background=upward_only,
        )
        self._high_pass = (
            CausalHighPass(cutoff_hz=highpass_cutoff_hz)
            if self.profile is FrontendProfile.HIGHPASS_AGC
            else None
        )
        self._speech_band = (
            _SpeechBandMeter()
            if self.profile in (
                FrontendProfile.SPEECH_BAND_AGC,
                FrontendProfile.SPEECH_BAND_UPWARD_AGC,
            )
            else None
        )

    def reset(self) -> None:
        self._conditioner.reset()
        if self._high_pass is not None:
            self._high_pass.reset()
        if self._speech_band is not None:
            self._speech_band.reset()

    def process(self, pcm: np.ndarray) -> tuple[np.ndarray, InputConditioningMetrics]:
        values = _validate_pcm(pcm)
        if self.profile is FrontendProfile.RAW:
            return values.copy(), _passthrough_metrics(values)
        if self.profile is FrontendProfile.PEAK_AGC:
            return self._conditioner.process(values)
        if self.profile is FrontendProfile.HIGHPASS_AGC:
            assert self._high_pass is not None
            return self._conditioner.process(self._high_pass.process(values))
        assert self._speech_band is not None
        conditioned, metrics = self._conditioner.process(
            values, control_rms_dbfs=self._speech_band.rms_dbfs(values)
        )
        if self.profile is FrontendProfile.SPEECH_BAND_UPWARD_AGC:
            if abs(metrics.applied_gain_db) <= 1e-7:
                return values.copy(), replace(
                    metrics,
                    conditioned_rms_dbfs=metrics.raw_rms_dbfs,
                    conditioned_peak_dbfs=metrics.raw_peak_dbfs,
                )
        return conditioned, metrics

    def process_vectorized(self, pcm: np.ndarray) -> tuple[np.ndarray, InputConditioningMetrics]:
        """Use SciPy only for offline replay while retaining the deployed state machine."""

        values = _validate_pcm(pcm)
        if self.profile in (FrontendProfile.RAW, FrontendProfile.PEAK_AGC):
            return self.process(values)
        if self.profile is FrontendProfile.HIGHPASS_AGC:
            assert self._high_pass is not None
            return self._conditioner.process(self._high_pass.process_vectorized(values))
        assert self._speech_band is not None
        conditioned, metrics = self._conditioner.process(
            values, control_rms_dbfs=self._speech_band.rms_dbfs_vectorized(values)
        )
        if self.profile is FrontendProfile.SPEECH_BAND_UPWARD_AGC:
            if abs(metrics.applied_gain_db) <= 1e-7:
                return values.copy(), replace(
                    metrics,
                    conditioned_rms_dbfs=metrics.raw_rms_dbfs,
                    conditioned_peak_dbfs=metrics.raw_peak_dbfs,
                )
        return conditioned, metrics


def _validate_pcm(pcm: np.ndarray) -> np.ndarray:
    values = np.asarray(pcm)
    if values.dtype != np.float32 or values.ndim != 1 or values.size == 0:
        raise ValueError("PCM must be nonempty float32 mono")
    if not np.all(np.isfinite(values)):
        raise ValueError("PCM must be finite")
    return values


def _passthrough_metrics(pcm: np.ndarray) -> InputConditioningMetrics:
    rms = math.sqrt(float(np.mean(np.square(pcm, dtype=np.float64))))
    peak = float(np.max(np.abs(pcm)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-8))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-8))
    return InputConditioningMetrics(
        raw_rms_dbfs=rms_dbfs,
        raw_peak_dbfs=peak_dbfs,
        conditioned_rms_dbfs=rms_dbfs,
        conditioned_peak_dbfs=peak_dbfs,
        applied_gain_db=0.0,
        noise_floor_dbfs=-60.0,
        source_clipped=bool(np.any(np.abs(pcm) >= 0.999)),
    )
