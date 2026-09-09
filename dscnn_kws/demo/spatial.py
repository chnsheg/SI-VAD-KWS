"""Low-latency spatial enhancement with a deterministic mono fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .audio import InputChannelMode


@dataclass(frozen=True)
class SpatialMetrics:
    mode: str
    selected_channel: int
    active_channels: int
    reason: str | None
    coherence: float | None
    estimated_snr_db: float | None


@dataclass(frozen=True)
class SpatialFrame:
    raw_mono: np.ndarray
    inference_mono: np.ndarray
    metrics: SpatialMetrics


class SpatialInputProcessor:
    """Select a reliable microphone or use an online MVDR estimate for a valid array."""

    _PROBE_SAMPLES = 8_000
    _MAX_CHANNELS = 4
    _FFT_SAMPLES = 320
    _HOP_SAMPLES = 160
    _WEIGHT_UPDATE_SAMPLES = 4_000

    def __init__(self, *, mode: InputChannelMode | str, sample_rate: int = 16000) -> None:
        try:
            self.mode = InputChannelMode(mode)
        except ValueError as error:
            raise ValueError("mode must be auto, mono, or array") from error
        if sample_rate != 16000:
            raise ValueError("spatial processor requires 16000 Hz PCM")
        self.sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:
        self._probe = np.empty((0, 0), dtype=np.float32)
        self._channel_count: int | None = None
        self._selected_channel = 0
        self._array_ready = False
        self._reason: str | None = None
        self._coherence: float | None = None
        self._vad_confidence: float | None = None
        self._noise_covariance: np.ndarray | None = None
        self._speech_covariance: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._samples_since_weight_update = 0

    def update_vad_confidence(self, previous_tick_score: float | None) -> None:
        if previous_tick_score is None:
            self._vad_confidence = None
            return
        if not isinstance(previous_tick_score, (int, float)) or not math.isfinite(float(previous_tick_score)):
            raise ValueError("previous_tick_score must be finite or None")
        self._vad_confidence = float(np.clip(previous_tick_score, 0.0, 1.0))

    def process(self, pcm: np.ndarray) -> SpatialFrame:
        values = np.asarray(pcm)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("PCM must be nonempty float32 channel-last audio")
        if values.shape[1] > self._MAX_CHANNELS:
            raise ValueError("spatial processor supports at most four channels")
        if not np.all(np.isfinite(values)):
            raise ValueError("PCM must contain only finite values")
        if self._channel_count is None:
            self._channel_count = values.shape[1]
        elif values.shape[1] != self._channel_count:
            raise ValueError("input channel count changed during one audio stream")
        if values.shape[1] == 1 or self.mode is InputChannelMode.MONO:
            mono = values[:, 0].copy()
            return SpatialFrame(mono, mono.copy(), SpatialMetrics("mono", 0, 1, None, None, None))

        self._append_probe(values)
        self._select_channels()
        reference = values[:, self._selected_channel].copy()
        if not self._array_ready:
            return SpatialFrame(
                reference,
                reference.copy(),
                SpatialMetrics(
                    "adaptive_select",
                    self._selected_channel,
                    values.shape[1],
                    self._reason or "probing",
                    self._coherence,
                    None,
                ),
            )

        enhanced = self._mvdr(values)
        reference_rms = _rms(reference)
        enhanced_rms = _rms(enhanced)
        if not np.all(np.isfinite(enhanced)) or (reference_rms > 1e-6 and not 0.1 <= enhanced_rms / reference_rms <= 4.0):
            self._array_ready = False
            self._reason = "mvdr_quality_fallback"
            return SpatialFrame(
                reference,
                reference.copy(),
                SpatialMetrics("adaptive_select", self._selected_channel, values.shape[1], self._reason, self._coherence, None),
            )
        snr_db = 20.0 * math.log10(max(enhanced_rms, 1e-8) / max(reference_rms * 0.1, 1e-8))
        return SpatialFrame(
            reference,
            enhanced,
            SpatialMetrics("mvdr", self._selected_channel, values.shape[1], None, self._coherence, snr_db),
        )

    def _append_probe(self, values: np.ndarray) -> None:
        if self._probe.size == 0:
            self._probe = values.copy()
        else:
            self._probe = np.concatenate((self._probe, values), axis=0)[-self._PROBE_SAMPLES :]

    def _select_channels(self) -> None:
        assert self._channel_count is not None
        probe = self._probe
        rms = np.sqrt(np.mean(np.square(probe, dtype=np.float64), axis=0))
        clipping = np.mean(np.abs(probe) >= 0.999, axis=0)
        valid = (rms > 1e-5) & (clipping < 0.01)
        if not np.any(valid):
            self._selected_channel = int(np.argmax(rms))
            self._array_ready = False
            self._reason = "no_usable_channel"
            self._coherence = None
            return
        self._selected_channel = int(np.argmax(np.where(valid, rms, -1.0)))
        if self.mode is InputChannelMode.MONO or probe.shape[0] < self._PROBE_SAMPLES:
            self._array_ready = False
            self._reason = "probing"
            self._coherence = None
            return
        valid_indices = np.flatnonzero(valid)
        correlations = []
        reference = probe[:, self._selected_channel]
        for index in valid_indices:
            if index == self._selected_channel:
                continue
            correlation = np.corrcoef(reference, probe[:, index])[0, 1]
            if math.isfinite(float(correlation)):
                correlations.append(abs(float(correlation)))
        self._coherence = max(correlations, default=0.0)
        self._array_ready = len(valid_indices) >= 2 and self._coherence >= 0.6
        self._reason = None if self._array_ready else "low_coherence"

    def _mvdr(self, values: np.ndarray) -> np.ndarray:
        channels = values.shape[1]
        if self._noise_covariance is None or self._noise_covariance.shape[0] != channels:
            identity = np.eye(channels, dtype=np.complex128)
            frequency_bins = self._FFT_SAMPLES // 2 + 1
            self._noise_covariance = np.repeat(identity[None, :, :], frequency_bins, axis=0)
            self._speech_covariance = np.repeat(identity[None, :, :], frequency_bins, axis=0)
        assert self._noise_covariance is not None
        assert self._speech_covariance is not None

        aligned = _align_channels(values, self._selected_channel)
        window = np.hanning(self._FFT_SAMPLES).astype(np.float64)
        output = np.zeros(values.shape[0], dtype=np.float64)
        normalizer = np.zeros(values.shape[0], dtype=np.float64)
        alpha = 0.97
        speech = self._vad_confidence is not None and self._vad_confidence >= 0.5
        for start in range(0, values.shape[0], self._HOP_SAMPLES):
            end = min(start + self._FFT_SAMPLES, values.shape[0])
            frame = np.zeros((self._FFT_SAMPLES, channels), dtype=np.float64)
            frame[: end - start] = aligned[start:end]
            spectrum = np.fft.rfft(frame * window[:, None], axis=0).T
            for frequency, vector in enumerate(spectrum.T):
                covariance = np.outer(vector, np.conjugate(vector))
                if speech:
                    self._speech_covariance[frequency] = alpha * self._speech_covariance[frequency] + (1.0 - alpha) * covariance
                else:
                    self._noise_covariance[frequency] = alpha * self._noise_covariance[frequency] + (1.0 - alpha) * covariance
            if self._weights is None or self._samples_since_weight_update >= self._WEIGHT_UPDATE_SAMPLES:
                self._weights = self._calculate_weights()
                self._samples_since_weight_update = 0
            weights = self._weights
            result_spectrum = np.sum(np.conjugate(weights) * spectrum, axis=0)
            result = np.fft.irfft(result_spectrum, n=self._FFT_SAMPLES)
            output[start:end] += result[: end - start] * window[: end - start]
            normalizer[start:end] += np.square(window[: end - start])
            self._samples_since_weight_update += self._HOP_SAMPLES
        return (output / np.maximum(normalizer, 1e-8)).astype(np.float32)

    def _calculate_weights(self) -> np.ndarray:
        assert self._noise_covariance is not None
        assert self._speech_covariance is not None
        frequency_bins, channels, _ = self._noise_covariance.shape
        weights = np.empty((channels, frequency_bins), dtype=np.complex128)
        for frequency in range(frequency_bins):
            speech_covariance = _hermitian(self._speech_covariance[frequency])
            noise_covariance = _hermitian(self._noise_covariance[frequency])
            values, vectors = np.linalg.eigh(speech_covariance)
            steering = vectors[:, int(np.argmax(values))]
            steering *= np.exp(-1j * np.angle(steering[self._selected_channel]))
            loading = max(float(np.trace(noise_covariance).real) / channels, 1e-8) * 1e-3
            loaded_noise = noise_covariance + loading * np.eye(channels)
            numerator = np.linalg.solve(loaded_noise, steering)
            denominator = np.vdot(steering, numerator)
            weights[:, frequency] = numerator / denominator if abs(denominator) > 1e-10 else steering
        return weights


def _align_channels(values: np.ndarray, reference_index: int) -> np.ndarray:
    aligned = values.astype(np.float64, copy=True)
    reference = aligned[:, reference_index]
    arrival_samples = np.zeros(values.shape[1], dtype=np.int64)
    for index in range(values.shape[1]):
        if index == reference_index:
            continue
        correlation = np.correlate(reference, aligned[:, index], mode="full")
        lag = int(np.argmax(np.abs(correlation)) - (values.shape[0] - 1))
        arrival_samples[index] = -lag
    latest_arrival = int(np.max(arrival_samples))
    for index, arrival in enumerate(arrival_samples):
        delay = latest_arrival - int(arrival)
        if delay > 0:
            aligned[delay:, index] = aligned[:-delay, index]
            aligned[:delay, index] = 0.0
    return aligned


def _hermitian(value: np.ndarray) -> np.ndarray:
    return (value + np.conjugate(value.T)) * 0.5


def _rms(values: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(values, dtype=np.float64))))
