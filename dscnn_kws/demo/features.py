"""PCM conversion and the local VAD log-Mel frontend contract."""

from __future__ import annotations

import math

import numpy as np
import torch
import torchaudio

from .contracts import VadContract


class AudioContractError(ValueError):
    """Raised when captured audio cannot satisfy the VAD frontend contract."""


class StreamingPcmConverter:
    """Stateful mono converter that preserves sample-time phase across chunks."""

    def __init__(self, target_sample_rate: int = 16000) -> None:
        _require_sample_rate("target_sample_rate", target_sample_rate)
        self.target_sample_rate = target_sample_rate
        self._input_sample_rate: int | None = None
        self._received_samples = 0
        self._next_input_position = 0.0
        self._tail = np.empty(0, dtype=np.float32)

    def convert(self, samples: np.ndarray, input_sample_rate: int) -> np.ndarray:
        _require_sample_rate("input_sample_rate", input_sample_rate)
        values = _mix_to_mono(samples)
        if self._input_sample_rate is None:
            self._input_sample_rate = input_sample_rate
        elif input_sample_rate != self._input_sample_rate:
            raise AudioContractError("input_sample_rate changed during one audio stream")
        if input_sample_rate == self.target_sample_rate:
            self._received_samples += values.size
            return values

        stream_start = self._received_samples
        stream_end = stream_start + values.size
        combined = values if self._tail.size == 0 else np.concatenate((self._tail, values))
        combined_start = stream_start - self._tail.size
        step = input_sample_rate / self.target_sample_rate
        output: list[float] = []
        while self._next_input_position + 1.0 < stream_end:
            left = int(np.floor(self._next_input_position))
            index = left - combined_start
            if index < 0 or index + 1 >= combined.size:
                break
            fraction = self._next_input_position - left
            output.append(float(combined[index] + fraction * (combined[index + 1] - combined[index])))
            self._next_input_position += step
        self._received_samples = stream_end
        self._tail = values[-1:].copy()
        return np.asarray(output, dtype=np.float32)

    def reset(self) -> None:
        self._input_sample_rate = None
        self._received_samples = 0
        self._next_input_position = 0.0
        self._tail = np.empty(0, dtype=np.float32)


_ANTI_ALIAS_TAP_COUNT = 127
_ANTI_ALIAS_KAISER_BETA = 8.6
_ANTI_ALIAS_MINIMUM_PHASE_FFT_SAMPLES = 4_096


def _anti_alias_filter(decimation_factor: int) -> np.ndarray:
    """Return a normalized minimum-phase low-pass for an integer downsampler."""

    cutoff = 0.5 / decimation_factor * 0.90
    center = (_ANTI_ALIAS_TAP_COUNT - 1) / 2.0
    index = np.arange(_ANTI_ALIAS_TAP_COUNT, dtype=np.float64) - center
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * index)
    taps *= np.kaiser(_ANTI_ALIAS_TAP_COUNT, _ANTI_ALIAS_KAISER_BETA)
    taps /= np.sum(taps)
    return _minimum_phase_filter(taps)


def _minimum_phase_filter(taps: np.ndarray) -> np.ndarray:
    """Convert a finite low-pass prototype to a same-length minimum-phase FIR."""

    spectrum = np.fft.rfft(taps, _ANTI_ALIAS_MINIMUM_PHASE_FFT_SAMPLES)
    log_magnitude = np.log(np.maximum(np.abs(spectrum), 1e-12))
    cepstrum = np.fft.irfft(log_magnitude, _ANTI_ALIAS_MINIMUM_PHASE_FFT_SAMPLES)
    minimum_cepstrum = np.zeros(_ANTI_ALIAS_MINIMUM_PHASE_FFT_SAMPLES, dtype=np.float64)
    half = _ANTI_ALIAS_MINIMUM_PHASE_FFT_SAMPLES // 2
    minimum_cepstrum[0] = cepstrum[0]
    minimum_cepstrum[1:half] = 2.0 * cepstrum[1:half]
    minimum_cepstrum[half] = cepstrum[half]
    response = np.exp(np.fft.rfft(minimum_cepstrum))
    result = np.fft.irfft(response, _ANTI_ALIAS_MINIMUM_PHASE_FFT_SAMPLES)[: taps.size]
    result /= np.sum(result)
    return result.astype(np.float32)


def _integer_downsample_factor(input_sample_rate: int, target_sample_rate: int) -> int | None:
    if input_sample_rate <= target_sample_rate:
        return None
    if input_sample_rate % target_sample_rate:
        return None
    factor = input_sample_rate // target_sample_rate
    return factor if factor >= 2 else None


class StreamingChannelConverter:
    """Resample channel-last PCM while retaining inter-channel sample alignment.

    Integer downsampling uses a stateful anti-alias filter before decimation.
    Non-integer ratios retain the historical linear interpolation path until a
    dedicated polyphase contract is defined for those devices.
    """

    def __init__(self, target_sample_rate: int = 16000) -> None:
        _require_sample_rate("target_sample_rate", target_sample_rate)
        self.target_sample_rate = target_sample_rate
        self.reset()

    def convert(self, samples: np.ndarray, input_sample_rate: int) -> np.ndarray:
        _require_sample_rate("input_sample_rate", input_sample_rate)
        values = np.asarray(samples)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise AudioContractError("audio must be nonempty float32 channel-last PCM")
        if not np.all(np.isfinite(values)):
            raise AudioContractError("audio must be finite")
        if self._input_sample_rate is None:
            self._input_sample_rate = input_sample_rate
            self._channel_count = values.shape[1]
            self._anti_alias_factor = _integer_downsample_factor(
                input_sample_rate, self.target_sample_rate
            )
            if self._anti_alias_factor is not None:
                self._anti_alias_taps = _anti_alias_filter(self._anti_alias_factor)
                self._anti_alias_tail = np.zeros(
                    (_ANTI_ALIAS_TAP_COUNT - 1, values.shape[1]), dtype=np.float32
                )
        elif input_sample_rate != self._input_sample_rate:
            raise AudioContractError("input_sample_rate changed during one audio stream")
        elif values.shape[1] != self._channel_count:
            raise AudioContractError("input channel count changed during one audio stream")
        if input_sample_rate == self.target_sample_rate:
            self._received_frames += values.shape[0]
            return values

        if self._anti_alias_factor is not None:
            return self._convert_integer_downsampled(values)

        stream_start = self._received_frames
        stream_end = stream_start + values.shape[0]
        combined = values if self._tail.size == 0 else np.concatenate((self._tail, values), axis=0)
        combined_start = stream_start - self._tail.shape[0]
        step = input_sample_rate / self.target_sample_rate
        output: list[np.ndarray] = []
        while self._next_input_position + 1.0 < stream_end:
            left = int(np.floor(self._next_input_position))
            index = left - combined_start
            if index < 0 or index + 1 >= combined.shape[0]:
                break
            fraction = np.float32(self._next_input_position - left)
            output.append(combined[index] + fraction * (combined[index + 1] - combined[index]))
            self._next_input_position += step
        self._received_frames = stream_end
        self._tail = values[-1:, :].copy()
        if not output:
            return np.empty((0, values.shape[1]), dtype=np.float32)
        return np.asarray(output, dtype=np.float32)

    def _convert_integer_downsampled(self, values: np.ndarray) -> np.ndarray:
        assert self._anti_alias_factor is not None
        assert self._anti_alias_taps is not None
        assert self._anti_alias_tail is not None
        combined = np.concatenate((self._anti_alias_tail, values), axis=0)
        filtered = np.empty_like(values)
        start = _ANTI_ALIAS_TAP_COUNT - 1
        stop = start + values.shape[0]
        for channel in range(values.shape[1]):
            full = np.convolve(combined[:, channel], self._anti_alias_taps, mode="full")
            filtered[:, channel] = full[start:stop].astype(np.float32, copy=False)
        self._anti_alias_tail = combined[-(_ANTI_ALIAS_TAP_COUNT - 1) :].copy()

        stream_start = self._received_frames
        stream_end = stream_start + values.shape[0]
        indices = np.arange(
            self._next_downsample_input_index,
            stream_end,
            self._anti_alias_factor,
            dtype=np.int64,
        )
        self._received_frames = stream_end
        if indices.size == 0:
            return np.empty((0, values.shape[1]), dtype=np.float32)
        self._next_downsample_input_index = int(indices[-1] + self._anti_alias_factor)
        return filtered[indices - stream_start]

    def reset(self) -> None:
        self._input_sample_rate: int | None = None
        self._channel_count: int | None = None
        self._received_frames = 0
        self._next_input_position = 0.0
        self._tail = np.empty((0, 0), dtype=np.float32)
        self._anti_alias_factor: int | None = None
        self._anti_alias_taps: np.ndarray | None = None
        self._anti_alias_tail: np.ndarray | None = None
        self._next_downsample_input_index = 0


class StreamingVadFeatureCache:
    """Cache causal raw log-Mel frames for the stateful VAD preprocessing rule."""

    def __init__(self, contract: VadContract) -> None:
        self.contract = contract
        self._frame_length = _samples_for_ms(contract.sample_rate, contract.frame_ms)
        self._hop_length = _samples_for_ms(contract.sample_rate, contract.hop_ms)
        self._window_samples = contract.sample_rate
        self._transform = _vad_mel_transform(contract, self._frame_length, self._hop_length)
        self.reset()

    def reset(self) -> None:
        self._pcm = np.empty(0, dtype=np.float32)
        self._pcm_start = 0
        self._next_frame_start = 0
        self._raw_starts = np.empty(0, dtype=np.int64)
        self._raw_frames = np.empty((0, self.contract.n_mels), dtype=np.float32)
        self._pending = np.empty((0, self.contract.n_mels), dtype=np.float32)
        self._initialized = False

    def rebuild(self, waveform: np.ndarray) -> np.ndarray:
        """Recreate the one-second history and return its bootstrap features."""

        values = _require_valid_vad_waveform(waveform)
        raw = _extract_vad_raw_logmel(values, self.contract, self._transform)
        count = raw.shape[0]
        starts = np.arange(count, dtype=np.int64) * self._hop_length
        self._raw_starts = starts
        self._raw_frames = raw
        self._pending = np.empty((0, self.contract.n_mels), dtype=np.float32)
        self._next_frame_start = count * self._hop_length
        self._pcm_start = self._next_frame_start
        self._pcm = values[self._pcm_start :].copy()
        self._initialized = True
        return normalize_vad_raw_logmel(raw, self.contract)

    def append_pcm(self, pcm: np.ndarray) -> None:
        """Compute and normalize each newly completed causal frame exactly once."""

        if not self._initialized:
            return
        values = np.asarray(pcm)
        if values.dtype != np.float32 or values.ndim != 1 or values.size == 0:
            raise AudioContractError("PCM must be a nonempty float32 mono array")
        if not np.all(np.isfinite(values)):
            raise AudioContractError("PCM must contain only finite values")
        self._pcm = np.concatenate((self._pcm, values))
        pcm_end = self._pcm_start + self._pcm.size
        emitted: list[np.ndarray] = []
        while self._next_frame_start + self._frame_length <= pcm_end:
            index = self._next_frame_start - self._pcm_start
            frame = self._pcm[index : index + self._frame_length]
            raw = _extract_vad_raw_logmel(frame, self.contract, self._transform)[0]
            self._raw_starts = np.append(self._raw_starts, self._next_frame_start)
            self._raw_frames = np.vstack((self._raw_frames, raw[None]))
            frame_end = self._next_frame_start + self._frame_length
            history_start = frame_end - self._window_samples
            retained = self._raw_starts >= history_start
            self._raw_starts = self._raw_starts[retained]
            self._raw_frames = self._raw_frames[retained]
            emitted.append(normalize_vad_raw_logmel(self._raw_frames, self.contract)[-1])
            self._next_frame_start += self._hop_length
        discard = self._next_frame_start - self._pcm_start
        if discard:
            self._pcm = self._pcm[discard:].copy()
            self._pcm_start = self._next_frame_start
        if emitted:
            new_values = np.asarray(emitted, dtype=np.float32)
            self._pending = np.concatenate((self._pending, new_values))

    def take_new_features(self) -> np.ndarray:
        """Return each normalized frame at most once in chronological order."""

        result = self._pending
        self._pending = np.empty((0, self.contract.n_mels), dtype=np.float32)
        return result


def ensure_mono_16k(
    samples: np.ndarray, input_sample_rate: int, target_sample_rate: int = 16000
) -> np.ndarray:
    """Mix channel-last PCM to mono and resample it to the target sample rate."""

    _require_sample_rate("input_sample_rate", input_sample_rate)
    _require_sample_rate("target_sample_rate", target_sample_rate)
    values = _mix_to_mono(samples)
    if input_sample_rate == target_sample_rate:
        return values.astype(np.float32, copy=False)
    try:
        tensor = torch.from_numpy(np.ascontiguousarray(values))
        resampled = torchaudio.functional.resample(tensor, input_sample_rate, target_sample_rate)
    except Exception as error:
        raise AudioContractError("could not resample captured audio") from error
    result = resampled.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise AudioContractError("resampling produced invalid audio")
    return result


def extract_vad_logmel(waveform: np.ndarray, contract: VadContract) -> np.ndarray:
    """Compute the causal VAD log-Mel features for one normalized PCM window."""

    return normalize_vad_raw_logmel(extract_vad_raw_logmel(waveform, contract), contract)


def extract_vad_raw_logmel(waveform: np.ndarray, contract: VadContract) -> np.ndarray:
    """Compute unnormalised causal log-Mel frames for the VAD feature contract."""

    values = _require_valid_vad_waveform(waveform)
    frame_length = _samples_for_ms(contract.sample_rate, contract.frame_ms)
    hop_length = _samples_for_ms(contract.sample_rate, contract.hop_ms)
    transform = _vad_mel_transform(contract, frame_length, hop_length)
    return _extract_vad_raw_logmel(values, contract, transform)


def normalize_vad_raw_logmel(raw_features: np.ndarray, contract: VadContract) -> np.ndarray:
    """Apply the VAD contract's per-window CMVN to raw log-Mel frames."""

    features = np.asarray(raw_features)
    if features.dtype != np.float32 or features.ndim != 2 or features.shape[0] < 1:
        raise AudioContractError("raw VAD features must be nonempty float32 [frames, n_mels]")
    if features.shape[1] != contract.n_mels or not np.all(np.isfinite(features)):
        raise AudioContractError("raw VAD features must be finite float32 [frames, n_mels]")
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    normalized = (features - mean) / np.maximum(std, contract.normalization_floor)
    return normalized.astype(np.float32, copy=False)


def _require_valid_vad_waveform(waveform: np.ndarray) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1:
        raise AudioContractError("waveform must be one-dimensional mono PCM")
    if values.size == 0:
        raise AudioContractError("waveform must not be empty")
    if not np.all(np.isfinite(values)):
        raise AudioContractError("waveform must be finite")
    return values


def _vad_mel_transform(
    contract: VadContract, frame_length: int, hop_length: int
) -> torchaudio.transforms.MelSpectrogram:
    try:
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=contract.sample_rate,
            n_fft=frame_length,
            win_length=frame_length,
            hop_length=hop_length,
            f_min=contract.f_min,
            f_max=contract.sample_rate / 2.0,
            n_mels=contract.n_mels,
            center=contract.center,
            power=contract.power,
        )
    except Exception as error:
        raise AudioContractError("could not configure VAD log-Mel frontend") from error


def _extract_vad_raw_logmel(
    values: np.ndarray, contract: VadContract, transform: torchaudio.transforms.MelSpectrogram
) -> np.ndarray:
    frame_length = _samples_for_ms(contract.sample_rate, contract.frame_ms)
    hop_length = _samples_for_ms(contract.sample_rate, contract.hop_ms)
    count = _frame_count(values.size, frame_length, hop_length)
    required_samples = (count - 1) * hop_length + frame_length
    values = values[:required_samples]

    try:
        with torch.no_grad():
            mel = transform(torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0))
        features = mel.squeeze(0).transpose(0, 1).cpu().numpy()
    except Exception as error:
        raise AudioContractError("could not compute VAD log-Mel features") from error

    features = np.log(np.maximum(features, contract.log_floor)).astype(np.float32)
    if len(features) > count:
        features = features[:count]
    elif len(features) < count:
        features = np.pad(features, ((0, count - len(features)), (0, 0)))
    return features.astype(np.float32, copy=False)


def _require_sample_rate(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AudioContractError(f"{name} must be a positive integer")


def _mix_to_mono(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    if values.ndim != 1:
        raise AudioContractError("audio must be mono or channel-last PCM")
    if values.size == 0:
        raise AudioContractError("audio must not be empty")
    if not np.all(np.isfinite(values)):
        raise AudioContractError("audio must be finite")
    return values


def _samples_for_ms(sample_rate: int, milliseconds: float) -> int:
    return max(1, int(round(sample_rate * milliseconds / 1000.0)))


def _frame_count(sample_count: int, frame_length: int, hop_length: int) -> int:
    if sample_count < frame_length:
        raise AudioContractError("waveform must contain one complete VAD frame")
    return (sample_count - frame_length) // hop_length + 1
