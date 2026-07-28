from __future__ import annotations

import math

import numpy as np
import torch
import torchaudio
from scipy.fftpack import dct


def samples_for_ms(sample_rate: int, ms: float) -> int:
    return max(1, int(round(sample_rate * ms / 1000.0)))


def frame_count(
    num_samples: int,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> int:
    frame_length = samples_for_ms(sample_rate, frame_ms)
    hop_length = samples_for_ms(sample_rate, hop_ms)
    if num_samples <= frame_length:
        return 1
    return int(math.ceil((num_samples - frame_length) / hop_length)) + 1


def pad_for_frame_count(
    waveform: np.ndarray,
    sample_rate: int,
    target_frames: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    frame_length = samples_for_ms(sample_rate, frame_ms)
    hop_length = samples_for_ms(sample_rate, hop_ms)
    required = (target_frames - 1) * hop_length + frame_length
    if len(waveform) >= required:
        return np.asarray(waveform[:required], dtype=np.float32)
    return np.pad(np.asarray(waveform, dtype=np.float32), (0, required - len(waveform)))


def frame_signal(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    count = frame_count(len(waveform), sample_rate, frame_ms, hop_ms)
    padded = pad_for_frame_count(waveform, sample_rate, count, frame_ms, hop_ms)
    frame_length = samples_for_ms(sample_rate, frame_ms)
    hop_length = samples_for_ms(sample_rate, hop_ms)
    frames = np.empty((count, frame_length), dtype=np.float32)
    for idx in range(count):
        start = idx * hop_length
        frames[idx] = padded[start : start + frame_length]
    return frames


def rms_zcr(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    frames = frame_signal(waveform, sample_rate, frame_ms, hop_ms)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    return rms.astype(np.float32), zcr.astype(np.float32)


def robust_normalize_01(values: np.ndarray, low_pct: float = 5.0, high_pct: float = 95.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def sample_mask_to_frame_labels(
    sample_mask: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    positive_ratio: float = 0.2,
) -> np.ndarray:
    frames = frame_signal(sample_mask.astype(np.float32), sample_rate, frame_ms, hop_ms)
    return (np.mean(frames, axis=1) >= positive_ratio).astype(np.uint8)


def align_length(values: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
    values = np.asarray(values)
    if len(values) == target_len:
        return values
    if len(values) > target_len:
        return values[:target_len]
    pad_width = [(0, target_len - len(values))]
    if values.ndim > 1:
        pad_width.extend((0, 0) for _ in values.shape[1:])
    return np.pad(values, pad_width, constant_values=pad_value)


def standardize_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-5)).astype(np.float32)


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    normalize: bool = True,
) -> np.ndarray:
    count = frame_count(len(waveform), sample_rate, frame_ms, hop_ms)
    padded = pad_for_frame_count(waveform, sample_rate, count, frame_ms, hop_ms)
    win_length = samples_for_ms(sample_rate, frame_ms)
    hop_length = samples_for_ms(sample_rate, hop_ms)
    tensor = torch.from_numpy(padded.astype(np.float32)).unsqueeze(0)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        f_min=20.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        center=False,
        power=2.0,
    )
    with torch.no_grad():
        mel = transform(tensor).squeeze(0).transpose(0, 1).cpu().numpy()
    feats = np.log(np.maximum(mel, 1e-10)).astype(np.float32)
    feats = align_length(feats, count)
    return standardize_features(feats) if normalize else feats


def mfcc_features(
    waveform: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 64,
    n_mels: int = 64,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    normalize: bool = True,
) -> np.ndarray:
    mel = log_mel_spectrogram(
        waveform,
        sample_rate,
        n_mels=n_mels,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        normalize=False,
    )
    coeffs = dct(mel, type=2, axis=1, norm="ortho")[:, :n_mfcc].astype(np.float32)
    return standardize_features(coeffs) if normalize else coeffs

