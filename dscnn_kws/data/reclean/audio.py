from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF


@dataclass(frozen=True)
class CanonicalAudioMetadata:
    source_sha256: str
    output_sha256: str
    original_sample_rate: int
    original_channels: int
    output_sample_rate: int
    output_frames: int


@dataclass(frozen=True)
class MixMetadata:
    requested_snr_db: float
    measured_active_snr_db: float
    peak_guard_gain: float
    clipped_before_guard: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mono_float32(path: Path) -> tuple[torch.Tensor, int, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    channels = int(waveform.shape[1])
    tensor = torch.from_numpy(waveform.T.copy()).mean(dim=0, keepdim=True)
    return tensor.clamp(-1.0, 1.0), int(sample_rate), channels


def _fit_length(waveform: torch.Tensor, sample_length: int) -> torch.Tensor:
    if waveform.shape[1] < sample_length:
        return F.pad(waveform, (0, sample_length - waveform.shape[1]))
    if waveform.shape[1] == sample_length:
        return waveform
    start = (waveform.shape[1] - sample_length) // 2
    return waveform.narrow(1, start, sample_length)


def save_pcm16_atomic(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = waveform.detach().cpu().to(torch.float32).clamp(-1.0, 1.0).squeeze(0).numpy()
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".tmp.wav", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        sf.write(temporary, payload, sample_rate, subtype="PCM_16")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonicalize_wav(
    source: Path,
    output: Path,
    sample_rate: int = 16000,
    sample_length: int | None = None,
) -> CanonicalAudioMetadata:
    source_sha256 = sha256_file(source)
    waveform, original_sample_rate, original_channels = load_mono_float32(source)
    if original_sample_rate != sample_rate:
        waveform = AF.resample(waveform, original_sample_rate, sample_rate)
    if sample_length is not None:
        waveform = _fit_length(waveform, sample_length)
    save_pcm16_atomic(output, waveform, sample_rate)
    return CanonicalAudioMetadata(
        source_sha256=source_sha256,
        output_sha256=sha256_file(output),
        original_sample_rate=original_sample_rate,
        original_channels=original_channels,
        output_sample_rate=sample_rate,
        output_frames=int(waveform.shape[1]),
    )


def active_rms(waveform: torch.Tensor, active_span: tuple[int, int]) -> torch.Tensor:
    start, end = active_span
    if start < 0 or end > waveform.shape[1] or start >= end:
        raise ValueError(f"Invalid active span {active_span} for waveform length {waveform.shape[1]}")
    active = waveform[:, start:end]
    return torch.sqrt(torch.mean(active.square()) + 1e-12)


def peak_guard(waveform: torch.Tensor, peak_limit: float = 0.99) -> tuple[torch.Tensor, float, bool]:
    peak = float(waveform.detach().abs().max().item())
    # Keep the stored float32 value strictly below the decimal contract limit.
    effective_limit = float(torch.nextafter(waveform.new_tensor(peak_limit), waveform.new_tensor(0.0)).item())
    if peak <= effective_limit:
        return waveform.clamp(-1.0, 1.0), 1.0, False
    gain = effective_limit / peak
    return (waveform * gain).clamp(-1.0, 1.0), gain, True


def mix_at_active_snr(
    speech: torch.Tensor,
    active_span: tuple[int, int],
    noise: torch.Tensor,
    snr_db: float,
) -> tuple[torch.Tensor, MixMetadata]:
    if speech.shape != noise.shape:
        raise ValueError(f"Speech/noise shape mismatch: {tuple(speech.shape)} != {tuple(noise.shape)}")
    speech_active_rms = active_rms(speech, active_span)
    noise_rms = torch.sqrt(torch.mean(noise.square()) + 1e-12)
    target_noise_rms = speech_active_rms / (10.0 ** (float(snr_db) / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    guarded, gain, clipped = peak_guard(speech + scaled_noise)
    start, end = active_span
    measured_noise_rms = torch.sqrt(torch.mean((scaled_noise[:, start:end] * gain).square()) + 1e-12)
    measured_speech_rms = torch.sqrt(torch.mean((speech[:, start:end] * gain).square()) + 1e-12)
    measured_snr = 20.0 * math.log10(float(measured_speech_rms / measured_noise_rms))
    return guarded, MixMetadata(
        requested_snr_db=float(snr_db),
        measured_active_snr_db=measured_snr,
        peak_guard_gain=gain,
        clipped_before_guard=clipped,
    )
