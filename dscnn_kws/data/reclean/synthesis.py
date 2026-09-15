from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from .audio import active_rms, peak_guard
from .recipes import Recipe


@dataclass(frozen=True)
class Boundary:
    start: int
    end: int

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2


@dataclass(frozen=True)
class CropMetadata:
    label: str
    coverage_ratio: float
    window_start: int
    source_active_span: tuple[int, int]
    output_active_span: tuple[int, int]


@dataclass(frozen=True)
class SynthesisMetadata:
    coverage_ratio: float
    active_rms_dbfs: float
    measured_snr_db: float | None
    measured_sir_db: float | None
    peak_guard_gain: float
    clipped_before_guard: bool


def _as_batch(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0).to(torch.float32)
    if waveform.ndim == 2:
        return waveform.to(torch.float32)
    raise ValueError(f"Expected [samples] or [batch, samples], got {tuple(waveform.shape)}")


def remap_boundary_for_speed(boundary: Boundary, speed: float) -> Boundary:
    if speed <= 0:
        raise ValueError("speed must be positive")
    return Boundary(start=round(boundary.start / speed), end=round(boundary.end / speed))


def time_stretch_preserve_pitch(waveform: torch.Tensor, speed: float, n_fft: int = 512, hop_length: int = 160) -> torch.Tensor:
    """Phase-vocoder speed change. `speed > 1` shortens the waveform."""
    waveform = _as_batch(waveform)
    if speed <= 0:
        raise ValueError("speed must be positive")
    if speed == 1.0:
        return waveform
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    phase_advance = torch.linspace(0, math.pi * hop_length, spectrum.shape[-2], device=waveform.device)[..., None]
    stretched = AF.phase_vocoder(spectrum, rate=speed, phase_advance=phase_advance)
    output_length = max(1, round(waveform.shape[-1] / speed))
    return torch.istft(stretched, n_fft=n_fft, hop_length=hop_length, window=window, length=output_length)


def _crop_with_padding(waveform: torch.Tensor, start: int, sample_length: int) -> torch.Tensor:
    output = waveform.new_zeros((waveform.shape[0], sample_length))
    source_start = max(0, start)
    source_end = min(waveform.shape[-1], start + sample_length)
    if source_end <= source_start:
        return output
    target_start = source_start - start
    target_end = target_start + source_end - source_start
    output[:, target_start:target_end] = waveform[:, source_start:source_end]
    return output


def crop_positive(
    waveform: torch.Tensor,
    boundary: Boundary,
    recipe: Recipe,
    sample_rate: int = 16000,
    sample_length: int = 16000,
) -> tuple[torch.Tensor, CropMetadata]:
    waveform = _as_batch(waveform)
    if not 0 <= boundary.start < boundary.end <= waveform.shape[-1]:
        raise ValueError(f"Invalid boundary {boundary} for waveform length {waveform.shape[-1]}")
    jitter_samples = round(recipe.jitter_ms * sample_rate / 1000)
    start = boundary.center - sample_length // 2 + jitter_samples
    cropped = _crop_with_padding(waveform, start, sample_length)
    overlap_start = max(boundary.start, start)
    overlap_end = min(boundary.end, start + sample_length)
    overlap = max(0, overlap_end - overlap_start)
    output_start = max(0, overlap_start - start)
    output_end = max(output_start, min(sample_length, overlap_end - start))
    return cropped, CropMetadata(
        label="positive",
        coverage_ratio=overlap / (boundary.end - boundary.start),
        window_start=start,
        source_active_span=(boundary.start, boundary.end),
        output_active_span=(output_start, output_end),
    )


def crop_negative(waveform: torch.Tensor, seed: int, sample_length: int = 16000) -> torch.Tensor:
    waveform = _as_batch(waveform)
    if waveform.shape[-1] <= sample_length:
        return _crop_with_padding(waveform, 0, sample_length)
    start = int(seed % (waveform.shape[-1] - sample_length + 1))
    return _crop_with_padding(waveform, start, sample_length)


def crop_complete_window(
    waveform: torch.Tensor,
    start_sample: int,
    *,
    sample_length: int = 16000,
) -> torch.Tensor:
    """Return a V3 source-time window, rejecting incomplete tail candidates."""
    waveform = _as_batch(waveform)
    if start_sample < 0 or start_sample + sample_length > waveform.shape[-1]:
        raise ValueError("V3 candidate is not a complete source window")
    return waveform[..., start_sample : start_sample + sample_length]


def fft_convolve_rir(waveform: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    waveform = _as_batch(waveform)
    rir = rir.to(device=waveform.device, dtype=waveform.dtype).flatten()
    if rir.numel() == 0:
        raise ValueError("RIR cannot be empty")
    length = waveform.shape[-1] + rir.numel() - 1
    convolved = torch.fft.irfft(torch.fft.rfft(waveform, n=length) * torch.fft.rfft(rir, n=length), n=length)
    return convolved[:, : waveform.shape[-1]]


def set_active_rms(waveform: torch.Tensor, active_span: tuple[int, int], target_dbfs: float) -> tuple[torch.Tensor, float]:
    waveform = _as_batch(waveform)
    current = active_rms(waveform, active_span)
    target = waveform.new_tensor(10.0 ** (float(target_dbfs) / 20.0))
    scaled = waveform * (target / current)
    measured = 20.0 * math.log10(float(active_rms(scaled, active_span).item()))
    return scaled, measured


def _mix_at_sir(
    speech: torch.Tensor,
    active_span: tuple[int, int],
    interferer: torch.Tensor,
    sir_db: float,
) -> tuple[torch.Tensor, float]:
    speech = _as_batch(speech)
    interferer = _as_batch(interferer)
    if speech.shape != interferer.shape:
        raise ValueError("Speech and interferer must have matching shapes")
    target_interferer_rms = active_rms(speech, active_span) / (10.0 ** (sir_db / 20.0))
    interferer_rms = torch.sqrt(torch.mean(interferer.square()) + 1e-12)
    scaled_interferer = interferer * (target_interferer_rms / interferer_rms)
    measured = 20.0 * math.log10(float(active_rms(speech, active_span) / active_rms(scaled_interferer, active_span)))
    return speech + scaled_interferer, measured


def synthesize_speech_noise(
    speech: torch.Tensor,
    active_span: tuple[int, int],
    noise: torch.Tensor | None,
    recipe: Recipe,
    interferer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SynthesisMetadata]:
    speech, active_rms_dbfs = set_active_rms(speech, active_span, float(recipe.active_rms_dbfs))
    mixed = speech
    measured_snr: float | None = None
    measured_sir: float | None = None
    if noise is not None and recipe.snr_db is not None:
        noise = _as_batch(noise).to(device=mixed.device, dtype=mixed.dtype)
        if noise.shape != mixed.shape:
            raise ValueError("Speech and noise must have matching shapes")
        target_noise_rms = active_rms(mixed, active_span) / (10.0 ** (recipe.snr_db / 20.0))
        scaled_noise = noise * (target_noise_rms / torch.sqrt(torch.mean(noise.square()) + 1e-12))
        mixed = mixed + scaled_noise
        measured_snr = 20.0 * math.log10(float(active_rms(mixed - scaled_noise, active_span) / active_rms(scaled_noise, active_span)))
    if interferer is not None and recipe.apply_interferer and recipe.sir_db is not None:
        mixed, measured_sir = _mix_at_sir(mixed, active_span, interferer, recipe.sir_db)
    guarded, gain, clipped = peak_guard(mixed)
    return guarded, SynthesisMetadata(
        coverage_ratio=1.0,
        active_rms_dbfs=active_rms_dbfs,
        measured_snr_db=measured_snr,
        measured_sir_db=measured_sir,
        peak_guard_gain=gain,
        clipped_before_guard=clipped,
    )


def to_pcm16(waveform: torch.Tensor) -> torch.Tensor:
    return torch.round(_as_batch(waveform).clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
