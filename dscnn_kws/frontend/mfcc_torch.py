from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .pwl_fit_utils import fit_piecewise_linear_log_from_samples


def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(freq_mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, freq_mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    filter_shape: str = "triangular",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if f_max is None:
        f_max = sample_rate / 2
    if not (0 <= f_min < f_max):
        raise ValueError(f"Invalid frequency range: f_min={f_min}, f_max={f_max}")

    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=dtype)

    mel_min = _hz_to_mel(torch.tensor(float(f_min), dtype=dtype))
    mel_max = _hz_to_mel(torch.tensor(float(f_max), dtype=dtype))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=dtype)
    hz_points = _mel_to_hz(mel_points)

    fb = torch.zeros(n_mels, n_freqs, dtype=dtype)
    eps = torch.tensor(1e-12, dtype=dtype)
    if filter_shape not in {"triangular", "rectangular"}:
        raise ValueError(f"Unsupported mel filter shape: {filter_shape}")

    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        if filter_shape == "triangular":
            lower = (fft_freqs - left) / torch.maximum(center - left, eps)
            upper = (right - fft_freqs) / torch.maximum(right - center, eps)
            fb[i] = torch.clamp(torch.minimum(lower, upper), min=0.0)
        else:
            fb[i] = ((fft_freqs >= left) & (fft_freqs <= right)).to(dtype)
    return fb


def apply_piecewise_linear(
    x: torch.Tensor,
    breakpoints: torch.Tensor,
    slopes: torch.Tensor,
    intercepts: torch.Tensor,
) -> torch.Tensor:
    if breakpoints.numel() != slopes.numel() + 1 or slopes.numel() != intercepts.numel():
        raise ValueError("Invalid piecewise parameters: len(breakpoints)=len(slopes)+1=len(intercepts)+1")

    x_min = breakpoints[0]
    x_max = breakpoints[-1]
    x_clip = torch.clamp(x, min=x_min, max=x_max)
    y = torch.empty_like(x_clip)

    for i in range(slopes.numel()):
        l = breakpoints[i]
        r = breakpoints[i + 1]
        if i == slopes.numel() - 1:
            mask = (x_clip >= l) & (x_clip <= r)
        else:
            mask = (x_clip >= l) & (x_clip < r)
        y[mask] = slopes[i] * x_clip[mask] + intercepts[i]
    return y


def load_log_pwl_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = {"breakpoints", "slopes", "intercepts"}
    if not required.issubset(cfg):
        raise ValueError(f"Invalid pwl json, required keys missing: {required - set(cfg.keys())}")
    return cfg


def create_dct_matrix(
    n_mfcc: int,
    n_mels: int,
    norm: str | None = "ortho",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    n = torch.arange(n_mels, dtype=dtype)
    k = torch.arange(n_mfcc, dtype=dtype).unsqueeze(1)
    dct = torch.cos(math.pi / n_mels * (n + 0.5) * k)

    if norm is None:
        return dct
    if norm != "ortho":
        raise ValueError(f"Unsupported DCT norm: {norm}")

    dct[0] *= math.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        dct[1:] *= math.sqrt(2.0 / n_mels)
    return dct


class TorchMFCC(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: float | None = None,
        center: bool = True,
        dct_norm: str | None = "ortho",
        mel_filter_shape: str = "triangular",
        log_offset: float = 1e-6,
        log_approx_mode: str = "exact",
        log_pwl_breakpoints: list[float] | None = None,
        log_pwl_slopes: list[float] | None = None,
        log_pwl_intercepts: list[float] | None = None,
        log_pwl_num_segments: int = 6,
        log_pwl_strategy: str = "uniform_logx",
        log_pwl_gamma: float = 1.0,
        log_input_clamp_min: float = 1e-12,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2
        self.center = center
        self.mel_filter_shape = mel_filter_shape
        self.log_offset = log_offset
        self.log_approx_mode = log_approx_mode
        self.log_input_clamp_min = log_input_clamp_min

        if self.log_approx_mode not in {"exact", "pwl"}:
            raise ValueError(f"Unsupported log_approx_mode: {self.log_approx_mode}")

        mel_fb = create_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
            filter_shape=mel_filter_shape,
        )
        dct_mat = create_dct_matrix(n_mfcc=n_mfcc, n_mels=n_mels, norm=dct_norm)
        window = torch.hann_window(win_length)

        self.register_buffer("mel_fb", mel_fb, persistent=False)
        self.register_buffer("dct_mat", dct_mat, persistent=False)
        self.register_buffer("window", window, persistent=False)

        if self.log_approx_mode == "pwl":
            if log_pwl_breakpoints is None or log_pwl_slopes is None or log_pwl_intercepts is None:
                default_fit = fit_piecewise_linear_log_from_samples(
                    x_samples=torch.logspace(-8, 2, steps=30000),
                    num_segments=log_pwl_num_segments,
                    strategy=log_pwl_strategy,
                    gamma=log_pwl_gamma,
                )
                log_pwl_breakpoints = default_fit["breakpoints"]
                log_pwl_slopes = default_fit["slopes"]
                log_pwl_intercepts = default_fit["intercepts"]

            bp = torch.tensor(log_pwl_breakpoints, dtype=torch.float32)
            slopes = torch.tensor(log_pwl_slopes, dtype=torch.float32)
            intercepts = torch.tensor(log_pwl_intercepts, dtype=torch.float32)
            self.register_buffer("log_pwl_breakpoints", bp, persistent=False)
            self.register_buffer("log_pwl_slopes", slopes, persistent=False)
            self.register_buffer("log_pwl_intercepts", intercepts, persistent=False)

    def _log_transform(self, mel_spec: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(mel_spec + self.log_offset, min=self.log_input_clamp_min)
        if self.log_approx_mode == "exact":
            return torch.log(x)

        bp = self.log_pwl_breakpoints.to(device=x.device, dtype=x.dtype)
        slopes = self.log_pwl_slopes.to(device=x.device, dtype=x.dtype)
        intercepts = self.log_pwl_intercepts.to(device=x.device, dtype=x.dtype)
        return apply_piecewise_linear(x, bp, slopes, intercepts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected x shape [B, T] or [B, 1, T], got {list(x.shape)}")

        window = self.window.to(device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            return_complex=True,
        )
        power_spec = stft.real.pow(2) + stft.imag.pow(2)

        mel_fb = self.mel_fb.to(device=x.device, dtype=power_spec.dtype)
        mel_spec = torch.matmul(mel_fb, power_spec)
        log_mel = self._log_transform(mel_spec)

        dct_mat = self.dct_mat.to(device=x.device, dtype=log_mel.dtype)
        mfcc = torch.matmul(dct_mat, log_mel)
        return mfcc
