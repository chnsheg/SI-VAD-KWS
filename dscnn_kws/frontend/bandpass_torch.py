from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mfcc_torch import apply_piecewise_linear
from .pwl_fit_utils import fit_piecewise_linear_log_from_samples


def _build_band_edges(
    n_bands: int,
    f_min: float,
    f_max: float,
    spacing: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_bands <= 0:
        raise ValueError(f"n_bands must be > 0, got {n_bands}")
    if not (0.0 < f_min < f_max):
        raise ValueError(f"Invalid band range: f_min={f_min}, f_max={f_max}")
    if spacing not in {"log", "linear"}:
        raise ValueError(f"Unsupported spacing: {spacing}")

    if spacing == "log":
        edges = torch.logspace(math.log10(f_min), math.log10(f_max), steps=n_bands + 1, dtype=dtype)
    else:
        edges = torch.linspace(f_min, f_max, steps=n_bands + 1, dtype=dtype)
    return edges


def _sinc_bandpass_kernel(
    sample_rate: int,
    low_hz: torch.Tensor,
    high_hz: torch.Tensor,
    kernel_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd for linear-phase FIR, got {kernel_size}")

    m = (kernel_size - 1) // 2
    n = torch.arange(-m, m + 1, dtype=dtype)
    fs = float(sample_rate)

    low = low_hz.unsqueeze(1)
    high = high_hz.unsqueeze(1)

    h_high = 2.0 * high / fs * torch.sinc(2.0 * high * n / fs)
    h_low = 2.0 * low / fs * torch.sinc(2.0 * low * n / fs)
    h_bp = h_high - h_low

    window = torch.hamming_window(kernel_size, periodic=False, dtype=dtype).unsqueeze(0)
    h_bp = h_bp * window

    norm = torch.sum(torch.abs(h_bp), dim=1, keepdim=True).clamp_min(1e-12)
    h_bp = h_bp / norm
    return h_bp


def create_fir_bandpass_filterbank(
    sample_rate: int,
    n_bands: int,
    f_min: float = 200.0,
    f_max: float | None = None,
    spacing: str = "log",
    kernel_size: int = 63,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    nyquist = sample_rate / 2.0
    if f_max is None:
        f_max = nyquist
    if f_max > nyquist:
        raise ValueError(f"f_max must be <= Nyquist({nyquist}), got {f_max}")

    edges = _build_band_edges(
        n_bands=n_bands,
        f_min=f_min,
        f_max=f_max,
        spacing=spacing,
        dtype=dtype,
    )
    low = edges[:-1]
    high = edges[1:]
    kernels = _sinc_bandpass_kernel(
        sample_rate=sample_rate,
        low_hz=low,
        high_hz=high,
        kernel_size=kernel_size,
        dtype=dtype,
    )
    return kernels.unsqueeze(1), edges


class TorchBandpass(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_bands: int,
        frame_length: int,
        frame_hop: int,
        f_min: float = 200.0,
        f_max: float | None = None,
        spacing: str = "log",
        kernel_size: int = 63,
        phase_count: int = 1,
        log_approx_mode: str = "exact",
        log_offset: float = 1e-6,
        log_input_clamp_min: float = 1e-12,
        log_pwl_breakpoints: list[float] | None = None,
        log_pwl_slopes: list[float] | None = None,
        log_pwl_intercepts: list[float] | None = None,
        log_pwl_num_segments: int = 6,
        log_pwl_strategy: str = "uniform_logx",
        log_pwl_gamma: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.frame_length = frame_length
        self.frame_hop = frame_hop
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2
        self.spacing = spacing
        self.kernel_size = kernel_size
        self.phase_count = phase_count
        self.log_approx_mode = log_approx_mode
        self.log_offset = log_offset
        self.log_input_clamp_min = log_input_clamp_min

        if frame_length <= 0 or frame_hop <= 0:
            raise ValueError(f"frame_length and frame_hop must be >0, got {frame_length}, {frame_hop}")
        if phase_count <= 0:
            raise ValueError(f"phase_count must be > 0, got {phase_count}")
        if self.log_approx_mode not in {"exact", "pwl"}:
            raise ValueError(f"Unsupported log_approx_mode: {self.log_approx_mode}")

        kernels, edges = create_fir_bandpass_filterbank(
            sample_rate=sample_rate,
            n_bands=n_bands,
            f_min=f_min,
            f_max=self.f_max,
            spacing=spacing,
            kernel_size=kernel_size,
        )
        self.register_buffer("bandpass_kernels", kernels, persistent=False)
        self.register_buffer("band_edges", edges, persistent=False)

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

            self.register_buffer("log_pwl_breakpoints", torch.tensor(log_pwl_breakpoints, dtype=torch.float32), persistent=False)
            self.register_buffer("log_pwl_slopes", torch.tensor(log_pwl_slopes, dtype=torch.float32), persistent=False)
            self.register_buffer("log_pwl_intercepts", torch.tensor(log_pwl_intercepts, dtype=torch.float32), persistent=False)

    def _log_transform(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x + self.log_offset, min=self.log_input_clamp_min)
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

        raw_len = x.size(-1)
        target_time_steps = raw_len // self.frame_hop + 1
        x = x.unsqueeze(1)
        kernels = self.bandpass_kernels.to(device=x.device, dtype=x.dtype)

        phase_powers: list[torch.Tensor] = []
        for p in range(self.phase_count):
            offset = int(round(p * self.frame_hop / self.phase_count))
            x_phase = x[..., offset:]
            if x_phase.size(-1) < self.kernel_size:
                x_phase = F.pad(x_phase, (0, self.kernel_size - x_phase.size(-1)), mode="constant", value=0.0)

            # 低算力路径：以 frame_hop 作为 stride，无 padding。
            y = F.conv1d(x_phase, kernels, stride=self.frame_hop, padding=0)
            power = y.pow(2)
            if power.size(-1) != target_time_steps:
                power = F.adaptive_avg_pool1d(power, output_size=target_time_steps)
            phase_powers.append(power)

        band_energy = torch.stack(phase_powers, dim=0).mean(dim=0)
        return self._log_transform(band_energy)
