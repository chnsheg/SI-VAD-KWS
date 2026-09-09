from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .mfcc_torch import apply_piecewise_linear, create_dct_matrix, create_mel_filterbank
from .pwl_fit_utils import fit_piecewise_linear_log_from_samples


@dataclass
class Int8StageSpec:
    scale: float | None = None
    qmin: int = -128
    qmax: int = 127
    nonnegative: bool = False

    def resolved_scale(self, observed_absmax: float, eps: float = 1e-12) -> float:
        if self.scale is not None and self.scale > 0:
            return float(self.scale)
        denom = self.qmax if self.nonnegative else max(abs(self.qmin), abs(self.qmax))
        return max(float(observed_absmax), eps) / float(denom)


@dataclass
class Int8MFCCScaleConfig:
    waveform: Int8StageSpec = field(default_factory=lambda: Int8StageSpec(scale=1.0 / 127.0))
    preemphasis: Int8StageSpec = field(default_factory=lambda: Int8StageSpec(scale=2.0 / 127.0))
    windowed: Int8StageSpec = field(default_factory=Int8StageSpec)
    power: Int8StageSpec = field(default_factory=lambda: Int8StageSpec(qmin=0, qmax=127, nonnegative=True))
    mel: Int8StageSpec = field(default_factory=lambda: Int8StageSpec(qmin=0, qmax=127, nonnegative=True))
    log_mel: Int8StageSpec = field(default_factory=Int8StageSpec)
    mfcc: Int8StageSpec = field(default_factory=Int8StageSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Int8MFCCScaleConfig":
        cfg = cls()
        for name, value in data.items():
            if not hasattr(cfg, name) or not isinstance(value, dict):
                continue
            setattr(
                cfg,
                name,
                Int8StageSpec(
                    scale=value.get("scale"),
                    qmin=int(value.get("qmin", 0 if value.get("nonnegative", False) else -128)),
                    qmax=int(value.get("qmax", 127)),
                    nonnegative=bool(value.get("nonnegative", False)),
                ),
            )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for name in self.__dataclass_fields__:
            spec = getattr(self, name)
            out[name] = {
                "scale": spec.scale,
                "qmin": spec.qmin,
                "qmax": spec.qmax,
                "nonnegative": spec.nonnegative,
            }
        return out


def _quantize_symmetric_coeff(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float]:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    amax = float(tensor.detach().abs().max().item())
    scale = max(amax, 1e-12) / float(qmax)
    q = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int32)
    return q, scale


def _scale_from_absmax(spec: Int8StageSpec, observed_absmax: float, eps: float = 1e-12) -> float:
    denom = spec.qmax if spec.nonnegative else max(abs(spec.qmin), abs(spec.qmax))
    return max(float(observed_absmax), eps) / float(denom)


def _reflect_pad_1d_int(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    if x.size(1) <= 1:
        return torch.nn.functional.pad(x, (pad, pad), mode="constant", value=0)
    left_count = min(pad, x.size(1) - 1)
    right_count = min(pad, x.size(1) - 1)
    left = x[:, 1 : left_count + 1].flip(1)
    right = x[:, -right_count - 1 : -1].flip(1)
    if left_count < pad:
        left = torch.nn.functional.pad(left, (pad - left_count, 0), mode="constant", value=0)
    if right_count < pad:
        right = torch.nn.functional.pad(right, (0, pad - right_count), mode="constant", value=0)
    return torch.cat([left, x, right], dim=1)


class Int8MFCCFrontend(nn.Module):
    """Integer-aware INT8 MFCC frontend for inference/calibration.

    This module quantizes coefficients and stage activations to INT8 grids.
    DFT, Mel, and DCT matrix products use integer tensors with int64
    accumulation in PyTorch, then requantize at stage boundaries. The log stage
    is evaluated on the dequantized Mel energy and requantized afterwards.

    It is intended as a reliable Python reference for fixed quantization rules,
    not as a fast training frontend.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 512,
        win_length: int = 512,
        hop_length: int = 512,
        n_mels: int = 40,
        f_min: float = 20.0,
        f_max: float | None = None,
        center: bool = True,
        dct_norm: str | None = "ortho",
        mel_filter_shape: str = "triangular",
        pre_emphasis: bool = True,
        pre_emphasis_coeff: float = 0.97,
        mel_log_mode: str = "torchaudio_db",
        log_offset: float = 1e-6,
        log_approx_mode: str = "pwl",
        log_pwl_num_segments: int = 8,
        log_pwl_strategy: str = "uniform_logx",
        log_pwl_gamma: float = 1.0,
        log_input_clamp_min: float = 1e-12,
        db_amin: float = 1e-10,
        top_db: float | None = 80.0,
        coeff_bits: int = 8,
        requantize_power: bool = False,
        requantize_mel: bool = False,
        scales: Int8MFCCScaleConfig | dict[str, Any] | None = None,
        observer_enabled: bool = False,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_mfcc = int(n_mfcc)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max if f_max is not None else sample_rate / 2)
        self.center = bool(center)
        self.dct_norm = dct_norm
        self.mel_filter_shape = mel_filter_shape
        self.pre_emphasis = bool(pre_emphasis)
        self.pre_emphasis_coeff = float(pre_emphasis_coeff)
        self.mel_log_mode = mel_log_mode
        self.log_offset = float(log_offset)
        self.log_approx_mode = log_approx_mode
        self.log_pwl_num_segments = int(log_pwl_num_segments)
        self.log_pwl_strategy = log_pwl_strategy
        self.log_pwl_gamma = float(log_pwl_gamma)
        self.log_input_clamp_min = float(log_input_clamp_min)
        self.db_amin = float(db_amin)
        self.top_db = None if top_db is None else float(top_db)
        self.coeff_bits = int(coeff_bits)
        self.requantize_power = bool(requantize_power)
        self.requantize_mel = bool(requantize_mel)
        self.observer_enabled = bool(observer_enabled)
        self.observed_absmax: dict[str, float] = {}

        if self.mel_log_mode not in {"natural_log", "torchaudio_db"}:
            raise ValueError(f"Unsupported mel_log_mode: {self.mel_log_mode}")

        if isinstance(scales, dict):
            self.scale_config = Int8MFCCScaleConfig.from_dict(scales)
        elif scales is None:
            self.scale_config = Int8MFCCScaleConfig()
        else:
            self.scale_config = scales
        self._fixed_scale_stages = {
            name
            for name in self.scale_config.__dataclass_fields__
            if getattr(self.scale_config, name).scale is not None
        }

        window = torch.hann_window(self.win_length)
        if self.win_length != self.n_fft:
            padded = torch.zeros(self.n_fft)
            padded[: self.win_length] = window
            window = padded
        mel_fb = create_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            filter_shape=mel_filter_shape,
        )
        dct_mat = create_dct_matrix(n_mfcc=self.n_mfcc, n_mels=self.n_mels, norm=dct_norm)

        n = torch.arange(self.n_fft, dtype=torch.float32)
        k = torch.arange(self.n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        angle = 2.0 * math.pi * k * n / float(self.n_fft)
        cos_mat = torch.cos(angle)
        neg_sin_mat = -torch.sin(angle)

        window_q, self.window_scale = _quantize_symmetric_coeff(window, bits=self.coeff_bits)
        cos_q, self.dft_cos_scale = _quantize_symmetric_coeff(cos_mat, bits=self.coeff_bits)
        sin_q, self.dft_sin_scale = _quantize_symmetric_coeff(neg_sin_mat, bits=self.coeff_bits)
        mel_q, self.mel_coeff_scale = _quantize_symmetric_coeff(mel_fb, bits=self.coeff_bits)
        dct_q, self.dct_coeff_scale = _quantize_symmetric_coeff(dct_mat, bits=self.coeff_bits)

        self.register_buffer("window_q", window_q, persistent=False)
        self.register_buffer("dft_cos_q", cos_q, persistent=False)
        self.register_buffer("dft_sin_q", sin_q, persistent=False)
        self.register_buffer("mel_fb_q", mel_q, persistent=False)
        self.register_buffer("dct_mat_q", dct_q, persistent=False)

        if self.log_approx_mode not in {"exact", "pwl"}:
            raise ValueError(f"Unsupported log_approx_mode: {self.log_approx_mode}")
        if self.log_approx_mode == "pwl":
            fit = fit_piecewise_linear_log_from_samples(
                x_samples=torch.logspace(-8, 2, steps=30000),
                num_segments=log_pwl_num_segments,
                strategy=log_pwl_strategy,
                gamma=log_pwl_gamma,
            )
            self.register_buffer("log_pwl_breakpoints", torch.tensor(fit["breakpoints"], dtype=torch.float32), persistent=False)
            self.register_buffer("log_pwl_slopes", torch.tensor(fit["slopes"], dtype=torch.float32), persistent=False)
            self.register_buffer("log_pwl_intercepts", torch.tensor(fit["intercepts"], dtype=torch.float32), persistent=False)

    @classmethod
    def from_scale_json(cls, path: str | Path, **kwargs) -> "Int8MFCCFrontend":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for key in (
            "sample_rate",
            "n_mfcc",
            "n_fft",
            "win_length",
            "hop_length",
            "n_mels",
            "f_min",
            "f_max",
            "center",
            "dct_norm",
            "mel_filter_shape",
            "pre_emphasis",
            "pre_emphasis_coeff",
            "mel_log_mode",
            "log_approx_mode",
            "log_pwl_num_segments",
            "log_pwl_strategy",
            "log_pwl_gamma",
            "log_offset",
            "log_input_clamp_min",
            "db_amin",
            "top_db",
            "coeff_bits",
            "requantize_power",
            "requantize_mel",
        ):
            if key in payload:
                kwargs.setdefault(key, payload[key])
        kwargs.setdefault("scales", payload["scales"])
        kwargs.setdefault("observer_enabled", False)
        return cls(**kwargs)

    def export_scale_json(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "frontend": "Int8MFCCFrontend",
            "coeff_bits": self.coeff_bits,
            "sample_rate": self.sample_rate,
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "center": self.center,
            "dct_norm": self.dct_norm,
            "mel_filter_shape": self.mel_filter_shape,
            "pre_emphasis": self.pre_emphasis,
            "pre_emphasis_coeff": self.pre_emphasis_coeff,
            "mel_log_mode": self.mel_log_mode,
            "log_approx_mode": self.log_approx_mode,
            "log_pwl_num_segments": self.log_pwl_num_segments,
            "log_pwl_strategy": self.log_pwl_strategy,
            "log_pwl_gamma": self.log_pwl_gamma,
            "log_offset": self.log_offset,
            "log_input_clamp_min": self.log_input_clamp_min,
            "db_amin": self.db_amin,
            "top_db": self.top_db,
            "requantize_power": self.requantize_power,
            "requantize_mel": self.requantize_mel,
            "observed_absmax": self.observed_absmax,
            "coefficient_scales": {
                "window": self.window_scale,
                "dft_cos": self.dft_cos_scale,
                "dft_sin": self.dft_sin_scale,
                "mel": self.mel_coeff_scale,
                "dct": self.dct_coeff_scale,
            },
            "scales": self.scale_config.to_dict(),
        }
        if extra:
            payload["extra"] = extra
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _spec(self, stage: str) -> Int8StageSpec:
        return getattr(self.scale_config, stage)

    def _observe_absmax(self, x: torch.Tensor, stage: str) -> float:
        spec = self._spec(stage)
        if spec.nonnegative:
            observed = float(x.detach().max().clamp_min(0).cpu().item())
        else:
            observed = float(x.detach().abs().max().cpu().item())
        if self.observer_enabled:
            self.observed_absmax[stage] = max(self.observed_absmax.get(stage, 0.0), observed)
            if stage not in self._fixed_scale_stages:
                spec.scale = _scale_from_absmax(spec, self.observed_absmax[stage])
        return observed

    def _quantize(self, x: torch.Tensor, stage: str) -> tuple[torch.Tensor, float]:
        observed = self._observe_absmax(x, stage)
        spec = self._spec(stage)
        scale = spec.resolved_scale(observed)
        q = torch.clamp(torch.round(x / scale), spec.qmin, spec.qmax).to(torch.int32)
        return q, scale

    def _log_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mel_log_mode == "torchaudio_db":
            x = torch.clamp(x, min=self.db_amin)
        else:
            x = torch.clamp(x + self.log_offset, min=self.log_input_clamp_min)

        if self.log_approx_mode == "exact":
            log_x = torch.log(x)
        else:
            log_x = apply_piecewise_linear(
                x,
                self.log_pwl_breakpoints.to(device=x.device, dtype=x.dtype),
                self.log_pwl_slopes.to(device=x.device, dtype=x.dtype),
                self.log_pwl_intercepts.to(device=x.device, dtype=x.dtype),
            )

        if self.mel_log_mode == "torchaudio_db":
            db = log_x * (10.0 / math.log(10.0))
            if self.top_db is not None:
                reduce_dims = tuple(range(1, db.dim()))
                cutoff = db.amax(dim=reduce_dims, keepdim=True) - self.top_db
                db = torch.maximum(db, cutoff)
            return db
        return log_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected x shape [B, T] or [B, 1, T], got {list(x.shape)}")

        x_q, x_scale = self._quantize(x.to(torch.float32), "waveform")
        if self.pre_emphasis:
            x_deq = x_q.to(torch.float32) * x_scale
            y = torch.empty_like(x_deq)
            y[:, 0] = x_deq[:, 0]
            y[:, 1:] = x_deq[:, 1:] - self.pre_emphasis_coeff * x_deq[:, :-1]
            pre_q, pre_scale = self._quantize(y, "preemphasis")
        else:
            pre_q, pre_scale = x_q, x_scale

        if self.center:
            pre_q = _reflect_pad_1d_int(pre_q, self.n_fft // 2)

        frames_q = pre_q.unfold(1, self.n_fft, self.hop_length)
        windowed_acc = frames_q.to(torch.int64) * self.window_q.to(device=x.device).view(1, 1, -1).to(torch.int64)
        windowed_float = windowed_acc.to(torch.float32) * float(pre_scale * self.window_scale)
        windowed_q, windowed_scale = self._quantize(windowed_float, "windowed")

        cos_q = self.dft_cos_q.to(device=x.device).t().to(torch.int64)
        sin_q = self.dft_sin_q.to(device=x.device).t().to(torch.int64)
        real_acc = torch.matmul(windowed_q.to(torch.int64), cos_q)
        imag_acc = torch.matmul(windowed_q.to(torch.int64), sin_q)
        dft_scale = float(windowed_scale * self.dft_cos_scale)
        power_acc = real_acc * real_acc + imag_acc * imag_acc
        power_scale = dft_scale * dft_scale
        power_float = power_acc.to(torch.float32) * float(power_scale)

        if self.requantize_power:
            power_q, power_requant_scale = self._quantize(power_float, "power")
            mel_acc = torch.matmul(power_q.to(torch.int64), self.mel_fb_q.to(device=x.device).t().to(torch.int64))
            mel_float = mel_acc.to(torch.float32) * float(power_requant_scale * self.mel_coeff_scale)
        else:
            self._observe_absmax(power_float, "power")
            mel_acc = torch.matmul(power_acc.to(torch.int64), self.mel_fb_q.to(device=x.device).t().to(torch.int64))
            mel_float = mel_acc.to(torch.float32) * float(power_scale * self.mel_coeff_scale)

        if self.requantize_mel:
            mel_q, mel_scale = self._quantize(mel_float, "mel")
            mel_for_log = mel_q.to(torch.float32) * float(mel_scale)
        else:
            self._observe_absmax(mel_float, "mel")
            mel_for_log = mel_float

        log_mel_float = self._log_transform(mel_for_log)
        log_q, log_scale = self._quantize(log_mel_float, "log_mel")

        mfcc_acc = torch.matmul(log_q.to(torch.int64), self.dct_mat_q.to(device=x.device).t().to(torch.int64))
        mfcc_float = mfcc_acc.to(torch.float32) * float(log_scale * self.dct_coeff_scale)
        mfcc_float = mfcc_float.transpose(1, 2)
        mfcc_q, mfcc_scale = self._quantize(mfcc_float, "mfcc")
        return mfcc_q.to(torch.float32) * float(mfcc_scale)
