from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .mfcc_torch import apply_piecewise_linear, create_dct_matrix, create_mel_filterbank
from .pwl_fit_utils import fit_piecewise_linear_log_from_samples


def _qrange(bits: int, signed: bool) -> tuple[int, int]:
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def _quantize_unsigned_unit(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    qmin, qmax = _qrange(bits, signed=False)
    scale = 1.0 / float(qmax)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q, q * scale, scale


def _quantize_signed_fractional(x: torch.Tensor, bits: int, frac_bits: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    qmin, qmax = _qrange(bits, signed=True)
    scale = 1.0 / float(1 << frac_bits)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q, q * scale, scale


BIT_ACCURATE_STAGE_BIT_ALIASES: dict[str, str] = {
    "PCM_W": "pcm",
    "PCM": "pcm",
    "MFCC_SAMPLE_W": "preemphasis",
    "PREEMPHASIS_W": "preemphasis",
    "PREEMPHASIS": "preemphasis",
    "HANN_COEFF_W": "hann_coeff",
    "HANN_COEFF": "hann_coeff",
    "FFT_IN_W": "windowed",
    "WINDOWED_W": "windowed",
    "WINDOWED": "windowed",
    "TWIDDLE_W": "twiddle_coeff",
    "TWIDDLE_COEFF_W": "twiddle_coeff",
    "TWIDDLE_COEFF": "twiddle_coeff",
    "FFT_DATA_W": "fft_data",
    "FFT_DATA": "fft_data",
    "POWER_W": "power",
    "POWER": "power",
    "MEL_ACC_W": "mel",
    "MEL_W": "mel",
    "MEL": "mel",
    "PWL_IN_W": "pwl_input",
    "PWL_INPUT_W": "pwl_input",
    "PWL_INPUT": "pwl_input",
    "LOG_W": "log_mel",
    "LOG_MEL_W": "log_mel",
    "LOG_MEL": "log_mel",
    "DCT_COEFF_W": "dct_coeff",
    "DCT_COEFF": "dct_coeff",
    "DCT_ACC_W": "dct",
    "DCT_W": "dct",
    "DCT": "dct",
    "MFCC_OUT_W": "mfcc",
    "MFCC_W": "mfcc",
    "MFCC": "mfcc",
}


@dataclass
class BitAccurateStageQuantConfig:
    enabled: bool = False
    bits: int = 8
    signed: bool = True
    scale: float | list[float] | None = None
    per_channel: bool = False
    channel_dim: int = 1
    name: str = ""

    @property
    def qmin(self) -> int:
        return _qrange(self.bits, self.signed)[0]

    @property
    def qmax(self) -> int:
        return _qrange(self.bits, self.signed)[1]

    @property
    def nonnegative(self) -> bool:
        return not self.signed

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BitAccurateStageQuantConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            bits=int(data.get("bits", 8)),
            signed=bool(data.get("signed", True)),
            scale=data.get("scale"),
            per_channel=bool(data.get("per_channel", False)),
            channel_dim=int(data.get("channel_dim", 1)),
            name=str(data.get("name", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_stage_quant() -> dict[str, BitAccurateStageQuantConfig]:
    return {
        "pcm": BitAccurateStageQuantConfig(enabled=True, bits=8, signed=True, name="pcm"),
        "preemphasis": BitAccurateStageQuantConfig(enabled=False, bits=12, signed=True, name="preemphasis"),
        "hann_coeff": BitAccurateStageQuantConfig(enabled=False, bits=16, signed=False, name="hann_coeff"),
        "windowed": BitAccurateStageQuantConfig(enabled=False, bits=18, signed=True, name="windowed"),
        "twiddle_coeff": BitAccurateStageQuantConfig(enabled=False, bits=16, signed=True, name="twiddle_coeff"),
        "fft_data": BitAccurateStageQuantConfig(enabled=False, bits=20, signed=True, name="fft_data"),
        "power": BitAccurateStageQuantConfig(enabled=False, bits=41, signed=False, name="power"),
        "mel": BitAccurateStageQuantConfig(enabled=False, bits=46, signed=False, name="mel"),
        "pwl_input": BitAccurateStageQuantConfig(enabled=False, bits=32, signed=False, name="pwl_input"),
        "log_mel": BitAccurateStageQuantConfig(enabled=False, bits=24, signed=True, name="log_mel"),
        "dct_coeff": BitAccurateStageQuantConfig(enabled=False, bits=8, signed=True, name="dct_coeff"),
        "dct": BitAccurateStageQuantConfig(enabled=False, bits=40, signed=True, name="dct"),
        "mfcc": BitAccurateStageQuantConfig(
            enabled=True,
            bits=8,
            signed=True,
            per_channel=True,
            channel_dim=1,
            name="mfcc",
        ),
    }


def normalize_stage_bit_name(name: str) -> str:
    key = str(name).strip()
    if not key:
        raise ValueError("Empty stage bit-width name.")
    lower = key.lower()
    default_names = set(_default_stage_quant())
    if lower in default_names:
        return lower
    upper = key.upper()
    if upper in BIT_ACCURATE_STAGE_BIT_ALIASES:
        return BIT_ACCURATE_STAGE_BIT_ALIASES[upper]
    supported = sorted(set(BIT_ACCURATE_STAGE_BIT_ALIASES) | default_names)
    raise ValueError(f"Unsupported bit-accurate MFCC stage bit-width name: {name}. Supported: {supported}")


def normalize_stage_bit_overrides(overrides: Mapping[str, int] | None) -> dict[str, int]:
    if not overrides:
        return {}
    normalized: dict[str, int] = {}
    for name, bits in overrides.items():
        stage = normalize_stage_bit_name(name)
        value = int(bits)
        if value <= 0:
            raise ValueError(f"Stage bit width must be positive: {name}={bits}")
        normalized[stage] = value
    return normalized


def _refresh_fixed_point_scales(stage_quant: dict[str, BitAccurateStageQuantConfig]) -> None:
    stage_quant["hann_coeff"].scale = 1.0 / float((1 << stage_quant["hann_coeff"].bits) - 1)
    stage_quant["twiddle_coeff"].scale = 1.0 / float(1 << (stage_quant["twiddle_coeff"].bits - 1))
    stage_quant["dct_coeff"].scale = 1.0 / float(1 << (stage_quant["dct_coeff"].bits - 1))


def apply_stage_bit_overrides(
    stage_quant: dict[str, BitAccurateStageQuantConfig],
    overrides: Mapping[str, int] | None,
) -> dict[str, int]:
    normalized = normalize_stage_bit_overrides(overrides)
    for stage, bits in normalized.items():
        stage_quant[stage].bits = int(bits)
    if normalized:
        _refresh_fixed_point_scales(stage_quant)
    return normalized


@dataclass
class BitAccurateMFCCConfig:
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 512
    n_mels: int = 40
    f_min: float = 20.0
    f_max: float | None = None
    center: bool = True
    dct_norm: str | None = "ortho"
    mel_filter_shape: str = "rectangular"
    frequency_transform: str = "torch_stft"
    pre_emphasis: bool = True
    pre_emphasis_coeff: float = 0.97
    log_approx_mode: str = "pwl"
    log_offset: float = 1e-6
    log_input_clamp_min: float = 1e-12
    log_pwl_num_segments: int = 8
    log_pwl_strategy: str = "uniform_logx"
    log_pwl_gamma: float = 1.0
    log_pwl_breakpoints: list[float] | None = None
    log_pwl_slopes: list[float] | None = None
    log_pwl_intercepts: list[float] | None = None
    constraint_profile: str = "upper_bound_s8_mfcc"
    stage_quant: dict[str, BitAccurateStageQuantConfig] = field(default_factory=_default_stage_quant)

    def __post_init__(self) -> None:
        if self.mel_filter_shape != "rectangular":
            raise ValueError("Bit-accurate MFCC design requires rectangular Mel filtering.")
        if self.log_approx_mode != "pwl":
            raise ValueError("Bit-accurate MFCC design requires PWL log; exact log is not allowed.")
        if self.frequency_transform not in {"torch_stft", "quantized_dft"}:
            raise ValueError(f"Unsupported frequency_transform: {self.frequency_transform}")
        normalized: dict[str, BitAccurateStageQuantConfig] = _default_stage_quant()
        for name, spec in self.stage_quant.items():
            normalized[name] = spec if isinstance(spec, BitAccurateStageQuantConfig) else BitAccurateStageQuantConfig.from_dict(spec)
            normalized[name].name = name
        self.stage_quant = normalized

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BitAccurateMFCCConfig":
        payload = dict(data)
        stage_quant = payload.get("stage_quant", {})
        payload["stage_quant"] = {
            name: BitAccurateStageQuantConfig.from_dict(value)
            for name, value in stage_quant.items()
        }
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage_quant"] = {
            name: spec.to_dict()
            for name, spec in self.stage_quant.items()
        }
        return payload


def make_bit_accurate_mfcc_config(
    *,
    constraint_profile: str = "upper_bound_s8_mfcc",
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 512,
    win_length: int | None = None,
    hop_length: int | None = None,
    n_mels: int = 40,
    f_min: float = 20.0,
    f_max: float | None = None,
    pre_emphasis: bool = True,
    pre_emphasis_coeff: float = 0.97,
    log_pwl_num_segments: int = 8,
    log_pwl_strategy: str = "uniform_logx",
    log_pwl_gamma: float = 1.0,
    log_offset: float = 1e-6,
    log_input_clamp_min: float = 1e-12,
    mfcc_scale: float | list[float] | None = None,
    mfcc_per_channel: bool = True,
    stage_bit_overrides: Mapping[str, int] | None = None,
) -> BitAccurateMFCCConfig:
    win_length = n_fft if win_length is None else int(win_length)
    hop_length = n_fft if hop_length is None else int(hop_length)
    stage_quant = _default_stage_quant()
    stage_quant["mfcc"].scale = mfcc_scale
    stage_quant["mfcc"].per_channel = bool(mfcc_per_channel)
    _refresh_fixed_point_scales(stage_quant)
    frequency_transform = "torch_stft"

    if constraint_profile == "upper_bound_s8_mfcc":
        stage_quant["pcm"].enabled = True
        stage_quant["mfcc"].enabled = True
    elif constraint_profile in {"frontend_fakequant", "wide_bit_accurate"}:
        for name in ("pcm", "pwl_input", "log_mel", "dct", "mfcc"):
            stage_quant[name].enabled = True
    elif constraint_profile == "hardware_baseline":
        for spec in stage_quant.values():
            spec.enabled = True
        for name in ("hann_coeff", "twiddle_coeff", "fft_data", "dct_coeff"):
            stage_quant[name].enabled = False
    elif constraint_profile == "hardware_coeff_baseline":
        for spec in stage_quant.values():
            spec.enabled = True
        frequency_transform = "quantized_dft"
    else:
        raise ValueError(f"Unsupported constraint_profile: {constraint_profile}")

    normalized_stage_bit_overrides = apply_stage_bit_overrides(stage_quant, stage_bit_overrides)

    return BitAccurateMFCCConfig(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        frequency_transform=frequency_transform,
        pre_emphasis=pre_emphasis,
        pre_emphasis_coeff=pre_emphasis_coeff,
        log_pwl_num_segments=log_pwl_num_segments,
        log_pwl_strategy=log_pwl_strategy,
        log_pwl_gamma=log_pwl_gamma,
        log_offset=log_offset,
        log_input_clamp_min=log_input_clamp_min,
        constraint_profile=(
            constraint_profile
            if not normalized_stage_bit_overrides
            else f"{constraint_profile}+stage_bit_overrides"
        ),
        stage_quant=stage_quant,
    )


class BitAccurateMFCCHighPrecisionFrontend(nn.Module):
    """Accuracy-upper-bound frontend: rectangular Mel, PWL log, S8 MFCC output.

    This module intentionally keeps the heavy signal-processing stages in float
    unless their stage quantizer is enabled in the config. Quantized stages use
    an STE dequantized value during training and a hard dequantized value during
    inference. The final output remains float because the PyTorch quantized
    backbone consumes float tensors at its QuantStub boundary.
    """

    def __init__(
        self,
        config: BitAccurateMFCCConfig | dict[str, Any] | None = None,
        *,
        observer_enabled: bool = True,
        ste: bool = True,
    ):
        super().__init__()
        if config is None:
            config = make_bit_accurate_mfcc_config()
        if isinstance(config, dict):
            config = BitAccurateMFCCConfig.from_dict(config)
        self.config = config
        self.observer_enabled = bool(observer_enabled)
        self.ste = bool(ste)
        self.observed_absmax: dict[str, torch.Tensor] = {}
        self.stage_stats: dict[str, dict[str, float]] = {}

        f_max = config.f_max if config.f_max is not None else config.sample_rate / 2
        window = torch.hann_window(config.win_length)
        if config.win_length != config.n_fft:
            padded_window = torch.zeros(config.n_fft)
            padded_window[: config.win_length] = window
            window = padded_window
        mel_fb = create_mel_filterbank(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=f_max,
            filter_shape=config.mel_filter_shape,
        )
        dct_mat = create_dct_matrix(config.n_mfcc, config.n_mels, norm=config.dct_norm)
        n = torch.arange(config.n_fft, dtype=torch.float32)
        k = torch.arange(config.n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        angle = 2.0 * math.pi * k * n / float(config.n_fft)
        dft_cos = torch.cos(angle).t().contiguous()
        dft_sin = (-torch.sin(angle)).t().contiguous()

        hann_q, hann_qdq, _ = _quantize_unsigned_unit(window, config.stage_quant["hann_coeff"].bits)
        dft_cos_q, dft_cos_qdq, _ = _quantize_signed_fractional(
            dft_cos,
            config.stage_quant["twiddle_coeff"].bits,
            config.stage_quant["twiddle_coeff"].bits - 1,
        )
        dft_sin_q, dft_sin_qdq, _ = _quantize_signed_fractional(
            dft_sin,
            config.stage_quant["twiddle_coeff"].bits,
            config.stage_quant["twiddle_coeff"].bits - 1,
        )
        dct_q, dct_qdq, _ = _quantize_signed_fractional(
            dct_mat,
            config.stage_quant["dct_coeff"].bits,
            config.stage_quant["dct_coeff"].bits - 1,
        )

        self.register_buffer("window", window, persistent=False)
        self.register_buffer("hann_coeff_q", hann_q, persistent=False)
        self.register_buffer("hann_coeff_qdq", hann_qdq, persistent=False)
        self.register_buffer("dft_cos_q", dft_cos_q, persistent=False)
        self.register_buffer("dft_sin_q", dft_sin_q, persistent=False)
        self.register_buffer("dft_cos_qdq", dft_cos_qdq, persistent=False)
        self.register_buffer("dft_sin_qdq", dft_sin_qdq, persistent=False)
        self.register_buffer("mel_fb", mel_fb, persistent=False)
        self.register_buffer("dct_mat", dct_mat, persistent=False)
        self.register_buffer("dct_mat_q", dct_q, persistent=False)
        self.register_buffer("dct_mat_qdq", dct_qdq, persistent=False)

        if config.log_pwl_breakpoints is None or config.log_pwl_slopes is None or config.log_pwl_intercepts is None:
            fit = fit_piecewise_linear_log_from_samples(
                x_samples=torch.logspace(-8, 2, steps=30000),
                num_segments=config.log_pwl_num_segments,
                strategy=config.log_pwl_strategy,
                gamma=config.log_pwl_gamma,
            )
            config.log_pwl_breakpoints = [float(v) for v in fit["breakpoints"]]
            config.log_pwl_slopes = [float(v) for v in fit["slopes"]]
            config.log_pwl_intercepts = [float(v) for v in fit["intercepts"]]

        self.register_buffer("log_pwl_breakpoints", torch.tensor(config.log_pwl_breakpoints, dtype=torch.float32), persistent=False)
        self.register_buffer("log_pwl_slopes", torch.tensor(config.log_pwl_slopes, dtype=torch.float32), persistent=False)
        self.register_buffer("log_pwl_intercepts", torch.tensor(config.log_pwl_intercepts, dtype=torch.float32), persistent=False)

    @classmethod
    def from_spec_json(cls, path: str | Path, **kwargs) -> "BitAccurateMFCCHighPrecisionFrontend":
        payload = _read_spec_json(path)
        kwargs.setdefault("config", payload["config"])
        kwargs.setdefault("observer_enabled", False)
        return cls(**kwargs)

    def set_observer_enabled(self, enabled: bool) -> None:
        self.observer_enabled = bool(enabled)

    def reset_observers(self) -> None:
        self.observed_absmax.clear()
        self.stage_stats.clear()

    def _record_stats(self, name: str, x: torch.Tensor, q: torch.Tensor | None = None, spec: BitAccurateStageQuantConfig | None = None) -> None:
        with torch.no_grad():
            detached = x.detach()
            stats = {
                "min": float(detached.min().cpu().item()),
                "max": float(detached.max().cpu().item()),
                "mean": float(detached.mean().cpu().item()),
                "std": float(detached.std(unbiased=False).cpu().item()),
                "zero_ratio": float((detached == 0).to(torch.float32).mean().cpu().item()),
            }
            if q is not None and spec is not None:
                sat = ((q <= spec.qmin) | (q >= spec.qmax)).to(torch.float32).mean()
                stats["saturation_ratio"] = float(sat.cpu().item())
            self.stage_stats[name] = stats

    def _record_coeff_stats(
        self,
        name: str,
        original: torch.Tensor,
        q: torch.Tensor,
        dequantized: torch.Tensor,
    ) -> None:
        spec = self.config.stage_quant[name]
        self._record_stats(name, original, q=q, spec=spec)
        with torch.no_grad():
            err = (dequantized.detach() - original.detach()).to(torch.float32)
            self.stage_stats[name].update(
                {
                    "quant_mae": float(err.abs().mean().cpu().item()),
                    "quant_rmse": float(torch.sqrt(torch.mean(err * err)).cpu().item()),
                    "quant_max_abs_error": float(err.abs().max().cpu().item()),
                }
            )

    def _observed_for_scale(self, name: str, x: torch.Tensor, spec: BitAccurateStageQuantConfig) -> torch.Tensor:
        detached = x.detach()
        if spec.nonnegative:
            value = detached.clamp_min(0).amax(dim=self._reduction_dims(detached, spec)) if spec.per_channel else detached.clamp_min(0).amax()
        else:
            value = detached.abs().amax(dim=self._reduction_dims(detached, spec)) if spec.per_channel else detached.abs().amax()
        if self.observer_enabled:
            old = self.observed_absmax.get(name)
            self.observed_absmax[name] = value if old is None else torch.maximum(old.to(value.device), value)
            return self.observed_absmax[name]
        old = self.observed_absmax.get(name)
        return old.to(value.device) if old is not None else value

    @staticmethod
    def _reduction_dims(x: torch.Tensor, spec: BitAccurateStageQuantConfig) -> tuple[int, ...]:
        channel_dim = spec.channel_dim if spec.channel_dim >= 0 else x.dim() + spec.channel_dim
        return tuple(idx for idx in range(x.dim()) if idx != channel_dim)

    @staticmethod
    def _scale_view(scale: torch.Tensor, x: torch.Tensor, spec: BitAccurateStageQuantConfig) -> torch.Tensor:
        if scale.dim() == 0:
            return scale
        channel_dim = spec.channel_dim if spec.channel_dim >= 0 else x.dim() + spec.channel_dim
        shape = [1] * x.dim()
        shape[channel_dim] = int(scale.numel())
        return scale.reshape(shape)

    def _resolve_scale(self, name: str, x: torch.Tensor, spec: BitAccurateStageQuantConfig) -> torch.Tensor:
        if spec.scale is not None:
            scale = torch.tensor(spec.scale, device=x.device, dtype=x.dtype)
        else:
            observed = self._observed_for_scale(name, x, spec).to(device=x.device, dtype=x.dtype)
            denom = float(max(1, spec.qmax if spec.signed else spec.qmax))
            scale = torch.clamp(observed / denom, min=torch.finfo(x.dtype).eps)
        return scale

    def _quant_dequant(self, name: str, x: torch.Tensor) -> torch.Tensor:
        spec = self.config.stage_quant[name]
        self._record_stats(name, x)
        if not spec.enabled:
            return x
        scale = self._resolve_scale(name, x, spec)
        scale_view = self._scale_view(scale, x, spec)
        q = torch.clamp(torch.round(x / scale_view), spec.qmin, spec.qmax)
        y = q * scale_view
        self._record_stats(name, x, q=q, spec=spec)
        if self.training and self.ste:
            return x + (y - x).detach()
        return y

    def _quant_dequant_complex_parts(
        self,
        name: str,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.config.stage_quant[name]
        combined = torch.cat([real.reshape(-1), imag.reshape(-1)])
        self._record_stats(name, combined)
        if not spec.enabled:
            return real, imag

        scale = self._resolve_scale(name, combined, spec)
        q_real = torch.clamp(torch.round(real / scale), spec.qmin, spec.qmax)
        q_imag = torch.clamp(torch.round(imag / scale), spec.qmin, spec.qmax)
        y_real = q_real * scale
        y_imag = q_imag * scale
        q_combined = torch.cat([q_real.reshape(-1), q_imag.reshape(-1)])
        self._record_stats(name, combined, q=q_combined, spec=spec)

        if self.training and self.ste:
            return real + (y_real - real).detach(), imag + (y_imag - imag).detach()
        return y_real, y_imag

    def _pre_emphasis(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.pre_emphasis:
            return x
        y = torch.empty_like(x)
        y[:, 0] = x[:, 0]
        y[:, 1:] = x[:, 1:] - self.config.pre_emphasis_coeff * x[:, :-1]
        return y

    def _pwl_log(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x + self.config.log_offset, min=self.config.log_input_clamp_min)
        x = self._quant_dequant("pwl_input", x)
        return apply_piecewise_linear(
            x,
            self.log_pwl_breakpoints.to(device=x.device, dtype=x.dtype),
            self.log_pwl_slopes.to(device=x.device, dtype=x.dtype),
            self.log_pwl_intercepts.to(device=x.device, dtype=x.dtype),
        )

    @staticmethod
    def _reflect_pad_1d(x: torch.Tensor, pad: int) -> torch.Tensor:
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

    def _quantized_dft(self, pre: torch.Tensor, stages: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.center:
            pre = self._reflect_pad_1d(pre, self.config.n_fft // 2)

        frames = pre.unfold(1, self.config.n_fft, self.config.hop_length)
        hann = self.hann_coeff_qdq.to(device=pre.device, dtype=pre.dtype)
        self._record_coeff_stats(
            "hann_coeff",
            self.window.to(device=pre.device, dtype=pre.dtype),
            self.hann_coeff_q.to(device=pre.device, dtype=pre.dtype),
            hann,
        )
        windowed = frames * hann.view(1, 1, -1)
        windowed = self._quant_dequant("windowed", windowed)
        stages["windowed"] = windowed

        cos = self.dft_cos_qdq.to(device=pre.device, dtype=pre.dtype)
        sin = self.dft_sin_qdq.to(device=pre.device, dtype=pre.dtype)
        twiddle_original = torch.cat(
            [
                torch.cos(2.0 * math.pi * torch.arange(self.config.n_fft, device=pre.device, dtype=pre.dtype).view(-1, 1)
                          * torch.arange(self.config.n_fft // 2 + 1, device=pre.device, dtype=pre.dtype).view(1, -1)
                          / float(self.config.n_fft)).reshape(-1),
                (-torch.sin(2.0 * math.pi * torch.arange(self.config.n_fft, device=pre.device, dtype=pre.dtype).view(-1, 1)
                            * torch.arange(self.config.n_fft // 2 + 1, device=pre.device, dtype=pre.dtype).view(1, -1)
                            / float(self.config.n_fft))).reshape(-1),
            ]
        )
        twiddle_q = torch.cat(
            [
                self.dft_cos_q.to(device=pre.device, dtype=pre.dtype).reshape(-1),
                self.dft_sin_q.to(device=pre.device, dtype=pre.dtype).reshape(-1),
            ]
        )
        twiddle_qdq = torch.cat([cos.reshape(-1), sin.reshape(-1)])
        self._record_coeff_stats("twiddle_coeff", twiddle_original, twiddle_q, twiddle_qdq)

        real = torch.matmul(windowed, cos)
        imag = torch.matmul(windowed, sin)
        real, imag = self._quant_dequant_complex_parts("fft_data", real, imag)
        stft = torch.complex(real.transpose(1, 2), imag.transpose(1, 2))
        return stft

    def forward_stages(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected x shape [B, T] or [B, 1, T], got {list(x.shape)}")
        x = x.to(torch.float32)

        stages: dict[str, torch.Tensor] = {"waveform": x}
        pcm = self._quant_dequant("pcm", x)
        stages["pcm_or_scaled_sample"] = pcm
        pre = self._quant_dequant("preemphasis", self._pre_emphasis(pcm))
        stages["preemphasis_or_sample_scale"] = pre

        if self.config.frequency_transform == "quantized_dft":
            stft = self._quantized_dft(pre, stages)
        else:
            window = self.window.to(device=x.device, dtype=x.dtype)
            stft = torch.stft(
                pre,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=window,
                center=self.config.center,
                return_complex=True,
            )
        stages["fft_complex"] = stft
        power = stft.real.pow(2) + stft.imag.pow(2)
        power = self._quant_dequant("power", power)
        stages["power"] = power

        mel_fb = self.mel_fb.to(device=x.device, dtype=x.dtype)
        mel = torch.matmul(mel_fb, power)
        mel = self._quant_dequant("mel", mel)
        stages["rectangular_mel"] = mel

        log_mel = self._pwl_log(mel)
        log_mel = self._quant_dequant("log_mel", log_mel)
        stages["pwl_log_mel"] = log_mel

        if self.config.stage_quant["dct_coeff"].enabled:
            dct_mat = self.dct_mat_qdq.to(device=x.device, dtype=x.dtype)
            self._record_coeff_stats(
                "dct_coeff",
                self.dct_mat.to(device=x.device, dtype=x.dtype),
                self.dct_mat_q.to(device=x.device, dtype=x.dtype),
                dct_mat,
            )
        else:
            dct_mat = self.dct_mat.to(device=x.device, dtype=x.dtype)
        mfcc_prequant = torch.matmul(dct_mat, log_mel)
        mfcc_prequant = self._quant_dequant("dct", mfcc_prequant)
        stages["dct_acc"] = mfcc_prequant
        stages["mfcc_prequant"] = mfcc_prequant
        mfcc = self._quant_dequant("mfcc", mfcc_prequant)
        stages["mfcc_int8"] = mfcc
        return stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(x)["mfcc_int8"]

    def resolved_config(self) -> BitAccurateMFCCConfig:
        cfg = copy.deepcopy(self.config)
        for name, spec in cfg.stage_quant.items():
            runtime = self.config.stage_quant[name]
            spec.scale = runtime.scale
            observed = self.observed_absmax.get(name)
            if runtime.scale is None and observed is not None and runtime.enabled:
                denom = float(max(1, runtime.qmax if runtime.signed else runtime.qmax))
                scale = torch.clamp(observed.detach().cpu() / denom, min=1e-12)
                spec.scale = [float(v) for v in scale.flatten().tolist()] if scale.dim() > 0 else float(scale.item())
        return cfg

    def export_spec_json(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frontend": self.__class__.__name__,
            "naming": "bit_accurate",
            "config": self.resolved_config().to_dict(),
            "observed_absmax": _tensor_dict_to_json(self.observed_absmax),
            "stage_stats": self.stage_stats,
        }
        if extra:
            payload["extra"] = extra
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tensor_dict_to_json(data: dict[str, torch.Tensor]) -> dict[str, float | list[float]]:
    out: dict[str, float | list[float]] = {}
    for name, value in data.items():
        cpu = value.detach().cpu()
        if cpu.dim() == 0:
            out[name] = float(cpu.item())
        else:
            out[name] = [float(v) for v in cpu.flatten().tolist()]
    return out


def _read_spec_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "config" not in payload:
        raise ValueError(f"Invalid bit-accurate MFCC spec, missing config: {path}")
    return payload
