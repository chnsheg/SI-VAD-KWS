from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .mfcc_torch import create_dct_matrix, create_mel_filterbank


def _qmin(bits: int, signed: bool) -> int:
    return -(1 << (bits - 1)) if signed else 0


def _qmax(bits: int, signed: bool) -> int:
    return (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1


def _saturate(x: torch.Tensor, bits: int, *, signed: bool) -> torch.Tensor:
    return torch.clamp(x, _qmin(bits, signed), _qmax(bits, signed))


def _round_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift <= 0:
        return x << (-shift)
    offset = 1 << (shift - 1)
    return torch.div(x + offset, 1 << shift, rounding_mode="floor")


def _round_shift_signed(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift <= 0:
        return x << (-shift)
    sign = torch.where(x < 0, -1, 1)
    mag = torch.abs(x)
    rounded = _round_shift(mag, shift)
    return rounded * sign


def _quantize_unsigned_q0(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = _qmax(bits, signed=False)
    return torch.clamp(torch.round(tensor * qmax), 0, qmax).to(torch.int64)


def _quantize_signed_q1_frac(tensor: torch.Tensor, bits: int, frac_bits: int) -> torch.Tensor:
    qmin = _qmin(bits, signed=True)
    qmax = _qmax(bits, signed=True)
    scale = float(1 << frac_bits)
    return torch.clamp(torch.round(tensor * scale), qmin, qmax).to(torch.int64)


@dataclass
class BitAccuracyMFCCConfig:
    pcm_w: int = 8
    mfcc_sample_w: int = 12
    hann_coeff_w: int = 16
    fft_in_w: int = 18
    fft_data_w: int = 20
    twiddle_w: int = 16
    power_w: int = 41
    mel_acc_w: int = 46
    pwl_in_w: int = 32
    log_w: int = 24
    dct_coeff_w: int = 8
    dct_acc_w: int = 40
    mfcc_out_w: int = 8
    sample_left_shift: int = 4
    window_shift: int = 10
    dft_shift: int = 24
    mel_drain_shift: int = 14
    log_frac_bits: int = 16
    dct_shift: int = 7
    mfcc_output_scale: float | None = None


class BitAccuracyMFCCFrontend(nn.Module):
    """Baseline bit-accurate-style MFCC frontend.

    This module follows ``README_MFCC_BitAccuracy_BASELINE.md``: S8 PCM,
    S12 internal samples, Q0.16 Hann, S18 windowed samples, S20 DFT output,
    U41 power, U46 Mel accumulator, U32 log input, S24 log, S8 DCT
    coefficients, S40 DCT accumulator, and S8 MFCC output.

    The first version uses a fixed-scaled integer DFT matrix as the Python
    reference for the FFT contract. It fixes the stage widths and rounding
    points first; a radix-2 FFT can later replace the DFT block while keeping
    the same input/output contract.
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
        mel_filter_shape: str = "rectangular",
        pre_emphasis: bool = True,
        pre_emphasis_coeff: float = 0.97,
        log_offset: float = 1e-6,
        log_input_clamp_min: float = 1e-12,
        config: BitAccuracyMFCCConfig | dict[str, Any] | None = None,
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
        self.log_offset = float(log_offset)
        self.log_input_clamp_min = float(log_input_clamp_min)
        self.observer_enabled = bool(observer_enabled)
        self.observed_absmax: dict[str, float] = {}
        if isinstance(config, dict):
            self.cfg = BitAccuracyMFCCConfig(**config)
        elif config is None:
            self.cfg = BitAccuracyMFCCConfig()
        else:
            self.cfg = config
        if self.mel_filter_shape != "rectangular":
            raise ValueError("BitAccuracyMFCCFrontend currently uses rectangular Mel filters only.")

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

        hann_q = _quantize_unsigned_q0(window, self.cfg.hann_coeff_w)
        cos_q = _quantize_signed_q1_frac(torch.cos(angle), self.cfg.twiddle_w, self.cfg.twiddle_w - 1)
        sin_q = _quantize_signed_q1_frac(-torch.sin(angle), self.cfg.twiddle_w, self.cfg.twiddle_w - 1)
        mel_q = (mel_fb > 0).to(torch.int64)
        self.mel_coeff_frac_bits = 0
        dct_q = _quantize_signed_q1_frac(dct_mat, self.cfg.dct_coeff_w, self.cfg.dct_coeff_w - 1)
        alpha_q = int(round(self.pre_emphasis_coeff * float(1 << 16)))

        self.register_buffer("hann_q", hann_q.to(torch.int64), persistent=False)
        self.register_buffer("dft_cos_q", cos_q.to(torch.int64), persistent=False)
        self.register_buffer("dft_sin_q", sin_q.to(torch.int64), persistent=False)
        self.register_buffer("mel_fb_q", mel_q.to(torch.int64), persistent=False)
        self.register_buffer("dct_mat_q", dct_q.to(torch.int64), persistent=False)
        self.register_buffer("alpha_q", torch.tensor(alpha_q, dtype=torch.int64), persistent=False)

    @classmethod
    def from_config_json(cls, path: str | Path, **kwargs: Any) -> "BitAccuracyMFCCFrontend":
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
            "log_offset",
            "log_input_clamp_min",
        ):
            if key in payload:
                kwargs.setdefault(key, payload[key])
        kwargs.setdefault("config", payload["config"])
        kwargs.setdefault("observer_enabled", False)
        return cls(**kwargs)

    def export_config_json(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "frontend": "BitAccuracyMFCCFrontend",
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
            "log_offset": self.log_offset,
            "log_input_clamp_min": self.log_input_clamp_min,
            "mel_coeff_frac_bits": self.mel_coeff_frac_bits,
            "observed_absmax": self.observed_absmax,
            "config": asdict(self.cfg),
        }
        if extra:
            payload["extra"] = extra
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _observe(self, name: str, x: torch.Tensor) -> None:
        if not self.observer_enabled:
            return
        value = float(x.detach().abs().max().cpu().item()) if x.numel() else 0.0
        self.observed_absmax[name] = max(self.observed_absmax.get(name, 0.0), value)
        if name == "mfcc_float" and self.cfg.mfcc_output_scale is None:
            self.cfg.mfcc_output_scale = max(value, 1e-12) / float(_qmax(self.cfg.mfcc_out_w, signed=True))

    def _reflect_pad_1d(self, x: torch.Tensor, pad: int) -> torch.Tensor:
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

    def _log_to_s24(self, mel_u32: torch.Tensor) -> torch.Tensor:
        mel_float = mel_u32.to(torch.float32) * float(1 << self.cfg.mel_drain_shift)
        mel_float = torch.clamp(mel_float + self.log_offset, min=self.log_input_clamp_min)
        log_float = torch.log(mel_float)
        log_q = torch.round(log_float * float(1 << self.cfg.log_frac_bits)).to(torch.int64)
        return _saturate(log_q, self.cfg.log_w, signed=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected x shape [B, T] or [B, 1, T], got {list(x.shape)}")

        pcm_q = torch.clamp(torch.round(x.to(torch.float32) * 127.0), -128, 127).to(torch.int64)
        sample_q = _saturate(pcm_q << self.cfg.sample_left_shift, self.cfg.mfcc_sample_w, signed=True)
        if self.pre_emphasis:
            prev = torch.nn.functional.pad(sample_q[:, :-1], (1, 0), value=0)
            emph_raw = (sample_q << 16) - prev * self.alpha_q.to(device=x.device)
            sample_q = _saturate(_round_shift_signed(emph_raw, 16), self.cfg.mfcc_sample_w, signed=True)

        if self.center:
            sample_q = self._reflect_pad_1d(sample_q, self.n_fft // 2)

        frames = sample_q.unfold(1, self.n_fft, self.hop_length)
        window_raw = frames * self.hann_q.to(device=x.device).view(1, 1, -1)
        windowed = _round_shift_signed(window_raw, self.cfg.window_shift)
        windowed = _saturate(windowed, self.cfg.fft_in_w, signed=True)

        cos_q = self.dft_cos_q.to(device=x.device).t()
        sin_q = self.dft_sin_q.to(device=x.device).t()
        real_raw = torch.matmul(windowed, cos_q)
        imag_raw = torch.matmul(windowed, sin_q)
        real = _saturate(_round_shift_signed(real_raw, self.cfg.dft_shift), self.cfg.fft_data_w, signed=True)
        imag = _saturate(_round_shift_signed(imag_raw, self.cfg.dft_shift), self.cfg.fft_data_w, signed=True)

        power = real * real + imag * imag
        power = _saturate(power, self.cfg.power_w, signed=False)

        mel_raw = torch.matmul(power, self.mel_fb_q.to(device=x.device).t())
        if self.mel_coeff_frac_bits:
            mel_raw = _round_shift(mel_raw, self.mel_coeff_frac_bits)
        mel_acc = _saturate(mel_raw, self.cfg.mel_acc_w, signed=False)
        mel_u32 = _saturate(_round_shift(mel_acc, self.cfg.mel_drain_shift), self.cfg.pwl_in_w, signed=False)

        log_q = self._log_to_s24(mel_u32)
        dct_raw = torch.matmul(log_q, self.dct_mat_q.to(device=x.device).t())
        dct_acc = _saturate(dct_raw, self.cfg.dct_acc_w, signed=True)
        mfcc_float = dct_acc.to(torch.float32) / float(1 << (self.cfg.log_frac_bits + self.cfg.dct_shift))
        mfcc_float = mfcc_float.transpose(1, 2)
        self._observe("mfcc_float", mfcc_float)

        mfcc_scale = self.cfg.mfcc_output_scale
        if mfcc_scale is None or mfcc_scale <= 0:
            observed = float(mfcc_float.detach().abs().max().cpu().item()) if mfcc_float.numel() else 1.0
            mfcc_scale = max(observed, 1e-12) / float(_qmax(self.cfg.mfcc_out_w, signed=True))
        mfcc_q = torch.clamp(
            torch.round(mfcc_float / float(mfcc_scale)),
            _qmin(self.cfg.mfcc_out_w, signed=True),
            _qmax(self.cfg.mfcc_out_w, signed=True),
        ).to(torch.int64)
        return mfcc_q.to(torch.float32) * float(mfcc_scale)
