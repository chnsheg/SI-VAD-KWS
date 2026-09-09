from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .int8_mfcc_frontend import Int8MFCCFrontend, _reflect_pad_1d_int


class Int8StreamingMFCCFrontend(Int8MFCCFrontend):
    """Integer-aware INT8 MFCC frontend with StreamingMFCC framing semantics.

    The older Int8MFCCFrontend mirrors the TorchMFCC/STFT style path. This class
    keeps the same integer coefficient and activation quantization rules, but
    frames audio the same way as StreamingMFCC: causal windows, optional
    left-history padding, and tail flushing to the next hop. It returns
    dequantized float MFCC tensors after every stage has been clamped to INT8.
    """

    def __init__(self, *args: Any, flush_tail: bool = True, **kwargs: Any) -> None:
        kwargs.setdefault("center", False)
        super().__init__(*args, **kwargs)
        self.flush_tail = bool(flush_tail)
        self.history_length = self.win_length - self.hop_length
        if self.history_length < 0:
            raise ValueError("Int8StreamingMFCCFrontend expects win_length >= hop_length")

    @classmethod
    def from_scale_json(cls, path: str | Path, **kwargs: Any) -> "Int8StreamingMFCCFrontend":
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
            "flush_tail",
        ):
            if key in payload:
                kwargs.setdefault(key, payload[key])
        kwargs.setdefault("scales", payload["scales"])
        kwargs.setdefault("observer_enabled", False)
        return cls(**kwargs)

    def export_scale_json(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        super().export_scale_json(path, extra=extra)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["frontend"] = "Int8StreamingMFCCFrontend"
        payload["flush_tail"] = self.flush_tail
        payload["history_length"] = self.history_length
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _frame_streaming_int(self, x_q: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count = x_q.shape
        if sample_count <= 0:
            return x_q.new_empty(batch_size, 0, self.n_fft)

        if self.flush_tail:
            right_pad = (self.hop_length - sample_count % self.hop_length) % self.hop_length
            x_q = F.pad(x_q, (self.history_length, right_pad), mode="constant", value=0)
        else:
            usable = (sample_count // self.hop_length) * self.hop_length
            if usable <= 0:
                return x_q.new_empty(batch_size, 0, self.n_fft)
            x_q = F.pad(x_q[:, :usable], (self.history_length, 0), mode="constant", value=0)

        if x_q.size(1) < self.win_length:
            return x_q.new_empty(batch_size, 0, self.n_fft)

        frames = x_q.unfold(dimension=1, size=self.win_length, step=self.hop_length)
        if self.win_length < self.n_fft:
            frames = F.pad(frames, (0, self.n_fft - self.win_length), mode="constant", value=0)
        elif self.win_length > self.n_fft:
            frames = frames[..., : self.n_fft]
        return frames

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
        else:
            frames_q = self._frame_streaming_int(pre_q)

        if frames_q.size(1) == 0:
            empty = x.new_empty(x.size(0), self.n_mfcc, 0, dtype=torch.float32)
            self._observe_absmax(empty, "mfcc")
            return empty

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
