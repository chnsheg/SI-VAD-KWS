from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dscnn_kws.frontend.mfcc_torch import create_dct_matrix, create_mel_filterbank


class ONNXFriendlyMFCC(nn.Module):
    """MFCC frontend implemented with real-valued ONNX-friendly operators."""

    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 20.0,
        f_max: float | None = None,
        center: bool = True,
        mel_filter_shape: str = "triangular",
        output_scale: str = "torchaudio_db",
        log_offset: float = 1e-6,
        log_input_clamp_min: float = 1e-12,
    ) -> None:
        super().__init__()
        if win_length != n_fft:
            raise ValueError("ONNXFriendlyMFCC currently expects win_length == n_fft")
        if output_scale not in {"torchaudio_db", "natural_log"}:
            raise ValueError(f"Unsupported output_scale: {output_scale}")

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.output_scale = output_scale
        self.log_offset = log_offset
        self.log_input_clamp_min = log_input_clamp_min

        f_max = f_max if f_max is not None else sample_rate / 2
        window = torch.hann_window(win_length)
        real_kernel, imag_kernel = self._build_dft_kernels(n_fft, window)
        mel_fb = create_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            filter_shape=mel_filter_shape,
        )
        dct_mat = create_dct_matrix(n_mfcc=n_mfcc, n_mels=n_mels, norm="ortho")

        self.register_buffer("real_kernel", real_kernel, persistent=False)
        self.register_buffer("imag_kernel", imag_kernel, persistent=False)
        self.register_buffer("mel_fb", mel_fb, persistent=False)
        self.register_buffer("dct_mat", dct_mat, persistent=False)

    @staticmethod
    def _build_dft_kernels(n_fft: int, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freq = torch.arange(n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        time = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        phase = 2.0 * math.pi * freq * time / float(n_fft)
        real = torch.cos(phase) * window.unsqueeze(0)
        imag = -torch.sin(phase) * window.unsqueeze(0)
        return real.unsqueeze(1), imag.unsqueeze(1)

    def _pre_emphasis(self, x: torch.Tensor, coeff: float) -> torch.Tensor:
        if coeff <= 0:
            return x
        first = x[:, :1]
        rest = x[:, 1:] - coeff * x[:, :-1]
        return torch.cat([first, rest], dim=1)

    def forward(self, x: torch.Tensor, pre_emphasis: bool = True, pre_emphasis_coeff: float = 0.97) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected x shape [B, T] or [B, 1, T], got {list(x.shape)}")

        if pre_emphasis:
            x = self._pre_emphasis(x, pre_emphasis_coeff)

        x = x.unsqueeze(1)
        if self.center:
            pad = self.n_fft // 2
            x = F.pad(x, (pad, pad), mode="reflect")

        real = F.conv1d(x, self.real_kernel.to(dtype=x.dtype), stride=self.hop_length)
        imag = F.conv1d(x, self.imag_kernel.to(dtype=x.dtype), stride=self.hop_length)
        power_spec = real * real + imag * imag

        mel_spec = torch.matmul(self.mel_fb.to(dtype=x.dtype).unsqueeze(0), power_spec)
        if self.output_scale == "torchaudio_db":
            log_mel = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
            max_per_item = torch.amax(log_mel, dim=(1, 2), keepdim=True)
            log_mel = torch.maximum(log_mel, max_per_item - 80.0)
        else:
            log_mel = torch.log(torch.clamp(mel_spec + self.log_offset, min=self.log_input_clamp_min))

        return torch.matmul(self.dct_mat.to(dtype=x.dtype).unsqueeze(0), log_mel)


class ONNXFriendlyMFCCDSCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        sample_rate: int,
        dct_coeff: int,
        window_size_ms: int,
        window_stride_ms: int,
        pre_emphasis: bool = True,
        pre_emphasis_coeff: float = 0.97,
        mfcc_scale: str = "torchaudio_db",
        mel_filter_shape: str = "triangular",
    ) -> None:
        super().__init__()
        n_fft = int(sample_rate * window_size_ms / 1000)
        hop_length = int(sample_rate * window_stride_ms / 1000)
        self.dct_coeff = dct_coeff
        self.pre_emphasis = pre_emphasis
        self.pre_emphasis_coeff = pre_emphasis_coeff
        self.feature_extractor = ONNXFriendlyMFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=40,
            f_min=20.0,
            f_max=sample_rate / 2,
            center=True,
            mel_filter_shape=mel_filter_shape,
            output_scale=mfcc_scale,
        )
        self.backbone = backbone

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mfcc = self.feature_extractor(
            waveform,
            pre_emphasis=self.pre_emphasis,
            pre_emphasis_coeff=self.pre_emphasis_coeff,
        )
        mfcc = mfcc[:, : self.dct_coeff, :]
        features = mfcc.permute(0, 2, 1).reshape(mfcc.size(0), -1)
        return self.backbone(features)
