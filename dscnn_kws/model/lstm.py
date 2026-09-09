from __future__ import annotations

import torch
import torch.nn as nn
from torchaudio.transforms import FrequencyMasking, MFCC, TimeMasking

from dscnn_kws.frontend import TorchBandpass, TorchMFCC
from dscnn_kws.utils import apply_pre_emphasis

# =============================================================================
# LSTM 模型配置参数
# =============================================================================
LSTM_UNITS = 128        # LSTM 隐藏层单元数
PROJECTION_UNITS = 64   # 投影层单元数
NUM_LAYERS = 2          # LSTM 层数
DROPOUT = 0.3           # Dropout 比例
# =============================================================================

class LSTM(nn.Module):
    """带投影层的 LSTM Backbone"""
    def __init__(self, input_dim: int, label_count: int, dct_coeff: int):
        super().__init__()
        self.dct_coeff = dct_coeff
        self.time_steps = input_dim // dct_coeff

        # 使用 PyTorch 原生 LSTM，并通过投影层实现 LSTMP 结构
        # 如果 NUM_LAYERS > 1，每一层都会经过投影
        self.lstm = nn.LSTM(
            input_size=dct_coeff,
            hidden_size=LSTM_UNITS,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0,
            bidirectional=False
        )

        self.projection = nn.Linear(LSTM_UNITS, PROJECTION_UNITS)
        self.dropout = nn.Dropout(DROPOUT)
        self.final_fc = nn.Linear(PROJECTION_UNITS, label_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, input_dim]
        batch_size = x.size(0)
        # reshape to [batch, time, freq]
        x = x.view(batch_size, self.time_steps, self.dct_coeff)

        # lstm_out shape: [batch, time, hidden_size]
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步
        last_out = lstm_out[:, -1, :]

        # 投影层
        proj_out = self.projection(last_out)
        proj_out = torch.relu(proj_out)
        proj_out = self.dropout(proj_out)

        return self.final_fc(proj_out)

class MFCCLSTM(nn.Module):
    """集成前端处理和 LSTM Backbone 的包装类"""
    def __init__(
        self,
        backbone: nn.Module,
        frontend: str,
        sample_rate: int,
        dct_coeff: int,
        window_size_ms: int,
        window_stride_ms: int,
        bandpass_n_bands: int,
        bandpass_f_min: float,
        bandpass_f_max: float,
        bandpass_spacing: str,
        bandpass_kernel_size: int,
        bandpass_phase_count: int,
        pre_emphasis: bool,
        pre_emphasis_coeff: float,
        spec_aug: bool,
        spec_aug_freq_mask_param: int,
        spec_aug_time_mask_param: int,
        spec_aug_num_freq_masks: int,
        spec_aug_num_time_masks: int,
        mfcc_impl: str,
        mel_filter_shape: str,
        log_approx_mode: str,
        log_pwl_num_segments: int,
        log_pwl_strategy: str,
        log_pwl_gamma: float,
        log_pwl_breakpoints: list[float] | None,
        log_pwl_slopes: list[float] | None,
        log_pwl_intercepts: list[float] | None,
        log_offset: float,
        log_input_clamp_min: float,
        mfcc_scale: str = "torchaudio_db",
    ):
        super().__init__()
        self.backbone = backbone
        self.frontend = frontend
        self.dct_coeff = dct_coeff
        self.pre_emphasis = pre_emphasis
        self.pre_emphasis_coeff = pre_emphasis_coeff
        self.spec_aug = spec_aug
        self.mfcc_impl = mfcc_impl
        if mfcc_scale not in {"natural_log", "torchaudio_db"}:
            raise ValueError("mfcc_scale must be 'natural_log' or 'torchaudio_db'")
        self.mel_filter_shape = mel_filter_shape

        n_fft = int(sample_rate * window_size_ms / 1000)
        hop_length = int(sample_rate * window_stride_ms / 1000)

        if frontend == "mfcc":
            if mfcc_impl == "torchaudio":
                self.feature_extractor = MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=40,
                    # Keep the recurrent model on the same batch-invariant
                    # fixed log-mel contract as MFCCDSCNN (see train.py).
                    log_mels=(mfcc_scale == "natural_log"),
                    melkwargs={
                        "n_fft": n_fft,
                        "win_length": n_fft,
                        "hop_length": hop_length,
                        "n_mels": 40,
                        "f_min": 20,
                        "f_max": int(sample_rate / 2),
                        "window_fn": torch.hann_window,
                        "center": True,
                    },
                )
            else:
                self.feature_extractor = TorchMFCC(
                    sample_rate=sample_rate,
                    n_mfcc=40,
                    n_fft=n_fft,
                    win_length=n_fft,
                    hop_length=hop_length,
                    n_mels=40,
                    f_min=20,
                    f_max=int(sample_rate / 2),
                    center=True,
                    dct_norm="ortho",
                    mel_filter_shape=mel_filter_shape,
                    log_approx_mode=log_approx_mode,
                    log_pwl_num_segments=log_pwl_num_segments,
                    log_pwl_strategy=log_pwl_strategy,
                    log_pwl_gamma=log_pwl_gamma,
                    log_pwl_breakpoints=log_pwl_breakpoints,
                    log_pwl_slopes=log_pwl_slopes,
                    log_pwl_intercepts=log_pwl_intercepts,
                    log_offset=log_offset,
                    log_input_clamp_min=log_input_clamp_min,
                )
        else:
            self.feature_extractor = TorchBandpass(
                sample_rate=sample_rate,
                n_bands=bandpass_n_bands,
                frame_length=n_fft,
                frame_hop=hop_length,
                f_min=bandpass_f_min,
                f_max=bandpass_f_max,
                spacing=bandpass_spacing,
                kernel_size=bandpass_kernel_size,
                phase_count=bandpass_phase_count,
                log_approx_mode=log_approx_mode,
                log_pwl_num_segments=log_pwl_num_segments,
                log_pwl_strategy=log_pwl_strategy,
                log_pwl_gamma=log_pwl_gamma,
                log_pwl_breakpoints=log_pwl_breakpoints,
                log_pwl_slopes=log_pwl_slopes,
                log_pwl_intercepts=log_pwl_intercepts,
                log_offset=log_offset,
                log_input_clamp_min=log_input_clamp_min,
            )

        self.freq_mask = FrequencyMasking(freq_mask_param=max(1, spec_aug_freq_mask_param))
        self.time_mask = TimeMasking(time_mask_param=max(1, spec_aug_time_mask_param))
        self.spec_aug_num_freq_masks = max(0, spec_aug_num_freq_masks)
        self.spec_aug_num_time_masks = max(0, spec_aug_num_time_masks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if self.pre_emphasis:
            x = apply_pre_emphasis(x, self.pre_emphasis_coeff)

        mfcc = self.feature_extractor(x)

        if self.training and self.spec_aug:
            for _ in range(self.spec_aug_num_freq_masks):
                mfcc = self.freq_mask(mfcc)
            for _ in range(self.spec_aug_num_time_masks):
                mfcc = self.time_mask(mfcc)

        mfcc = mfcc[:, : self.dct_coeff, :]
        mfcc = mfcc.permute(0, 2, 1).reshape(mfcc.size(0), -1)

        return self.backbone(mfcc)
