from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dscnn_kws.frontend import StreamingMFCC, TorchBandpass, TorchMFCC
from dscnn_kws.utils import apply_pre_emphasis


@dataclass
class StreamingCRNNState:
    """Runtime state needed by frame-by-frame inference."""

    cnn_caches: list[torch.Tensor | None]
    gru_hidden: torch.Tensor | None


class CausalDepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable 2D conv with causal padding on the time axis.

    Input format is [B, C, T, F]. The block preserves T and F when stride is 1.
    In streaming inference it consumes one new time frame [B, C, 1, F] and keeps
    the previous kernel_time - 1 frames in a cache.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_time: int = 5,
        kernel_freq: int = 3,
        bn_momentum: float = 0.04,
    ) -> None:
        super().__init__()
        if kernel_time < 1 or kernel_freq < 1:
            raise ValueError("kernel_time and kernel_freq must be positive")

        self.kernel_time = int(kernel_time)
        self.kernel_freq = int(kernel_freq)
        self.time_left = self.kernel_time - 1
        self.freq_left = self.kernel_freq // 2
        self.freq_right = self.kernel_freq - 1 - self.freq_left

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(self.kernel_time, self.kernel_freq),
            stride=(1, 1),
            padding=0,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(in_channels, momentum=bn_momentum)
        self.bn_pointwise = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

    def _pad_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.freq_left, self.freq_right, self.time_left, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_sequence(x)
        x = F.relu(self.bn_depthwise(self.depthwise(x)))
        x = F.relu(self.bn_pointwise(self.pointwise(x)))
        return x

    @torch.no_grad()
    def forward_step(
        self,
        x_t: torch.Tensor,
        cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.training:
            raise RuntimeError("forward_step is intended for eval/inference mode")
        if x_t.dim() != 4 or x_t.size(2) != 1:
            raise ValueError(f"Expected x_t shape [B, C, 1, F], got {tuple(x_t.shape)}")

        if self.time_left == 0:
            context = x_t
            new_cache = None
        else:
            if cache is None:
                cache = x_t.new_zeros(x_t.size(0), x_t.size(1), self.time_left, x_t.size(3))
            context = torch.cat([cache, x_t], dim=2)
            new_cache = context[:, :, -self.time_left :, :].detach()

        context = F.pad(context, (self.freq_left, self.freq_right, 0, 0))
        y_t = F.relu(self.bn_depthwise(self.depthwise(context)))
        y_t = F.relu(self.bn_pointwise(self.pointwise(y_t)))
        return y_t, new_cache


class StreamingDSCNNGRUBackbone(nn.Module):
    """Streaming-friendly CRNN backbone.

    The CNN part is a stack of causal depthwise-separable 2D convolution blocks.
    The RNN part is a GRU that carries temporal state across feature frames.
    """

    def __init__(
        self,
        input_features: int,
        label_count: int,
        # cnn_channels: tuple[int, ...] = (64, 64, 64, 64, 64),
        cnn_channels: tuple[int, ...] = (24, 24, 24, 24, 24),
        kernel_time: int = 5,
        kernel_freq: int = 3,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_features <= 0:
            raise ValueError("input_features must be positive")
        if not cnn_channels:
            raise ValueError("cnn_channels must not be empty")

        self.input_features = int(input_features)
        self.label_count = int(label_count)
        self.cnn_channels = tuple(int(c) for c in cnn_channels)
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)

        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in self.cnn_channels:
            blocks.append(
                CausalDepthwiseSeparableConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_time=kernel_time,
                    kernel_freq=kernel_freq,
                )
            )
            in_channels = out_channels
        self.cnn = nn.ModuleList(blocks)
        self.gru = nn.GRU(
            input_size=self.cnn_channels[-1],
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.gru_hidden, self.label_count)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"Expected features [B, T, F], got {tuple(features.shape)}")
        if features.size(2) != self.input_features:
            raise ValueError(f"Expected F={self.input_features}, got {features.size(2)}")

        x = features.unsqueeze(1)
        for block in self.cnn:
            x = block(x)
        x = x.mean(dim=3)
        return x.transpose(1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence = self.forward_features(features)
        output, _ = self.gru(sequence)
        last = output[:, -1, :]
        return self.fc(self.dropout(last))

    def init_stream_state(self) -> StreamingCRNNState:
        return StreamingCRNNState(cnn_caches=[None for _ in self.cnn], gru_hidden=None)

    @torch.no_grad()
    def forward_stream_frame(
        self,
        feature_frame: torch.Tensor,
        state: StreamingCRNNState | None = None,
    ) -> tuple[torch.Tensor, StreamingCRNNState]:
        if self.training:
            raise RuntimeError("forward_stream_frame is intended for eval/inference mode")
        if feature_frame.dim() != 2:
            raise ValueError(f"Expected feature_frame [B, F], got {tuple(feature_frame.shape)}")
        if feature_frame.size(1) != self.input_features:
            raise ValueError(f"Expected F={self.input_features}, got {feature_frame.size(1)}")

        if state is None:
            state = self.init_stream_state()

        x_t = feature_frame.unsqueeze(1).unsqueeze(2)
        next_caches: list[torch.Tensor | None] = []
        for block, cache in zip(self.cnn, state.cnn_caches):
            x_t, next_cache = block.forward_step(x_t, cache)
            next_caches.append(next_cache)

        gru_in = x_t.mean(dim=3).transpose(1, 2)
        gru_out, next_hidden = self.gru(gru_in, state.gru_hidden)
        logits = self.fc(gru_out[:, -1, :])
        return logits, StreamingCRNNState(cnn_caches=next_caches, gru_hidden=next_hidden.detach())


class StreamingKWSModel(nn.Module):
    """Waveform-to-logits wrapper for training a streaming-style CRNN.

    Training still uses complete utterances for efficiency. Deployment should
    split the frontend output into frames and call backbone.forward_stream_frame.
    """

    def __init__(
        self,
        sample_rate: int,
        label_count: int,
        frontend: str = "mfcc",
        dct_coeff: int = 10,
        window_size_ms: int = 32,
        window_stride_ms: int = 32,
        pre_emphasis: bool = True,
        pre_emphasis_coeff: float = 0.97,
        # cnn_channels: tuple[int, ...] = (64, 64, 64, 64, 64),
        cnn_channels: tuple[int, ...] = (24, 24, 24, 24, 24),
        kernel_time: int = 5,
        kernel_freq: int = 3,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.2,
        bandpass_n_bands: int = 10,
        bandpass_f_min: float = 200.0,
        bandpass_f_max: float = 4000.0,
        bandpass_spacing: str = "log",
        bandpass_kernel_size: int = 63,
        bandpass_phase_count: int = 1,
        mfcc_center: bool = False,
        streaming_mfcc: bool = True,
        mel_filter_shape: str = "triangular",
        log_approx_mode: str = "exact",
        log_pwl_num_segments: int = 6,
        log_pwl_strategy: str = "uniform_logx",
        log_pwl_gamma: float = 1.0,
        log_pwl_breakpoints: list[float] | None = None,
        log_pwl_slopes: list[float] | None = None,
        log_pwl_intercepts: list[float] | None = None,
        log_offset: float = 1e-6,
        log_input_clamp_min: float = 1e-12,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.dct_coeff = int(dct_coeff)
        self.pre_emphasis = bool(pre_emphasis)
        self.pre_emphasis_coeff = float(pre_emphasis_coeff)
        self.streaming_mfcc = bool(streaming_mfcc)

        n_fft = int(sample_rate * window_size_ms / 1000)
        hop_length = int(sample_rate * window_stride_ms / 1000)
        if frontend == "mfcc":
            mfcc_kwargs = dict(
                sample_rate=sample_rate,
                n_mfcc=40,
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=hop_length,
                n_mels=40,
                f_min=20,
                f_max=int(sample_rate / 2),
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
            if self.streaming_mfcc:
                self.feature_extractor = StreamingMFCC(**mfcc_kwargs, flush_tail=True)
            else:
                self.feature_extractor = TorchMFCC(**mfcc_kwargs, center=mfcc_center)
            input_features = self.dct_coeff
        elif frontend == "bandpass":
            if self.dct_coeff != int(bandpass_n_bands):
                raise ValueError("For bandpass frontend, dct_coeff must equal bandpass_n_bands")
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
            input_features = self.dct_coeff
        else:
            raise ValueError(f"Unsupported frontend: {frontend}")

        self.backbone = StreamingDSCNNGRUBackbone(
            input_features=input_features,
            label_count=label_count,
            cnn_channels=cnn_channels,
            kernel_time=kernel_time,
            kernel_freq=kernel_freq,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if self.pre_emphasis:
            x = apply_pre_emphasis(x, self.pre_emphasis_coeff)
        features = self.feature_extractor(x)
        features = features[:, : self.dct_coeff, :]
        return features.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.extract_features(x))

    @torch.no_grad()
    def stream_features(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Reference frame-by-frame execution from a precomputed feature tensor."""
        if features.dim() != 3:
            raise ValueError(f"Expected features [B, T, F], got {tuple(features.shape)}")
        self.eval()
        state: StreamingCRNNState | None = None
        logits_per_frame: list[torch.Tensor] = []
        for t in range(features.size(1)):
            logits, state = self.backbone.forward_stream_frame(features[:, t, :], state)
            logits_per_frame.append(logits)
        return logits_per_frame


def parse_cnn_channels(raw: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in raw.replace(",", " ").split() if item.strip())
    if not values:
        raise ValueError("cnn_channels must contain at least one integer")
    return values


def model_config_dict(model: StreamingKWSModel) -> dict[str, Any]:
    return {
        "frontend": model.frontend,
        "dct_coeff": model.dct_coeff,
        "streaming_mfcc": model.streaming_mfcc,
        "feature_extractor": model.feature_extractor.__class__.__name__,
        "cnn_channels": model.backbone.cnn_channels,
        "gru_hidden": model.backbone.gru_hidden,
        "gru_layers": model.backbone.gru_layers,
        "input_features": model.backbone.input_features,
        "label_count": model.backbone.label_count,
    }
