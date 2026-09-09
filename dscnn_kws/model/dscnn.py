from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: tuple[int, int]):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(in_channels, momentum=0.04)
        self.bn_pointwise = nn.BatchNorm2d(out_channels, momentum=0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_depthwise(self.depthwise(x)))
        x = F.relu(self.bn_pointwise(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        label_count: int,
        model_size_info: List[int],
        dct_coeff: int,
        *,
        pooling: str = "global",
        temporal_bins: int = 4,
    ):
        super().__init__()
        self.num_layers = model_size_info[0]
        self.dct_coeff = dct_coeff
        self.input_time_size = input_dim // dct_coeff
        self.input_frequency_size = dct_coeff
        if pooling not in {"global", "temporal"}:
            raise ValueError("pooling must be 'global' or 'temporal'")
        if isinstance(temporal_bins, bool) or not isinstance(temporal_bins, int) or temporal_bins < 2:
            raise ValueError("temporal_bins must be an integer >= 2")
        self.pooling = pooling
        self.temporal_bins = int(temporal_bins)

        layers_params = []
        idx = 1
        for _ in range(self.num_layers):
            feat, kt, kw, st, sw = model_size_info[idx : idx + 5]
            layers_params.append((feat, kt, kw, st, sw))
            idx += 5

        self.conv_layers = nn.ModuleList()
        for layer_no in range(self.num_layers):
            feat, kt, kw, st, sw = layers_params[layer_no]
            if layer_no == 0:
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv2d(1, feat, kernel_size=(kt, kw), stride=(st, sw), padding=(kt // 2, kw // 2), bias=False),
                        nn.BatchNorm2d(feat, momentum=0.04),
                        nn.ReLU(),
                    )
                )
            else:
                self.conv_layers.append(
                    DepthwiseSeparableConv2d(
                        in_channels=layers_params[layer_no - 1][0],
                        out_channels=feat,
                        kernel_size=(kt, kw),
                        stride=(st, sw),
                    )
                )

        # The historical global pool is retained as the default so old
        # checkpoints load byte-for-byte.  ``temporal`` keeps an ordered set
        # of coarse time bins (frequency is still averaged), preventing a
        # keyword prefix/suffix from being treated as an order-free match.
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool2d((self.temporal_bins, 1))
        self.dropout = nn.Dropout(0.3)
        self.final_fc = nn.Linear(layers_params[-1][0], label_count)
        if self.pooling == "temporal":
            self.temporal_fc = nn.Linear(layers_params[-1][0] * self.temporal_bins, label_count)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"DSCNN expects [batch, input_dim], got {tuple(x.shape)}")

        batch_size = x.size(0)
        expected_input_dim = self.input_time_size * self.input_frequency_size
        if x.size(1) != expected_input_dim:
            raise ValueError(f"DSCNN input_dim mismatch: expected {expected_input_dim}, got {x.size(1)}")

        x = x.reshape(batch_size, 1, self.input_time_size, self.input_frequency_size)
        for layer in self.conv_layers:
            x = layer(x)
        if self.pooling == "temporal":
            x = self.temporal_pool(x).squeeze(-1)
            x = x.flatten(1)
            x = self.dropout(x)
            return self.temporal_fc(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.final_fc(x)


def calculate_time_steps(sample_rate: int, window_stride_ms: int, audio_duration_ms: int = 1000) -> int:
    stride_samples = int(sample_rate * window_stride_ms / 1000)
    audio_samples = int(sample_rate * audio_duration_ms / 1000)
    if stride_samples <= 0:
        return 1
    return math.floor(audio_samples / stride_samples) + 1
