from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDepthwiseTemporalBlock(nn.Module):
    """Residual depthwise-separable temporal block with length-preserving padding."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation < 1:
            raise ValueError("dilation must be positive")
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn_depthwise = nn.BatchNorm1d(channels, momentum=0.04)
        self.bn_pointwise = nn.BatchNorm1d(channels, momentum=0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn_depthwise(self.depthwise(x)))
        x = self.bn_pointwise(self.pointwise(x))
        return F.relu(x + residual)


class CepstralTCN(nn.Module):
    """Compact temporal KWS backbone that treats MFCC coefficients as channels."""

    def __init__(
        self,
        input_dim: int,
        label_count: int,
        dct_coeff: int,
        *,
        channels: int = 68,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 1, 2, 2),
        temporal_bins: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        if isinstance(dct_coeff, bool) or not isinstance(dct_coeff, int) or dct_coeff < 1:
            raise ValueError("dct_coeff must be a positive integer")
        if input_dim < 1 or input_dim % dct_coeff:
            raise ValueError("input_dim must be positive and divisible by dct_coeff")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or num_blocks < 1:
            raise ValueError("num_blocks must be a positive integer")
        if len(dilations) != num_blocks or any(int(value) < 1 for value in dilations):
            raise ValueError("dilations must contain one positive value per temporal block")
        if isinstance(temporal_bins, bool) or not isinstance(temporal_bins, int) or temporal_bins < 2:
            raise ValueError("temporal_bins must be an integer >= 2")

        self.dct_coeff = dct_coeff
        self.input_time_size = input_dim // dct_coeff
        if temporal_bins > self.input_time_size:
            raise ValueError("temporal_bins cannot exceed the input frame count")
        self.channels = channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dilations = tuple(int(value) for value in dilations)
        self.temporal_bins = temporal_bins

        self.stem = nn.Sequential(
            nn.Conv1d(dct_coeff, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.04),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualDepthwiseTemporalBlock(channels, kernel_size, dilation)
            for dilation in self.dilations
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(temporal_bins)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * temporal_bins, label_count)
        self._initialize_weights()

    @property
    def temporal_receptive_field_frames(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"CepstralTCN expects [batch, input_dim], got {tuple(x.shape)}")
        expected_input_dim = self.input_time_size * self.dct_coeff
        if x.size(1) != expected_input_dim:
            raise ValueError(f"CepstralTCN input_dim mismatch: expected {expected_input_dim}, got {x.size(1)}")
        x = x.reshape(x.size(0), self.input_time_size, self.dct_coeff).transpose(1, 2)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.temporal_pool(x).flatten(1)
        return self.classifier(self.dropout(x))


__all__ = ["CepstralTCN", "ResidualDepthwiseTemporalBlock"]
