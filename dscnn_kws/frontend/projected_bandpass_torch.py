from __future__ import annotations

import math

import torch
import torch.nn as nn

from .bandpass_torch import TorchBandpass


def create_dct_projection(out_features: int, in_features: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create an orthonormal DCT-II style projection matrix."""
    rows = []
    n = torch.arange(in_features, dtype=dtype)
    for k in range(out_features):
        scale = math.sqrt(1.0 / in_features) if k == 0 else math.sqrt(2.0 / in_features)
        rows.append(scale * torch.cos(math.pi / in_features * (n + 0.5) * k))
    return torch.stack(rows, dim=0)


class ProjectedBandpass(nn.Module):
    """Bandpass filterbank frontend with a learnable per-frame projection.

    The internal bandpass frontend can use many bands, for example 40, while
    the projected output keeps the original DSCNN input width, for example 10.
    Output shape is [batch, output_bands, time_steps].
    """

    def __init__(
        self,
        sample_rate: int,
        internal_bands: int,
        output_bands: int,
        frame_length: int,
        frame_hop: int,
        f_min: float = 80.0,
        f_max: float | None = None,
        spacing: str = "log",
        kernel_size: int = 255,
        phase_count: int = 4,
        log_approx_mode: str = "pwl",
        log_offset: float = 1e-6,
        log_input_clamp_min: float = 1e-12,
        log_pwl_breakpoints: list[float] | None = None,
        log_pwl_slopes: list[float] | None = None,
        log_pwl_intercepts: list[float] | None = None,
        log_pwl_num_segments: int = 8,
        log_pwl_strategy: str = "uniform_logx",
        log_pwl_gamma: float = 1.0,
        projection_init: str = "dct",
        trainable_projection: bool = True,
    ):
        super().__init__()
        if internal_bands <= 0 or output_bands <= 0:
            raise ValueError("internal_bands and output_bands must be positive")
        if internal_bands < output_bands:
            raise ValueError(
                f"internal_bands must be >= output_bands, got {internal_bands} < {output_bands}"
            )
        if projection_init not in {"dct", "average", "random"}:
            raise ValueError(f"Unsupported projection_init={projection_init}")

        self.internal_bands = internal_bands
        self.output_bands = output_bands
        self.projection_init = projection_init

        self.bandpass = TorchBandpass(
            sample_rate=sample_rate,
            n_bands=internal_bands,
            frame_length=frame_length,
            frame_hop=frame_hop,
            f_min=f_min,
            f_max=f_max,
            spacing=spacing,
            kernel_size=kernel_size,
            phase_count=phase_count,
            log_approx_mode=log_approx_mode,
            log_offset=log_offset,
            log_input_clamp_min=log_input_clamp_min,
            log_pwl_breakpoints=log_pwl_breakpoints,
            log_pwl_slopes=log_pwl_slopes,
            log_pwl_intercepts=log_pwl_intercepts,
            log_pwl_num_segments=log_pwl_num_segments,
            log_pwl_strategy=log_pwl_strategy,
            log_pwl_gamma=log_pwl_gamma,
        )
        self.projection = nn.Linear(internal_bands, output_bands, bias=True)
        self._initialize_projection(projection_init)
        self.projection.weight.requires_grad_(trainable_projection)
        self.projection.bias.requires_grad_(trainable_projection)

    def _initialize_projection(self, projection_init: str) -> None:
        with torch.no_grad():
            if projection_init == "dct":
                weight = create_dct_projection(self.output_bands, self.internal_bands)
                self.projection.weight.copy_(weight)
                self.projection.bias.zero_()
            elif projection_init == "average":
                self.projection.weight.zero_()
                edges = torch.linspace(0, self.internal_bands, self.output_bands + 1).round().to(torch.int64)
                for i in range(self.output_bands):
                    start = int(edges[i].item())
                    end = max(start + 1, int(edges[i + 1].item()))
                    self.projection.weight[i, start:end] = 1.0 / (end - start)
                self.projection.bias.zero_()
            else:
                nn.init.xavier_uniform_(self.projection.weight)
                self.projection.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.bandpass(x)
        features = features.transpose(1, 2)
        features = self.projection(features)
        return features.transpose(1, 2)
