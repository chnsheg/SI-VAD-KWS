from __future__ import annotations

import math
from typing import Any

import torch


def _fit_line(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    if x.numel() == 0:
        return 0.0, 0.0
    if x.numel() == 1:
        xv = float(x.item())
        yv = float(y.item())
        return 0.0, yv if xv == 0 else yv - 0.0 * xv

    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).mean()
    if float(var_x.item()) < 1e-20:
        slope = 0.0
    else:
        slope = float((((x - x_mean) * (y - y_mean)).mean() / var_x).item())
    intercept = float((y_mean - slope * x_mean).item())
    return slope, intercept


def fit_piecewise_linear_log_from_samples(
    x_samples: torch.Tensor,
    num_segments: int,
    strategy: str = "uniform_logx",
    x_min: float | None = None,
    x_max: float | None = None,
    gamma: float = 1.0,
) -> dict[str, Any]:
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    if strategy not in {"uniform_logx", "quantile", "powerlaw"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    x = x_samples.detach().float().reshape(-1)
    if x.numel() == 0:
        raise ValueError("x_samples is empty")

    eps = 1e-12
    x = torch.clamp(x, min=eps)
    if x_min is None:
        x_min = float(torch.quantile(x, 0.001).item())
    if x_max is None:
        x_max = float(torch.quantile(x, 0.999).item())
    x_min = max(float(x_min), eps)
    x_max = max(float(x_max), x_min * (1.0 + 1e-6))

    x_fit = torch.clamp(x, min=x_min, max=x_max)
    y_fit = torch.log(x_fit)

    if strategy == "uniform_logx":
        bp = torch.logspace(math.log10(x_min), math.log10(x_max), steps=num_segments + 1)
    elif strategy == "quantile":
        q = torch.linspace(0.0, 1.0, steps=num_segments + 1)
        bp = torch.quantile(x_fit, q)
        bp[0] = x_min
        bp[-1] = x_max
        for i in range(1, bp.numel()):
            if bp[i] <= bp[i - 1]:
                bp[i] = bp[i - 1] * (1.0 + 1e-6)
    else:
        t = torch.linspace(0.0, 1.0, steps=num_segments + 1)
        t_gamma = torch.pow(t, gamma)
        ratio = x_max / x_min
        bp = x_min * torch.pow(torch.tensor(ratio, dtype=x_fit.dtype), t_gamma)

    slopes: list[float] = []
    intercepts: list[float] = []
    seg_counts: list[int] = []
    y_hat = torch.empty_like(y_fit)

    for i in range(num_segments):
        l = bp[i]
        r = bp[i + 1]
        if i == num_segments - 1:
            mask = (x_fit >= l) & (x_fit <= r)
        else:
            mask = (x_fit >= l) & (x_fit < r)

        xi = x_fit[mask]
        yi = y_fit[mask]
        if xi.numel() == 0:
            x0 = torch.tensor([(float(l) + float(r)) / 2.0], dtype=x_fit.dtype)
            y0 = torch.log(x0)
            slope, intercept = _fit_line(x0, y0)
            y_hat[mask] = slope * xi + intercept
        else:
            slope, intercept = _fit_line(xi, yi)
            y_hat[mask] = slope * xi + intercept
        slopes.append(float(slope))
        intercepts.append(float(intercept))
        seg_counts.append(int(xi.numel()))

    mae = float(torch.mean(torch.abs(y_hat - y_fit)).item())
    max_ae = float(torch.max(torch.abs(y_hat - y_fit)).item())

    return {
        "strategy": strategy,
        "num_segments": num_segments,
        "gamma": gamma,
        "x_min": x_min,
        "x_max": x_max,
        "breakpoints": [float(v) for v in bp.tolist()],
        "slopes": slopes,
        "intercepts": intercepts,
        "segment_counts": seg_counts,
        "fit_mae": mae,
        "fit_max_ae": max_ae,
    }
