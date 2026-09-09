from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field


OPS = ("conv2d", "dsconv2d", "dsconv1d", "eca")
CHANNELS = (8, 12, 16, 24, 32, 48)
ECA_KERNELS = (3, 5, 7)
MFCC_WINDOWS_MS = (16, 32, 64, 128)
MFCC_STRIDES_MS = (8, 16, 32, 64, 128)
MFCC_N_MFCC = (10, 11, 12, 13, 14, 15)
STEM_KERNEL_T = (5, 7)
STEM_KERNEL_F = (5, 7)
MID_KERNEL_T = (3, 5)
MID_KERNEL_F = (3, 5)
LATE_KERNEL_T = (3,)
LATE_KERNEL_F = (3, 5)
STEM0_STRIDE_T = (4, 5, 6, 7)
STEM1_STRIDE_T = (2, 3, 4, 5)
MID_STRIDE_T = (1, 2, 3)
LATE_STRIDE_T = (1, 2)
STEM_STRIDE_F = (1, 2)
MID_STRIDE_F = (1, 2)
LATE_STRIDE_F = (1,)


@dataclass
class LayerGene:
    op: str
    channels: int
    kernel_t: int
    kernel_f: int
    stride_t: int
    stride_f: int
    eca_kernel: int = 3


@dataclass
class NASArchitecture:
    layers: list[LayerGene]
    mfcc_window_ms: int = 32
    mfcc_stride_ms: int = 32
    mfcc_n_mfcc: int = 13
    t_target: int = 32
    f_target: int = 13
    uid: str = field(default_factory=lambda: "")


def estimate_mfcc_time_steps(sample_rate: int, window_stride_ms: int, audio_duration_ms: int = 1000) -> int:
    stride_samples = int(sample_rate * window_stride_ms / 1000)
    audio_samples = int(sample_rate * audio_duration_ms / 1000)
    if stride_samples <= 0:
        return 1
    return audio_samples // stride_samples + 1


def _sample_mfcc_frontend(rng: random.Random, sample_rate: int = 8000) -> tuple[int, int, int, int]:
    window_ms = rng.choice(MFCC_WINDOWS_MS)
    valid_strides = tuple(s for s in MFCC_STRIDES_MS if s <= window_ms)
    stride_ms = rng.choice(valid_strides)
    n_mfcc = rng.choice(MFCC_N_MFCC)
    t_target = estimate_mfcc_time_steps(sample_rate=sample_rate, window_stride_ms=stride_ms)
    return window_ms, stride_ms, n_mfcc, t_target


def _sample_stride_t(layer_idx: int, rng: random.Random) -> int:
    if layer_idx == 0:
        return rng.choice(STEM0_STRIDE_T)
    if layer_idx == 1:
        return rng.choice(STEM1_STRIDE_T)
    if layer_idx < 4:
        return rng.choice(MID_STRIDE_T)
    return rng.choice(LATE_STRIDE_T)


def _sample_kernel_t(layer_idx: int, rng: random.Random) -> int:
    if layer_idx < 2:
        return rng.choice(STEM_KERNEL_T)
    if layer_idx < 4:
        return rng.choice(MID_KERNEL_T)
    return rng.choice(LATE_KERNEL_T)


def _sample_kernel_f(layer_idx: int, rng: random.Random) -> int:
    if layer_idx < 2:
        return rng.choice(STEM_KERNEL_F)
    if layer_idx < 4:
        return rng.choice(MID_KERNEL_F)
    return rng.choice(LATE_KERNEL_F)


def _sample_stride_f(layer_idx: int, rng: random.Random) -> int:
    if layer_idx < 2:
        return rng.choice(STEM_STRIDE_F)
    if layer_idx < 4:
        return rng.choice(MID_STRIDE_F)
    return rng.choice(LATE_STRIDE_F)


def _sample_op(layer_idx: int, rng: random.Random) -> str:
    # stem 前两层禁用 dsconv1d（其沿频率卷积，不利于快速收敛时间维）。
    if layer_idx < 2:
        return rng.choice(("conv2d", "dsconv2d"))
    return rng.choice(OPS)


def _choose_stem_strides(t_in: int, t_target: int, rng: random.Random) -> tuple[int, int]:
    s0 = rng.choice(STEM0_STRIDE_T)
    t1 = (t_in + s0 - 1) // s0
    best_s1 = STEM1_STRIDE_T[0]
    best_err = 10**9
    for s1 in STEM1_STRIDE_T:
        t2 = (t1 + s1 - 1) // s1
        err = abs(t2 - t_target)
        if err < best_err:
            best_err = err
            best_s1 = s1
    return s0, best_s1


def _nearest_allowed_channel(value: int, low: float, high: float) -> int:
    candidates = [c for c in CHANNELS if low <= c <= high]
    if not candidates:
        candidates = [min(CHANNELS, key=lambda c: abs(c - max(low, min(high, c))))]
    return min(candidates, key=lambda c: abs(c - value))


def _sanitize_layer(layer: LayerGene, layer_idx: int, prev_channels: int | None) -> LayerGene:
    if layer.op == "dsconv1d":
        layer.kernel_t = 1
        layer.stride_t = 1
    else:
        layer.kernel_t = _sample_kernel_t(layer_idx, random.Random(layer.kernel_t + 17 * (layer_idx + 1))) if layer.kernel_t <= 0 else layer.kernel_t
        if layer.stride_t > layer.kernel_t:
            layer.stride_t = layer.kernel_t

    if layer.stride_f > layer.kernel_f:
        layer.stride_f = layer.kernel_f

    if prev_channels is not None:
        low = max(8.0, prev_channels * 0.5)
        high = min(48.0, prev_channels * 2.0)
        if not (low <= layer.channels <= high):
            layer.channels = _nearest_allowed_channel(layer.channels, low, high)
    return layer


def sanitize_architecture(arch: NASArchitecture) -> NASArchitecture:
    out = copy.deepcopy(arch)
    prev_c: int | None = None
    for i, layer in enumerate(out.layers):
        out.layers[i] = _sanitize_layer(layer, i, prev_c)
        prev_c = out.layers[i].channels
    out.t_target = estimate_mfcc_time_steps(sample_rate=8000, window_stride_ms=out.mfcc_stride_ms)
    out.f_target = out.mfcc_n_mfcc
    return out


def sample_random_architecture(
    num_layers: int,
    rng: random.Random,
    t_target: int = 32,
    f_target: int = 16,
    sample_rate: int = 8000,
) -> NASArchitecture:
    window_ms, stride_ms, n_mfcc, real_t_target = _sample_mfcc_frontend(rng=rng, sample_rate=sample_rate)
    layers: list[LayerGene] = []
    t_in = max(1, real_t_target)
    stem_s0, stem_s1 = _choose_stem_strides(t_in=t_in, t_target=min(real_t_target, 32), rng=rng)
    for i in range(num_layers):
        op = _sample_op(i, rng)
        stride_t = _sample_stride_t(i, rng)
        if i == 0:
            stride_t = stem_s0
        elif i == 1:
            stride_t = stem_s1
        layers.append(
            LayerGene(
                op=op,
                channels=rng.choice(CHANNELS),
                kernel_t=1 if op == "dsconv1d" else _sample_kernel_t(i, rng),
                kernel_f=_sample_kernel_f(i, rng),
                stride_t=1 if op == "dsconv1d" else stride_t,
                stride_f=_sample_stride_f(i, rng),
                eca_kernel=rng.choice(ECA_KERNELS),
            )
        )
    return sanitize_architecture(NASArchitecture(
        layers=layers,
        mfcc_window_ms=window_ms,
        mfcc_stride_ms=stride_ms,
        mfcc_n_mfcc=n_mfcc,
        t_target=real_t_target,
        f_target=n_mfcc,
    ))


def mutate_architecture(arch: NASArchitecture, rng: random.Random, mutation_prob: float = 0.2) -> NASArchitecture:
    out = copy.deepcopy(arch)
    if rng.random() < mutation_prob:
        key = rng.choice(("mfcc_window_ms", "mfcc_stride_ms", "mfcc_n_mfcc"))
        if key == "mfcc_window_ms":
            out.mfcc_window_ms = rng.choice(MFCC_WINDOWS_MS)
            valid_strides = tuple(s for s in MFCC_STRIDES_MS if s <= out.mfcc_window_ms)
            if out.mfcc_stride_ms not in valid_strides:
                out.mfcc_stride_ms = rng.choice(valid_strides)
        elif key == "mfcc_stride_ms":
            valid_strides = tuple(s for s in MFCC_STRIDES_MS if s <= out.mfcc_window_ms)
            out.mfcc_stride_ms = rng.choice(valid_strides)
        else:
            out.mfcc_n_mfcc = rng.choice(MFCC_N_MFCC)
        out.t_target = estimate_mfcc_time_steps(sample_rate=8000, window_stride_ms=out.mfcc_stride_ms)
        out.f_target = out.mfcc_n_mfcc

    for i, layer in enumerate(out.layers):
        if rng.random() >= mutation_prob:
            continue
        key = rng.choice(("op", "channels", "kernel_t", "kernel_f", "stride_t", "stride_f", "eca_kernel"))
        if key == "op":
            layer.op = _sample_op(i, rng)
            if layer.op == "dsconv1d":
                layer.kernel_t = 1
                layer.stride_t = 1
        elif key == "channels":
            prev_c = out.layers[i - 1].channels if i > 0 else None
            if prev_c is None:
                layer.channels = rng.choice(CHANNELS)
            else:
                low = max(8.0, prev_c * 0.5)
                high = min(48.0, prev_c * 2.0)
                candidates = [c for c in CHANNELS if low <= c <= high]
                layer.channels = rng.choice(candidates or CHANNELS)
        elif key == "kernel_t":
            if layer.op != "dsconv1d":
                layer.kernel_t = _sample_kernel_t(i, rng)
        elif key == "kernel_f":
            layer.kernel_f = _sample_kernel_f(i, rng)
        elif key == "stride_t":
            if layer.op != "dsconv1d":
                layer.stride_t = _sample_stride_t(i, rng)
        elif key == "stride_f":
            layer.stride_f = _sample_stride_f(i, rng)
        elif key == "eca_kernel":
            layer.eca_kernel = rng.choice(ECA_KERNELS)
    return sanitize_architecture(out)
