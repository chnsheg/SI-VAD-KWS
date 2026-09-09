from __future__ import annotations

from dataclasses import dataclass

from .search_space import LayerGene, NASArchitecture, estimate_mfcc_time_steps


@dataclass
class LayerStats:
    idx: int
    op: str
    t: int
    f: int
    c: int
    mults: int
    params: int


@dataclass
class ArchCost:
    total_mults: int
    frontend_mults: int
    backbone_mults: int
    params: int
    layer_stats: list[LayerStats]


def _estimate_mfcc_frontend_mults(arch: NASArchitecture, sample_rate: int) -> int:
    win_length = int(sample_rate * arch.mfcc_window_ms / 1000)
    n_fft = 1 if win_length <= 1 else 1 << (win_length - 1).bit_length()
    n_frames = estimate_mfcc_time_steps(sample_rate=sample_rate, window_stride_ms=arch.mfcc_stride_ms)
    n_freqs = n_fft // 2 + 1
    fft_stages = max(1, n_fft.bit_length() - 1)

    hann_mults = n_frames * n_fft
    fft_mults = n_frames * (n_fft // 2) * fft_stages * 4
    power_mults = n_frames * n_freqs * 2
    mel_mults = 0
    log_mults = n_frames * n_freqs
    dct_mults = n_frames * arch.mfcc_n_mfcc * n_freqs
    return hann_mults + fft_mults + power_mults + mel_mults + log_mults + dct_mults


def _has_invalid_channel_jump(layers: list[LayerGene]) -> bool:
    for i in range(1, len(layers)):
        prev_c = layers[i - 1].channels
        cur_c = layers[i].channels
        if cur_c < prev_c * 0.5 or cur_c > prev_c * 2.0:
            return True
    return False


def _has_invalid_stride_kernel(layers: list[LayerGene]) -> bool:
    for layer in layers:
        if layer.op != "dsconv1d" and layer.stride_t > layer.kernel_t:
            return True
        if layer.stride_f > layer.kernel_f:
            return True
    return False


def _same_conv_out(x: int, stride: int) -> int:
    if stride <= 0:
        return -1
    return (x + stride - 1) // stride


def _layer_cost(layer: LayerGene, c_in: int, t_in: int, f_in: int) -> tuple[int, int, int, int, int]:
    t_out = _same_conv_out(t_in, layer.stride_t)
    f_out = _same_conv_out(f_in, layer.stride_f)
    c_out = layer.channels
    if t_out <= 0 or f_out <= 0:
        return c_out, t_out, f_out, -1, -1

    if layer.op == "conv2d":
        mults = c_out * t_out * f_out * (c_in * layer.kernel_t * layer.kernel_f)
        params = c_out * c_in * layer.kernel_t * layer.kernel_f
    elif layer.op == "dsconv2d":
        dw_mults = c_in * t_out * f_out * (layer.kernel_t * layer.kernel_f)
        pw_mults = c_out * t_out * f_out * c_in
        mults = dw_mults + pw_mults
        params = c_in * layer.kernel_t * layer.kernel_f + c_out * c_in
    elif layer.op == "dsconv1d":
        dw_mults = c_in * t_out * f_out * layer.kernel_f
        pw_mults = c_out * t_out * f_out * c_in
        mults = dw_mults + pw_mults
        params = c_in * layer.kernel_f + c_out * c_in
    elif layer.op == "eca":
        mults = c_in * layer.eca_kernel + c_in * t_in * f_in
        params = layer.eca_kernel
        c_out, t_out, f_out = c_in, t_in, f_in
    else:
        return c_out, t_out, f_out, -1, -1
    return c_out, t_out, f_out, mults, params


def estimate_arch_cost(arch: NASArchitecture, sample_rate: int = 8000, c_in: int = 1) -> ArchCost:
    stats: list[LayerStats] = []
    t_in = estimate_mfcc_time_steps(sample_rate=sample_rate, window_stride_ms=arch.mfcc_stride_ms)
    f_in = max(1, int(arch.mfcc_n_mfcc))
    t, f, c = t_in, f_in, c_in
    total_mults = 0
    total_params = 0

    for i, layer in enumerate(arch.layers):
        c, t, f, mults, params = _layer_cost(layer, c, t, f)
        if mults < 0 or params < 0:
            return ArchCost(total_mults=-1, frontend_mults=-1, backbone_mults=-1, params=-1, layer_stats=[])
        stats.append(LayerStats(idx=i, op=layer.op, t=t, f=f, c=c, mults=mults, params=params))
        total_mults += mults
        total_params += params

    frontend_mults = _estimate_mfcc_frontend_mults(arch, sample_rate)
    return ArchCost(
        total_mults=frontend_mults + total_mults,
        frontend_mults=frontend_mults,
        backbone_mults=total_mults,
        params=total_params,
        layer_stats=stats,
    )


def check_arch_constraints(
    arch: NASArchitecture,
    mult_limit: int = 2_200_000,
    mult_limit_parent: int | None = 3_000_000,
    param_limit: int = 120_000,
    stem_t_range: tuple[int, int] | None = None,
    stem_f_range: tuple[int, int] | None = None,
    stem_end_idx: int = 2,
    sample_rate: int = 8000,
) -> tuple[bool, str, ArchCost]:
    cost = estimate_arch_cost(arch, sample_rate=sample_rate)
    if cost.total_mults < 0:
        return False, "invalid_op_or_shape", cost
    if _has_invalid_stride_kernel(arch.layers):
        return False, "stride_kernel_invalid", cost
    if _has_invalid_channel_jump(arch.layers):
        return False, "channel_jump_invalid", cost

    if mult_limit_parent is not None and cost.total_mults > mult_limit_parent:
        return False, f"mult_exceed:{cost.total_mults}", cost

    if cost.total_mults > mult_limit:
        return True, "mult_soft_exceed", cost

    if not cost.layer_stats:
        return False, "empty_arch", cost

    return True, "ok", cost
