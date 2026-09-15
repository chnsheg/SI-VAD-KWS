"""ONNX 可追踪的 V6.1 严格整数 MFCC 前端。

背景
----
原版 ``StrictIntegerMFCCFrontend`` / ``StrictIntegerMFCCFloatAdapter``
（dscnn_kws/quantization/bit_accurate_mfcc_experiments_v6_strict_integer_mfcc/strict_integer_mfcc.py）
在推理语义上是正确且位精确的，但内部包含 ``.item()``、``torch.bucketize``、
就地索引赋值、``Tensor.unfold`` 等无法被 ``torch.onnx.export`` 追踪的算子，
因此不能直接导出。

本模块以“位精确等价”为目标重写同一套整数核心，仅替换实现手段，数值与
``StrictIntegerMFCCFloatAdapter.forward`` 保持逐位一致：

- 用 ``F.pad(mode='reflect')`` 替代手写的 reflect pad；
- 用 gather（``index_select`` + 显式帧索引）替代 ``unfold``；
- 用 ``torch.where`` + 分段掩码替代 ``torch.bucketize``；
- 用整数截断除法 ``torch.div(..., rounding_mode='trunc')`` 替代 int64 位右移；
- 去掉所有 ``.item()`` 统计与 range-check（它们只用于诊断，不影响输出）；
- 用 ``torch.where`` 替代 FFT 中 trivial-twiddle 的就地索引赋值。

输入输出契约
------------
输入：波形 float32 ``[batch, samples]``（默认 samples=16000），会先经
``round(waveform / pcm_scale)`` 量化为 int8 PCM code。
输出：float32 MFCC ``[batch, 40, frames]``（``mfcc_code * mfcc_scale``），
与 ``StrictIntegerMFCCFloatAdapter.forward`` 一致，供后续 backbone 使用。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def qrange(bits: int, signed: bool) -> tuple[int, int]:
    bits = int(bits)
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def shift_toward_zero(x: torch.Tensor, shift: int) -> torch.Tensor:
    """int64 向零截断右移（等价于原版 bitwise_right_shift(abs) 的实现）。"""
    x = x.to(torch.int64)
    shift = int(shift)
    if shift < 0:
        return x * (1 << (-shift))
    if shift == 0:
        return x
    # 输入非负：trunc 除法与 bitwise_right_shift 逐位等价（ONNX 直接支持）。
    mag = torch.div(torch.abs(x), 1 << shift, rounding_mode="trunc")
    return torch.where(x < 0, -mag, mag)


def shift_round_nearest_away_zero(x: torch.Tensor, shift: int) -> torch.Tensor:
    """int64 就近取整（远离零）右移，等价于原版 round_shift_nearest_away_zero。"""
    x = x.to(torch.int64)
    shift = int(shift)
    if shift < 0:
        return x * (1 << (-shift))
    if shift == 0:
        return x
    # 输入非负：trunc 除法与 bitwise_right_shift 逐位等价（ONNX 直接支持）。
    mag = torch.div(torch.abs(x) + (1 << (shift - 1)), 1 << shift, rounding_mode="trunc")
    return torch.where(x < 0, -mag, mag)


def _arith_shift_right(x: torch.Tensor, shift: int) -> torch.Tensor:
    """精确的算术右移（floor 语义）。

    PyTorch ONNX 导出器不支持 int64 bitwise_or / bitwise_left_shift，
    且会把 int64 bitwise_right_shift 降级为 trunc 语义的 onnx::Div（负数错误）。
    因此用纯算术实现：trunc 除法 + 余数修正还原 floor 语义，
    仅依赖 ONNX Div/Mul/Sub/Less/And/Cast（全部 opset 7 起支持）。
    """
    x = x.to(torch.int64)
    shift = int(shift)
    if shift == 0:
        return x
    divisor = 1 << shift
    q = torch.div(x, divisor, rounding_mode="trunc")  # ONNX int Div 原生 trunc 语义
    r = x - q * divisor                                # 余数（x<0 时 r <= 0）
    correction = ((x < 0) & (r != 0)).to(torch.int64)  # floor = trunc - (负且除不尽)
    return q - correction


def _saturate_unsigned(x: torch.Tensor, bits: int) -> torch.Tensor:
    """无符号饱和到 bits 位（x>=0）。overflow 为小值，比较/Where 精确。"""
    overflow = _arith_shift_right(x, bits)
    m = torch.tensor((1 << bits) - 1, dtype=torch.int64)
    return torch.where(overflow == 0, x, m)


def _saturate_signed(x: torch.Tensor, bits: int) -> torch.Tensor:
    """有符号饱和到 bits 位。算术右移把溢出压成小值后再比较。"""
    hi = (1 << (bits - 1)) - 1
    lo = -(1 << (bits - 1))
    arith = _arith_shift_right(x, bits - 1)
    pos_ovf = arith >= 1
    neg_ovf = arith <= -2
    out = torch.where(pos_ovf, torch.tensor(hi, dtype=torch.int64), x)
    out = torch.where(neg_ovf, torch.tensor(lo, dtype=torch.int64), out)
    return out


def _clamp(x: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
    if signed:
        return _saturate_signed(x, bits)
    return _saturate_unsigned(x, bits)


class OnnxStrictIntegerMFCCFrontend(nn.Module):
    """ONNX-traceable strict-integer MFCC frontend (float waveform -> float MFCC)."""

    def __init__(self, spec: dict[str, Any] | str | Path):
        super().__init__()
        if isinstance(spec, (str, Path)):
            spec = json.loads(Path(spec).read_text(encoding="utf-8"))
        if "runtime" not in spec:
            raise ValueError("V6.1 spec 缺少 runtime 段")

        self.runtime = spec["runtime"]
        self.audit = spec.get("audit", {})
        signal = self.runtime["signal"]
        self.n_fft = int(signal["n_fft"])
        self.hop_length = int(signal["hop_length"])
        self.center = bool(signal["center"])
        self.n_mels = int(signal["n_mels"])
        self.n_mfcc = int(signal["n_mfcc"])

        self.register_buffer(
            "hann_q",
            torch.tensor(self.runtime["hann_coefficients"], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "twiddle_real",
            torch.tensor(self.runtime["fft"]["twiddle_real"], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "twiddle_imag",
            torch.tensor(self.runtime["fft"]["twiddle_imag"], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "dct_q",
            torch.tensor(self.runtime["dct_coefficients"], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "pcm_scale",
            torch.tensor(float(self.audit["stage_scales"]["pcm"]), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "mfcc_scales",
            torch.tensor(self.audit["stage_scales"]["mfcc"], dtype=torch.float32),
            persistent=False,
        )

        self.mel_bands = [list(band) for band in self.runtime["mel_bands"]]

        pwl = self.runtime["pwl"]
        self.a_frac = int(pwl["a_frac"])
        self.y0_bits = int(pwl["y0_bits"])
        # PWL 段边界只用于 Python 层面的常量控制流，避免在导出时对 buffer 调用 .item()。
        self._pwl_bp_list = [int(x) for x in pwl["breakpoints"]]
        self.register_buffer(
            "pwl_bp", torch.tensor(pwl["breakpoints"], dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "pwl_slopes", torch.tensor(pwl["slopes"], dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "pwl_anchors", torch.tensor(pwl["anchors"], dtype=torch.int64), persistent=False
        )

        self.register_buffer("bitrev", self._bit_reverse_indices(self.n_fft), persistent=False)

    @staticmethod
    def _bit_reverse_indices(n: int) -> torch.Tensor:
        bits = int(math.log2(n))
        indices: list[int] = []
        for value in range(n):
            source = value
            rev = 0
            for _ in range(bits):
                rev = (rev << 1) | (source & 1)
                source >>= 1
            indices.append(rev)
        return torch.tensor(indices, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # 输入量化
    # ------------------------------------------------------------------ #
    def pcm_codes(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        q = torch.clamp(torch.round(waveform.to(torch.float32) / self.pcm_scale), -128, 127)
        return q.to(torch.int8)

    # ------------------------------------------------------------------ #
    # 逐级整数运算（位精确，无统计/无 .item）
    # ------------------------------------------------------------------ #
    def _preemphasis(self, pcm_q: torch.Tensor) -> torch.Tensor:
        cfg = self.runtime["preemphasis"]
        frac = int(cfg["coeff_frac"])
        shift_left = 1 << frac
        first = pcm_q[:, :1].to(torch.int64) * shift_left
        rest = (
            pcm_q[:, 1:].to(torch.int64) * shift_left
            - int(cfg["coeff_int"]) * pcm_q[:, :-1].to(torch.int64)
        )
        raw = torch.cat([first, rest], dim=1)
        product = raw * int(cfg["requant_multiplier"])
        scaled = shift_round_nearest_away_zero(product, frac + int(cfg["requant_shift"]))
        return _clamp(scaled, int(cfg["output_bits"]), True)

    @staticmethod
    def _reflect_pad_1d(x: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return x
        # torch 2.0 CPU 的 reflection_pad1d 不支持 int64，且 ONNX 无对应算子；
        # 用 gather（ONNX Gather）实现不含边界的 reflect 填充，语义与 F.pad 一致。
        n = x.size(-1)
        idx = torch.arange(-pad, n + pad, device=x.device, dtype=torch.int64)
        idx = torch.where(idx < 0, -idx, idx)
        idx = torch.where(idx >= n, 2 * (n - 1) - idx, idx)
        return x.index_select(-1, idx)

    def _requant(self, x: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
        x = x.to(torch.int64)
        product = x * int(spec["multiplier"]) + int(spec.get("addend", 0))
        if str(spec["rounding"]) == "round_to_nearest_away_from_zero":
            scaled = shift_round_nearest_away_zero(product, int(spec["shift"]))
        else:
            scaled = shift_toward_zero(product, int(spec["shift"]))
        return _clamp(scaled, int(spec["output_bits"]), bool(spec["output_signed"]))

    def _radix2_fft(self, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fft = self.runtime["fft"]
        ar = real.to(torch.int64).index_select(-1, self.bitrev)
        ai = torch.zeros_like(ar)
        frac = int(fft["twiddle_w"]) - 1
        internal_w = int(fft["internal_data_w"])
        stage_lo, stage_hi = qrange(internal_w, True)

        for stage, shift_value in enumerate(fft["stage_shift_schedule"]):
            m = 1 << (stage + 1)
            half = m // 2
            offset = sum(1 << s for s in range(stage))
            wrs = self.twiddle_real[offset : offset + half]
            wis = self.twiddle_imag[offset : offset + half]
            blocks = self.n_fft // m

            shaped_r = ar.reshape(ar.size(0), ar.size(1), blocks, m)
            shaped_i = ai.reshape(ai.size(0), ai.size(1), blocks, m)
            u_r, v_r = shaped_r[..., :half], shaped_r[..., half:]
            u_i, v_i = shaped_i[..., :half], shaped_i[..., half:]

            view = (1,) * (v_r.dim() - 1) + (half,)
            wr = wrs.view(view)
            wi = wis.view(view)

            t_r = shift_toward_zero(v_r * wr - v_i * wi, frac)
            t_i = shift_toward_zero(v_r * wi + v_i * wr, frac)

            # V5.1 exact_trivial 语义：修正 trivial twiddle 位置。
            nd = t_r.dim()
            pos = torch.arange(half, device=t_r.device)
            v0 = (pos == 0).view(*(1,) * (nd - 1), half)
            t_r = torch.where(v0, v_r, t_r)
            t_i = torch.where(v0, v_i, t_i)
            if m % 4 == 0:
                k = m // 4
                vk = (pos == k).view(*(1,) * (nd - 1), half)
                t_r = torch.where(vk, v_i, t_r)
                t_i = torch.where(vk, -v_r, t_i)

            shift = int(shift_value)
            b0r = shift_toward_zero(u_r + t_r, shift)
            b0i = shift_toward_zero(u_i + t_i, shift)
            b1r = shift_toward_zero(u_r - t_r, shift)
            b1i = shift_toward_zero(u_i - t_i, shift)

            b0r = torch.clamp(b0r, stage_lo, stage_hi)
            b0i = torch.clamp(b0i, stage_lo, stage_hi)
            b1r = torch.clamp(b1r, stage_lo, stage_hi)
            b1i = torch.clamp(b1i, stage_lo, stage_hi)

            ar = torch.cat([b0r, b1r], dim=-1).reshape(ar.size(0), ar.size(1), self.n_fft)
            ai = torch.cat([b0i, b1i], dim=-1).reshape(ai.size(0), ai.size(1), self.n_fft)

        return ar, ai

    def _pwl(self, xq: torch.Tensor) -> torch.Tensor:
        bp = self.pwl_bp
        slopes = self.pwl_slopes
        anchors = self.pwl_anchors
        bp_list = self._pwl_bp_list
        xq = xq.to(torch.int64)

        # 位精确地把 xq 饱和到 [0, bp[-1]-1]（xq 本身非负）。
        domain_hi = bp_list[-1] - 1
        # 注意：不要用右移 63 位取符号——导出时降级为 trunc Div，负数会得 0。
        xq = torch.where(xq < bp_list[-1], xq, torch.tensor(domain_hi, dtype=torch.int64))

        # 段选择只用内部边界（均 <2^21，比较在 ORT 中精确）。
        n_seg = len(bp_list) - 1
        idx = torch.zeros_like(xq)
        for i in range(1, n_seg):
            boundary = bp_list[i]
            idx = torch.where(xq >= boundary, torch.full_like(xq, i), idx)

        bp_lower = bp[:-1]
        bp_sel = bp_lower[idx]
        slope_sel = slopes[idx]
        anchor_sel = anchors[idx]
        dx = xq - bp_sel
        product = slope_sel * dx
        delta = shift_toward_zero(product, self.a_frac)
        raw = anchor_sel + delta
        return _clamp(raw, self.y0_bits, True)

    def _mel(self, power: torch.Tensor) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for band in self.mel_bands:
            idx = torch.tensor(band, dtype=torch.long)
            rows.append(power.index_select(1, idx).sum(dim=1))
        return torch.stack(rows, dim=1)

    def _dct(self, log_mel: torch.Tensor) -> torch.Tensor:
        # log_mel: [B, 40, frames]；向量化 DCT，不在导出图中遍历 buffer。
        return (log_mel.unsqueeze(1) * self.dct_q.unsqueeze(-1)).sum(dim=2)

    def forward_codes(self, pcm_q: torch.Tensor) -> dict[str, torch.Tensor]:
        if pcm_q.dim() == 3 and pcm_q.size(1) == 1:
            pcm_q = pcm_q.squeeze(1)
        codes: dict[str, torch.Tensor] = {"pcm": pcm_q.to(torch.int64)}

        pre = self._preemphasis(pcm_q)
        codes["preemphasis"] = pre

        framed_source = self._reflect_pad_1d(pre, self.n_fft // 2) if self.center else pre
        length = framed_source.size(1)
        num_frames = (length - self.n_fft) // self.hop_length + 1
        frame_idx = (
            torch.arange(num_frames, device=framed_source.device, dtype=torch.long).unsqueeze(1)
            * self.hop_length
            + torch.arange(self.n_fft, device=framed_source.device, dtype=torch.long).unsqueeze(0)
        )
        frames_flat = framed_source.index_select(1, frame_idx.reshape(-1))
        frames = frames_flat.reshape(framed_source.size(0), num_frames, self.n_fft)

        window_acc = frames * self.hann_q.view(1, 1, -1)
        windowed = self._requant(window_acc, self.runtime["requants"]["window"])
        codes["windowed"] = windowed

        fft_real_full, fft_imag_full = self._radix2_fft(windowed)
        fft_real_raw = fft_real_full[..., : self.n_fft // 2 + 1]
        fft_imag_raw = fft_imag_full[..., : self.n_fft // 2 + 1]

        bridge = self.runtime["requants"]["fft_bridge"]
        bridge_real = self._requant(fft_real_raw, bridge)
        bridge_imag = self._requant(fft_imag_raw, bridge)
        fft_real = bridge_real.transpose(1, 2)
        fft_imag = bridge_imag.transpose(1, 2)
        codes["fft_real"] = fft_real
        codes["fft_imag"] = fft_imag

        power_acc = fft_real.square() + fft_imag.square()
        power = self._requant(power_acc, self.runtime["requants"]["power"])
        codes["power"] = power

        mel_acc = self._mel(power)
        mel = self._requant(mel_acc, self.runtime["requants"]["mel"])
        codes["mel"] = mel

        pwl_input = self._requant(mel, self.runtime["requants"]["pwl_input"])
        codes["pwl_input"] = pwl_input
        log_mel = self._pwl(pwl_input)
        codes["log_mel"] = log_mel

        dct_acc = self._dct(log_mel)
        dct = self._requant(dct_acc, self.runtime["requants"]["dct"])
        codes["dct"] = dct

        mfcc_rows: list[torch.Tensor] = []
        for channel, rq in enumerate(self.runtime["requants"]["mfcc"]):
            mfcc_rows.append(self._requant(dct[:, channel, :], rq))
        mfcc = torch.stack(mfcc_rows, dim=1)
        codes["mfcc"] = mfcc
        return codes

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        pcm_q = self.pcm_codes(waveform)
        mfcc_codes = self.forward_codes(pcm_q)["mfcc"]
        return mfcc_codes.to(torch.float32) * self.mfcc_scales.view(1, -1, 1)


__all__ = [
    "OnnxStrictIntegerMFCCFrontend",
    "qrange",
    "shift_round_nearest_away_zero",
    "shift_toward_zero",
]