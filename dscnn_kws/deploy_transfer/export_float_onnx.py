# -*- coding: utf-8 -*-
"""将浮点 MFCC 前端 + 浮点 DSCNN 检查点导出为 full ONNX（waveform -> logits）。

与 export_v6_1_strict_int8_onnx.py 同一部署规范的浮点版本：
- 输入 ``waveform [1, 16000]`` float32，输出 ``logits [1, 2]`` float32（demo 应用契约）。
- 前端：训练用 TorchMFCC 浮点路径的逐算子精确复刻（pre_emphasis 0.97 -> reflect pad
  256 -> hann 加窗 DFT（conv1d 实/虚核）-> power -> 40 三角 mel[20,8000] ->
  log(clamp(mel+1e-6, 1e-12)) -> ortho DCT 取前 13 维 -> [1,13,32]）。
  不含 PCMN/PCEN/PWL 近似，不含任何相位补偿。
- 后端：浮点 DSCNN（5x64，56 个权重），从交付检查点的 ``backbone.*`` state_dict 加载。
- 导出固定 batch=1（demo contracts.validate_kws_model 要求静态 [1,16000]）。

用法
----
    python export_float_onnx.py --checkpoint <best.pt> [--output_dir models_float]
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 自包含的 DSCNN 定义（与训练用 dscnn.py 逐字段一致，仅用于读取卷积结构，
# 避免依赖 sklearn/torchaudio/dataset 等训练期依赖）
# --------------------------------------------------------------------------- #
def calculate_time_steps(sample_rate: int, window_stride_ms: int, audio_duration_ms: int = 1000) -> int:
    stride_samples = int(sample_rate * window_stride_ms / 1000)
    audio_samples = int(sample_rate * audio_duration_ms / 1000)
    if stride_samples <= 0:
        return 1
    return math.floor(audio_samples / stride_samples) + 1


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: tuple[int, int]):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2), groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(in_channels, momentum=0.04)
        self.bn_pointwise = nn.BatchNorm2d(out_channels, momentum=0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_depthwise(self.depthwise(x)))
        x = F.relu(self.bn_pointwise(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    def __init__(self, input_dim: int, label_count: int, model_size_info: list[int], dct_coeff: int):
        super().__init__()
        self.num_layers = model_size_info[0]
        self.dct_coeff = dct_coeff
        self.input_time_size = input_dim // dct_coeff
        self.input_frequency_size = dct_coeff

        layers_params = []
        idx = 1
        for _ in range(self.num_layers):
            feat, kt, kw, st, sw = model_size_info[idx:idx + 5]
            layers_params.append((feat, kt, kw, st, sw))
            idx += 5

        self.conv_layers = nn.ModuleList()
        for layer_no in range(self.num_layers):
            feat, kt, kw, st, sw = layers_params[layer_no]
            if layer_no == 0:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(1, feat, kernel_size=(kt, kw), stride=(st, sw), padding=(kt // 2, kw // 2), bias=False),
                    nn.BatchNorm2d(feat, momentum=0.04),
                    nn.ReLU(),
                ))
            else:
                self.conv_layers.append(DepthwiseSeparableConv2d(
                    in_channels=layers_params[layer_no - 1][0], out_channels=feat,
                    kernel_size=(kt, kw), stride=(st, sw),
                ))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.final_fc = nn.Linear(layers_params[-1][0], label_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.reshape(batch_size, 1, self.input_time_size, self.input_frequency_size)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.final_fc(x)


def make_model_size_info(num_layers: int, channels: int) -> list[int]:
    info = [num_layers, channels, 10, 4, 2, 2]
    for _ in range(num_layers - 1):
        info += [channels, 3, 3, 1, 1]
    return info


# --------------------------------------------------------------------------- #
# 训练 TorchMFCC（frontend/mfcc_torch.py，log_approx_mode="exact"、无 PCMN/PCEN）
# 的逐算子复刻：mel filterbank / DCT / 加窗 DFT 全部使用相同公式。
# --------------------------------------------------------------------------- #
def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(freq_mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, freq_mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    filter_shape: str = "triangular",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if f_max is None:
        f_max = sample_rate / 2
    if not (0 <= f_min < f_max):
        raise ValueError(f"Invalid frequency range: f_min={f_min}, f_max={f_max}")
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=dtype)
    mel_min = _hz_to_mel(torch.tensor(float(f_min), dtype=dtype))
    mel_max = _hz_to_mel(torch.tensor(float(f_max), dtype=dtype))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=dtype)
    hz_points = _mel_to_hz(mel_points)
    fb = torch.zeros(n_mels, n_freqs, dtype=dtype)
    eps = torch.tensor(1e-12, dtype=dtype)
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        lower = (fft_freqs - left) / torch.maximum(center - left, eps)
        upper = (right - fft_freqs) / torch.maximum(right - center, eps)
        fb[i] = torch.clamp(torch.minimum(lower, upper), min=0.0)
    return fb


def create_dct_matrix(
    n_mfcc: int,
    n_mels: int,
    norm: str | None = "ortho",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    n = torch.arange(n_mels, dtype=dtype)
    k = torch.arange(n_mfcc, dtype=dtype).unsqueeze(1)
    dct = torch.cos(math.pi / n_mels * (n + 0.5) * k)
    if norm is None:
        return dct
    if norm != "ortho":
        raise ValueError(f"Unsupported DCT norm: {norm}")
    dct[0] *= math.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        dct[1:] *= math.sqrt(2.0 / n_mels)
    return dct


def apply_pre_emphasis(x: torch.Tensor, coeff: float) -> torch.Tensor:
    first = x[:, :1]
    rest = x[:, 1:] - coeff * x[:, :-1]
    return torch.cat([first, rest], dim=1)


class OnnxFloatMFCCFrontend(nn.Module):
    """TorchMFCC（exact log、无 PCMN/PCEN）浮点前端的 ONNX 可导出精确复刻。"""

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
        log_offset: float = 1e-6,
        log_input_clamp_min: float = 1e-12,
        pre_emphasis: bool = True,
        pre_emphasis_coeff: float = 0.97,
    ):
        super().__init__()
        if win_length != n_fft:
            raise ValueError("OnnxFloatMFCCFrontend expects win_length == n_fft")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.log_offset = log_offset
        self.log_input_clamp_min = log_input_clamp_min
        self.pre_emphasis = pre_emphasis
        self.pre_emphasis_coeff = pre_emphasis_coeff

        f_max = f_max if f_max is not None else sample_rate / 2
        window = torch.hann_window(win_length)
        freq = torch.arange(n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        time = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        phase = 2.0 * math.pi * freq * time / float(n_fft)
        real = torch.cos(phase) * window.unsqueeze(0)
        imag = -torch.sin(phase) * window.unsqueeze(0)
        mel_fb = create_mel_filterbank(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels,
            f_min=f_min, f_max=f_max, filter_shape=mel_filter_shape,
        )
        dct_mat = create_dct_matrix(n_mfcc=n_mfcc, n_mels=n_mels, norm="ortho")
        self.register_buffer("real_kernel", real.unsqueeze(1), persistent=False)
        self.register_buffer("imag_kernel", imag.unsqueeze(1), persistent=False)
        self.register_buffer("mel_fb", mel_fb, persistent=False)
        self.register_buffer("dct_mat", dct_mat, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if self.pre_emphasis:
            x = apply_pre_emphasis(x, self.pre_emphasis_coeff)
        x = x.unsqueeze(1)
        if self.center:
            pad = self.n_fft // 2
            x = F.pad(x, (pad, pad), mode="reflect")
        real = F.conv1d(x, self.real_kernel, stride=self.hop_length)
        imag = F.conv1d(x, self.imag_kernel, stride=self.hop_length)
        power_spec = real * real + imag * imag
        mel_spec = torch.matmul(self.mel_fb, power_spec)
        log_mel = torch.log(torch.clamp(mel_spec + self.log_offset, min=self.log_input_clamp_min))
        return torch.matmul(self.dct_mat, log_mel)


class OnnxFloatFullModel(nn.Module):
    """浮点前端 + 浮点 DSCNN 的完整导出模型。"""

    def __init__(self, frontend: nn.Module, backbone: nn.Module, dct_coeff: int):
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone
        self.dct_coeff = int(dct_coeff)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.frontend(waveform)
        features = features[:, : self.dct_coeff, :]
        features = features.permute(0, 2, 1).reshape(features.size(0), -1)
        return self.backbone(features)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _long_path(path: Path) -> str:
    s = str(path.resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def _inline_external_data(onnx_path: Path) -> bool:
    import onnx

    data_path = Path(str(onnx_path) + ".data")
    if not data_path.exists():
        return False
    from onnx.external_data_helper import convert_model_from_external_data

    proto = onnx.load(_long_path(onnx_path), load_external_data=True)
    convert_model_from_external_data(proto)
    onnx.save_model(proto, _long_path(onnx_path), save_as_external_data=False)
    data_path.unlink(missing_ok=True)
    return True


def _load_state_dict(path: Path) -> dict[str, Any]:
    load_path = _long_path(path)
    try:
        obj = torch.load(load_path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(load_path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
        state = dict(state.items()) if hasattr(state, "items") else state
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        state = dict(state.items()) if hasattr(state, "items") else state
    else:
        state = obj
    return {k: v for k, v in state.items() if not k.startswith("module.")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export float MFCC + float DSCNN to full ONNX")
    parser.add_argument("--checkpoint", required=True, help="交付 best.pt（含 backbone.* 56 权重）")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "models_float"))
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--label_count", type=int, default=2)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--window_size_ms", type=int, default=32)
    parser.add_argument("--window_stride_ms", type=int, default=32)
    parser.add_argument("--n_mels", type=int, default=40)
    parser.add_argument("--dct_coeff", type=int, default=13)
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--f_max", type=float, default=8000.0)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=0.97)
    parser.add_argument("--log_offset", type=float, default=1e-6)
    parser.add_argument("--log_input_clamp_min", type=float, default=1e-12)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic_batch", action=argparse.BooleanOptionalAction, default=False,
                        help="demo 契约要求静态 [1,16000]，默认关闭")
    parser.add_argument("--check_onnx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check_onnxruntime", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_fft = int(args.sample_rate * args.window_size_ms / 1000)
    hop_length = int(args.sample_rate * args.window_stride_ms / 1000)
    time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
    input_dim = time_steps * args.dct_coeff

    backbone = DSCNN(
        input_dim=input_dim,
        label_count=args.label_count,
        model_size_info=make_model_size_info(args.layers, args.channels),
        dct_coeff=args.dct_coeff,
    )
    state = _load_state_dict(Path(args.checkpoint))
    bb_state = {k.replace("backbone.", "", 1): v for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/backbone mismatch: missing={missing} unexpected={unexpected}")
    print(f"[OK] loaded {len(bb_state)} backbone tensors from {args.checkpoint}")

    frontend = OnnxFloatMFCCFrontend(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mels,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
        center=True,
        mel_filter_shape="triangular",
        log_offset=args.log_offset,
        log_input_clamp_min=args.log_input_clamp_min,
        pre_emphasis=True,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
    )
    model = OnnxFloatFullModel(frontend, backbone, args.dct_coeff).cpu().eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(args.checkpoint).stem)
    onnx_path = output_dir / f"{stem}_float.onnx"

    dummy = torch.randn(1, args.sample_rate, dtype=torch.float32)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"waveform": {0: "batch"}, "logits": {0: "batch"}}

    with torch.no_grad():
        model.eval()
        torch.onnx.export(
            model,
            dummy,
            _long_path(onnx_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["waveform"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )

    if _inline_external_data(onnx_path):
        print(f"[INFO] merged .onnx.data into single ONNX: {onnx_path}")

    if args.check_onnx:
        import onnx

        onnx_model = onnx.load(_long_path(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"[OK] ONNX checker passed: {onnx_path}")

    if args.check_onnxruntime:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(_long_path(onnx_path), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        max_diff = 0.0
        with torch.no_grad():
            for scale in (0.02, 0.2, 1.0):
                wav = torch.from_numpy((rng.standard_normal(args.sample_rate) * scale).astype(np.float32))
                torch_out = model(wav.unsqueeze(0)).detach().cpu().numpy()
                ort_out = session.run(None, {"waveform": wav.unsqueeze(0).numpy()})[0]
                max_diff = max(max_diff, float(np.max(np.abs(torch_out - ort_out))))
        print(f"[ORT] max_abs_diff={max_diff:.6e}")
        if max_diff > 1e-4:
            raise RuntimeError("ONNXRuntime parity failed")

    print(f"[DONE] exported: {onnx_path.resolve()}")


if __name__ == "__main__":
    main()
