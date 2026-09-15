"""将 V6.1 严格整数 MFCC + int8 QAT DSCNN 导出为 ONNX（full：waveform -> logits）。

目标
----
输入原始波形（float32 ``[batch, 16000]``），输出 logits（``[batch, label_count]``）：

    waveform -> OnnxStrictIntegerMFCCFrontend -> MFCC -> int8 QDQ backbone -> logits

导出策略
--------
1. 前端：使用 ``OnnxStrictIntegerMFCCFrontend``（严格整数 MFCC 的 ONNX-traceable
   位精确重实现，见 onnx_strict_integer_mfcc.py）。
2. 后端：直接从 ``*_mfcc_int8_backbone.pt`` 的 ``state_dict`` 解析每个已量化
   Conv/Linear 的 weight/bias/scale/zero_point，重建为 ONNX QDQ 形式：
   ``QuantizeLinear -> DequantizeLinear -> float Conv/Linear``。权重按 channel
   反量化为 float（等价于 QDQ 的 weight 侧），激活逐层 per-tensor Q/DQ。

    说明：这是 ONNX 标准的 QDQ 表示，不需要再走 ``torch.ao.quantization`` 的
    ``prepare_qat``/``convert``/``load_state_dict``，因此避免了量化 packed-params
    在不同 torch 版本间 key 命名不一致（``_packed_params`` vs ``weight``）导致的
    反序列化失败，也避免了需要 sklearn/torchaudio 等训练期依赖。

用法
----
    python dscnn_kws/ONNX/export_v6_1_strict_int8_onnx.py \
        --spec <mfcc_spec.json> \
        --checkpoint <..._mfcc_int8_backbone.pt> \
        --output_dir <dir>
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

from onnx_strict_integer_mfcc import OnnxStrictIntegerMFCCFrontend


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


def build_backbone(
    *,
    num_layers: int,
    channels: int,
    label_count: int,
    sample_rate: int,
    window_stride_ms: int,
    dct_coeff: int,
) -> DSCNN:
    time_steps = calculate_time_steps(sample_rate, window_stride_ms)
    input_dim = time_steps * dct_coeff
    return DSCNN(
        input_dim=input_dim,
        label_count=label_count,
        model_size_info=make_model_size_info(num_layers, channels),
        dct_coeff=dct_coeff,
    )


# --------------------------------------------------------------------------- #
# QDQ 模块定义
# --------------------------------------------------------------------------- #
def _fake_quantize_per_tensor_affine(
    x: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int
) -> torch.Tensor:
    """用基础浮点算子实现 per-tensor 仿射 fake-quant（等价于 ONNX Q/DQ 的往返）。

    torch.quantize_per_tensor / torch.dequantize / torch.fake_quantize_per_tensor_affine
    在 torch 2.13 的 dynamo exporter 下均无 Meta/fake kernel，无法导出；这里用等价的
    ``round(x/scale)+zp -> clamp -> (q-zp)*scale`` 纯浮点实现，位语义一致。
    """
    q = torch.clamp(torch.round(x / scale) + zero_point, quant_min, quant_max)
    return (q - zero_point) * scale


class QDQConvReLU(nn.Module):
    """int8 QDQ 表示的一层量化卷积 + ReLU（激活输出侧 Q/DQ）。"""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int,
        out_scale: float,
        out_zp: int,
    ):
        super().__init__()
        self.register_buffer("weight", weight.to(torch.float32).contiguous())
        if bias is None:
            self.register_buffer("bias", torch.zeros(weight.size(0)))
        else:
            self.register_buffer("bias", bias.to(torch.float32).contiguous())
        self.stride = (int(stride[0]), int(stride[1]))
        self.padding = (int(padding[0]), int(padding[1]))
        self.groups = int(groups)
        self.out_scale = float(out_scale)
        self.out_zp = int(out_zp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, groups=self.groups)
        y = F.relu(y)
        y = _fake_quantize_per_tensor_affine(y, self.out_scale, self.out_zp, 0, 255)
        return y


class QDQLinear(nn.Module):
    """int8 QDQ 表示的量化全连接层（输入与输出两侧 Q/DQ）。"""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        in_scale: float,
        in_zp: int,
        out_scale: float,
        out_zp: int,
    ):
        super().__init__()
        self.register_buffer("weight", weight.to(torch.float32).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float32).contiguous())
        else:
            self.bias = None
        self.in_scale = float(in_scale)
        self.in_zp = int(in_zp)
        self.out_scale = float(out_scale)
        self.out_zp = int(out_zp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _fake_quantize_per_tensor_affine(x, self.in_scale, self.in_zp, 0, 255)
        y = F.linear(x, self.weight, self.bias)
        y = _fake_quantize_per_tensor_affine(y, self.out_scale, self.out_zp, 0, 255)
        return y


class QDQBackbone(nn.Module):
    """DSCNN backbone 的 QDQ 表示（输入波形特征 -> 已反量化 logits）。"""

    def __init__(
        self,
        conv_layers: nn.ModuleList,
        final_fc: QDQLinear,
        input_scale: float,
        input_zp: int,
        input_time_size: int,
        input_frequency_size: int,
    ):
        super().__init__()
        self.conv_layers = conv_layers
        self.final_fc = final_fc
        self.input_scale = float(input_scale)
        self.input_zp = int(input_zp)
        self.input_time_size = int(input_time_size)
        self.input_frequency_size = int(input_frequency_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _fake_quantize_per_tensor_affine(x, self.input_scale, self.input_zp, 0, 255)
        x = x.reshape(x.size(0), 1, self.input_time_size, self.input_frequency_size)
        for layer in self.conv_layers:
            x = layer(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        return self.final_fc(x)


class OnnxV61FullModel(nn.Module):
    """前端 + QDQ backbone 的完整导出模型。"""

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
# int8 状态直接解析 -> QDQ
# --------------------------------------------------------------------------- #
def _long_path(path: Path) -> str:
    """规避 Windows 260 字符 MAX_PATH：绝对路径加长路径前缀。"""
    s = str(path.resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def _inline_external_data(onnx_path: Path) -> bool:
    """把旁挂的 ``<name>.onnx.data`` 外部数据合并回单个 .onnx 文件。

    torch.onnx.export 对含大量 int64 常量的图（严格整数 MFCC）可能把部分
    initializer 写到 external data；重新 load 后以 save_as_external_data=False
    再保存即内联回单一文件。返回是否发生了内联。
    """
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
        # 量化 state_dict 内含 packed-params，不是 weights_only 允许的安全类型。
        obj = torch.load(load_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        state = dict(state.items()) if hasattr(state, "items") else state
    else:
        state = obj
    return {k: v for k, v in state.items() if not k.startswith("module.")}


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        v = value.detach().cpu()
        return float(v.item()) if v.numel() == 1 else float(v.flatten()[0].item())
    return float(value)


def _dequant_weight(weight: Any) -> torch.Tensor:
    if hasattr(weight, "dequantize"):
        return weight.dequantize().detach().to(torch.float32).contiguous()
    return weight.detach().to(torch.float32).contiguous()


def build_qdq_backbone(
    state: dict[str, Any],
    *,
    num_layers: int,
    channels: int,
    label_count: int,
    sample_rate: int,
    window_stride_ms: int,
    dct_coeff: int,
) -> tuple[QDQBackbone, list[dict[str, Any]]]:
    """从原始 state_dict 直接解析量化参数并构建 QDQ backbone。"""
    in_scale = _scalar(state["backbone.quant.scale"])
    in_zp = int(_scalar(state["backbone.quant.zero_point"]))

    # 用浮点 DSCNN 读每层卷积的结构（stride/padding/groups），与训练完全一致。
    backbone = build_backbone(
        num_layers=num_layers, channels=channels, label_count=label_count,
        sample_rate=sample_rate, window_stride_ms=window_stride_ms, dct_coeff=dct_coeff,
    )
    first_conv = backbone.conv_layers[0][0]
    conv_specs: list[tuple[tuple[int, int], tuple[int, int], int]] = [
        (first_conv.stride, first_conv.padding, first_conv.groups)
    ]
    for layer in backbone.conv_layers[1:]:
        conv_specs.append((layer.depthwise.stride, layer.depthwise.padding, layer.depthwise.groups))
        conv_specs.append((layer.pointwise.stride, layer.pointwise.padding, layer.pointwise.groups))

    conv_keys: list[str] = ["backbone.backbone.conv_layers.0.0"]
    for i in range(1, num_layers):
        conv_keys.append(f"backbone.backbone.conv_layers.{i}.depthwise")
        conv_keys.append(f"backbone.backbone.conv_layers.{i}.pointwise")

    conv_layers = nn.ModuleList()
    report: list[dict[str, Any]] = []
    prev_scale, prev_zp = in_scale, in_zp
    for (stride, padding, groups), key in zip(conv_specs, conv_keys):
        weight = _dequant_weight(state[key + ".weight"])
        bias_val = state.get(key + ".bias")
        bias = bias_val.detach().to(torch.float32).contiguous() if bias_val is not None else None
        out_scale = _scalar(state[key + ".scale"])
        out_zp = int(_scalar(state[key + ".zero_point"]))
        conv_layers.append(QDQConvReLU(weight, bias, stride, padding, groups, out_scale, out_zp))
        report.append({
            "key": key,
            "weight_shape": tuple(weight.shape),
            "stride": stride,
            "padding": padding,
            "groups": groups,
            "in_scale": prev_scale,
            "in_zp": prev_zp,
            "out_scale": out_scale,
            "out_zp": out_zp,
        })
        prev_scale, prev_zp = out_scale, out_zp

    # final_fc：旧版本 torch 以 _packed_params._packed_params = (weight_qint8, bias) 保存。
    packed = state["backbone.backbone.final_fc._packed_params._packed_params"]
    fc_weight = _dequant_weight(packed[0])
    fc_bias = packed[1].detach().to(torch.float32).contiguous()
    fc_out_scale = _scalar(state["backbone.backbone.final_fc.scale"])
    fc_out_zp = int(_scalar(state["backbone.backbone.final_fc.zero_point"]))
    final_fc = QDQLinear(
        fc_weight, fc_bias,
        in_scale=prev_scale, in_zp=prev_zp,
        out_scale=fc_out_scale, out_zp=fc_out_zp,
    )
    report.append({
        "key": "backbone.backbone.final_fc",
        "weight_shape": tuple(fc_weight.shape),
        "in_scale": prev_scale,
        "in_zp": prev_zp,
        "out_scale": fc_out_scale,
        "out_zp": fc_out_zp,
    })

    time_steps = calculate_time_steps(sample_rate, window_stride_ms)
    qdq = QDQBackbone(
        conv_layers=conv_layers,
        final_fc=final_fc,
        input_scale=in_scale,
        input_zp=in_zp,
        input_time_size=time_steps,
        input_frequency_size=dct_coeff,
    )
    return qdq, report


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V6.1 strict-integer MFCC + int8 QAT DSCNN to ONNX")
    parser.add_argument("--spec", required=True, help="V6.1 *_bit_accurate_mfcc_spec.json 路径")
    parser.add_argument("--checkpoint", required=True, help="*_mfcc_int8_backbone.pt 路径")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "models_v6_1_strict_int8"))
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--label_count", type=int, default=2)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--window_stride_ms", type=int, default=32)
    parser.add_argument("--dct_coeff", type=int, default=10)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic_batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check_onnx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check_onnxruntime", action="store_true", default=False)
    parser.add_argument("--inspect", action="store_true", default=False, help="仅打印解析出的量化 backbone 参数后退出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frontend = OnnxStrictIntegerMFCCFrontend(args.spec)

    state = _load_state_dict(Path(args.checkpoint))
    qdq_backbone, report = build_qdq_backbone(
        state,
        num_layers=args.layers,
        channels=args.channels,
        label_count=args.label_count,
        sample_rate=args.sample_rate,
        window_stride_ms=args.window_stride_ms,
        dct_coeff=args.dct_coeff,
    )

    if args.inspect:
        print("[INSPECT] input quant: scale / zero_point =", report[0]["in_scale"], report[0]["in_zp"])
        for row in report:
            print(
                f"  {row['key']:58s} w={str(row['weight_shape']):22s} "
                f"g={row.get('groups', 1):>2} in=({row['in_scale']:.6g}, {row['in_zp']:>3}) "
                f"out=({row['out_scale']:.6g}, {row['out_zp']:>3})"
            )
        return

    model = OnnxV61FullModel(frontend, qdq_backbone, args.dct_coeff).cpu().eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(args.checkpoint).stem)
    onnx_path = output_dir / f"{stem}_v6_1_strict_int8.onnx"

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
        with torch.no_grad():
            torch_out = model(dummy).detach().cpu().numpy()
        ort_out = session.run(None, {"waveform": dummy.cpu().numpy()})[0]
        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
        print(f"[ORT] max_abs_diff={max_abs_diff:.6e}")
        print("[ORT] torch_out=", torch_out)
        print("[ORT] ort_out=", ort_out)

    print(f"[DONE] exported: {onnx_path.resolve()}")


if __name__ == "__main__":
    main()