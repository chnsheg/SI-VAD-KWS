"""Export a v6.1 converted INT8 KWS checkpoint as a bit-exact ONNX graph.

The checkpoint is loaded into PyTorch's converted FBGEMM modules.  A QAT
skeleton is required only to instantiate those modules; this exporter never
reloads the checkpoint into a floating DSCNN model.

The default exact graph stores the original INT8 checkpoint codes and lowers
FBGEMM requantization explicitly.  The FLOAT Conv/Gemm nodes only accumulate
centered integer codes: all products and sums are exactly representable by
float32 for this fixed DSCNN.  They are not a floating-model export.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
V6_ROOT = REPO_ROOT / "dscnn_kws" / "quantization" / "bit_accurate_mfcc_experiments_v6_strict_integer_mfcc"
V61_ROOT = REPO_ROOT / "dscnn_kws" / "quantization" / "bit_accurate_mfcc_experiments_v6_1_strict_scale_calibration"
for _directory in (REPO_ROOT, V6_ROOT, V61_ROOT):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from dscnn_kws.quantization import qat_bit_accurate_mfcc_accuracy_first as base
from strict_integer_mfcc import StrictIntegerMFCCFloatAdapter


WAVEFORM_SHAPE = [1, 16_000]
LOGITS_SHAPE = [1, 2]
OPSET_VERSION = 17
EXACT_FORMAT = "ONNX opset 17 direct INT8 parameters with exact FBGEMM lowering"
EXPORTER_VERSION = "v6.1-direct-int8-exact-1"
_STAGE_LABELS_BY_MODULE_NAME = {
    "conv_0": "conv0",
    "conv_1_depthwise": "conv1_depthwise",
    "conv_1_pointwise": "conv1_pointwise",
    "conv_2_depthwise": "conv2_depthwise",
    "conv_2_pointwise": "conv2_pointwise",
    "conv_3_depthwise": "conv3_depthwise",
    "conv_3_pointwise": "conv3_pointwise",
    "conv_4_depthwise": "conv4_depthwise",
    "conv_4_pointwise": "conv4_pointwise",
}


class StrictParityError(RuntimeError):
    """Raised when the materialized graph is not bit-exact with FBGEMM."""

    def __init__(self, report: dict[str, Any]) -> None:
        nonzero_stages = {
            name: count
            for name, count in report.get("per_stage_code_mismatch_counts", {}).items()
            if count
        }
        super().__init__(
            "strict v6.1 INT8 parity failed: "
            f"max_abs_error={report['max_abs_error']!r}, "
            f"float_word_mismatch_count={report['float_word_mismatch_count']}, "
            f"nonzero_stage_code_mismatches={nonzero_stages}"
        )
        self.report = report


class OutputPathLockError(RuntimeError):
    """Raised when another exporter owns the requested output path."""


class _FixedShapeDSCNN(nn.Module):
    """Quantized DSCNN tail with the fixed v6.1 one-second input contract."""

    def __init__(self, dscnn: nn.Module) -> None:
        super().__init__()
        self.input_time_size = int(dscnn.input_time_size)
        self.input_frequency_size = int(dscnn.input_frequency_size)
        self.conv_layers = dscnn.conv_layers
        self.avg_pool = dscnn.avg_pool
        self.dropout = dscnn.dropout
        self.final_fc = dscnn.final_fc

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.reshape(1, 1, self.input_time_size, self.input_frequency_size)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.final_fc(x)


class FixedShapeV61Int8Model(nn.Module):
    """Fixed-shape waveform-to-logits wrapper over converted v6.1 modules."""

    def __init__(self, quantized_model: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = quantized_model.feature_extractor
        self.dct_coeff = int(quantized_model.dct_coeff)
        self.quant = quantized_model.backbone.quant
        self.dscnn = _FixedShapeDSCNN(quantized_model.backbone.backbone)
        self.dequant = quantized_model.backbone.dequant

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(waveform)
        features = features[:, : self.dct_coeff, :]
        features = features.permute(0, 2, 1).reshape(1, -1)
        return self.dequant(self.dscnn(self.quant(features)))


def _qat_args() -> argparse.Namespace:
    return argparse.Namespace(
        sample_rate=16_000,
        window_stride_ms=32,
        dct_coeff=10,
        backend="fbgemm",
        quantize_frontend=False,
    )


def _filesystem_path(path: Path) -> str:
    """Return a Windows extended-length path when the source path needs it."""

    resolved = str(path.resolve())
    if os.name == "nt" and len(resolved) >= 248 and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def _require_file(path: Path, argument_name: str) -> None:
    if not os.path.isfile(_filesystem_path(path)):
        raise FileNotFoundError(f"{argument_name} is not a file: {path}")


def build_reference(checkpoint: Path, spec: Path) -> nn.Module:
    """Recreate the converted FBGEMM reference and strictly load its state."""

    _require_file(checkpoint, "checkpoint")
    _require_file(spec, "spec")

    if "fbgemm" not in torch.backends.quantized.supported_engines:
        raise RuntimeError(
            "v6.1 converted checkpoint requires FBGEMM, but FBGEMM is not available in "
            f"{torch.backends.quantized.supported_engines}"
        )

    # v6.1's direct INT8 graph is defined for the historical global-pooling
    # DSCNN only.  A temporal head has a different classifier shape and a
    # different reduction (AdaptiveAvgPool2d(B, 1)); silently feeding such a
    # checkpoint into this exporter would either fail deep in torch or, worse,
    # produce a graph whose logits do not match the checkpoint.  Fail early
    # with an actionable message and use export_full_onnx_friendly.py for a
    # temporal FP32/ONNX export until a temporal INT8 lowering is implemented.
    try:
        payload = torch.load(_filesystem_path(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(_filesystem_path(checkpoint), map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise TypeError(f"checkpoint does not contain a state_dict: {checkpoint}")
    state_dict = payload["state_dict"]
    if isinstance(state_dict, dict) and any("temporal_fc." in str(key) for key in state_dict):
        raise ValueError(
            "v6.1 direct INT8 exporter does not support temporal DSCNN pooling; "
            "use export_full_onnx_friendly.py for temporal models or export a global-pooling checkpoint"
        )

    previous_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = "fbgemm"
    try:
        args = _qat_args()
        backbone = base.build_backbone(args, num_layers=5, channels=64, label_count=2)
        qat_model = base.BitAccurateMFCCBackboneModel(
            frontend=StrictIntegerMFCCFloatAdapter(Path(_filesystem_path(spec))),
            backbone=backbone,
            dct_coeff=args.dct_coeff,
        )
        prepared = base.prepare_model_for_qat(qat_model, args)
        converted = base.quantize_qat_model(prepared)
        reference = base.BitAccurateMFCCQuantizedBackboneModel(
            frontend=StrictIntegerMFCCFloatAdapter(Path(_filesystem_path(spec))),
            quantized_backbone=converted.backbone,
            dct_coeff=args.dct_coeff,
        )
        reference.load_state_dict(payload["state_dict"], strict=True)
        return reference.eval()
    finally:
        torch.backends.quantized.engine = previous_engine


def deterministic_inputs(parity_samples: int = 19) -> list[torch.Tensor]:
    """Return a reproducible corpus with extrema and boundary candidates.

    The fixed waveforms exercise silence, tonal audio, noise, input extrema,
    saturation, alternating polarity, and PCM/requantization boundary
    candidates. They are targeted regression inputs, not a mathematical proof
    that every waveform-level boundary is covered.
    """

    fixed_count = 9
    if parity_samples < fixed_count:
        raise ValueError(
            f"parity_samples must be at least {fixed_count} to include the fixed regression corpus"
        )

    sample_index = torch.arange(WAVEFORM_SHAPE[1], dtype=torch.float32)
    sine = 0.25 * torch.sin(2.0 * math.pi * 440.0 * sample_index / 16_000.0)
    noise_generator = torch.Generator().manual_seed(20_260_728)
    noise = torch.rand(WAVEFORM_SHAPE, generator=noise_generator, dtype=torch.float32) * 0.5 - 0.25
    alternating = torch.where(
        sample_index.remainder(2) == 0,
        torch.ones_like(sample_index),
        -torch.ones_like(sample_index),
    ).unsqueeze(0)
    pcm_tie_candidates = torch.tensor(
        [-1.5, -0.5, 0.5, 1.5], dtype=torch.float32
    ).repeat(WAVEFORM_SHAPE[1] // 4) / float(1 << 15)
    waveforms = [
        torch.zeros(WAVEFORM_SHAPE, dtype=torch.float32),
        sine.unsqueeze(0),
        noise,
        torch.ones(WAVEFORM_SHAPE, dtype=torch.float32),
        -torch.ones(WAVEFORM_SHAPE, dtype=torch.float32),
        torch.full(WAVEFORM_SHAPE, 1.25, dtype=torch.float32),
        torch.full(WAVEFORM_SHAPE, -1.25, dtype=torch.float32),
        alternating,
        pcm_tie_candidates.unsqueeze(0),
    ]
    for seed in range(20_260_729, 20_260_729 + parity_samples - fixed_count):
        generator = torch.Generator().manual_seed(seed)
        waveforms.append(
            torch.rand(WAVEFORM_SHAPE, generator=generator, dtype=torch.float32) * 0.5 - 0.25
        )
    return waveforms


def _static_shape(value_info: Any) -> list[int]:
    dimensions = value_info.shape
    if any(not isinstance(dimension, int) for dimension in dimensions):
        raise AssertionError(f"expected static I/O dimensions, got {dimensions}")
    return [int(dimension) for dimension in dimensions]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_filesystem_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_onnx_path(output_path: Path, role: str) -> Path:
    suffix = output_path.suffix or ".onnx"
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output_path.stem}.{role}.",
        suffix=suffix,
        dir=_filesystem_path(output_path.parent),
    )
    os.close(descriptor)
    return Path(raw_path)


def _remove_owned_file(path: Path) -> None:
    filesystem_path = _filesystem_path(path)
    if os.path.exists(filesystem_path):
        os.unlink(filesystem_path)


def _output_lock_path(output_path: Path) -> Path:
    resolved_output = output_path.resolve()
    return resolved_output.with_name(f".{resolved_output.name}.lock")


@contextmanager
def _exclusive_output_lock(output_path: Path) -> Iterable[Path]:
    """Acquire a kernel-released output lock without waiting for a competitor."""

    lock_path = _output_lock_path(output_path)
    descriptor = os.open(
        _filesystem_path(lock_path),
        os.O_CREAT | os.O_RDWR,
    )

    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            raise OutputPathLockError(f"output path is already locked: {output_path}") from error
        raise

    try:
        yield lock_path
    finally:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _same_resolved_filesystem_path(first: Path, second: Path) -> bool:
    first_identity = os.path.normcase(os.path.normpath(str(first.resolve())))
    second_identity = os.path.normcase(os.path.normpath(str(second.resolve())))
    return first_identity == second_identity


def _export_staging(candidate: nn.Module, waveform: torch.Tensor, staging_path: Path) -> None:
    with torch.no_grad():
        torch.onnx.export(
            candidate,
            waveform,
            _filesystem_path(staging_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["waveform"],
            output_names=["logits"],
            dynamo=False,
        )


def _materialize_qoperator(staging_path: Path, output_path: Path) -> None:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session_options.optimized_model_filepath = _filesystem_path(output_path)
    ort.InferenceSession(
        _filesystem_path(staging_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def _quantized_conv_layers(dscnn: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return the converted convolution modules in the model's forward order."""

    layers: list[tuple[str, nn.Module]] = []
    for layer_index, layer in enumerate(dscnn.conv_layers):
        if layer_index == 0:
            layers.append((f"conv_{layer_index}", layer[0]))
        else:
            layers.extend(
                [
                    (f"conv_{layer_index}_depthwise", layer.depthwise),
                    (f"conv_{layer_index}_pointwise", layer.pointwise),
                ]
            )
    if len(layers) != 9:
        raise AssertionError(f"expected nine converted convolution layers, found {len(layers)}")
    return layers


def _numpy_scalar(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1)[0].item()
    return np.asarray(value, dtype=dtype)


def _append_initializer(
    initializers: list[onnx.TensorProto],
    name: str,
    value: np.ndarray,
) -> None:
    if any(initializer.name == name for initializer in initializers):
        raise AssertionError(f"duplicate ONNX initializer name: {name}")
    initializers.append(numpy_helper.from_array(np.ascontiguousarray(value), name=name))


def _require_exact_float32_accumulator(
    *,
    prefix: str,
    weight_codes: np.ndarray,
    weight_zero_points: np.ndarray,
    input_zero_point: int,
) -> int:
    """Reject checkpoints whose centered dot products exceed float32's exact range."""

    centered_weights = weight_codes.astype(np.int64) - weight_zero_points.astype(np.int64).reshape(
        (weight_codes.shape[0],) + (1,) * (weight_codes.ndim - 1)
    )
    max_centered_input = max(int(input_zero_point), 255 - int(input_zero_point))
    maximum_accumulator = int(
        np.max(np.sum(np.abs(centered_weights), axis=tuple(range(1, weight_codes.ndim))))
        * max_centered_input
    )
    if maximum_accumulator >= (1 << 24):
        raise RuntimeError(
            f"{prefix} may exceed exact float32 integer accumulation: "
            f"bound={maximum_accumulator}, limit={1 << 24}"
        )
    return maximum_accumulator


def _frontend_code_prefix(
    staging_path: Path,
) -> tuple[list[onnx.ValueInfoProto], list[onnx.NodeProto], list[onnx.TensorProto], str]:
    """Take the exact strict-MFCC slice ending at the first UINT8 backbone input."""

    source = onnx.load(_filesystem_path(staging_path))
    producer_by_value: dict[str, int] = {}
    for index, node in enumerate(source.graph.node):
        for output in node.output:
            if output:
                producer_by_value[output] = index

    raw_convs = [
        (index, node) for index, node in enumerate(source.graph.node) if node.op_type == "Conv"
    ]
    if len(raw_convs) != 9:
        raise AssertionError(
            "staging graph must contain exactly the nine raw QDQ backbone convolutions, found "
            f"{len(raw_convs)}"
        )
    first_conv_index, _ = raw_convs[0]
    frontend_dequantizers = [
        node
        for node in source.graph.node[:first_conv_index]
        if node.op_type == "DequantizeLinear"
    ]
    if not frontend_dequantizers or not frontend_dequantizers[0].input[0]:
        raise AssertionError("staging graph is missing the quantized feature-code boundary")
    # Quantized reshape exports as DQ -> Reshape -> Q.  Reshape the original
    # UINT8 codes ourselves, so this boundary remains code-domain only.
    activation_codes = frontend_dequantizers[0].input[0]

    required_values = [activation_codes]
    required_nodes: set[int] = set()
    while required_values:
        value = required_values.pop()
        producer_index = producer_by_value.get(value)
        if producer_index is None or producer_index in required_nodes:
            continue
        required_nodes.add(producer_index)
        required_values.extend(
            input_name for input_name in source.graph.node[producer_index].input if input_name
        )

    prefix_nodes = [
        copy.deepcopy(node)
        for index, node in enumerate(source.graph.node)
        if index in required_nodes
    ]
    required_initializer_names = {
        input_name for node in prefix_nodes for input_name in node.input if input_name
    }
    prefix_initializers = [
        copy.deepcopy(initializer)
        for initializer in source.graph.initializer
        if initializer.name in required_initializer_names
    ]
    return (
        [copy.deepcopy(value_info) for value_info in source.graph.input],
        prefix_nodes,
        prefix_initializers,
        activation_codes,
    )


def _append_exact_conv(
    *,
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    prefix: str,
    input_codes: str,
    input_scale: float,
    input_zero_point: int,
    module: nn.Module,
) -> tuple[str, float, int, int]:
    """Append the code-domain equivalent of one FBGEMM quantized convolution."""

    weight, bias = module._weight_bias()
    if bias is None:
        raise AssertionError(f"{prefix} unexpectedly has no folded float32 bias")
    if weight.qscheme() != torch.per_channel_affine or weight.q_per_channel_axis() != 0:
        raise AssertionError(f"{prefix} must use per-output-channel affine INT8 weights")

    weight_codes = weight.int_repr().detach().cpu().numpy().astype(np.int8, copy=False)
    weight_scales = weight.q_per_channel_scales().detach().cpu().numpy().astype(np.float32)
    weight_zero_points = weight.q_per_channel_zero_points().detach().cpu().numpy().astype(np.int8)
    output_scale = np.float32(float(module.scale))
    output_zero_point = int(module.zero_point)
    output_channels = int(weight_codes.shape[0])
    if weight_scales.shape != (output_channels,) or weight_zero_points.shape != (output_channels,):
        raise AssertionError(f"{prefix} per-channel qparams do not match its weight layout")
    accumulator_bound = _require_exact_float32_accumulator(
        prefix=prefix,
        weight_codes=weight_codes,
        weight_zero_points=weight_zero_points,
        input_zero_point=input_zero_point,
    )

    input_scale_f32 = np.float32(input_scale)
    bias_f32 = bias.detach().cpu().numpy().astype(np.float32, copy=False)
    if bias_f32.shape != (output_channels,):
        raise AssertionError(f"{prefix} bias layout does not match its weight layout")

    _append_initializer(initializers, f"backbone_{prefix}_weight_codes", weight_codes)
    _append_initializer(initializers, f"backbone_{prefix}_weight_scales", weight_scales)
    _append_initializer(initializers, f"backbone_{prefix}_weight_zero_points", weight_zero_points)
    _append_initializer(initializers, f"backbone_{prefix}_bias_f32", bias_f32)
    _append_initializer(
        initializers,
        f"backbone_{prefix}_input_scale",
        _numpy_scalar(input_scale_f32, np.float32),
    )
    _append_initializer(
        initializers,
        f"backbone_{prefix}_input_zero_point",
        _numpy_scalar(input_zero_point, np.uint8),
    )
    _append_initializer(
        initializers,
        f"backbone_{prefix}_output_scale",
        _numpy_scalar(output_scale, np.float32),
    )
    _append_initializer(
        initializers,
        f"backbone_{prefix}_output_zero_point",
        _numpy_scalar(output_zero_point, np.uint8),
    )
    _append_initializer(
        initializers,
        f"backbone_{prefix}_weight_channel_shape",
        np.asarray([output_channels, 1, 1, 1], dtype=np.int64),
    )
    _append_initializer(
        initializers,
        f"backbone_{prefix}_output_channel_shape",
        np.asarray([1, output_channels, 1, 1], dtype=np.int64),
    )
    _append_initializer(
        initializers,
        f"backbone_{prefix}_clip_maximum",
        _numpy_scalar(255.0, np.float32),
    )

    input_f32 = f"{prefix}_input_f32"
    input_zp_f32 = f"{prefix}_input_zp_f32"
    input_centered = f"{prefix}_input_centered"
    weight_f32 = f"{prefix}_weight_f32"
    weight_zp_f32 = f"{prefix}_weight_zp_f32"
    weight_zp_broadcast = f"{prefix}_weight_zp_broadcast"
    weight_centered = f"{prefix}_weight_centered"
    accumulator = f"{prefix}_accumulator"
    multiplier = f"{prefix}_requant_multiplier"
    multiplier_broadcast = f"{prefix}_requant_multiplier_broadcast"
    scaled_accumulator = f"{prefix}_scaled_accumulator"
    bias_scaled = f"{prefix}_bias_scaled"
    bias_broadcast = f"{prefix}_bias_broadcast"
    biased = f"{prefix}_biased"
    rounded = f"{prefix}_rounded"
    output_zp_f32 = f"{prefix}_output_zp_f32"
    shifted = f"{prefix}_shifted"
    clipped = f"{prefix}_clipped"
    output_codes = f"{prefix}_output_codes"

    nodes.extend(
        [
            helper.make_node("Cast", [input_codes], [input_f32], name=f"exact_{prefix}_cast_input", to=TensorProto.FLOAT),
            helper.make_node(
                "Cast",
                [f"backbone_{prefix}_input_zero_point"],
                [input_zp_f32],
                name=f"exact_{prefix}_cast_input_zp",
                to=TensorProto.FLOAT,
            ),
            helper.make_node("Sub", [input_f32, input_zp_f32], [input_centered], name=f"exact_{prefix}_center_input"),
            helper.make_node(
                "Cast",
                [f"backbone_{prefix}_weight_codes"],
                [weight_f32],
                name=f"exact_{prefix}_cast_weight",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Cast",
                [f"backbone_{prefix}_weight_zero_points"],
                [weight_zp_f32],
                name=f"exact_{prefix}_cast_weight_zp",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Reshape",
                [weight_zp_f32, f"backbone_{prefix}_weight_channel_shape"],
                [weight_zp_broadcast],
                name=f"exact_{prefix}_reshape_weight_zp",
            ),
            helper.make_node(
                "Sub",
                [weight_f32, weight_zp_broadcast],
                [weight_centered],
                name=f"exact_{prefix}_center_weight",
            ),
            helper.make_node(
                "Conv",
                [input_centered, weight_centered],
                [accumulator],
                name=f"exact_fbgemm_{prefix}",
                kernel_shape=list(module.kernel_size),
                strides=list(module.stride),
                pads=[*module.padding, *module.padding],
                dilations=list(module.dilation),
                group=int(module.groups),
            ),
            helper.make_node(
                "Mul",
                [f"backbone_{prefix}_input_scale", f"backbone_{prefix}_weight_scales"],
                [multiplier],
                name=f"exact_{prefix}_multiply_scales",
            ),
            helper.make_node(
                "Div",
                [multiplier, f"backbone_{prefix}_output_scale"],
                [multiplier_broadcast],
                name=f"exact_{prefix}_divide_output_scale",
            ),
            helper.make_node(
                "Reshape",
                [multiplier_broadcast, f"backbone_{prefix}_output_channel_shape"],
                [scaled_accumulator],
                name=f"exact_{prefix}_reshape_multiplier",
            ),
            helper.make_node(
                "Mul",
                [accumulator, scaled_accumulator],
                [scaled_accumulator + "_applied"],
                name=f"exact_{prefix}_requantize_accumulator",
            ),
            helper.make_node(
                "Div",
                [f"backbone_{prefix}_bias_f32", f"backbone_{prefix}_output_scale"],
                [bias_scaled],
                name=f"exact_{prefix}_scale_bias",
            ),
            helper.make_node(
                "Reshape",
                [bias_scaled, f"backbone_{prefix}_output_channel_shape"],
                [bias_broadcast],
                name=f"exact_{prefix}_reshape_bias",
            ),
            helper.make_node(
                "Add",
                [scaled_accumulator + "_applied", bias_broadcast],
                [biased],
                name=f"exact_{prefix}_add_bias",
            ),
            helper.make_node("Round", [biased], [rounded], name=f"exact_{prefix}_round_even"),
            helper.make_node(
                "Cast",
                [f"backbone_{prefix}_output_zero_point"],
                [output_zp_f32],
                name=f"exact_{prefix}_cast_output_zp",
                to=TensorProto.FLOAT,
            ),
            helper.make_node("Add", [rounded, output_zp_f32], [shifted], name=f"exact_{prefix}_add_output_zp"),
            helper.make_node(
                "Clip",
                [shifted, output_zp_f32, f"backbone_{prefix}_clip_maximum"],
                [clipped],
                name=f"exact_{prefix}_relu_saturate",
            ),
            helper.make_node(
                "Cast",
                [clipped],
                [output_codes],
                name=f"exact_{prefix}_cast_output",
                to=TensorProto.UINT8,
            ),
        ]
    )
    return output_codes, float(output_scale), output_zero_point, accumulator_bound


def _append_exact_global_average_pool(
    *,
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    input_codes: str,
    input_zero_point: int,
) -> str:
    """Lower quantized adaptive global average pooling in the UINT8 code domain."""

    _append_initializer(
        initializers,
        "exact_pool_input_zero_point",
        _numpy_scalar(input_zero_point, np.uint8),
    )
    _append_initializer(initializers, "backbone_pool_clip_minimum", _numpy_scalar(0.0, np.float32))
    _append_initializer(initializers, "backbone_pool_clip_maximum", _numpy_scalar(255.0, np.float32))
    nodes.extend(
        [
            helper.make_node("Cast", [input_codes], ["pool_input_f32"], name="exact_pool_cast_input", to=TensorProto.FLOAT),
            helper.make_node(
                "Cast",
                ["exact_pool_input_zero_point"],
                ["pool_input_zero_point_f32"],
                name="exact_pool_cast_input_zero_point",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Sub",
                ["pool_input_f32", "pool_input_zero_point_f32"],
                ["pool_centered_codes"],
                name="exact_pool_center_input",
            ),
            helper.make_node(
                "GlobalAveragePool",
                ["pool_centered_codes"],
                ["pool_centered_mean"],
                name="exact_pool_mean_centered_codes",
            ),
            helper.make_node(
                "Round",
                ["pool_centered_mean"],
                ["pool_rounded_centered_codes"],
                name="exact_pool_round_centered_even",
            ),
            helper.make_node(
                "Add",
                ["pool_rounded_centered_codes", "pool_input_zero_point_f32"],
                ["pool_shifted_codes"],
                name="exact_pool_restore_zero_point",
            ),
            helper.make_node(
                "Clip",
                ["pool_shifted_codes", "backbone_pool_clip_minimum", "backbone_pool_clip_maximum"],
                ["pool_clipped"],
                name="exact_pool_saturate",
            ),
            helper.make_node("Cast", ["pool_clipped"], ["pool_codes"], name="exact_pool_cast_output", to=TensorProto.UINT8),
        ]
    )
    return "pool_codes"


def _append_exact_linear(
    *,
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    input_codes: str,
    input_scale: float,
    input_zero_point: int,
    module: nn.Module,
) -> tuple[str, int]:
    """Append the code-domain equivalent of the final FBGEMM quantized linear."""

    weight, bias = module._weight_bias()
    if bias is None:
        raise AssertionError("final_fc unexpectedly has no float32 bias")
    if weight.qscheme() != torch.per_channel_affine or weight.q_per_channel_axis() != 0:
        raise AssertionError("final_fc must use per-output-channel affine INT8 weights")

    weight_codes = weight.int_repr().detach().cpu().numpy().astype(np.int8, copy=False)
    weight_scales = weight.q_per_channel_scales().detach().cpu().numpy().astype(np.float32)
    weight_zero_points = weight.q_per_channel_zero_points().detach().cpu().numpy().astype(np.int8)
    output_scale = np.float32(float(module.scale))
    output_zero_point = int(module.zero_point)
    output_channels = int(weight_codes.shape[0])
    input_features = int(weight_codes.shape[1])
    accumulator_bound = _require_exact_float32_accumulator(
        prefix="final_fc",
        weight_codes=weight_codes,
        weight_zero_points=weight_zero_points,
        input_zero_point=input_zero_point,
    )

    _append_initializer(initializers, "backbone_final_fc_weight_codes", weight_codes)
    _append_initializer(initializers, "backbone_final_fc_weight_scales", weight_scales)
    _append_initializer(initializers, "backbone_final_fc_weight_zero_points", weight_zero_points)
    _append_initializer(
        initializers,
        "backbone_final_fc_bias_f32",
        bias.detach().cpu().numpy().astype(np.float32, copy=False),
    )
    _append_initializer(initializers, "backbone_final_fc_input_scale", _numpy_scalar(np.float32(input_scale), np.float32))
    _append_initializer(initializers, "backbone_final_fc_input_zero_point", _numpy_scalar(input_zero_point, np.uint8))
    _append_initializer(initializers, "backbone_final_fc_output_scale", _numpy_scalar(output_scale, np.float32))
    _append_initializer(initializers, "backbone_final_fc_output_zero_point", _numpy_scalar(output_zero_point, np.uint8))
    _append_initializer(initializers, "backbone_final_fc_input_shape", np.asarray([1, input_features], dtype=np.int64))
    _append_initializer(initializers, "backbone_final_fc_weight_channel_shape", np.asarray([output_channels, 1], dtype=np.int64))
    _append_initializer(initializers, "backbone_final_fc_output_channel_shape", np.asarray([1, output_channels], dtype=np.int64))
    _append_initializer(initializers, "backbone_final_fc_clip_minimum", _numpy_scalar(0.0, np.float32))
    _append_initializer(initializers, "backbone_final_fc_clip_maximum", _numpy_scalar(255.0, np.float32))

    nodes.extend(
        [
            helper.make_node("Reshape", [input_codes, "backbone_final_fc_input_shape"], ["fc_input_codes"], name="exact_fc_reshape_input"),
            helper.make_node("Cast", ["fc_input_codes"], ["fc_input_f32"], name="exact_fc_cast_input", to=TensorProto.FLOAT),
            helper.make_node("Cast", ["backbone_final_fc_input_zero_point"], ["fc_input_zp_f32"], name="exact_fc_cast_input_zp", to=TensorProto.FLOAT),
            helper.make_node("Sub", ["fc_input_f32", "fc_input_zp_f32"], ["fc_input_centered"], name="exact_fc_center_input"),
            helper.make_node("Cast", ["backbone_final_fc_weight_codes"], ["fc_weight_f32"], name="exact_fc_cast_weight", to=TensorProto.FLOAT),
            helper.make_node("Cast", ["backbone_final_fc_weight_zero_points"], ["fc_weight_zp_f32"], name="exact_fc_cast_weight_zp", to=TensorProto.FLOAT),
            helper.make_node("Reshape", ["fc_weight_zp_f32", "backbone_final_fc_weight_channel_shape"], ["fc_weight_zp_broadcast"], name="exact_fc_reshape_weight_zp"),
            helper.make_node("Sub", ["fc_weight_f32", "fc_weight_zp_broadcast"], ["fc_weight_centered"], name="exact_fc_center_weight"),
            helper.make_node("Gemm", ["fc_input_centered", "fc_weight_centered"], ["fc_accumulator"], name="exact_fbgemm_final_fc", transB=1),
            helper.make_node("Mul", ["backbone_final_fc_input_scale", "backbone_final_fc_weight_scales"], ["fc_requant_multiplier"], name="exact_fc_multiply_scales"),
            helper.make_node("Div", ["fc_requant_multiplier", "backbone_final_fc_output_scale"], ["fc_requant_multiplier_broadcast"], name="exact_fc_divide_output_scale"),
            helper.make_node("Reshape", ["fc_requant_multiplier_broadcast", "backbone_final_fc_output_channel_shape"], ["fc_multiplier"], name="exact_fc_reshape_multiplier"),
            helper.make_node("Mul", ["fc_accumulator", "fc_multiplier"], ["fc_scaled_accumulator"], name="exact_fc_requantize_accumulator"),
            helper.make_node("Div", ["backbone_final_fc_bias_f32", "backbone_final_fc_output_scale"], ["fc_bias_scaled"], name="exact_fc_scale_bias"),
            helper.make_node("Reshape", ["fc_bias_scaled", "backbone_final_fc_output_channel_shape"], ["fc_bias_broadcast"], name="exact_fc_reshape_bias"),
            helper.make_node("Add", ["fc_scaled_accumulator", "fc_bias_broadcast"], ["fc_biased"], name="exact_fc_add_bias"),
            helper.make_node("Round", ["fc_biased"], ["fc_rounded"], name="exact_fc_round_even"),
            helper.make_node("Cast", ["backbone_final_fc_output_zero_point"], ["fc_output_zp_f32"], name="exact_fc_cast_output_zp", to=TensorProto.FLOAT),
            helper.make_node("Add", ["fc_rounded", "fc_output_zp_f32"], ["fc_shifted"], name="exact_fc_add_output_zp"),
            helper.make_node("Clip", ["fc_shifted", "backbone_final_fc_clip_minimum", "backbone_final_fc_clip_maximum"], ["fc_clipped"], name="exact_fc_saturate"),
            helper.make_node("Cast", ["fc_clipped"], ["logit_codes"], name="exact_fc_cast_output", to=TensorProto.UINT8),
            helper.make_node("DequantizeLinear", ["logit_codes", "backbone_final_fc_output_scale", "backbone_final_fc_output_zero_point"], ["logits"], name="exact_fc_dequantize_output"),
        ]
    )
    return "logit_codes", accumulator_bound


def _materialize_exact_fbgemm(
    staging_path: Path,
    output_path: Path,
    reference: nn.Module,
    *,
    checkpoint_sha256: str,
    spec_sha256: str,
    verification_scope: str,
) -> tuple[dict[str, str], dict[str, int]]:
    """Replace the exported QOperator tail with exact direct-INT8 lowering."""

    graph_inputs, prefix_nodes, initializers, activation_codes = _frontend_code_prefix(staging_path)
    nodes = list(prefix_nodes)
    converted = reference.backbone
    _append_initializer(
        initializers,
        "backbone_input_code_shape",
        np.asarray(
            [
                1,
                1,
                int(converted.backbone.input_time_size),
                int(converted.backbone.input_frequency_size),
            ],
            dtype=np.int64,
        ),
    )
    nodes.append(
        helper.make_node(
            "Reshape",
            [activation_codes, "backbone_input_code_shape"],
            ["backbone_input_codes"],
            name="exact_backbone_reshape_feature_codes",
        )
    )
    input_scale = float(converted.quant.scale.detach().cpu().reshape(-1)[0].item())
    input_zero_point = int(converted.quant.zero_point.detach().cpu().reshape(-1)[0].item())
    codes = "backbone_input_codes"
    stage_tensor_names = {"input_quant": activation_codes}
    float32_accumulator_bounds: dict[str, int] = {}
    for name, module in _quantized_conv_layers(converted.backbone):
        codes, input_scale, input_zero_point, accumulator_bound = _append_exact_conv(
            nodes=nodes,
            initializers=initializers,
            prefix=name,
            input_codes=codes,
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            module=module,
        )
        float32_accumulator_bounds[name] = accumulator_bound
        stage_tensor_names[_STAGE_LABELS_BY_MODULE_NAME[name]] = codes
    codes = _append_exact_global_average_pool(
        nodes=nodes,
        initializers=initializers,
        input_codes=codes,
        input_zero_point=input_zero_point,
    )
    stage_tensor_names["global_avg_pool"] = codes
    stage_tensor_names["fc_input"] = "fc_input_codes"
    final_codes, final_accumulator_bound = _append_exact_linear(
        nodes=nodes,
        initializers=initializers,
        input_codes=codes,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        module=converted.backbone.final_fc,
    )
    float32_accumulator_bounds["final_fc"] = final_accumulator_bound
    stage_tensor_names["final_fc"] = final_codes
    graph = helper.make_graph(
        nodes,
        "v6_1_exact_fbgemm_int8",
        graph_inputs,
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, LOGITS_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="v6_1_direct_int8_export",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    for key, value in {
        "format": EXACT_FORMAT,
        "exporter_version": EXPORTER_VERSION,
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_spec_sha256": spec_sha256,
        "verification_scope": verification_scope,
        "checkpoint_parameter_source": "converted FBGEMM INT8 checkpoint",
        "exact_logit_codes": final_codes,
    }.items():
        model.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(model)
    onnx.save(model, _filesystem_path(output_path))
    return stage_tensor_names, float32_accumulator_bounds


def _graph_report(output_path: Path, *, mode: str) -> dict[str, Any]:
    model = onnx.load(_filesystem_path(output_path))
    onnx.checker.check_model(model)
    op_counts = Counter(node.op_type for node in model.graph.node)
    domain_op_counts = Counter((node.domain, node.op_type) for node in model.graph.node)
    nonportable_nodes = [
        {"domain": domain, "op_type": op_type, "count": count}
        for (domain, op_type), count in sorted(domain_op_counts.items())
        if domain
    ]

    qlinear_conv_count = op_counts["QLinearConv"]
    qlinear_global_average_pool_count = op_counts["QLinearGlobalAveragePool"]
    onnx_conv_count = op_counts["Conv"]
    onnx_gemm_count = op_counts["Gemm"]
    qgemm_count = domain_op_counts[("com.microsoft", "QGemm")]
    exact_conv_count = sum(
        node.op_type == "Conv" and node.name.startswith("exact_fbgemm_") for node in model.graph.node
    )
    exact_gemm_count = sum(
        node.op_type == "Gemm" and node.name.startswith("exact_fbgemm_") for node in model.graph.node
    )
    direct_int8_weights = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name.startswith("backbone_")
        and initializer.name.endswith("_weight_codes")
        and initializer.data_type == TensorProto.INT8
    ]
    checkpoint_parameter_suffixes = (
        "_weight_codes",
        "_weight_scales",
        "_weight_zero_points",
        "_bias_f32",
        "_input_scale",
        "_input_zero_point",
        "_output_scale",
        "_output_zero_point",
    )
    checkpoint_parameter_hashes = {
        initializer.name: hashlib.sha256(numpy_helper.to_array(initializer).tobytes()).hexdigest()
        for initializer in model.graph.initializer
        if initializer.name.startswith("backbone_")
        and initializer.name.endswith(checkpoint_parameter_suffixes)
    }

    if mode == "qoperator":
        if qlinear_conv_count != 9:
            raise AssertionError(f"expected exactly nine QLinearConv nodes, found {qlinear_conv_count}")
        if qlinear_global_average_pool_count != 1:
            raise AssertionError(
                "expected exactly one QLinearGlobalAveragePool node, found "
                f"{qlinear_global_average_pool_count}"
            )
        if onnx_conv_count or onnx_gemm_count:
            raise AssertionError(
                "optimized quantized DSCNN contains float operators: "
                f"Conv={onnx_conv_count}, Gemm={onnx_gemm_count}"
            )
    elif mode == "exact":
        if qlinear_conv_count or qlinear_global_average_pool_count or qgemm_count:
            raise AssertionError("exact graph must not retain QLinearConv, QLinearGlobalAveragePool, or QGemm")
        if exact_conv_count != 9 or exact_gemm_count != 1:
            raise AssertionError(
                "exact FBGEMM lowering has the wrong backbone operator count: "
                f"Conv={exact_conv_count}, Gemm={exact_gemm_count}"
            )
        if onnx_conv_count != exact_conv_count or onnx_gemm_count != exact_gemm_count:
            raise AssertionError("exact graph contains a floating backbone operator outside the code-domain lowerer")
        if len(direct_int8_weights) != 10:
            raise AssertionError(
                "exact graph must retain nine convolution and one FC INT8 weight tensors, found "
                f"{len(direct_int8_weights)}"
            )
        if op_counts["DequantizeLinear"] != 1:
            raise AssertionError("exact graph must expose only the final output DequantizeLinear")
    else:
        raise ValueError(f"unsupported export mode: {mode}")

    report = {
        "op_counts": dict(sorted(op_counts.items())),
        "domain_op_counts": {
            f"{domain or 'ai.onnx'}::{op_type}": count
            for (domain, op_type), count in sorted(domain_op_counts.items())
        },
        "qlinear_conv_node_count": qlinear_conv_count,
        "qlinear_global_average_pool_node_count": qlinear_global_average_pool_count,
        "onnx_conv_node_count": onnx_conv_count,
        "onnx_gemm_node_count": onnx_gemm_count,
        "exact_integer_lowering_conv_node_count": exact_conv_count,
        "exact_integer_lowering_gemm_node_count": exact_gemm_count,
        "floating_backbone_conv_node_count": onnx_conv_count - exact_conv_count,
        "floating_backbone_gemm_node_count": onnx_gemm_count - exact_gemm_count,
        "qgemm_node_count": qgemm_count,
        "direct_int8_weight_initializer_count": len(direct_int8_weights),
        "direct_int8_weight_code_count": sum(
            int(np.prod(numpy_helper.to_array(initializer).shape)) for initializer in direct_int8_weights
        ),
        "nonportable_nodes": nonportable_nodes,
        "standard_onnx_syntax": not nonportable_nodes,
        "portable_standard_onnx": mode != "exact" and not nonportable_nodes,
        "direct_checkpoint_parameter_sha256": dict(sorted(checkpoint_parameter_hashes.items())),
    }
    if mode == "exact":
        report["format"] = EXACT_FORMAT
        report["exact_validation_provider"] = f"ONNX Runtime CPUExecutionProvider {ort.__version__}"
        report["requires_target_bit_exact_certification"] = True
    return report


def _run_and_compare(
    reference: nn.Module,
    output_path: Path,
    waveforms: Iterable[torch.Tensor],
    *,
    stage_tensor_names: dict[str, str] | None = None,
    reference_stage_code_fn: Callable[[torch.Tensor], dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if (stage_tensor_names is None) != (reference_stage_code_fn is None):
        raise ValueError(
            "stage_tensor_names and reference_stage_code_fn must either both be set or both be unset"
        )
    waveform_list = list(waveforms)
    reference_outputs: list[np.ndarray] = []
    reference_stage_codes: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for waveform in waveform_list:
            reference_outputs.append(reference(waveform).detach().cpu().numpy())
            if reference_stage_code_fn is not None:
                reference_stage_codes.append(reference_stage_code_fn(waveform))

    stage_labels: tuple[str, ...] = ()
    if stage_tensor_names is None:
        session = ort.InferenceSession(_filesystem_path(output_path), providers=["CPUExecutionProvider"])
    else:
        stage_labels = tuple(stage_tensor_names)
        if not stage_labels:
            raise AssertionError("exact graph must expose at least one stage code for strict parity")
        for stage_codes in reference_stage_codes:
            if tuple(stage_codes) != stage_labels:
                raise AssertionError(
                    "eager stage labels do not match exact graph stage labels: "
                    f"{tuple(stage_codes)!r} != {stage_labels!r}"
                )
        model = onnx.load(_filesystem_path(output_path))
        for label in stage_labels:
            codes = reference_stage_codes[0][label]
            if codes.dtype != np.dtype(np.uint8):
                raise AssertionError(f"eager stage {label} must be uint8, got {codes.dtype}")
            model.graph.output.append(
                helper.make_tensor_value_info(stage_tensor_names[label], TensorProto.UINT8, list(codes.shape))
            )
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_shape = _static_shape(input_meta)
    output_shape = _static_shape(output_meta)
    if input_meta.name != "waveform" or output_meta.name != "logits":
        raise AssertionError(f"unexpected I/O names: {input_meta.name!r}, {output_meta.name!r}")
    if input_meta.type != "tensor(float)" or output_meta.type != "tensor(float)":
        raise AssertionError(f"unexpected I/O types: {input_meta.type!r}, {output_meta.type!r}")
    if input_shape != WAVEFORM_SHAPE or output_shape != LOGITS_SHAPE:
        raise AssertionError(f"unexpected I/O contract: {input_shape!r}, {output_shape!r}")

    ort_outputs: list[np.ndarray] = []
    ort_stage_codes: list[dict[str, np.ndarray]] = []
    for waveform in waveform_list:
        outputs = session.run(None, {input_meta.name: waveform.numpy()})
        ort_outputs.append(outputs[0])
        if stage_tensor_names is not None:
            ort_stage_codes.append(
                {label: outputs[index + 1] for index, label in enumerate(stage_labels)}
            )

    def float_words(array: np.ndarray, label: str) -> np.ndarray:
        if array.dtype != np.dtype(np.float32):
            raise AssertionError(f"{label} must be float32, got {array.dtype}")
        if not array.flags.c_contiguous:
            raise AssertionError(f"{label} must be C-contiguous")
        return array.view(np.uint32)

    word_pairs = [
        (float_words(reference_output, "reference logits"), float_words(ort_output, "ORT logits"))
        for reference_output, ort_output in zip(reference_outputs, ort_outputs)
    ]
    max_abs_error = max(
        float(np.max(np.abs(reference_output - ort_output)))
        for reference_output, ort_output in zip(reference_outputs, ort_outputs)
    )
    logit_mismatch_count = sum(
        int(np.count_nonzero(reference_output != ort_output))
        for reference_output, ort_output in zip(reference_outputs, ort_outputs)
    )
    mismatch_samples = [
        index
        for index, (reference_words, ort_words) in enumerate(word_pairs)
        if not np.array_equal(reference_words, ort_words)
    ]
    per_stage_code_mismatch_counts = {
        label: sum(
            int(np.count_nonzero(reference_codes[label] != ort_codes[label]))
            for reference_codes, ort_codes in zip(reference_stage_codes, ort_stage_codes)
        )
        for label in stage_labels
    }
    code_mismatch_count = (
        sum(per_stage_code_mismatch_counts.values()) if stage_tensor_names is not None else None
    )
    return {
        "input_name": input_meta.name,
        "input_shape": input_shape,
        "input_type": input_meta.type,
        "output_name": output_meta.name,
        "output_shape": output_shape,
        "output_type": output_meta.type,
        "parity_samples": len(reference_outputs),
        "logit_mismatch_count": logit_mismatch_count,
        "code_mismatch_count": code_mismatch_count,
        "per_stage_code_mismatch_counts": per_stage_code_mismatch_counts,
        "float_word_mismatch_count": sum(
            int(np.count_nonzero(reference_words != ort_words))
            for reference_words, ort_words in word_pairs
        ),
        "mismatch_sample_indices": mismatch_samples,
        "max_abs_error": max_abs_error,
    }


def _quantized_uint8_codes(value: torch.Tensor, stage: str) -> np.ndarray:
    if not value.is_quantized or value.dtype != torch.quint8:
        raise AssertionError(f"converted FBGEMM stage {stage} did not return quint8 codes")
    return value.int_repr().detach().cpu().numpy()


def _eager_stage_codes(
    candidate: FixedShapeV61Int8Model,
    waveform: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Capture the converted FBGEMM code tensors in exact backbone order."""

    features = candidate.feature_extractor(waveform)
    features = features[:, : candidate.dct_coeff, :]
    features = features.permute(0, 2, 1).reshape(1, -1)
    x = candidate.quant(features)
    stages = {"input_quant": _quantized_uint8_codes(x, "input_quant")}
    x = x.reshape(1, 1, candidate.dscnn.input_time_size, candidate.dscnn.input_frequency_size)
    x = candidate.dscnn.conv_layers[0](x)
    stages["conv0"] = _quantized_uint8_codes(x, "conv0")
    for layer_index, layer in enumerate(candidate.dscnn.conv_layers[1:], start=1):
        x = torch.relu(layer.bn_depthwise(layer.depthwise(x)))
        stages[f"conv{layer_index}_depthwise"] = _quantized_uint8_codes(
            x, f"conv{layer_index}_depthwise"
        )
        x = torch.relu(layer.bn_pointwise(layer.pointwise(x)))
        stages[f"conv{layer_index}_pointwise"] = _quantized_uint8_codes(
            x, f"conv{layer_index}_pointwise"
        )
    x = candidate.dscnn.avg_pool(x)
    stages["global_avg_pool"] = _quantized_uint8_codes(x, "global_avg_pool")
    x = torch.flatten(x, 1)
    stages["fc_input"] = _quantized_uint8_codes(x, "fc_input")
    x = candidate.dscnn.dropout(x)
    x = candidate.dscnn.final_fc(x)
    stages["final_fc"] = _quantized_uint8_codes(x, "final_fc")
    return stages


def _strict_parity_passed(comparison: dict[str, Any]) -> bool:
    stage_counts = comparison["per_stage_code_mismatch_counts"]
    return (
        comparison["float_word_mismatch_count"] == 0
        and comparison["logit_mismatch_count"] == 0
        and comparison["max_abs_error"] == 0.0
        and comparison["code_mismatch_count"] == 0
        and bool(stage_counts)
        and all(count == 0 for count in stage_counts.values())
    )


def export_and_verify(
    *,
    checkpoint: Path,
    spec: Path,
    output_path: Path,
    parity_samples: int = 19,
    keep_staging: bool = False,
    mode: str = "exact",
) -> dict[str, Any]:
    """Export and gate direct INT8 ONNX against eager v6.1 FBGEMM logits.

    ``exact`` is the default bit-exact direct-INT8 artifact.  ``qoperator`` is
    retained only as a diagnostic baseline and is expected to fail strict parity
    for this FBGEMM checkpoint.
    """

    checkpoint = Path(checkpoint)
    spec = Path(spec)
    output_path = Path(output_path)
    _require_file(checkpoint, "checkpoint")
    _require_file(spec, "spec")
    if mode not in {"exact", "qoperator"}:
        raise ValueError(f"unsupported export mode: {mode}")
    checkpoint_sha256 = _sha256_file(checkpoint)
    spec_sha256 = _sha256_file(spec)
    waveforms = deterministic_inputs(parity_samples)
    verification_scope = (
        f"{parity_samples} deterministic fixed-shape waveforms; float32 logit words and UINT8 "
        "stage codes against PyTorch v6.1 FBGEMM on ONNX Runtime CPU"
    )

    reference = build_reference(checkpoint, spec)
    candidate = FixedShapeV61Int8Model(reference).eval()
    with torch.no_grad():
        for waveform in waveforms:
            if not np.array_equal(
                reference(waveform).detach().cpu().numpy(),
                candidate(waveform).detach().cpu().numpy(),
            ):
                raise AssertionError("fixed-shape wrapper changed eager converted-INT8 logits")

    os.makedirs(_filesystem_path(output_path.parent), exist_ok=True)
    output_preexisted = os.path.exists(_filesystem_path(output_path))
    artifact_published = False
    staging_path: Path | None = None
    candidate_path: Path | None = None
    float32_accumulator_bounds: dict[str, int] = {}
    try:
        staging_path = _temporary_onnx_path(output_path, "staging")
        candidate_path = _temporary_onnx_path(output_path, "candidate")
        _export_staging(candidate, waveforms[0], staging_path)
        if mode == "exact":
            stage_tensor_names, float32_accumulator_bounds = _materialize_exact_fbgemm(
                staging_path,
                candidate_path,
                reference,
                checkpoint_sha256=checkpoint_sha256,
                spec_sha256=spec_sha256,
                verification_scope=verification_scope,
            )
            graph = _graph_report(candidate_path, mode=mode)
            comparison = _run_and_compare(
                reference,
                candidate_path,
                waveforms,
                stage_tensor_names=stage_tensor_names,
                reference_stage_code_fn=lambda waveform: _eager_stage_codes(candidate, waveform),
            )
        else:
            _materialize_qoperator(staging_path, candidate_path)
            graph = _graph_report(candidate_path, mode=mode)
            comparison = _run_and_compare(reference, candidate_path, waveforms)

        strict_parity_passed = (
            _strict_parity_passed(comparison)
            if mode == "exact"
            else comparison["float_word_mismatch_count"] == 0
        )
        with _exclusive_output_lock(output_path):
            candidate_sha256 = _sha256_file(candidate_path)
            if strict_parity_passed:
                os.replace(_filesystem_path(candidate_path), _filesystem_path(output_path))
                artifact_published = True

            report = {
                "status": "success" if strict_parity_passed else "failed",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "spec": str(spec.resolve()),
                "spec_sha256": spec_sha256,
                "onnx": str(output_path.resolve()),
                "onnx_sha256": (
                    _sha256_file(output_path)
                    if os.path.exists(_filesystem_path(output_path))
                    else None
                ),
                "candidate_onnx_sha256": candidate_sha256,
                "artifact_published": artifact_published,
                "opset": OPSET_VERSION,
                "pytorch_version": torch.__version__,
                "onnx_version": onnx.__version__,
                "onnxruntime_version": ort.__version__,
                "strict_parity_passed": strict_parity_passed,
                "mode": mode,
                "float32_accumulator_bounds": float32_accumulator_bounds,
                **graph,
                **comparison,
            }
    finally:
        if not keep_staging and staging_path is not None:
            _remove_owned_file(staging_path)
        if candidate_path is not None:
            _remove_owned_file(candidate_path)

    if not strict_parity_passed:
        raise StrictParityError(report)
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    os.makedirs(_filesystem_path(path.parent), exist_ok=True)
    with open(_filesystem_path(path), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=True) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a v6.1 converted INT8 KWS checkpoint to ORT QOperator ONNX"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parity-samples", type=int, default=19)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["exact", "qoperator"],
        default="exact",
        help="exact is bit-exact direct INT8; qoperator is a nonexact diagnostic baseline",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if _same_resolved_filesystem_path(args.output, args.report):
        print("--output and --report must refer to different files", file=sys.stderr)
        return 2
    try:
        report = export_and_verify(
            checkpoint=args.checkpoint,
            spec=args.spec,
            output_path=args.output,
            parity_samples=args.parity_samples,
            keep_staging=args.keep_staging,
            mode=args.mode,
        )
    except Exception as error:
        if isinstance(error, StrictParityError):
            failure_report = dict(error.report)
        else:
            failure_report = {
                "status": "failed",
                "checkpoint": str(args.checkpoint.resolve()),
                "spec": str(args.spec.resolve()),
                "onnx": str(args.output.resolve()),
                "strict_parity_passed": False,
            }
        failure_report["status"] = "failed"
        failure_report["exception_type"] = type(error).__name__
        failure_report["exception_message"] = str(error)
        _write_report(args.report, failure_report)
        print(json.dumps(failure_report, ensure_ascii=True), file=sys.stderr)
        return 1

    _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
