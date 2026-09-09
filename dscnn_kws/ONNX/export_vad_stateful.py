"""Inspect supported causal-CRNN VAD ONNX graphs before stateful export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from dscnn_kws.vad.state import VadStreamState


class VadExportError(RuntimeError):
    """Raised when a VAD graph cannot satisfy the stateful export contract."""


@dataclass(frozen=True)
class CausalPadSite:
    pad_node_index: int
    conv_node_index: int
    source_value: str
    channels: int


@dataclass(frozen=True)
class VadGraphLayout:
    input_name: str
    output_name: str
    input_shape: tuple[int, str, int]
    gru_node_index: int
    gru_hidden_size: int
    causal_sites: tuple[CausalPadSite, CausalPadSite, CausalPadSite]
    context_channels: tuple[int, int, int]
    context_frames: int

    @property
    def context_shape(self) -> tuple[int, int, int]:
        return (1, sum(self.context_channels), self.context_frames)


@dataclass(frozen=True)
class VadStatefulArtifact:
    """Stateful ONNX artifact details derived from a supported source graph."""

    source_path: Path
    output_path: Path
    layout: VadGraphLayout

    def zero_state(self) -> VadStreamState:
        return VadStreamState(
            cnn_context=np.zeros(self.layout.context_shape, dtype=np.float32),
            gru_hidden=np.zeros((1, 1, self.layout.gru_hidden_size), dtype=np.float32),
        )


def inspect_vad_layout(source_path: Path) -> VadGraphLayout:
    """Return the fixed causal-CRNN layout or reject an incompatible VAD graph."""

    path = Path(source_path)
    if not path.is_file():
        raise VadExportError(f"VAD source graph does not exist: {path}")
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as error:
        raise VadExportError(f"could not load a valid VAD ONNX graph: {path}") from error

    input_name, input_shape = _require_input(model)
    output_name = _require_output(model)
    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    consumers = _consumers(model.graph.node)
    pad_nodes = [(index, node) for index, node in enumerate(model.graph.node) if node.op_type == "Pad"]
    if len(pad_nodes) != 3:
        raise VadExportError(f"unsupported causal context: expected three Pad nodes, found {len(pad_nodes)}")

    pad_values = _evaluate_pad_values(model, input_name, input_shape, pad_nodes)
    sites: list[CausalPadSite] = []
    for pad_index, pad in pad_nodes:
        if len(pad.input) < 2 or not pad.input[0] or not pad.input[1] or len(pad.output) != 1 or not pad.output[0]:
            raise VadExportError("unsupported causal context: Pad node has incomplete inputs or outputs")
        downstream = consumers.get(pad.output[0], [])
        if len(downstream) != 1 or downstream[0][1].op_type != "Conv":
            raise VadExportError("unsupported causal context: Pad output must feed exactly one Conv")
        conv_index, conv = downstream[0]
        pads = pad_values[pad.input[1]]
        if not np.array_equal(pads, np.asarray([0, 0, 4, 0, 0, 0], dtype=np.int64)):
            raise VadExportError(f"unsupported causal context: expected left pad 4, got {pads.tolist()}")
        if _attribute_ints(conv, "kernel_shape") != (5,) or _attribute_ints(conv, "strides") != (1,):
            raise VadExportError("unsupported causal context: expected Conv kernel 5 and stride 1")
        if len(conv.input) < 2 or conv.input[1] not in initializer_by_name:
            raise VadExportError("unsupported causal context: Conv weight must be an initializer")
        weight = numpy_helper.to_array(initializer_by_name[conv.input[1]])
        if weight.ndim != 3:
            raise VadExportError("unsupported causal context: expected one-dimensional Conv weights")
        group = _attribute_int(conv, "group", default=1)
        sites.append(
            CausalPadSite(
                pad_node_index=pad_index,
                conv_node_index=conv_index,
                source_value=pad.input[0],
                channels=int(weight.shape[1]) * group,
            )
        )

    sites.sort(key=lambda site: site.conv_node_index)
    context_channels = tuple(site.channels for site in sites)
    if context_channels != (64, 32, 32):
        raise VadExportError(f"unsupported causal context channels: {context_channels}")

    gru_nodes = [(index, node) for index, node in enumerate(model.graph.node) if node.op_type == "GRU"]
    if len(gru_nodes) != 1:
        raise VadExportError(f"unsupported VAD recurrent topology: expected one GRU, found {len(gru_nodes)}")
    gru_index, gru = gru_nodes[0]
    if _attribute_text(gru, "direction", default="forward") != "forward":
        raise VadExportError("unsupported VAD recurrent topology: GRU must be forward")
    hidden_size = _attribute_int(gru, "hidden_size", default=0)
    if hidden_size != 40:
        raise VadExportError(f"unsupported VAD recurrent topology: GRU hidden_size must be 40, got {hidden_size}")
    if len(gru.input) < 6 or not gru.input[5] or len(gru.output) < 2 or not gru.output[1]:
        raise VadExportError("unsupported VAD recurrent topology: GRU state I/O is absent")

    return VadGraphLayout(
        input_name=input_name,
        output_name=output_name,
        input_shape=input_shape,
        gru_node_index=gru_index,
        gru_hidden_size=hidden_size,
        causal_sites=(sites[0], sites[1], sites[2]),
        context_channels=(64, 32, 32),
        context_frames=4,
    )


def export_stateful_vad(source_path: Path, output_path: Path) -> VadStatefulArtifact:
    """Externalize causal CNN and GRU state without altering source weights."""

    source = Path(source_path)
    destination = Path(output_path)
    layout = inspect_vad_layout(source)
    model = onnx.load(str(source))
    original_input_name = layout.input_name
    model.graph.input[0].name = "log_mel_chunk"
    for node in model.graph.node:
        for position, value in enumerate(node.input):
            if value == original_input_name:
                node.input[position] = "log_mel_chunk"

    model.graph.input.extend(
        [
            helper.make_tensor_value_info("cnn_context_in", TensorProto.FLOAT, list(layout.context_shape)),
            helper.make_tensor_value_info(
                "gru_hidden_in", TensorProto.FLOAT, [1, 1, layout.gru_hidden_size]
            ),
        ]
    )
    gru = model.graph.node[layout.gru_node_index]
    gru.input[5] = "gru_hidden_in"

    site_by_conv = {site.conv_node_index: site for site in layout.causal_sites}
    pad_indexes = {site.pad_node_index for site in layout.causal_sites}
    replacements: list[onnx.NodeProto] = []
    context_tails: list[str] = []
    channel_start = 0
    initializers = list(model.graph.initializer)
    _append_i64(initializers, "stateful_context_axis", [1])
    _append_i64(initializers, "stateful_time_axis", [2])
    _append_i64(initializers, "stateful_slice_step", [1])
    _append_i64(initializers, "stateful_tail_start", [-layout.context_frames])
    _append_i64(initializers, "stateful_tail_end", [np.iinfo(np.int64).max])

    for index, node in enumerate(model.graph.node):
        if index in pad_indexes:
            continue
        site = site_by_conv.get(index)
        if site is not None:
            channel_end = channel_start + site.channels
            context_slice = f"stateful_context_{index}"
            padded_input = f"stateful_padded_input_{index}"
            context_tail = f"stateful_context_tail_{index}"
            start_name = f"stateful_context_start_{index}"
            end_name = f"stateful_context_end_{index}"
            _append_i64(initializers, start_name, [channel_start])
            _append_i64(initializers, end_name, [channel_end])
            replacements.extend(
                [
                    helper.make_node(
                        "Slice",
                        ["cnn_context_in", start_name, end_name, "stateful_context_axis", "stateful_slice_step"],
                        [context_slice],
                        name=f"stateful_context_slice_{index}",
                    ),
                    helper.make_node(
                        "Concat",
                        [context_slice, site.source_value],
                        [padded_input],
                        name=f"stateful_context_concat_{index}",
                        axis=2,
                    ),
                    helper.make_node(
                        "Slice",
                        [
                            padded_input,
                            "stateful_tail_start",
                            "stateful_tail_end",
                            "stateful_time_axis",
                            "stateful_slice_step",
                        ],
                        [context_tail],
                        name=f"stateful_context_tail_{index}",
                    ),
                ]
            )
            node.input[0] = padded_input
            context_tails.append(context_tail)
            channel_start = channel_end
        replacements.append(node)

    replacements.extend(
        [
            helper.make_node(
                "Concat",
                context_tails,
                ["cnn_context_out"],
                name="stateful_context_output",
                axis=1,
            ),
            helper.make_node(
                "Identity",
                [layout.output_name],
                ["probability_chunk"],
                name="stateful_probability_output",
            ),
            helper.make_node(
                "Identity",
                [gru.output[1]],
                ["gru_hidden_out"],
                name="stateful_hidden_output",
            ),
        ]
    )
    model.graph.ClearField("node")
    model.graph.node.extend(replacements)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)
    model.graph.ClearField("output")
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("probability_chunk", TensorProto.FLOAT, [1, "time"]),
            helper.make_tensor_value_info("cnn_context_out", TensorProto.FLOAT, list(layout.context_shape)),
            helper.make_tensor_value_info(
                "gru_hidden_out", TensorProto.FLOAT, [1, 1, layout.gru_hidden_size]
            ),
        ]
    )
    try:
        onnx.checker.check_model(model)
    except Exception as error:
        raise VadExportError("stateful VAD graph validation failed") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return VadStatefulArtifact(source_path=source.resolve(), output_path=destination.resolve(), layout=layout)


def _require_input(model: onnx.ModelProto) -> tuple[str, tuple[int, str, int]]:
    if len(model.graph.input) != 1:
        raise VadExportError("VAD source graph must have one public input")
    value = model.graph.input[0]
    tensor_type = value.type.tensor_type
    dimensions = tensor_type.shape.dim
    if tensor_type.elem_type != TensorProto.FLOAT or len(dimensions) != 3:
        raise VadExportError("VAD input must be float32 [1,time,64]")
    if dimensions[0].dim_value != 1 or dimensions[1].dim_param != "time" or dimensions[2].dim_value != 64:
        raise VadExportError("VAD input must be float32 [1,time,64]")
    return value.name, (1, "time", 64)


def _require_output(model: onnx.ModelProto) -> str:
    if len(model.graph.output) != 1:
        raise VadExportError("VAD source graph must have one public output")
    value = model.graph.output[0]
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type != TensorProto.FLOAT or len(tensor_type.shape.dim) != 2:
        raise VadExportError("VAD output must be float32 [1,time]")
    return value.name


def _evaluate_pad_values(
    model: onnx.ModelProto,
    input_name: str,
    input_shape: tuple[int, str, int],
    pad_nodes: Sequence[tuple[int, onnx.NodeProto]],
) -> dict[str, np.ndarray]:
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    names = [node.input[1] for _, node in pad_nodes]
    for name in names:
        probe.graph.output.append(helper.make_tensor_value_info(name, TensorProto.INT64, [6]))
    try:
        session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
        values = session.run(names, {input_name: np.zeros((1, 5, input_shape[2]), dtype=np.float32)})
    except Exception as error:
        raise VadExportError("could not evaluate VAD Pad constants") from error
    return {name: np.asarray(value, dtype=np.int64).reshape(-1) for name, value in zip(names, values)}


def _consumers(nodes: Sequence[onnx.NodeProto]) -> dict[str, list[tuple[int, onnx.NodeProto]]]:
    result: dict[str, list[tuple[int, onnx.NodeProto]]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            if name:
                result.setdefault(name, []).append((index, node))
    return result


def _attribute_int(node: onnx.NodeProto, name: str, *, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(onnx.helper.get_attribute_value(attribute))
    return default


def _attribute_ints(node: onnx.NodeProto, name: str) -> tuple[int, ...]:
    for attribute in node.attribute:
        if attribute.name == name:
            return tuple(int(value) for value in onnx.helper.get_attribute_value(attribute))
    return ()


def _attribute_text(node: onnx.NodeProto, name: str, *, default: str) -> str:
    for attribute in node.attribute:
        if attribute.name == name:
            value = onnx.helper.get_attribute_value(attribute)
            return value.decode("ascii") if isinstance(value, bytes) else str(value)
    return default


def _append_i64(initializers: list[onnx.TensorProto], name: str, values: Sequence[int]) -> None:
    if any(initializer.name == name for initializer in initializers):
        raise VadExportError(f"duplicate stateful VAD initializer: {name}")
    initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
