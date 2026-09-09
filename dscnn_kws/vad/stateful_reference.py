"""PyTorch reference for the supported stateful VAD ONNX graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch.nn import functional as F

from dscnn_kws.ONNX.export_vad_stateful import VadExportError, inspect_vad_layout


@dataclass(frozen=True)
class VadReferenceResult:
    probability: torch.Tensor
    cnn_context: torch.Tensor
    gru_hidden: torch.Tensor


@dataclass(frozen=True)
class _ConvParameters:
    weight: np.ndarray
    bias: np.ndarray | None
    groups: int


class StatefulVadReference(torch.nn.Module):
    """Stateful float32 VAD reference reconstructed from ONNX initializers."""

    def __init__(
        self,
        *,
        conv0: _ConvParameters,
        depthwise1: _ConvParameters,
        pointwise1: _ConvParameters,
        depthwise2: _ConvParameters,
        pointwise2: _ConvParameters,
        gru_weight: np.ndarray,
        gru_recurrent_weight: np.ndarray,
        gru_bias: np.ndarray,
        classifier_weight: np.ndarray,
        classifier_bias: np.ndarray,
    ) -> None:
        super().__init__()
        self._conv_groups = (conv0.groups, depthwise1.groups, pointwise1.groups, depthwise2.groups, pointwise2.groups)
        self._register_array("conv0_weight", conv0.weight)
        self._register_optional_array("conv0_bias", conv0.bias)
        self._register_array("depthwise1_weight", depthwise1.weight)
        self._register_optional_array("depthwise1_bias", depthwise1.bias)
        self._register_array("pointwise1_weight", pointwise1.weight)
        self._register_optional_array("pointwise1_bias", pointwise1.bias)
        self._register_array("depthwise2_weight", depthwise2.weight)
        self._register_optional_array("depthwise2_bias", depthwise2.bias)
        self._register_array("pointwise2_weight", pointwise2.weight)
        self._register_optional_array("pointwise2_bias", pointwise2.bias)
        self._register_array("gru_weight", gru_weight)
        self._register_array("gru_recurrent_weight", gru_recurrent_weight)
        self._register_array("gru_bias", gru_bias)
        self._register_array("classifier_weight", classifier_weight)
        self._register_array("classifier_bias", classifier_bias)

    @classmethod
    def from_onnx(cls, source_path: Path) -> "StatefulVadReference":
        """Load Conv, GRU and classifier parameters from a supported VAD ONNX graph."""

        layout = inspect_vad_layout(source_path)
        model = onnx.load(str(source_path))
        initializers = {item.name: numpy_helper.to_array(item).astype(np.float32, copy=False) for item in model.graph.initializer}
        nodes = list(model.graph.node)
        consumers = _consumers(nodes)

        conv0 = _conv_parameters(nodes[layout.causal_sites[0].conv_node_index], initializers)
        depthwise1_node = nodes[layout.causal_sites[1].conv_node_index]
        depthwise2_node = nodes[layout.causal_sites[2].conv_node_index]
        depthwise1 = _conv_parameters(depthwise1_node, initializers)
        depthwise2 = _conv_parameters(depthwise2_node, initializers)
        pointwise1 = _conv_parameters(_single_consumer(depthwise1_node, consumers, "Conv"), initializers)
        pointwise2 = _conv_parameters(_single_consumer(depthwise2_node, consumers, "Conv"), initializers)

        gru = nodes[layout.gru_node_index]
        if len(gru.input) < 4:
            raise VadExportError("supported VAD GRU has incomplete parameter inputs")
        if _attribute_int(gru, "linear_before_reset", default=0) != 1:
            raise VadExportError("supported VAD GRU must use linear_before_reset=1")
        try:
            gru_weight = initializers[gru.input[1]]
            gru_recurrent_weight = initializers[gru.input[2]]
            gru_bias = initializers[gru.input[3]]
        except KeyError as error:
            raise VadExportError("supported VAD GRU parameters must be initializers") from error
        if gru_weight.shape != (1, 120, 32) or gru_recurrent_weight.shape != (1, 120, 40) or gru_bias.shape != (1, 240):
            raise VadExportError("supported VAD GRU parameter shapes do not match the state contract")

        matmuls = [node for node in nodes if node.op_type == "MatMul"]
        if len(matmuls) != 1:
            raise VadExportError("supported VAD graph must have one classifier MatMul")
        classifier_matmul = matmuls[0]
        if len(classifier_matmul.input) != 2 or classifier_matmul.input[1] not in initializers:
            raise VadExportError("supported VAD classifier weight must be an initializer")
        classifier_weight = initializers[classifier_matmul.input[1]]
        classifier_add = _single_consumer(classifier_matmul, consumers, "Add")
        bias_input = next((name for name in classifier_add.input if name != classifier_matmul.output[0]), "")
        if bias_input not in initializers:
            raise VadExportError("supported VAD classifier bias must be an initializer")
        classifier_bias = initializers[bias_input]
        if classifier_weight.shape != (40, 1) or classifier_bias.shape != (1,):
            raise VadExportError("supported VAD classifier parameter shapes do not match the state contract")

        return cls(
            conv0=conv0,
            depthwise1=depthwise1,
            pointwise1=pointwise1,
            depthwise2=depthwise2,
            pointwise2=pointwise2,
            gru_weight=gru_weight,
            gru_recurrent_weight=gru_recurrent_weight,
            gru_bias=gru_bias,
            classifier_weight=classifier_weight,
            classifier_bias=classifier_bias,
        )

    def forward(
        self,
        log_mel_chunk: torch.Tensor,
        cnn_context_in: torch.Tensor,
        gru_hidden_in: torch.Tensor,
    ) -> VadReferenceResult:
        _require_log_mel(log_mel_chunk)
        _require_tensor("cnn_context_in", cnn_context_in, channels=128, frames=4)
        _require_tensor("gru_hidden_in", gru_hidden_in, channels=1, frames=40)
        if log_mel_chunk.shape[0] != 1 or cnn_context_in.shape[0] != 1 or gru_hidden_in.shape != (1, 1, 40):
            raise ValueError("stateful VAD reference only accepts batch size one")

        x = log_mel_chunk.transpose(1, 2)
        padded0 = torch.cat((cnn_context_in[:, :64], x), dim=2)
        context0 = padded0[:, :, -4:]
        x0 = torch.relu(F.conv1d(padded0, self.conv0_weight, self.conv0_bias, groups=self._conv_groups[0]))

        padded1 = torch.cat((cnn_context_in[:, 64:96], x0), dim=2)
        context1 = padded1[:, :, -4:]
        depth1 = F.conv1d(
            padded1,
            self.depthwise1_weight,
            self.depthwise1_bias,
            groups=self._conv_groups[1],
        )
        residual1 = x0 + torch.relu(
            F.conv1d(depth1, self.pointwise1_weight, self.pointwise1_bias, groups=self._conv_groups[2])
        )

        padded2 = torch.cat((cnn_context_in[:, 96:128], residual1), dim=2)
        context2 = padded2[:, :, -4:]
        depth2 = F.conv1d(
            padded2,
            self.depthwise2_weight,
            self.depthwise2_bias,
            groups=self._conv_groups[3],
        )
        residual2 = residual1 + torch.relu(
            F.conv1d(depth2, self.pointwise2_weight, self.pointwise2_bias, groups=self._conv_groups[4])
        )

        sequence, hidden = self._gru_forward(residual2.transpose(1, 2), gru_hidden_in.squeeze(0))
        probability = torch.sigmoid(torch.matmul(sequence, self.classifier_weight) + self.classifier_bias).squeeze(-1)
        return VadReferenceResult(
            probability=probability,
            cnn_context=torch.cat((context0, context1, context2), dim=1),
            gru_hidden=hidden.unsqueeze(0),
        )

    def _gru_forward(self, sequence: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.gru_weight.squeeze(0)
        recurrent = self.gru_recurrent_weight.squeeze(0)
        input_bias, recurrent_bias = self.gru_bias.squeeze(0).split(120)
        wz, wr, wn = weights.split(40)
        rz, rr, rn = recurrent.split(40)
        wbz, wbr, wbn = input_bias.split(40)
        rbz, rbr, rbn = recurrent_bias.split(40)
        outputs: list[torch.Tensor] = []
        for value in sequence.unbind(dim=1):
            update = torch.sigmoid(value @ wz.T + hidden @ rz.T + wbz + rbz)
            reset = torch.sigmoid(value @ wr.T + hidden @ rr.T + wbr + rbr)
            candidate = torch.tanh(value @ wn.T + reset * (hidden @ rn.T + rbn) + wbn)
            hidden = (1.0 - update) * candidate + update * hidden
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), hidden

    def _register_array(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.from_numpy(np.array(value, dtype=np.float32, copy=True, order="C")))

    def _register_optional_array(self, name: str, value: np.ndarray | None) -> None:
        if value is None:
            self.register_buffer(name, None)
        else:
            self._register_array(name, value)


def _conv_parameters(node: onnx.NodeProto, initializers: dict[str, np.ndarray]) -> _ConvParameters:
    if node.op_type != "Conv" or len(node.input) < 2 or node.input[1] not in initializers:
        raise VadExportError("supported VAD Conv weight must be an initializer")
    bias = initializers[node.input[2]] if len(node.input) > 2 and node.input[2] else None
    return _ConvParameters(
        weight=initializers[node.input[1]],
        bias=bias,
        groups=_attribute_int(node, "group", default=1),
    )


def _single_consumer(
    node: onnx.NodeProto,
    consumers: dict[str, list[onnx.NodeProto]],
    expected_op_type: str,
) -> onnx.NodeProto:
    if len(node.output) != 1:
        raise VadExportError("supported VAD node must have one output")
    matches = [candidate for candidate in consumers.get(node.output[0], []) if candidate.op_type == expected_op_type]
    if len(matches) != 1:
        raise VadExportError(f"supported VAD node must feed one {expected_op_type}")
    return matches[0]


def _consumers(nodes: list[onnx.NodeProto]) -> dict[str, list[onnx.NodeProto]]:
    result: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for name in node.input:
            if name:
                result.setdefault(name, []).append(node)
    return result


def _attribute_int(node: onnx.NodeProto, name: str, *, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(onnx.helper.get_attribute_value(attribute))
    return default


def _require_tensor(name: str, value: torch.Tensor, *, channels: int, frames: int | None) -> None:
    if value.dtype != torch.float32 or value.ndim != 3 or value.shape[1] != channels:
        raise ValueError(f"{name} has an invalid stateful VAD shape")
    if frames is not None and value.shape[2] != frames:
        raise ValueError(f"{name} has an invalid stateful VAD frame count")
    if value.shape[2] < 1 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite and nonempty")


def _require_log_mel(value: torch.Tensor) -> None:
    if value.dtype != torch.float32 or value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 1 or value.shape[2] != 64:
        raise ValueError("log_mel_chunk must be finite float32 [1,frames,64]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("log_mel_chunk must be finite float32 [1,frames,64]")
