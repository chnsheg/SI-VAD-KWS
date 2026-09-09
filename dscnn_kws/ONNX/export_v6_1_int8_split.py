"""Split a certified v6.1 exact KWS ONNX graph at its UINT8 MFCC boundary."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

# The deployment virtual environment uses python311._pth and omits script/CWD paths.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from export_v6_1_int8_qoperator import (
    EXACT_FORMAT,
    _filesystem_path,
    _remove_owned_file,
    _sha256_file,
    _temporary_onnx_path,
    deterministic_inputs,
    export_and_verify,
)


BOUNDARY_INPUT_NAME = "strict_mfcc_codes"
BOUNDARY_OUTPUT_NAME = "strict_mfcc_codes"


def export_split_and_verify(
    *,
    checkpoint: Path,
    spec: Path,
    frontend_path: Path,
    backbone_path: Path,
    parity_samples: int = 19,
) -> dict[str, Any]:
    """Publish exact frontend/backbone graphs only after composition parity passes."""

    frontend_path = Path(frontend_path)
    backbone_path = Path(backbone_path)
    if frontend_path.resolve() == backbone_path.resolve():
        raise ValueError("frontend_path and backbone_path must be different files")
    if frontend_path.suffix.lower() != ".onnx" or backbone_path.suffix.lower() != ".onnx":
        raise ValueError("frontend_path and backbone_path must have .onnx suffixes")
    if parity_samples < 9:
        raise ValueError("parity_samples must be at least 9")

    frontend_path.parent.mkdir(parents=True, exist_ok=True)
    backbone_path.parent.mkdir(parents=True, exist_ok=True)
    frontend_candidate = _temporary_onnx_path(frontend_path, "candidate")
    backbone_candidate = _temporary_onnx_path(backbone_path, "candidate")
    try:
        with tempfile.TemporaryDirectory(prefix="v6_1_exact_split_") as temporary_directory:
            root = Path(temporary_directory)
            full_path = root / "full_exact.onnx"
            source_report = export_and_verify(
                checkpoint=Path(checkpoint),
                spec=Path(spec),
                output_path=full_path,
                parity_samples=parity_samples,
                mode="exact",
            )
            boundary_shape = _split_exact_graph(full_path, frontend_candidate, backbone_candidate)
            comparison = _verify_composition(
                full_path,
                frontend_candidate,
                backbone_candidate,
                deterministic_inputs(parity_samples),
            )
            strict_parity_passed = (
                comparison["frontend_code_mismatch_count"] == 0
                and comparison["logit_word_mismatch_count"] == 0
                and comparison["max_abs_error"] == 0.0
            )
            if not strict_parity_passed:
                raise RuntimeError(f"exact KWS split composition parity failed: {comparison}")
            os.replace(_filesystem_path(frontend_candidate), _filesystem_path(frontend_path))
            os.replace(_filesystem_path(backbone_candidate), _filesystem_path(backbone_path))
    finally:
        _remove_owned_file(frontend_candidate)
        _remove_owned_file(backbone_candidate)

    return {
        "status": "success",
        "strict_parity_passed": True,
        "checkpoint_sha256": source_report["checkpoint_sha256"],
        "spec_sha256": source_report["spec_sha256"],
        "source_full_onnx_sha256": source_report["onnx_sha256"],
        "frontend_path": str(frontend_path.resolve()),
        "frontend_sha256": _sha256_file(frontend_path),
        "backbone_path": str(backbone_path.resolve()),
        "backbone_sha256": _sha256_file(backbone_path),
        "boundary_name": BOUNDARY_OUTPUT_NAME,
        "boundary_shape": list(boundary_shape),
        "boundary_dtype": "uint8",
        "parity_samples": parity_samples,
        **comparison,
    }


def _split_exact_graph(full_path: Path, frontend_path: Path, backbone_path: Path) -> tuple[int, ...]:
    model = onnx.load(_filesystem_path(full_path))
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if metadata.get("format") != EXACT_FORMAT:
        raise ValueError("source ONNX is not a certified v6.1 exact graph")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("exact KWS graph must have one public input and one public output")
    quantizers = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    if len(quantizers) != 1 or len(quantizers[0].output) != 1:
        raise ValueError("exact KWS graph must contain exactly one MFCC QuantizeLinear boundary")
    boundary = quantizers[0].output[0]
    boundary_shape = _boundary_shape(model, boundary)
    if boundary_shape != (1, 320):
        raise ValueError(f"unexpected strict MFCC boundary shape: {boundary_shape}")

    frontend_nodes = _backward_slice(model.graph.node, (boundary,))
    backbone_nodes = _backward_slice(model.graph.node, (model.graph.output[0].name,), stop_values={boundary})
    frontend = _make_model(
        source=model,
        nodes=frontend_nodes,
        inputs=[copy.deepcopy(model.graph.input[0])],
        outputs=[helper.make_tensor_value_info(BOUNDARY_OUTPUT_NAME, TensorProto.UINT8, list(boundary_shape))],
        original_boundary=boundary,
        rename_boundary_output=True,
        split_kind="strict_frontend",
    )
    backbone = _make_model(
        source=model,
        nodes=backbone_nodes,
        inputs=[helper.make_tensor_value_info(BOUNDARY_INPUT_NAME, TensorProto.UINT8, list(boundary_shape))],
        outputs=[copy.deepcopy(model.graph.output[0])],
        original_boundary=boundary,
        rename_boundary_input=True,
        split_kind="exact_backbone",
    )
    onnx.save(frontend, _filesystem_path(frontend_path))
    onnx.save(backbone, _filesystem_path(backbone_path))
    return boundary_shape


def _boundary_shape(model: onnx.ModelProto, boundary: str) -> tuple[int, ...]:
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    probe.graph.output.append(helper.make_tensor_value_info(boundary, TensorProto.UINT8, None))
    session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    (codes,) = session.run([boundary], {model.graph.input[0].name: np.zeros((1, 16000), np.float32)})
    if codes.dtype != np.uint8 or codes.ndim != 2 or not all(size > 0 for size in codes.shape):
        raise ValueError("strict MFCC boundary must be nonempty UINT8 rank-2 codes")
    return tuple(int(size) for size in codes.shape)


def _backward_slice(
    nodes: Iterable[onnx.NodeProto], outputs: tuple[str, ...], *, stop_values: set[str] | None = None
) -> list[onnx.NodeProto]:
    source = list(nodes)
    producers = {name: index for index, node in enumerate(source) for name in node.output if name}
    required = list(outputs)
    selected: set[int] = set()
    stopped = stop_values or set()
    while required:
        value = required.pop()
        if value in stopped:
            continue
        index = producers.get(value)
        if index is None or index in selected:
            continue
        selected.add(index)
        required.extend(name for name in source[index].input if name)
    return [copy.deepcopy(node) for index, node in enumerate(source) if index in selected]


def _make_model(
    *,
    source: onnx.ModelProto,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    original_boundary: str,
    split_kind: str,
    rename_boundary_output: bool = False,
    rename_boundary_input: bool = False,
) -> onnx.ModelProto:
    if rename_boundary_output:
        for node in nodes:
            node.output[:] = [
                BOUNDARY_OUTPUT_NAME if name == original_boundary else name for name in node.output
            ]
    if rename_boundary_input:
        for node in nodes:
            node.input[:] = [
                BOUNDARY_INPUT_NAME if name == original_boundary else name for name in node.input
            ]
    used_inputs = {name for node in nodes for name in node.input if name}
    initializers = [copy.deepcopy(value) for value in source.graph.initializer if value.name in used_inputs]
    graph = helper.make_graph(nodes, f"v6_1_{split_kind}", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, producer_name="v6_1_direct_int8_split", opset_imports=source.opset_import)
    model.ir_version = source.ir_version
    properties = {entry.key: entry.value for entry in source.metadata_props}
    properties["split_kind"] = split_kind
    properties["split_boundary_name"] = BOUNDARY_OUTPUT_NAME
    for key, value in properties.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    return model


def _original_boundary_name(source: onnx.ModelProto) -> str:
    quantizer = next(node for node in source.graph.node if node.op_type == "QuantizeLinear")
    return quantizer.output[0]


def _verify_composition(
    full_path: Path,
    frontend_path: Path,
    backbone_path: Path,
    waveforms: Iterable[torch.Tensor],
) -> dict[str, Any]:
    full = onnx.load(_filesystem_path(full_path))
    boundary = _original_boundary_name(full)
    full.graph.output.append(helper.make_tensor_value_info(boundary, TensorProto.UINT8, [1, 320]))
    full_session = ort.InferenceSession(full.SerializeToString(), providers=["CPUExecutionProvider"])
    frontend = ort.InferenceSession(_filesystem_path(frontend_path), providers=["CPUExecutionProvider"])
    backbone = ort.InferenceSession(_filesystem_path(backbone_path), providers=["CPUExecutionProvider"])
    code_mismatches = 0
    word_mismatches = 0
    max_abs_error = 0.0
    for waveform in waveforms:
        values = waveform.detach().cpu().numpy()
        full_logits, full_codes = full_session.run(["logits", boundary], {"waveform": values})
        (split_codes,) = frontend.run(None, {"waveform": values})
        (split_logits,) = backbone.run(None, {BOUNDARY_INPUT_NAME: split_codes})
        code_mismatches += int(np.count_nonzero(full_codes != split_codes))
        word_mismatches += int(np.count_nonzero(full_logits.view(np.uint32) != split_logits.view(np.uint32)))
        max_abs_error = max(max_abs_error, float(np.max(np.abs(full_logits - split_logits))))
    return {
        "frontend_code_mismatch_count": code_mismatches,
        "logit_word_mismatch_count": word_mismatches,
        "max_abs_error": max_abs_error if math.isfinite(max_abs_error) else float("inf"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split certified v6.1 exact INT8 KWS ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parity-samples", type=int, default=19)
    args = parser.parse_args(argv)
    try:
        report = export_split_and_verify(
            checkpoint=args.checkpoint,
            spec=args.spec,
            frontend_path=args.frontend,
            backbone_path=args.backbone,
            parity_samples=args.parity_samples,
        )
    except Exception as error:
        report = {"status": "failed", "strict_parity_passed": False, "error": str(error)}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
