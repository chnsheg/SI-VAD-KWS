"""Export the exact v6.1 KWS boundary-frame repair frontend.

The full strict frontend uses 32 non-overlapping 512-sample frames.  A 96 ms
KWS advance is three frames, so only the leading reflected frame and four
trailing frames need recomputing.  This exporter preserves the direct v6.1
INT8 checkpoint quantizer and publishes a repair graph only after its UINT8
codes match the certified full exact graph on ORT CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn

# The deployment virtual environment uses python311._pth and omits script/CWD paths.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from export_v6_1_int8_qoperator import (
    EXACT_FORMAT,
    WAVEFORM_SHAPE,
    _exclusive_output_lock,
    _filesystem_path,
    _remove_owned_file,
    _sha256_file,
    _temporary_onnx_path,
    build_reference,
    deterministic_inputs,
    export_and_verify,
)
from strict_integer_mfcc import StrictIntegerMFCCFloatAdapter


REPAIR_OUTPUT_NAME = "strict_mfcc_repair_codes"
REPAIR_FRAME_INDICES = (0, 28, 29, 30, 31)
FRAME_COUNT = 32
MFCC_PER_FRAME = 10
FRAME_SAMPLES = 512


class _StrictMfccRepairFeatures(nn.Module):
    """Calculate exactly the five non-reusable v6.1 strict MFCC frames."""

    def __init__(self, spec: Path) -> None:
        super().__init__()
        self.adapter = StrictIntegerMFCCFloatAdapter(_filesystem_path(spec))
        self.dct_coeff = MFCC_PER_FRAME

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        pcm_q = self.adapter.pcm_codes(waveform)
        preemphasis = self.adapter.core._preemphasis(pcm_q)
        padded = self.adapter.core._reflect_pad_1d(preemphasis, self.adapter.core.n_fft // 2)
        frame_sources = [
            padded[:, index * self.adapter.core.hop_length : index * self.adapter.core.hop_length + self.adapter.core.n_fft]
            for index in REPAIR_FRAME_INDICES
        ]
        selected = torch.cat(frame_sources, dim=1)
        stages = self.adapter.core._forward_from_framed_preemphasis(selected)
        mfcc = stages["mfcc"][:, : self.dct_coeff, :].to(torch.float32)
        features = mfcc * self.adapter.mfcc_scales[: self.dct_coeff].view(1, -1, 1)
        return features.permute(0, 2, 1).reshape(1, -1)


def export_repair_and_verify(
    *,
    checkpoint: Path,
    spec: Path,
    output_path: Path,
    parity_samples: int = 19,
) -> dict[str, Any]:
    """Publish the repair graph only after exact full-frontend code parity."""

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("output_path must have a .onnx suffix")
    if parity_samples < 9:
        raise ValueError("parity_samples must be at least 9")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = _temporary_onnx_path(output_path, "candidate")
    try:
        with tempfile.TemporaryDirectory(prefix="v6_1_exact_repair_") as temporary_directory:
            root = Path(temporary_directory)
            full_path = root / "full_exact.onnx"
            source_report = export_and_verify(
                checkpoint=Path(checkpoint),
                spec=Path(spec),
                output_path=full_path,
                parity_samples=parity_samples,
                mode="exact",
            )
            reference = build_reference(Path(checkpoint), Path(spec))
            quant_scale = float(reference.backbone.quant.scale.reshape(-1)[0].item())
            quant_zero_point = int(reference.backbone.quant.zero_point.reshape(-1)[0].item())
            _export_repair_candidate(Path(spec), quant_scale, quant_zero_point, candidate_path)
            comparison = _verify_repair_codes(full_path, candidate_path, deterministic_inputs(parity_samples))
            strict_parity_passed = comparison["code_mismatch_count"] == 0
            if not strict_parity_passed:
                raise RuntimeError(f"strict MFCC repair code parity failed: {comparison}")
            with _exclusive_output_lock(output_path):
                os.replace(_filesystem_path(candidate_path), _filesystem_path(output_path))
    finally:
        _remove_owned_file(candidate_path)

    return {
        "status": "success",
        "strict_parity_passed": True,
        "checkpoint_sha256": source_report["checkpoint_sha256"],
        "spec_sha256": source_report["spec_sha256"],
        "source_full_onnx_sha256": source_report["onnx_sha256"],
        "repair_path": str(output_path.resolve()),
        "repair_sha256": _sha256_file(output_path),
        "input_shape": list(WAVEFORM_SHAPE),
        "output_name": REPAIR_OUTPUT_NAME,
        "output_shape": [1, len(REPAIR_FRAME_INDICES) * MFCC_PER_FRAME],
        "output_dtype": "uint8",
        "frame_count": FRAME_COUNT,
        "frame_samples": FRAME_SAMPLES,
        "frame_indices": list(REPAIR_FRAME_INDICES),
        "parity_samples": parity_samples,
        **comparison,
    }


def _export_repair_candidate(spec: Path, scale: float, zero_point: int, output_path: Path) -> None:
    candidate = _StrictMfccRepairFeatures(spec).eval()
    staged_path = output_path.with_suffix(".staging.onnx")
    try:
        with torch.no_grad():
            torch.onnx.export(
                candidate,
                torch.zeros(WAVEFORM_SHAPE, dtype=torch.float32),
                _filesystem_path(staged_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["waveform"],
                output_names=["strict_mfcc_repair_features"],
                dynamo=False,
            )
        model = onnx.load(_filesystem_path(staged_path))
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise AssertionError("repair staging graph must have one input and one output")
        feature_output = model.graph.output[0].name
        if feature_output != "strict_mfcc_repair_features":
            raise AssertionError("repair staging graph output name changed unexpectedly")
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray(scale, dtype=np.float32), "repair_input_scale"),
                numpy_helper.from_array(np.asarray(zero_point, dtype=np.uint8), "repair_input_zero_point"),
            ]
        )
        model.graph.node.append(
            helper.make_node(
                "QuantizeLinear",
                [feature_output, "repair_input_scale", "repair_input_zero_point"],
                [REPAIR_OUTPUT_NAME],
                name="exact_repair_input_quant",
            )
        )
        del model.graph.output[:]
        model.graph.output.append(
            helper.make_tensor_value_info(
                REPAIR_OUTPUT_NAME,
                TensorProto.UINT8,
                [1, len(REPAIR_FRAME_INDICES) * MFCC_PER_FRAME],
            )
        )
        metadata = {entry.key: entry.value for entry in model.metadata_props}
        metadata["format"] = EXACT_FORMAT
        metadata["artifact_kind"] = "strict_mfcc_boundary_repair"
        metadata["repair_frame_indices"] = ",".join(str(index) for index in REPAIR_FRAME_INDICES)
        del model.metadata_props[:]
        for key, value in metadata.items():
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
        onnx.checker.check_model(model)
        onnx.save(model, _filesystem_path(output_path))
    finally:
        if staged_path.exists():
            staged_path.unlink()


def _verify_repair_codes(
    full_path: Path, repair_path: Path, waveforms: Iterable[torch.Tensor]
) -> dict[str, int]:
    full = onnx.load(_filesystem_path(full_path))
    quantizers = [node for node in full.graph.node if node.op_type == "QuantizeLinear"]
    if len(quantizers) != 1:
        raise AssertionError("certified full graph must contain exactly one MFCC QuantizeLinear")
    boundary = quantizers[0].output[0]
    full.graph.output.append(helper.make_tensor_value_info(boundary, TensorProto.UINT8, [1, 320]))
    full_session = ort.InferenceSession(full.SerializeToString(), providers=["CPUExecutionProvider"])
    repair_session = ort.InferenceSession(_filesystem_path(repair_path), providers=["CPUExecutionProvider"])
    mismatch_count = 0
    for waveform in waveforms:
        values = waveform.detach().cpu().numpy()
        (full_codes,) = full_session.run([boundary], {"waveform": values})
        (repair_codes,) = repair_session.run([REPAIR_OUTPUT_NAME], {"waveform": values})
        expected = full_codes.reshape(1, FRAME_COUNT, MFCC_PER_FRAME)[:, REPAIR_FRAME_INDICES, :].reshape(
            1, -1
        )
        mismatch_count += int(np.count_nonzero(expected != repair_codes))
    return {"code_mismatch_count": mismatch_count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export exact v6.1 KWS boundary-frame repair ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parity-samples", type=int, default=19)
    args = parser.parse_args(argv)
    try:
        report = export_repair_and_verify(
            checkpoint=args.checkpoint,
            spec=args.spec,
            output_path=args.output,
            parity_samples=args.parity_samples,
        )
    except Exception as error:
        report = {"status": "failed", "strict_parity_passed": False, "error": str(error)}
        exit_code = 1
    else:
        exit_code = 0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
