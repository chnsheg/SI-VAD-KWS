"""Export a reusable exact v6.1 KWS runtime bundle for the streaming demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The deployment virtual environment uses python311._pth and omits script/CWD paths.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from export_v6_1_int8_frame_repair import export_repair_and_verify
from export_v6_1_int8_split import export_split_and_verify


def export_runtime_bundle(
    *,
    checkpoint: Path,
    spec: Path,
    output_dir: Path,
    stem: str = "v6_1_hi_xiaowen_exact",
    parity_samples: int = 19,
) -> dict[str, Any]:
    """Export the exact frontend/backbone/repair trio with shared provenance."""

    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a nonempty filename stem")
    if parity_samples < 19:
        raise ValueError("parity_samples must be at least 19 for a runtime bundle")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frontend_path = root / f"{stem}_frontend.onnx"
    backbone_path = root / f"{stem}_backbone.onnx"
    repair_path = root / f"{stem}_frame_repair.onnx"
    split = export_split_and_verify(
        checkpoint=Path(checkpoint),
        spec=Path(spec),
        frontend_path=frontend_path,
        backbone_path=backbone_path,
        parity_samples=parity_samples,
    )
    repair = export_repair_and_verify(
        checkpoint=Path(checkpoint),
        spec=Path(spec),
        output_path=repair_path,
        parity_samples=parity_samples,
    )
    _require_shared_provenance(split, repair)
    split_report_path = root / f"{stem}_split.report.json"
    repair_report_path = root / f"{stem}_frame_repair.report.json"
    _write_report(split_report_path, split)
    _write_report(repair_report_path, repair)
    return {
        "status": "success",
        "strict_parity_passed": True,
        "checkpoint_sha256": split["checkpoint_sha256"],
        "spec_sha256": split["spec_sha256"],
        "source_full_onnx_sha256": split["source_full_onnx_sha256"],
        "frontend": str(frontend_path.resolve()),
        "backbone": str(backbone_path.resolve()),
        "frame_repair": str(repair_path.resolve()),
        "split_report": str(split_report_path.resolve()),
        "repair_report": str(repair_report_path.resolve()),
        "parity_samples": parity_samples,
    }


def _require_shared_provenance(split: dict[str, Any], repair: dict[str, Any]) -> None:
    for name in ("checkpoint_sha256", "spec_sha256", "source_full_onnx_sha256"):
        if split.get(name) != repair.get(name):
            raise RuntimeError(f"split and repair {name} values do not match")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export exact v6.1 streaming KWS ONNX runtime bundle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="v6_1_hi_xiaowen_exact")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--parity-samples", type=int, default=19)
    args = parser.parse_args(argv)
    report_path = args.report or args.output_dir / f"{args.stem}_runtime_bundle.report.json"
    try:
        report = export_runtime_bundle(
            checkpoint=args.checkpoint,
            spec=args.spec,
            output_dir=args.output_dir,
            stem=args.stem,
            parity_samples=args.parity_samples,
        )
    except Exception as error:
        report = {"status": "failed", "strict_parity_passed": False, "error": str(error)}
        exit_code = 1
    else:
        exit_code = 0
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path, report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
