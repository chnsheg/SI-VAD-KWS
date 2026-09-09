from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

from onnx_friendly_mfcc import ONNXFriendlyMFCCDSCNN
from dscnn_kws.configs import CLASS_LIST
from dscnn_kws.model import DSCNN
from dscnn_kws.model.dscnn import calculate_time_steps


ARCH_RE = re.compile(r"L(?P<layers>\d+)_C(?P<channels>\d+)", re.IGNORECASE)


def make_model_size_info(num_layers: int, channels: int) -> list[int]:
    info = [num_layers]
    info += [channels, 10, 4, 2, 2]
    for _ in range(num_layers - 1):
        info += [channels, 3, 3, 1, 1]
    return info


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} is not a state_dict or checkpoint dict")

    state = {}
    for key, value in obj.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    return state


def infer_arch_from_name(path: Path) -> tuple[int | None, int | None]:
    match = ARCH_RE.search(path.name) or ARCH_RE.search(path.parent.name)
    if not match:
        return None, None
    return int(match.group("layers")), int(match.group("channels"))


def infer_arch_from_state_dict(state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    conv_indices = set()
    channels = None
    label_count = None

    for key, value in state.items():
        if not hasattr(value, "shape"):
            continue
        match = re.search(r"(?:^|backbone\.)conv_layers\.(\d+)\.", key)
        if match:
            conv_indices.add(int(match.group(1)))
        if key.endswith("conv_layers.0.0.weight"):
            channels = int(value.shape[0])
        if key.endswith("final_fc.weight"):
            label_count = int(value.shape[0])

    if not conv_indices:
        raise ValueError("Cannot infer DSCNN layer count from state_dict keys")
    if channels is None:
        raise ValueError("Cannot infer DSCNN channel count from first conv weight")
    if label_count is None:
        label_count = len(CLASS_LIST)
    return max(conv_indices) + 1, channels, label_count


def extract_backbone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("backbone.") for key in state):
        return {
            key[len("backbone.") :]: value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
    return state


def build_model(
    *,
    state: dict[str, torch.Tensor],
    checkpoint: Path,
    args: argparse.Namespace,
) -> ONNXFriendlyMFCCDSCNN:
    inferred_layers, inferred_channels, label_count = infer_arch_from_state_dict(state)
    named_layers, named_channels = infer_arch_from_name(checkpoint)
    num_layers = args.layers or named_layers or inferred_layers
    channels = args.channels or named_channels or inferred_channels

    time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
    backbone = DSCNN(
        input_dim=time_steps * args.dct_coeff,
        label_count=label_count,
        model_size_info=make_model_size_info(num_layers, channels),
        dct_coeff=args.dct_coeff,
        pooling=getattr(args, "pooling", "global"),
        temporal_bins=int(getattr(args, "temporal_bins", 4)),
    )
    backbone.load_state_dict(extract_backbone_state(state), strict=True)

    model = ONNXFriendlyMFCCDSCNN(
        backbone=backbone,
        sample_rate=args.sample_rate,
        dct_coeff=args.dct_coeff,
        window_size_ms=args.window_size_ms,
        window_stride_ms=args.window_stride_ms,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        mfcc_scale=args.mfcc_scale,
        mel_filter_shape=args.mel_filter_shape,
    )
    model.eval()
    model.export_info = {
        "layers": num_layers,
        "channels": channels,
        "label_count": label_count,
        "time_steps": time_steps,
    }
    return model


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        paths = [Path(p) for p in args.checkpoints]
    elif args.input_dir:
        paths = sorted(Path(args.input_dir).glob(args.pattern))
    else:
        sweep_dir = REPO_ROOT / "dscnn_kws" / "runs" / "sweep_best_models"
        if sweep_dir.exists():
            paths = sorted(sweep_dir.glob("*.pt"))
        else:
            paths = sorted((REPO_ROOT / "dscnn_kws" / "runs").glob("**/best.pt"))
    return [p.resolve() for p in paths if p.exists()]


def safe_stem(path: Path) -> str:
    if path.name == "best.pt":
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.parent.name)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def export_one(checkpoint: Path, args: argparse.Namespace) -> dict[str, Any]:
    state = load_state_dict(checkpoint)
    model = build_model(state=state, checkpoint=checkpoint, args=args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{safe_stem(checkpoint)}_full_onnx_friendly.onnx"

    dummy = torch.randn(args.batch_size, args.sample_rate, dtype=torch.float32)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "waveform": {0: "batch"},
            "logits": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["waveform"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )

    if args.check_onnx:
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

    if args.check_onnxruntime:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        torch_out = model(dummy).detach().cpu().numpy()
        ort_out = session.run(None, {"waveform": dummy.cpu().numpy()})[0]
        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    else:
        max_abs_diff = None

    info = model.export_info
    return {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path.resolve()),
        "layers": info["layers"],
        "channels": info["channels"],
        "label_count": info["label_count"],
        "sample_rate": args.sample_rate,
        "dct_coeff": args.dct_coeff,
        "time_steps": info["time_steps"],
        "input_shape": f"{args.batch_size}x{args.sample_rate}",
        "mfcc_scale": args.mfcc_scale,
        "opset": args.opset,
        "onnxruntime_max_abs_diff": max_abs_diff,
    }


def write_manifest(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    manifest_path = output_dir / "full_onnx_friendly_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] manifest saved to: {manifest_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export waveform-to-logits DSCNN with ONNX-friendly MFCC")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--pattern", default="*.pt")
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "models_full"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pooling", choices=["global", "temporal"], default="global")
    parser.add_argument("--temporal_bins", type=int, default=4)
    parser.add_argument("--dynamic_batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check_onnx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check_onnxruntime", action="store_true", default=False)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--dct_coeff", type=int, default=10)
    parser.add_argument("--window_size_ms", type=int, default=32)
    parser.add_argument("--window_stride_ms", type=int, default=32)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=0.97)
    parser.add_argument("--mfcc_scale", choices=["torchaudio_db", "natural_log"], default="torchaudio_db")
    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default="triangular")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = discover_checkpoints(args)
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found. Use --input_dir or --checkpoints to specify best.pt files.")

    print(f"[INFO] found {len(checkpoints)} checkpoint(s)")
    rows = []
    for checkpoint in checkpoints:
        print(f"[EXPORT] {checkpoint}")
        try:
            row = export_one(checkpoint, args)
        except Exception as exc:
            print(f"[ERROR] failed: {checkpoint}: {exc}")
            if len(checkpoints) == 1:
                raise
            continue
        rows.append(row)
        print(f"[OK] {row['onnx']}")

    write_manifest(rows, Path(args.output_dir))
    print(f"[DONE] exported {len(rows)}/{len(checkpoints)} model(s)")


if __name__ == "__main__":
    main()
