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

import torch
import torch.nn as nn
import torch.nn.functional as F

from dscnn_kws.configs import CLASS_LIST
from dscnn_kws.frontend.bandpass_torch import create_fir_bandpass_filterbank
from dscnn_kws.frontend.pwl_fit_utils import fit_piecewise_linear_log_from_samples
from dscnn_kws.model import DSCNN
from dscnn_kws.model.dscnn import calculate_time_steps
from dscnn_kws.train import MFCCDSCNN


ARCH_RE = re.compile(r"L(?P<layers>\d+)_C(?P<channels>\d+)", re.IGNORECASE)

# =============================================================================
# Default export configuration
# =============================================================================
# Edit this block when you want to batch-export a different sweep without typing
# a long command every time. Command-line arguments still override these values.
DEFAULT_INPUT_DIR = REPO_ROOT / "dscnn_kws" / "runs" / "bandpass_pwl_snr_scene_arch_sweep_best_models"
DEFAULT_FALLBACK_INPUT_DIRS = [
    REPO_ROOT / "dscnn_kws" / "runs" / "bandpass_snr_scene_arch_sweep_best_models",
    REPO_ROOT / "dscnn_kws" / "runs" / "sweep_best_models",
    REPO_ROOT / "dscnn_kws" / "runs",
]
DEFAULT_PATTERN = "*.pt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "models_bandpass_pwl_full"

DEFAULT_EXPORT_MODE = "full"  # "backbone" or "full"
DEFAULT_FRONTEND = "bandpass"  # "mfcc" or "bandpass"
DEFAULT_OPSET = 17
DEFAULT_BATCH_SIZE = 1
DEFAULT_DYNAMIC_BATCH = True
DEFAULT_CHECK_ONNX = True

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DCT_COEFF = 10
DEFAULT_WINDOW_SIZE_MS = 32
DEFAULT_WINDOW_STRIDE_MS = 32

DEFAULT_MFCC_IMPL = "torchaudio"
DEFAULT_MEL_FILTER_SHAPE = "triangular"
DEFAULT_PRE_EMPHASIS = True
DEFAULT_PRE_EMPHASIS_COEFF = 0.97

DEFAULT_BANDPASS_N_BANDS = 10
DEFAULT_BANDPASS_F_MIN = 200.0
DEFAULT_BANDPASS_F_MAX = 4000.0
DEFAULT_BANDPASS_SPACING = "log"
DEFAULT_BANDPASS_KERNEL_SIZE = 63
DEFAULT_BANDPASS_PHASE_COUNT = 1

DEFAULT_LOG_APPROX_MODE = "pwl"
DEFAULT_LOG_PWL_NUM_SEGMENTS = 6
DEFAULT_LOG_PWL_STRATEGY = "uniform_logx"
DEFAULT_LOG_PWL_GAMMA = 1.0
DEFAULT_LOG_OFFSET = 1e-6
DEFAULT_LOG_INPUT_CLAMP_MIN = 1e-12


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
        if key.endswith("conv_layers.0.0.weight") or key.endswith("conv_layers.0.depthwise.weight"):
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


def build_backbone(
    *,
    num_layers: int,
    channels: int,
    label_count: int,
    sample_rate: int,
    window_stride_ms: int,
    dct_coeff: int,
    pooling: str = "global",
    temporal_bins: int = 4,
) -> DSCNN:
    time_steps = calculate_time_steps(sample_rate, window_stride_ms)
    input_dim = time_steps * dct_coeff
    return DSCNN(
        input_dim=input_dim,
        label_count=label_count,
        model_size_info=make_model_size_info(num_layers, channels),
        dct_coeff=dct_coeff,
        pooling=pooling,
        temporal_bins=temporal_bins,
    )


class ExportableBandpassFrontend(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.sample_rate = args.sample_rate
        self.n_bands = args.bandpass_n_bands
        self.frame_hop = int(args.sample_rate * args.window_stride_ms / 1000)
        self.kernel_size = args.bandpass_kernel_size
        self.phase_count = args.bandpass_phase_count
        self.target_time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
        self.log_approx_mode = args.log_approx_mode
        self.log_offset = args.log_offset
        self.log_input_clamp_min = args.log_input_clamp_min
        self.pre_emphasis = args.pre_emphasis
        self.pre_emphasis_coeff = args.pre_emphasis_coeff

        kernels, _ = create_fir_bandpass_filterbank(
            sample_rate=args.sample_rate,
            n_bands=args.bandpass_n_bands,
            f_min=args.bandpass_f_min,
            f_max=args.bandpass_f_max,
            spacing=args.bandpass_spacing,
            kernel_size=args.bandpass_kernel_size,
        )
        self.register_buffer("bandpass_kernels", kernels, persistent=False)

        if args.log_approx_mode == "pwl":
            fit = fit_piecewise_linear_log_from_samples(
                x_samples=torch.logspace(-8, 2, steps=30000),
                num_segments=args.log_pwl_num_segments,
                strategy=args.log_pwl_strategy,
                gamma=args.log_pwl_gamma,
            )
            self.register_buffer("log_pwl_breakpoints", torch.tensor(fit["breakpoints"], dtype=torch.float32), persistent=False)
            self.register_buffer("log_pwl_slopes", torch.tensor(fit["slopes"], dtype=torch.float32), persistent=False)
            self.register_buffer("log_pwl_intercepts", torch.tensor(fit["intercepts"], dtype=torch.float32), persistent=False)

    def _pre_emphasis(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pre_emphasis or self.pre_emphasis_coeff <= 0:
            return x
        first = x[:, :1]
        rest = x[:, 1:] - self.pre_emphasis_coeff * x[:, :-1]
        return torch.cat([first, rest], dim=1)

    def _log_transform(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x + self.log_offset, min=self.log_input_clamp_min)
        if self.log_approx_mode == "exact":
            return torch.log(x)

        bp = self.log_pwl_breakpoints.to(device=x.device, dtype=x.dtype)
        slopes = self.log_pwl_slopes.to(device=x.device, dtype=x.dtype)
        intercepts = self.log_pwl_intercepts.to(device=x.device, dtype=x.dtype)
        y = slopes[-1] * x + intercepts[-1]
        for i in range(slopes.numel() - 2, -1, -1):
            yi = slopes[i] * x + intercepts[i]
            y = torch.where(x < bp[i + 1], yi, y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self._pre_emphasis(x)
        x = x.unsqueeze(1)
        kernels = self.bandpass_kernels.to(device=x.device, dtype=x.dtype)

        required_len = (self.target_time_steps - 1) * self.frame_hop + self.kernel_size
        phase_powers = []
        for p in range(self.phase_count):
            offset = int(round(p * self.frame_hop / self.phase_count))
            x_phase = x[..., offset:]
            pad_right = max(0, required_len - int(x_phase.size(-1)))
            if pad_right > 0:
                x_phase = F.pad(x_phase, (0, pad_right), mode="constant", value=0.0)
            y = F.conv1d(x_phase, kernels, stride=self.frame_hop, padding=0)
            y = y[..., : self.target_time_steps]
            phase_powers.append(y * y)

        band_energy = torch.stack(phase_powers, dim=0).mean(dim=0)
        return self._log_transform(band_energy)


class ExportableBandpassDSCNN(nn.Module):
    def __init__(self, backbone: nn.Module, args: argparse.Namespace) -> None:
        super().__init__()
        self.feature_extractor = ExportableBandpassFrontend(args)
        self.backbone = backbone
        self.dct_coeff = args.dct_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = features[:, : self.dct_coeff, :]
        features = features.permute(0, 2, 1).reshape(features.size(0), -1)
        return self.backbone(features)


def build_full_model(backbone: nn.Module, args: argparse.Namespace) -> nn.Module:
    if args.frontend == "bandpass":
        return ExportableBandpassDSCNN(backbone, args)

    return MFCCDSCNN(
        backbone=backbone,
        frontend=args.frontend,
        sample_rate=args.sample_rate,
        dct_coeff=args.dct_coeff,
        window_size_ms=args.window_size_ms,
        window_stride_ms=args.window_stride_ms,
        bandpass_n_bands=args.bandpass_n_bands,
        bandpass_f_min=args.bandpass_f_min,
        bandpass_f_max=args.bandpass_f_max,
        bandpass_spacing=args.bandpass_spacing,
        bandpass_kernel_size=args.bandpass_kernel_size,
        bandpass_phase_count=args.bandpass_phase_count,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        spec_aug=False,
        spec_aug_freq_mask_param=1,
        spec_aug_time_mask_param=1,
        spec_aug_num_freq_masks=0,
        spec_aug_num_time_masks=0,
        mfcc_impl=args.mfcc_impl,
        mel_filter_shape=args.mel_filter_shape,
        log_approx_mode=args.log_approx_mode,
        log_pwl_num_segments=args.log_pwl_num_segments,
        log_pwl_strategy=args.log_pwl_strategy,
        log_pwl_gamma=args.log_pwl_gamma,
        log_pwl_breakpoints=None,
        log_pwl_slopes=None,
        log_pwl_intercepts=None,
        log_offset=args.log_offset,
        log_input_clamp_min=args.log_input_clamp_min,
    )


def select_model_and_state(
    state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    num_layers: int,
    channels: int,
    label_count: int,
) -> tuple[nn.Module, dict[str, torch.Tensor], tuple[int, ...], list[str], list[str]]:
    backbone = build_backbone(
        num_layers=num_layers,
        channels=channels,
        label_count=label_count,
        sample_rate=args.sample_rate,
        window_stride_ms=args.window_stride_ms,
        dct_coeff=args.dct_coeff,
        pooling=getattr(args, "pooling", "global"),
        temporal_bins=int(getattr(args, "temporal_bins", 4)),
    )

    if args.export_mode == "backbone":
        if any(key.startswith("backbone.") for key in state):
            state = {
                key[len("backbone.") :]: value
                for key, value in state.items()
                if key.startswith("backbone.")
            }
        dummy_shape = (args.batch_size, calculate_time_steps(args.sample_rate, args.window_stride_ms) * args.dct_coeff)
        return backbone, state, dummy_shape, ["features"], ["logits"]

    model = build_full_model(backbone, args)
    dummy_shape = (args.batch_size, args.sample_rate)
    return model, state, dummy_shape, ["waveform"], ["logits"]


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        paths = [Path(p) for p in args.checkpoints]
    elif args.input_dir:
        paths = sorted(Path(args.input_dir).glob(args.pattern))
    else:
        paths = []
        if DEFAULT_INPUT_DIR.exists():
            paths = sorted(DEFAULT_INPUT_DIR.glob(args.pattern))
        if not paths:
            for fallback_dir in DEFAULT_FALLBACK_INPUT_DIRS:
                if not fallback_dir.exists():
                    continue
                if fallback_dir.name == "runs":
                    paths = sorted(fallback_dir.glob("**/best.pt"))
                else:
                    paths = sorted(fallback_dir.glob(args.pattern))
                if paths:
                    break

    return [p.resolve() for p in paths if p.exists()]


def safe_stem(path: Path) -> str:
    if path.name == "best.pt":
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.parent.name)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def export_one(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    state = load_state_dict(path)
    inferred_layers, inferred_channels, label_count = infer_arch_from_state_dict(state)
    named_layers, named_channels = infer_arch_from_name(path)
    num_layers = args.layers or named_layers or inferred_layers
    channels = args.channels or named_channels or inferred_channels

    model, model_state, dummy_shape, input_names, output_names = select_model_and_state(
        state,
        args,
        num_layers=num_layers,
        channels=channels,
        label_count=label_count,
    )
    model.load_state_dict(model_state, strict=True)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{safe_stem(path)}_{args.export_mode}.onnx"

    dummy = torch.randn(*dummy_shape, dtype=torch.float32)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            input_names[0]: {0: "batch"},
            output_names[0]: {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
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
        ort_out = session.run(None, {input_names[0]: dummy.cpu().numpy()})[0]
        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    else:
        max_abs_diff = None

    return {
        "checkpoint": str(path),
        "onnx": str(onnx_path.resolve()),
        "export_mode": args.export_mode,
        "frontend": args.frontend,
        "layers": num_layers,
        "channels": channels,
        "label_count": label_count,
        "sample_rate": args.sample_rate,
        "dct_coeff": args.dct_coeff,
        "bandpass_n_bands": args.bandpass_n_bands if args.frontend == "bandpass" else None,
        "bandpass_f_min": args.bandpass_f_min if args.frontend == "bandpass" else None,
        "bandpass_f_max": args.bandpass_f_max if args.frontend == "bandpass" else None,
        "bandpass_spacing": args.bandpass_spacing if args.frontend == "bandpass" else None,
        "bandpass_kernel_size": args.bandpass_kernel_size if args.frontend == "bandpass" else None,
        "bandpass_phase_count": args.bandpass_phase_count if args.frontend == "bandpass" else None,
        "log_approx_mode": args.log_approx_mode,
        "log_pwl_num_segments": args.log_pwl_num_segments,
        "log_pwl_strategy": args.log_pwl_strategy,
        "log_pwl_gamma": args.log_pwl_gamma,
        "log_offset": args.log_offset,
        "log_input_clamp_min": args.log_input_clamp_min,
        "window_size_ms": args.window_size_ms,
        "window_stride_ms": args.window_stride_ms,
        "input_shape": "x".join(str(x) for x in dummy_shape),
        "opset": args.opset,
        "onnxruntime_max_abs_diff": max_abs_diff,
    }


def write_manifest(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    manifest_path = output_dir / "export_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] manifest saved to: {manifest_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export swept DSCNN best.pt models to ONNX")
    parser.add_argument("--input_dir", default=None, help="Directory containing .pt checkpoints")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern used with --input_dir")
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Explicit checkpoint paths")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--export_mode", choices=["backbone", "full"], default=DEFAULT_EXPORT_MODE)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dynamic_batch", action=argparse.BooleanOptionalAction, default=DEFAULT_DYNAMIC_BATCH)
    parser.add_argument("--check_onnx", action=argparse.BooleanOptionalAction, default=DEFAULT_CHECK_ONNX)
    parser.add_argument("--check_onnxruntime", action="store_true", default=False)

    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--dct_coeff", type=int, default=DEFAULT_DCT_COEFF)
    parser.add_argument("--window_size_ms", type=int, default=DEFAULT_WINDOW_SIZE_MS)
    parser.add_argument("--window_stride_ms", type=int, default=DEFAULT_WINDOW_STRIDE_MS)
    parser.add_argument("--layers", type=int, default=None, help="Override inferred layer count")
    parser.add_argument("--channels", type=int, default=None, help="Override inferred channel count")
    parser.add_argument("--pooling", choices=["global", "temporal"], default="global")
    parser.add_argument("--temporal_bins", type=int, default=4)

    parser.add_argument("--frontend", choices=["mfcc", "bandpass"], default=DEFAULT_FRONTEND)
    parser.add_argument("--mfcc_impl", choices=["torchaudio", "torch"], default=DEFAULT_MFCC_IMPL)
    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default=DEFAULT_MEL_FILTER_SHAPE)
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=DEFAULT_PRE_EMPHASIS)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=DEFAULT_PRE_EMPHASIS_COEFF)
    parser.add_argument("--bandpass_n_bands", type=int, default=DEFAULT_BANDPASS_N_BANDS)
    parser.add_argument("--bandpass_f_min", type=float, default=DEFAULT_BANDPASS_F_MIN)
    parser.add_argument("--bandpass_f_max", type=float, default=DEFAULT_BANDPASS_F_MAX)
    parser.add_argument("--bandpass_spacing", choices=["log", "linear"], default=DEFAULT_BANDPASS_SPACING)
    parser.add_argument("--bandpass_kernel_size", type=int, default=DEFAULT_BANDPASS_KERNEL_SIZE)
    parser.add_argument("--bandpass_phase_count", type=int, default=DEFAULT_BANDPASS_PHASE_COUNT)
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default=DEFAULT_LOG_APPROX_MODE)
    parser.add_argument("--log_pwl_num_segments", type=int, default=DEFAULT_LOG_PWL_NUM_SEGMENTS)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default=DEFAULT_LOG_PWL_STRATEGY)
    parser.add_argument("--log_pwl_gamma", type=float, default=DEFAULT_LOG_PWL_GAMMA)
    parser.add_argument("--log_offset", type=float, default=DEFAULT_LOG_OFFSET)
    parser.add_argument("--log_input_clamp_min", type=float, default=DEFAULT_LOG_INPUT_CLAMP_MIN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frontend == "bandpass" and args.dct_coeff != args.bandpass_n_bands:
        raise ValueError(
            "For bandpass full export, --dct_coeff must equal --bandpass_n_bands. "
            f"Got dct_coeff={args.dct_coeff}, bandpass_n_bands={args.bandpass_n_bands}."
        )

    checkpoints = discover_checkpoints(args)
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found. Use --input_dir or --checkpoints to specify best.pt files.")

    print(f"[INFO] export_mode={args.export_mode}")
    print(f"[INFO] frontend={args.frontend}")
    print(f"[INFO] output_dir={Path(args.output_dir).resolve()}")
    print(
        f"[INFO] sample_rate={args.sample_rate}, dct_coeff={args.dct_coeff}, "
        f"window_size_ms={args.window_size_ms}, window_stride_ms={args.window_stride_ms}"
    )
    if args.frontend == "bandpass":
        print(
            "[INFO] bandpass="
            f"n_bands={args.bandpass_n_bands}, "
            f"f_min={args.bandpass_f_min}, "
            f"f_max={args.bandpass_f_max}, "
            f"spacing={args.bandpass_spacing}, "
            f"kernel_size={args.bandpass_kernel_size}, "
            f"phase_count={args.bandpass_phase_count}"
        )
    print(
        "[INFO] log="
        f"mode={args.log_approx_mode}, "
        f"pwl_segments={args.log_pwl_num_segments}, "
        f"pwl_strategy={args.log_pwl_strategy}, "
        f"pwl_gamma={args.log_pwl_gamma}"
    )
    print(f"[INFO] found {len(checkpoints)} checkpoint(s)")
    rows = []
    for ckpt in checkpoints:
        print(f"[EXPORT] {ckpt}")
        try:
            row = export_one(ckpt, args)
        except Exception as exc:
            print(f"[ERROR] failed: {ckpt}: {exc}")
            if len(checkpoints) == 1:
                raise
            continue
        rows.append(row)
        print(f"[OK] {row['onnx']}")

    write_manifest(rows, Path(args.output_dir))
    print(f"[DONE] exported {len(rows)}/{len(checkpoints)} model(s)")


if __name__ == "__main__":
    main()
