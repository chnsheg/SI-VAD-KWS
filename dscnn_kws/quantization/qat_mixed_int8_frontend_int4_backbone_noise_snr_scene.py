from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader

try:
    from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
    from torch.ao.quantization import disable_observer, fuse_modules_qat, prepare_qat
    from torch.ao.quantization.qconfig import QConfig
except ImportError:
    from torch.quantization import FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
    from torch.quantization import disable_observer, fuse_modules_qat, prepare_qat
    from torch.quantization.qconfig import QConfig

from dscnn_kws.quantization.qat_int8_noise_snr_scene import (
    DepthwiseSeparableConv2d,
    build_model,
    build_scene_loader,
    build_test_list_loader,
    build_train_loader,
    build_valid_loader,
    choose_backend,
    copy_state_dict_cpu,
    count_usable_noise_files,
    evaluate,
    expected_params,
    freeze_qat_bn_stats,
    infer_arch_from_name,
    infer_arch_from_state_dict,
    infer_dataset_from_name,
    load_model_weights,
    load_state_dict,
    load_state_dict_to_model,
    safe_stem,
    train_one_epoch,
)
from dscnn_kws.utils import apply_pre_emphasis


SUPPORTED_FRONTEND_BITS = {8}
SUPPORTED_BACKBONE_BITS = {3, 4}
SUPPORTED_WEIGHT_QSCHEMES = {"per_channel", "per_tensor"}

TAU_SCENES = [
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram",
]


# =============================================================================
# Code-level configuration
# =============================================================================
# Edit this block directly, then run:
#   python dscnn_kws/quantization/qat_mixed_int8_frontend_int4_backbone_noise_snr_scene.py
#
# Command-line flags are still supported and override these defaults.

# Quantization. Frontend uses INT8 fake quant; backbone uses INT4 fake quant.
FRONTEND_BIT_WIDTH = 8
BACKBONE_BIT_WIDTH = 4
BACKBONE_WEIGHT_QSCHEME = "per_channel"  # per_channel or per_tensor

# Input / output.
INPUT_DIR = "/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models_full"
PATTERN = "*.pt"
CHECKPOINTS = None
OUTPUT_DIR = (
    f"./dscnn_kws/quantization/qat_frontend_int{FRONTEND_BIT_WIDTH}_"
    f"backbone_int{BACKBONE_BIT_WIDTH}_noise_snr_scene_models"
)
TRAIN_RESULTS_CSV = (
    f"./dscnn_kws/quantization/qat_frontend_int{FRONTEND_BIT_WIDTH}_"
    f"backbone_int{BACKBONE_BIT_WIDTH}_noise_snr_scene_train_results.csv"
)
GRID_RESULTS_CSV = (
    f"./dscnn_kws/quantization/qat_frontend_int{FRONTEND_BIT_WIDTH}_"
    f"backbone_int{BACKBONE_BIT_WIDTH}_noise_snr_scene_grid_results.csv"
)
LIMIT = 0

# Data / runtime.
ROOT = "/root/kws/dscnn_kws/dscnn_kws/data"
DATASET = None
BATCH = 256
NUM_WORKERS = 8
GPU = 1
SEED = 42

# QAT. PyTorch has no true INT4 convert backend here; this script uses
# mixed fake-quant QAT and exports qint tensors for reference/hardware work.
# Frontend path: INT8 activation fake quant.
# Backbone: INT4 activation fake quant + INT4 Conv/Linear weight fake quant.
QAT_EPOCHS = 30
LR = 5e-5
WEIGHT_DECAY = 1e-6
BACKEND = "fbgemm"
FREEZE_BN_AFTER_EPOCH = 8
DISABLE_OBSERVER_AFTER_EPOCH = 18
QUANTIZE_FRONTEND = True
WEIGHT_CH_AXIS = 0

# Noise training / validation / testing.
TRAIN_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_train.txt"]
VALID_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_valid.txt"]
TEST_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_test.txt"]
TRAIN_NOISE_PROB = 0.8
TRAIN_SNR_MIN_DB = -5.0
TRAIN_SNR_MAX_DB = 20.0
VALID_NOISE_PROB = 1.0
VALID_SNR_DB = 5.0
SCENE_TEST_ROOT = "./dscnn_kws/noise/tau"
SCENE_NAMES = TAU_SCENES
TEST_SNRS = [20.0, 10.0, 5.0, 0.0, -5.0]

# Model shape / family.
MODEL_FAMILY = "standard"  # standard or projected_bandpass
SAMPLE_RATE = 16000
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
LAYERS = None
CHANNELS = None

# Standard MFCCDSCNN frontend.
FRONTEND = "mfcc"  # mfcc or bandpass
MFCC_IMPL = "torchaudio"  # torchaudio or torch
MEL_FILTER_SHAPE = "triangular"  # triangular or rectangular
PRE_EMPHASIS = True
PRE_EMPHASIS_COEFF = 0.97
SPEC_AUG = False
SPEC_AUG_FREQ_MASK_PARAM = 1
SPEC_AUG_TIME_MASK_PARAM = 1
SPEC_AUG_NUM_FREQ_MASKS = 1
SPEC_AUG_NUM_TIME_MASKS = 1

# Bandpass / projected-bandpass frontend.
BANDPASS_N_BANDS = 10
BANDPASS_INTERNAL_BANDS = 40
BANDPASS_F_MIN = 80.0
BANDPASS_F_MAX = 6000.0
BANDPASS_SPACING = "log"  # log or linear
BANDPASS_KERNEL_SIZE = 255
BANDPASS_PHASE_COUNT = 4
PROJECTION_INIT = "dct"  # dct, average, or random
TRAINABLE_PROJECTION = True

# Log / PWL log.
LOG_APPROX_MODE = "exact"  # exact or pwl
LOG_PWL_NUM_SEGMENTS = 6
LOG_PWL_STRATEGY = "uniform_logx"  # uniform_logx, quantile, or powerlaw
LOG_PWL_GAMMA = 1.0
LOG_OFFSET = 1e-6
LOG_INPUT_CLAMP_MIN = 1e-12


def signed_qrange(bit_width: int) -> tuple[int, int]:
    if bit_width < 2 or bit_width > 8:
        raise ValueError(f"Signed fake quant bit width must be in [2, 8], got bit_width={bit_width}")
    return -(1 << (bit_width - 1)), (1 << (bit_width - 1)) - 1


def format_name(frontend_bit_width: int, backbone_bit_width: int) -> str:
    return f"frontend_int{frontend_bit_width}_backbone_int{backbone_bit_width}"


def default_output_dir(frontend_bit_width: int, backbone_bit_width: int) -> str:
    return (
        f"./dscnn_kws/quantization/qat_frontend_int{frontend_bit_width}_"
        f"backbone_int{backbone_bit_width}_noise_snr_scene_models"
    )


def default_train_csv(frontend_bit_width: int, backbone_bit_width: int) -> str:
    return (
        f"./dscnn_kws/quantization/qat_frontend_int{frontend_bit_width}_"
        f"backbone_int{backbone_bit_width}_noise_snr_scene_train_results.csv"
    )


def default_grid_csv(frontend_bit_width: int, backbone_bit_width: int) -> str:
    return (
        f"./dscnn_kws/quantization/qat_frontend_int{frontend_bit_width}_"
        f"backbone_int{backbone_bit_width}_noise_snr_scene_grid_results.csv"
    )


def make_symmetric_fake_quant(bit_width: int) -> FakeQuantize:
    qmin, qmax = signed_qrange(bit_width)
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=qmin,
        quant_max=qmax,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )()


def make_per_channel_weight_fake_quant(bit_width: int) -> FakeQuantize:
    qmin, qmax = signed_qrange(bit_width)
    return FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=qmin,
        quant_max=qmax,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=WEIGHT_CH_AXIS,
        reduce_range=False,
    )


def make_per_tensor_weight_fake_quant(bit_width: int) -> FakeQuantize:
    qmin, qmax = signed_qrange(bit_width)
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=qmin,
        quant_max=qmax,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )


def make_backbone_qconfig(bit_width: int, weight_qscheme: str) -> QConfig:
    if weight_qscheme not in SUPPORTED_WEIGHT_QSCHEMES:
        raise ValueError(f"Unsupported weight_qscheme={weight_qscheme}")
    activation_fake_quant = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=signed_qrange(bit_width)[0],
        quant_max=signed_qrange(bit_width)[1],
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )
    if weight_qscheme == "per_channel":
        weight_fake_quant = make_per_channel_weight_fake_quant(bit_width)
    else:
        weight_fake_quant = make_per_tensor_weight_fake_quant(bit_width)
    return QConfig(activation=activation_fake_quant, weight=weight_fake_quant)


class BackboneFakeQuantWrapper(nn.Module):
    """Keep fake-quant active around the backbone for mixed-bit QAT evaluation."""

    def __init__(self, backbone: nn.Module, bit_width: int):
        super().__init__()
        self.input_fake_quant = make_symmetric_fake_quant(bit_width)
        self.backbone = backbone
        self.output_fake_quant = make_symmetric_fake_quant(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_fake_quant(x)
        x = self.backbone(x)
        return self.output_fake_quant(x)


class FrontendInt8QATWrapper(nn.Module):
    """Add signed symmetric INT8 fake-quant points around the frontend path."""

    def __init__(self, model: nn.Module, bit_width: int):
        super().__init__()
        required = ["feature_extractor", "backbone", "dct_coeff"]
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise AttributeError(f"Cannot wrap frontend for low-bit QAT; missing attributes: {missing}")

        self.feature_extractor = model.feature_extractor
        self.backbone = model.backbone
        self.dct_coeff = model.dct_coeff
        self.pre_emphasis = bool(getattr(model, "pre_emphasis", True))
        self.pre_emphasis_coeff = float(getattr(model, "pre_emphasis_coeff", 0.97))
        self.spec_aug = bool(getattr(model, "spec_aug", False))
        self.freq_mask = getattr(model, "freq_mask", nn.Identity())
        self.time_mask = getattr(model, "time_mask", nn.Identity())
        self.spec_aug_num_freq_masks = int(getattr(model, "spec_aug_num_freq_masks", 0))
        self.spec_aug_num_time_masks = int(getattr(model, "spec_aug_num_time_masks", 0))

        self.waveform_fake_quant = make_symmetric_fake_quant(bit_width)
        self.pre_emphasis_fake_quant = make_symmetric_fake_quant(bit_width)
        self.frontend_output_fake_quant = make_symmetric_fake_quant(bit_width)
        self.backbone_input_fake_quant = make_symmetric_fake_quant(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.waveform_fake_quant(x)
        if self.pre_emphasis:
            x = apply_pre_emphasis(x, self.pre_emphasis_coeff)
            x = self.pre_emphasis_fake_quant(x)

        features = self.feature_extractor(x)
        features = self.frontend_output_fake_quant(features)
        if self.training and self.spec_aug:
            for _ in range(self.spec_aug_num_freq_masks):
                features = self.freq_mask(features)
            for _ in range(self.spec_aug_num_time_masks):
                features = self.time_mask(features)
        features = features[:, : self.dct_coeff, :]
        features = features.permute(0, 2, 1).reshape(features.size(0), -1)
        features = self.backbone_input_fake_quant(features)
        return self.backbone(features)


def fuse_backbone_for_qat(backbone: nn.Module) -> None:
    backbone.train()
    for idx, layer in enumerate(backbone.conv_layers):
        if isinstance(layer, nn.Sequential):
            fuse_modules_qat(layer, ["0", "1", "2"], inplace=True)
        elif isinstance(layer, DepthwiseSeparableConv2d):
            fuse_modules_qat(layer, ["depthwise", "bn_depthwise"], inplace=True)
            fuse_modules_qat(layer, ["pointwise", "bn_pointwise"], inplace=True)
        else:
            raise TypeError(f"Unsupported DSCNN layer at conv_layers.{idx}: {type(layer)}")


def prepare_model_for_mixed_qat(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if args.quantize_frontend:
        model = FrontendInt8QATWrapper(model, args.frontend_bit_width)
    if not hasattr(model, "backbone"):
        raise AttributeError("Mixed-bit QAT script expects model.backbone")
    fuse_backbone_for_qat(model.backbone)
    model.backbone = BackboneFakeQuantWrapper(model.backbone, args.backbone_bit_width)
    model.qconfig = None
    model.backbone.qconfig = make_backbone_qconfig(args.backbone_bit_width, args.backbone_weight_qscheme)
    model.train()
    prepare_qat(model, inplace=True)
    return model


def quantize_tensor_to_int(tensor: torch.Tensor, bit_width: int, eps: float = 1e-12) -> tuple[torch.Tensor, float]:
    qmin, qmax = signed_qrange(bit_width)
    amax = float(tensor.detach().abs().max().cpu().item())
    scale = max(amax, eps) / float(qmax)
    q = torch.clamp(torch.round(tensor.detach().cpu() / scale), qmin, qmax).to(torch.int8)
    return q, scale


def is_per_channel_weight_tensor(key: str, tensor: torch.Tensor) -> bool:
    if tensor.dim() < 2:
        return False
    return key.startswith("backbone.") and key.endswith(".weight")


def quantize_weight_per_channel_to_int(
    tensor: torch.Tensor,
    bit_width: int,
    ch_axis: int = WEIGHT_CH_AXIS,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmin, qmax = signed_qrange(bit_width)
    value = tensor.detach().cpu()
    reduce_dims = [dim for dim in range(value.dim()) if dim != ch_axis]
    amax = value.abs().amax(dim=reduce_dims)
    scales = torch.clamp(amax / float(qmax), min=eps)
    view_shape = [1] * value.dim()
    view_shape[ch_axis] = scales.numel()
    q = torch.clamp(torch.round(value / scales.view(view_shape)), qmin, qmax).to(torch.int8)
    return q, scales


def infer_tensor_bit_width(key: str, args: argparse.Namespace) -> int:
    frontend_prefixes = (
        "feature_extractor.",
        "waveform_fake_quant.",
        "pre_emphasis_fake_quant.",
        "frontend_output_fake_quant.",
        "backbone_input_fake_quant.",
    )
    if key.startswith(frontend_prefixes):
        return args.frontend_bit_width
    return args.backbone_bit_width


def export_mixed_state_dict(model: nn.Module, args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    q_state: dict[str, torch.Tensor] = {}
    q_specs: dict[str, dict[str, Any]] = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            bit_width = infer_tensor_bit_width(key, args)
            qmin, qmax = signed_qrange(bit_width)
            if args.backbone_weight_qscheme == "per_channel" and is_per_channel_weight_tensor(key, value):
                q, scales = quantize_weight_per_channel_to_int(value, bit_width)
                q_state[key] = q
                q_specs[key] = {
                    "scale": scales.tolist(),
                    "zero_point": [0 for _ in range(scales.numel())],
                    "qmin": qmin,
                    "qmax": qmax,
                    "dtype": f"int{bit_width}_stored_int8",
                    "qscheme": "per_channel_symmetric",
                    "ch_axis": WEIGHT_CH_AXIS,
                    "bit_width": bit_width,
                }
            else:
                q, scale = quantize_tensor_to_int(value, bit_width)
                q_state[key] = q
                q_specs[key] = {
                    "scale": scale,
                    "zero_point": 0,
                    "qmin": qmin,
                    "qmax": qmax,
                    "dtype": f"int{bit_width}_stored_int8",
                    "qscheme": "per_tensor_symmetric",
                    "bit_width": bit_width,
                }
        elif torch.is_tensor(value):
            q_state[key] = value.detach().cpu()
    return q_state, q_specs


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] csv saved to: {path.resolve()}")


def save_mixed_artifacts(
    *,
    output_dir: Path,
    stem: str,
    source_checkpoint: Path,
    qat_model: nn.Module,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = format_name(args.frontend_bit_width, args.backbone_bit_width)
    qat_path = output_dir / f"{stem}_qat_{tag}_prepared_best.pt"
    fake_quant_path = output_dir / f"{stem}_qat_{tag}_fake_quant.pt"
    qint_state, q_specs = export_mixed_state_dict(qat_model, args)
    frontend_qmin, frontend_qmax = signed_qrange(args.frontend_bit_width)
    backbone_qmin, backbone_qmax = signed_qrange(args.backbone_bit_width)
    common = {
        "source_checkpoint": str(source_checkpoint),
        "metadata": row,
        "qat": {
            "frontend_bit_width": args.frontend_bit_width,
            "frontend_qrange": [frontend_qmin, frontend_qmax],
            "backbone_bit_width": args.backbone_bit_width,
            "backbone_qrange": [backbone_qmin, backbone_qmax],
            "epochs": args.qat_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "freeze_bn_after_epoch": args.freeze_bn_after_epoch,
            "disable_observer_after_epoch": args.disable_observer_after_epoch,
            "quantize_frontend": args.quantize_frontend,
            "activation_qscheme": "per_tensor_symmetric",
            "backbone_weight_qscheme": args.backbone_weight_qscheme,
            "weight_ch_axis": WEIGHT_CH_AXIS,
            "scope": (
                f"frontend int{args.frontend_bit_width} activation fake-quant QAT + "
                f"DSCNN backbone int{args.backbone_bit_width} activation/weight fake-quant"
                if args.quantize_frontend
                else f"DSCNN backbone int{args.backbone_bit_width} activation/weight fake-quant; frontend remains float"
            ),
        },
        "note": (
            "This is a mixed-bit fake-quant QAT checkpoint. It keeps float modules "
            "with fake-quant simulation for evaluation and stores qint tensors for reference."
        ),
    }
    torch.save({**common, "state_dict": qat_model.state_dict(), "format": f"prepared_qat_{tag}_float"}, qat_path)
    torch.save(
        {
            **common,
            "state_dict": qat_model.state_dict(),
            "state_dict_qint": qint_state,
            "quant_specs": q_specs,
            "format": f"fake_quant_{tag}_with_qint_export",
        },
        fake_quant_path,
    )
    return qat_path, fake_quant_path


def run_scene_grid(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    dataset: str,
    arch: str,
    layers: int,
    channels: int,
    expected_param_count: int,
    source_checkpoint: Path,
    quantized_checkpoint: Path,
) -> list[dict[str, Any]]:
    device = torch.device("cpu")
    rows = []
    for scene_idx, scene in enumerate(args.scene_names):
        scene_root = Path(args.scene_test_root) / scene
        usable = count_usable_noise_files([str(scene_root)])
        if usable <= 0:
            print(f"[WARN] scene skipped, no usable wavs: {scene_root}")
            continue
        for snr_idx, snr_db in enumerate(args.test_snrs):
            seed = 500000 + scene_idx * 10007 + snr_idx * 101
            loader = build_scene_loader(args, dataset, scene, snr_db, seed)
            metrics = evaluate(model, loader, device)
            row = {
                "dataset": dataset,
                "arch": arch,
                "layers": layers,
                "channels": channels,
                "expected_params": expected_param_count,
                "scene": scene,
                "snr_db": snr_db,
                "scene_noise_root": str(scene_root),
                "usable_noise_files": usable,
                "acc": metrics.acc,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "num_samples": metrics.num_samples,
                "source_checkpoint": str(source_checkpoint),
                "quantized_checkpoint": str(quantized_checkpoint),
                "frontend_bit_width": args.frontend_bit_width,
                "backbone_bit_width": args.backbone_bit_width,
                "activation_qscheme": "per_tensor_symmetric",
                "backbone_weight_qscheme": args.backbone_weight_qscheme,
                "quantized_backend": "mixed_fake_quant_float_ops",
            }
            rows.append(row)
            print(
                f"[GRID-{format_name(args.frontend_bit_width, args.backbone_bit_width).upper()}] {dataset} | {arch} | "
                f"{scene:<18} | snr={snr_db:>5} | acc={metrics.acc:.4f} | f1={metrics.f1:.4f}"
            )
    return rows


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        paths = [Path(p).resolve() for p in args.checkpoints]
    else:
        paths = sorted(Path(args.input_dir).glob(args.pattern))
        paths = [p.resolve() for p in paths if p.is_file()]
    if args.limit > 0:
        paths = paths[: args.limit]
    return paths


def run_one_checkpoint(path: Path, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = load_state_dict(path)
    inferred_layers, inferred_channels, label_count = infer_arch_from_state_dict(state)
    named_layers, named_channels = infer_arch_from_name(path)
    layers = args.layers or named_layers or inferred_layers
    channels = args.channels or named_channels or inferred_channels
    dataset = args.dataset or infer_dataset_from_name(path)
    if not dataset:
        raise ValueError(f"Cannot infer dataset from {path}; pass --dataset for explicit single-dataset runs.")

    tag = format_name(args.frontend_bit_width, args.backbone_bit_width)
    arch = f"L{layers}_C{channels}"
    param_count = expected_params(layers, channels, label_count)
    print("\n" + "=" * 100)
    print(f"[QAT-{tag.upper()}] dataset={dataset}, arch={arch}, ckpt={path}")

    model = build_model(args, num_layers=layers, channels=channels, label_count=label_count)
    load_model_weights(model, state)
    model = prepare_model_for_mixed_qat(model, args).to(device)

    train_loader = build_train_loader(args, dataset)
    valid_loader = build_valid_loader(args, dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.qat_epochs), eta_min=args.lr * 0.05)

    best_acc = -1.0
    best_epoch = 0
    best_state = copy_state_dict_cpu(model)
    train_loss = None
    train_acc = None
    valid_loss = None
    valid_acc = None
    for epoch in range(1, args.qat_epochs + 1):
        if args.freeze_bn_after_epoch > 0 and epoch == args.freeze_bn_after_epoch + 1:
            model.apply(freeze_qat_bn_stats)
            print(f"[INFO] froze QAT BN stats after epoch {args.freeze_bn_after_epoch}")
        if args.disable_observer_after_epoch > 0 and epoch == args.disable_observer_after_epoch + 1:
            model.apply(disable_observer)
            print(f"[INFO] disabled QAT observers after epoch {args.disable_observer_after_epoch}")

        train_m = train_one_epoch(model, train_loader, optimizer, device)
        valid_m = evaluate(model, valid_loader, device)
        scheduler.step()
        train_loss, train_acc = train_m.loss, train_m.acc
        valid_loss, valid_acc = valid_m.loss, valid_m.acc
        print(
            f"Epoch {epoch}/{args.qat_epochs} | "
            f"train_loss={train_m.loss:.4f} acc={train_m.acc:.4f} | "
            f"valid_loss={valid_m.loss:.4f} acc={valid_m.acc:.4f} f1={valid_m.f1:.4f}"
        )
        if valid_m.acc > best_acc:
            best_acc = valid_m.acc
            best_epoch = epoch
            best_state = copy_state_dict_cpu(model)

    load_state_dict_to_model(model, best_state)
    model.eval()
    test_list_loader = build_test_list_loader(args, dataset)
    qat_test_m = evaluate(model, test_list_loader, device)

    fake_quant_model = copy.deepcopy(model).cpu().eval()
    fake_quant_test_m = evaluate(fake_quant_model, test_list_loader, torch.device("cpu"))

    stem = safe_stem(path)
    artifact_row = {
        "dataset": dataset,
        "arch": arch,
        "layers": layers,
        "channels": channels,
        "expected_params": param_count,
        "source_checkpoint": str(path),
        "best_epoch": best_epoch,
        "best_valid_acc": best_acc,
        "last_train_loss": train_loss,
        "last_train_acc": train_acc,
        "last_valid_loss": valid_loss,
        "last_valid_acc": valid_acc,
        "qat_test_tau_list_acc": qat_test_m.acc,
        "qat_test_tau_list_f1": qat_test_m.f1,
        "quantized_test_tau_list_acc": fake_quant_test_m.acc,
        "quantized_test_tau_list_f1": fake_quant_test_m.f1,
        "frontend_bit_width": args.frontend_bit_width,
        "backbone_bit_width": args.backbone_bit_width,
        "quant_format": tag,
        "activation_qscheme": "per_tensor_symmetric",
        "backbone_weight_qscheme": args.backbone_weight_qscheme,
        "weight_ch_axis": WEIGHT_CH_AXIS,
        "quantized_backend": "mixed_fake_quant_float_ops",
        "model_family": args.model_family,
        "frontend": args.frontend if args.model_family == "standard" else "projected_bandpass",
        "quantize_frontend": args.quantize_frontend,
    }
    qat_path, fake_quant_path = save_mixed_artifacts(
        output_dir=Path(args.output_dir),
        stem=stem,
        source_checkpoint=path,
        qat_model=model.cpu(),
        row=artifact_row,
        args=args,
    )
    artifact_row["qat_prepared_checkpoint"] = str(qat_path.resolve())
    artifact_row["quantized_checkpoint"] = str(fake_quant_path.resolve())
    artifact_row["qat_prepared_size_bytes"] = qat_path.stat().st_size
    artifact_row["quantized_size_bytes"] = fake_quant_path.stat().st_size

    grid_rows = run_scene_grid(
        args,
        model=fake_quant_model,
        dataset=dataset,
        arch=arch,
        layers=layers,
        channels=channels,
        expected_param_count=param_count,
        source_checkpoint=path,
        quantized_checkpoint=fake_quant_path,
    )
    for row in grid_rows:
        row.update(
            {
                "best_epoch": best_epoch,
                "best_valid_acc": best_acc,
                "model_family": args.model_family,
                "frontend": args.frontend if args.model_family == "standard" else "projected_bandpass",
                "quantize_frontend": args.quantize_frontend,
            }
        )
    return artifact_row, grid_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mixed fake-quant QAT for noise-scene DSCNN checkpoints: "
            f"frontend INT{FRONTEND_BIT_WIDTH}, backbone INT{BACKBONE_BIT_WIDTH}."
        )
    )
    parser.add_argument("--frontend_bit_width", type=int, choices=sorted(SUPPORTED_FRONTEND_BITS), default=FRONTEND_BIT_WIDTH)
    parser.add_argument("--backbone_bit_width", type=int, choices=sorted(SUPPORTED_BACKBONE_BITS), default=BACKBONE_BIT_WIDTH)
    parser.add_argument("--backbone_weight_qscheme", choices=sorted(SUPPORTED_WEIGHT_QSCHEMES), default=BACKBONE_WEIGHT_QSCHEME)
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--pattern", default=PATTERN)
    parser.add_argument("--checkpoints", nargs="*", default=CHECKPOINTS)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--train_results_csv", default=TRAIN_RESULTS_CSV)
    parser.add_argument("--grid_results_csv", default=GRID_RESULTS_CSV)
    parser.add_argument("--limit", type=int, default=LIMIT)

    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--gpu", type=int, default=GPU)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--qat_epochs", type=int, default=QAT_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--freeze_bn_after_epoch", type=int, default=FREEZE_BN_AFTER_EPOCH)
    parser.add_argument("--disable_observer_after_epoch", type=int, default=DISABLE_OBSERVER_AFTER_EPOCH)
    parser.add_argument("--quantize_frontend", action=argparse.BooleanOptionalAction, default=QUANTIZE_FRONTEND)

    parser.add_argument("--train_noise_roots", nargs="+", default=TRAIN_NOISE_ROOTS)
    parser.add_argument("--valid_noise_roots", nargs="+", default=VALID_NOISE_ROOTS)
    parser.add_argument("--test_noise_roots", nargs="+", default=TEST_NOISE_ROOTS)
    parser.add_argument("--train_noise_prob", type=float, default=TRAIN_NOISE_PROB)
    parser.add_argument("--train_snr_min_db", type=float, default=TRAIN_SNR_MIN_DB)
    parser.add_argument("--train_snr_max_db", type=float, default=TRAIN_SNR_MAX_DB)
    parser.add_argument("--valid_noise_prob", type=float, default=VALID_NOISE_PROB)
    parser.add_argument("--valid_snr_db", type=float, default=VALID_SNR_DB)
    parser.add_argument("--scene_test_root", default=SCENE_TEST_ROOT)
    parser.add_argument("--scene_names", nargs="+", default=SCENE_NAMES)
    parser.add_argument("--test_snrs", nargs="+", type=float, default=TEST_SNRS)

    parser.add_argument("--model_family", choices=["standard", "projected_bandpass"], default=MODEL_FAMILY)
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--dct_coeff", type=int, default=DCT_COEFF)
    parser.add_argument("--window_size_ms", type=int, default=WINDOW_SIZE_MS)
    parser.add_argument("--window_stride_ms", type=int, default=WINDOW_STRIDE_MS)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--channels", type=int, default=CHANNELS)

    parser.add_argument("--frontend", choices=["mfcc", "bandpass"], default=FRONTEND)
    parser.add_argument("--mfcc_impl", choices=["torchaudio", "torch"], default=MFCC_IMPL)
    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default=MEL_FILTER_SHAPE)
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=PRE_EMPHASIS)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=PRE_EMPHASIS_COEFF)
    parser.add_argument("--spec_aug", action=argparse.BooleanOptionalAction, default=SPEC_AUG)
    parser.add_argument("--spec_aug_freq_mask_param", type=int, default=SPEC_AUG_FREQ_MASK_PARAM)
    parser.add_argument("--spec_aug_time_mask_param", type=int, default=SPEC_AUG_TIME_MASK_PARAM)
    parser.add_argument("--spec_aug_num_freq_masks", type=int, default=SPEC_AUG_NUM_FREQ_MASKS)
    parser.add_argument("--spec_aug_num_time_masks", type=int, default=SPEC_AUG_NUM_TIME_MASKS)

    parser.add_argument("--bandpass_n_bands", type=int, default=BANDPASS_N_BANDS)
    parser.add_argument("--bandpass_internal_bands", type=int, default=BANDPASS_INTERNAL_BANDS)
    parser.add_argument("--bandpass_f_min", type=float, default=BANDPASS_F_MIN)
    parser.add_argument("--bandpass_f_max", type=float, default=BANDPASS_F_MAX)
    parser.add_argument("--bandpass_spacing", choices=["log", "linear"], default=BANDPASS_SPACING)
    parser.add_argument("--bandpass_kernel_size", type=int, default=BANDPASS_KERNEL_SIZE)
    parser.add_argument("--bandpass_phase_count", type=int, default=BANDPASS_PHASE_COUNT)
    parser.add_argument("--projection_init", choices=["dct", "average", "random"], default=PROJECTION_INIT)
    parser.add_argument("--trainable_projection", action=argparse.BooleanOptionalAction, default=TRAINABLE_PROJECTION)

    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default=LOG_APPROX_MODE)
    parser.add_argument("--log_pwl_num_segments", type=int, default=LOG_PWL_NUM_SEGMENTS)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default=LOG_PWL_STRATEGY)
    parser.add_argument("--log_pwl_gamma", type=float, default=LOG_PWL_GAMMA)
    parser.add_argument("--log_offset", type=float, default=LOG_OFFSET)
    parser.add_argument("--log_input_clamp_min", type=float, default=LOG_INPUT_CLAMP_MIN)
    args = parser.parse_args()

    # Keep default output names aligned with explicitly changed bit widths.
    if args.frontend_bit_width != FRONTEND_BIT_WIDTH or args.backbone_bit_width != BACKBONE_BIT_WIDTH:
        if args.output_dir == OUTPUT_DIR:
            args.output_dir = default_output_dir(args.frontend_bit_width, args.backbone_bit_width)
        if args.train_results_csv == TRAIN_RESULTS_CSV:
            args.train_results_csv = default_train_csv(args.frontend_bit_width, args.backbone_bit_width)
        if args.grid_results_csv == GRID_RESULTS_CSV:
            args.grid_results_csv = default_grid_csv(args.frontend_bit_width, args.backbone_bit_width)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.frontend_bit_width not in SUPPORTED_FRONTEND_BITS:
        raise ValueError(f"Only frontend INT8 is supported here, got int{args.frontend_bit_width}")
    if args.backbone_bit_width not in SUPPORTED_BACKBONE_BITS:
        raise ValueError(f"Only backbone INT3/INT4 are supported here, got int{args.backbone_bit_width}")
    if args.backbone_weight_qscheme not in SUPPORTED_WEIGHT_QSCHEMES:
        raise ValueError(f"Unsupported backbone_weight_qscheme={args.backbone_weight_qscheme}")
    signed_qrange(args.frontend_bit_width)
    signed_qrange(args.backbone_bit_width)
    if args.model_family == "standard" and args.frontend == "bandpass" and args.dct_coeff != args.bandpass_n_bands:
        raise ValueError("--frontend bandpass requires --dct_coeff == --bandpass_n_bands")
    if args.model_family == "projected_bandpass" and args.bandpass_internal_bands < args.dct_coeff:
        raise ValueError("--bandpass_internal_bands must be >= --dct_coeff")
    for name, roots in [
        ("train_noise_roots", args.train_noise_roots),
        ("valid_noise_roots", args.valid_noise_roots),
        ("test_noise_roots", args.test_noise_roots),
    ]:
        usable = count_usable_noise_files(roots)
        print(f"[INFO] {name}={roots}, usable_noise_files={usable}")
        if usable <= 0:
            raise FileNotFoundError(f"No usable wav files found for {name}={roots}")


def main() -> None:
    args = parse_args()
    args.backend = choose_backend(args.backend)
    validate_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.gpu > 0 and torch.cuda.is_available() else "cpu")
    print(
        f"[INFO] device={device}, backend={args.backend}, model_family={args.model_family}, "
        f"quant={format_name(args.frontend_bit_width, args.backbone_bit_width)}, "
        f"frontend_qrange={signed_qrange(args.frontend_bit_width)}, "
        f"backbone_qrange={signed_qrange(args.backbone_bit_width)}, "
        f"backbone_weight_qscheme={args.backbone_weight_qscheme}"
    )

    checkpoints = discover_checkpoints(args)
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found. Use --input_dir or --checkpoints.")
    print(f"[INFO] checkpoints={len(checkpoints)}")

    train_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        try:
            train_row, rows = run_one_checkpoint(ckpt, args, device)
        except Exception as exc:
            print(f"[ERROR] failed: {ckpt}: {exc}")
            if len(checkpoints) == 1:
                raise
            continue
        train_rows.append(train_row)
        grid_rows.extend(rows)
        save_csv(train_rows, Path(args.train_results_csv))
        save_csv(grid_rows, Path(args.grid_results_csv))

    save_csv(train_rows, Path(args.train_results_csv))
    save_csv(grid_rows, Path(args.grid_results_csv))
    print(
        f"[DONE] {format_name(args.frontend_bit_width, args.backbone_bit_width)} "
        f"qat_models={len(train_rows)}, grid_rows={len(grid_rows)}"
    )


if __name__ == "__main__":
    main()
