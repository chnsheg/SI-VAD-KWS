from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver
    from torch.ao.quantization import QuantStub, DeQuantStub
    from torch.ao.quantization import convert, disable_observer, get_default_qat_qconfig, prepare_qat
    from torch.ao.quantization import fuse_modules_qat
except ImportError:
    from torch.quantization import FakeQuantize, MovingAverageMinMaxObserver
    from torch.quantization import QuantStub, DeQuantStub
    from torch.quantization import convert, disable_observer, get_default_qat_qconfig, prepare_qat
    from torch.quantization import fuse_modules_qat

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
from dscnn_kws.data.dataset import SpeechCommandDataset
from dscnn_kws.model import DSCNN
from dscnn_kws.model.dscnn import DepthwiseSeparableConv2d, calculate_time_steps
from dscnn_kws.train import MFCCDSCNN
from dscnn_kws.utils import apply_pre_emphasis


ARCH_RE = re.compile(r"L(?P<layers>\d+)_C(?P<channels>\d+)", re.IGNORECASE)
DATASET_RE = re.compile(r"(?P<dataset>.+?)_L\d+_C\d+", re.IGNORECASE)

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
# Normally you can edit this block directly, then run:
#   python dscnn_kws/quantization/qat_int8_noise_snr_scene.py
#
# Command-line flags are still supported and override these defaults.

# Input / output.
INPUT_DIR = "/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models"
PATTERN = "*.pt"
CHECKPOINTS = None
OUTPUT_DIR = "./dscnn_kws/quantization/qat_int8_noise_snr_scene_models"
TRAIN_RESULTS_CSV = "./dscnn_kws/quantization/qat_int8_noise_snr_scene_train_results.csv"
GRID_RESULTS_CSV = "./dscnn_kws/quantization/qat_int8_noise_snr_scene_grid_results.csv"
LIMIT = 0

# Data / runtime.
ROOT = "/root/kws/dscnn_kws/dscnn_kws/data"
DATASET = None
BATCH = 256
NUM_WORKERS = 8
GPU = 1
SEED = 42

# QAT.
QAT_EPOCHS = 10
LR = 5e-5
WEIGHT_DECAY = 1e-6
BACKEND = "fbgemm"
FREEZE_BN_AFTER_EPOCH = 3
DISABLE_OBSERVER_AFTER_EPOCH = 5
QUANTIZE_FRONTEND = True

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


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    precision: float
    recall: float
    f1: float
    num_samples: int


class QuantizedBackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.backbone = backbone
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.backbone(x)
        return self.dequant(x)


def make_symmetric_int8_fake_quant() -> FakeQuantize:
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )()


class FrontendInt8QATWrapper(nn.Module):
    """Add INT8 fake-quant points around the frontend path.

    PyTorch eager-mode quantization cannot convert STFT/log/PWL and functional
    bandpass operations into integer kernels. This wrapper makes those frontend
    tensors participate in QAT by applying fake-quant to the waveform,
    pre-emphasis output, frontend output, and flattened backbone input.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        required = ["feature_extractor", "backbone", "dct_coeff"]
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise AttributeError(f"Cannot wrap frontend for QAT; missing attributes: {missing}")

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

        self.waveform_fake_quant = make_symmetric_int8_fake_quant()
        self.pre_emphasis_fake_quant = make_symmetric_int8_fake_quant()
        self.frontend_output_fake_quant = make_symmetric_int8_fake_quant()
        self.backbone_input_fake_quant = make_symmetric_int8_fake_quant()

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


def make_model_size_info(num_layers: int, channels: int) -> list[int]:
    info = [num_layers]
    info += [channels, 10, 4, 2, 2]
    for _ in range(num_layers - 1):
        info += [channels, 3, 3, 1, 1]
    return info


def expected_params(num_layers: int, channels: int, num_classes: int = 2) -> int:
    c = channels
    n = num_layers
    return (n - 1) * c * c + (42 + 13 * (n - 1) + num_classes) * c + num_classes


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


def infer_dataset_from_name(path: Path) -> str | None:
    match = DATASET_RE.search(path.name) or DATASET_RE.search(path.parent.name)
    if not match:
        return None
    return match.group("dataset")


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


def safe_stem(path: Path) -> str:
    raw = path.parent.name if path.name == "best.pt" else path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def load_model_weights(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def build_backbone(args: argparse.Namespace, num_layers: int, channels: int, label_count: int) -> DSCNN:
    time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
    input_dim = time_steps * args.dct_coeff
    return DSCNN(
        input_dim=input_dim,
        label_count=label_count,
        model_size_info=make_model_size_info(num_layers, channels),
        dct_coeff=args.dct_coeff,
    )


def build_standard_model(
    args: argparse.Namespace,
    *,
    num_layers: int,
    channels: int,
    label_count: int,
) -> MFCCDSCNN:
    backbone = build_backbone(args, num_layers, channels, label_count)
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
        spec_aug=args.spec_aug,
        spec_aug_freq_mask_param=args.spec_aug_freq_mask_param,
        spec_aug_time_mask_param=args.spec_aug_time_mask_param,
        spec_aug_num_freq_masks=args.spec_aug_num_freq_masks,
        spec_aug_num_time_masks=args.spec_aug_num_time_masks,
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


def build_projected_bandpass_model(
    args: argparse.Namespace,
    *,
    num_layers: int,
    channels: int,
    label_count: int,
) -> nn.Module:
    from dscnn_kws.train_projected_bandpass import ProjectedBandpassDSCNN

    backbone = build_backbone(args, num_layers, channels, label_count)
    return ProjectedBandpassDSCNN(
        backbone=backbone,
        sample_rate=args.sample_rate,
        dct_coeff=args.dct_coeff,
        window_size_ms=args.window_size_ms,
        window_stride_ms=args.window_stride_ms,
        bandpass_internal_bands=args.bandpass_internal_bands,
        bandpass_f_min=args.bandpass_f_min,
        bandpass_f_max=args.bandpass_f_max,
        bandpass_spacing=args.bandpass_spacing,
        bandpass_kernel_size=args.bandpass_kernel_size,
        bandpass_phase_count=args.bandpass_phase_count,
        projection_init=args.projection_init,
        trainable_projection=args.trainable_projection,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        spec_aug=args.spec_aug,
        spec_aug_freq_mask_param=args.spec_aug_freq_mask_param,
        spec_aug_time_mask_param=args.spec_aug_time_mask_param,
        spec_aug_num_freq_masks=args.spec_aug_num_freq_masks,
        spec_aug_num_time_masks=args.spec_aug_num_time_masks,
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


def build_model(args: argparse.Namespace, *, num_layers: int, channels: int, label_count: int) -> nn.Module:
    if args.model_family == "standard":
        return build_standard_model(args, num_layers=num_layers, channels=channels, label_count=label_count)
    if args.model_family == "projected_bandpass":
        return build_projected_bandpass_model(args, num_layers=num_layers, channels=channels, label_count=label_count)
    raise ValueError(args.model_family)


def fuse_backbone_for_qat(backbone: DSCNN) -> None:
    backbone.train()
    for idx, layer in enumerate(backbone.conv_layers):
        if isinstance(layer, nn.Sequential):
            fuse_modules_qat(layer, ["0", "1", "2"], inplace=True)
        elif isinstance(layer, DepthwiseSeparableConv2d):
            fuse_modules_qat(layer, ["depthwise", "bn_depthwise"], inplace=True)
            fuse_modules_qat(layer, ["pointwise", "bn_pointwise"], inplace=True)
        else:
            raise TypeError(f"Unsupported DSCNN layer at conv_layers.{idx}: {type(layer)}")


def prepare_model_for_qat(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if args.quantize_frontend:
        model = FrontendInt8QATWrapper(model)
    if not hasattr(model, "backbone"):
        raise AttributeError("QAT script expects model.backbone")
    fuse_backbone_for_qat(model.backbone)
    model.backbone = QuantizedBackboneWrapper(model.backbone)
    model.qconfig = None
    model.backbone.qconfig = get_default_qat_qconfig(args.backend)
    model.train()
    prepare_qat(model, inplace=True)
    return model


def freeze_qat_bn_stats(module: nn.Module) -> None:
    if hasattr(module, "freeze_bn_stats"):
        module.freeze_bn_stats()


def choose_backend(requested_backend: str) -> str:
    supported = list(torch.backends.quantized.supported_engines)
    if requested_backend in supported:
        torch.backends.quantized.engine = requested_backend
        return requested_backend
    for backend in ("fbgemm", "x86", "qnnpack"):
        if backend in supported:
            print(f"[WARN] backend '{requested_backend}' unavailable; using '{backend}', supported={supported}")
            torch.backends.quantized.engine = backend
            return backend
    raise RuntimeError(f"No quantized backend is available. supported={supported}")


def count_usable_noise_files(noise_roots: list[str]) -> int:
    count = 0
    for raw_root in noise_roots:
        root = Path(raw_root)
        if root.is_file():
            base = root.parent
            for line in root.read_text(encoding="utf-8").splitlines():
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = base / path
                if path.suffix.lower() == ".wav" and path.exists() and path.stat().st_size > 44:
                    count += 1
        elif root.is_dir():
            count += sum(1 for p in root.rglob("*.wav") if p.stat().st_size > 44)
    return count


def build_loader(
    args: argparse.Namespace,
    *,
    dataset: str,
    split: str,
    is_training: bool,
    noise_roots: list[str],
    noise_prob: float,
    snr_min_db: float,
    snr_max_db: float,
    deterministic_noise: bool,
    random_seed: int,
) -> DataLoader:
    manifest = {
        "train": "train_manifest.json",
        "validation": "validation_manifest.json",
        "test": "test_manifest.json",
    }[split]
    data_path = Path(args.root) / dataset
    ds = SpeechCommandDataset(
        dataset_path=str(data_path),
        json_filename=str(data_path / manifest),
        is_training=is_training,
        class_list=CLASS_LIST,
        class_encoding=CLASS_ENCODING,
        sample_rate=args.sample_rate,
        noise_aug=True,
        noise_roots=noise_roots,
        noise_prob=noise_prob,
        noise_snr_min_db=snr_min_db,
        noise_snr_max_db=snr_max_db,
        deterministic_noise=deterministic_noise,
        random_seed=random_seed,
        allow_online_resample=True,
        strict_sample_rate=False,
    )
    workers = args.num_workers if split == "train" else max(0, args.num_workers // 2)
    return DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=is_training,
        drop_last=is_training,
        num_workers=workers,
        pin_memory=args.gpu > 0 and torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def build_train_loader(args: argparse.Namespace, dataset: str) -> DataLoader:
    return build_loader(
        args,
        dataset=dataset,
        split="train",
        is_training=True,
        noise_roots=args.train_noise_roots,
        noise_prob=args.train_noise_prob,
        snr_min_db=args.train_snr_min_db,
        snr_max_db=args.train_snr_max_db,
        deterministic_noise=False,
        random_seed=args.seed,
    )


def build_valid_loader(args: argparse.Namespace, dataset: str) -> DataLoader:
    return build_loader(
        args,
        dataset=dataset,
        split="validation",
        is_training=False,
        noise_roots=args.valid_noise_roots,
        noise_prob=args.valid_noise_prob,
        snr_min_db=args.valid_snr_db,
        snr_max_db=args.valid_snr_db,
        deterministic_noise=True,
        random_seed=args.seed + 100000,
    )


def build_test_list_loader(args: argparse.Namespace, dataset: str) -> DataLoader:
    return build_loader(
        args,
        dataset=dataset,
        split="test",
        is_training=False,
        noise_roots=args.test_noise_roots,
        noise_prob=args.valid_noise_prob,
        snr_min_db=args.valid_snr_db,
        snr_max_db=args.valid_snr_db,
        deterministic_noise=True,
        random_seed=args.seed + 200000,
    )


def build_scene_loader(args: argparse.Namespace, dataset: str, scene: str, snr_db: float, seed: int) -> DataLoader:
    return build_loader(
        args,
        dataset=dataset,
        split="test",
        is_training=False,
        noise_roots=[str(Path(args.scene_test_root) / scene)],
        noise_prob=1.0,
        snr_min_db=snr_db,
        snr_max_db=snr_db,
        deterministic_noise=True,
        random_seed=seed,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EpochMetrics:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    preds_all: list[int] = []
    labels_all: list[int] = []
    for waveform, labels in loader:
        waveform = waveform.to(device)
        labels = labels.to(device)
        logits = model(waveform)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item())
        total += int(labels.numel())
        correct += int((preds == labels).sum().item())
        preds_all.extend(preds.cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
    return EpochMetrics(
        loss=total_loss / max(1, len(loader)),
        acc=correct / max(1, total),
        precision=precision_score(labels_all, preds_all, average="macro", zero_division=0),
        recall=recall_score(labels_all, preds_all, average="macro", zero_division=0),
        f1=f1_score(labels_all, preds_all, average="macro", zero_division=0),
        num_samples=total,
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device) -> EpochMetrics:
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    preds_all: list[int] = []
    labels_all: list[int] = []
    for waveform, labels in tqdm(loader, desc="qat-train", leave=False):
        waveform = waveform.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(waveform)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits.detach(), dim=1)
        total_loss += float(loss.item())
        total += int(labels.numel())
        correct += int((preds == labels).sum().item())
        preds_all.extend(preds.cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
    return EpochMetrics(
        loss=total_loss / max(1, len(loader)),
        acc=correct / max(1, total),
        precision=precision_score(labels_all, preds_all, average="macro", zero_division=0),
        recall=recall_score(labels_all, preds_all, average="macro", zero_division=0),
        f1=f1_score(labels_all, preds_all, average="macro", zero_division=0),
        num_samples=total,
    )


def copy_state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_state_dict_to_model(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def quantize_qat_model(qat_model: nn.Module) -> nn.Module:
    quantized_model = copy.deepcopy(qat_model).cpu()
    quantized_model.eval()
    convert(quantized_model.backbone, inplace=True)
    quantized_model.eval()
    return quantized_model


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


def save_qat_artifacts(
    *,
    output_dir: Path,
    stem: str,
    source_checkpoint: Path,
    qat_model: nn.Module,
    quantized_model: nn.Module,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qat_path = output_dir / f"{stem}_qat_prepared_best.pt"
    quantized_path = output_dir / f"{stem}_qat_int8_quantized.pt"
    common = {
        "source_checkpoint": str(source_checkpoint),
        "metadata": row,
        "qat": {
            "backend": args.backend,
            "epochs": args.qat_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "freeze_bn_after_epoch": args.freeze_bn_after_epoch,
            "disable_observer_after_epoch": args.disable_observer_after_epoch,
            "quantize_frontend": args.quantize_frontend,
            "scope": (
                "frontend INT8 fake-quant QAT + converted INT8 DSCNN backbone"
                if args.quantize_frontend
                else "converted INT8 DSCNN backbone; frontend remains float"
            ),
        },
    }
    torch.save({**common, "state_dict": qat_model.state_dict(), "format": "prepared_qat_float"}, qat_path)
    torch.save({**common, "state_dict": quantized_model.state_dict(), "format": "converted_int8"}, quantized_path)
    return qat_path, quantized_path


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
            }
            rows.append(row)
            print(f"[GRID] {dataset} | {arch} | {scene:<18} | snr={snr_db:>5} | acc={metrics.acc:.4f} | f1={metrics.f1:.4f}")
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

    arch = f"L{layers}_C{channels}"
    param_count = expected_params(layers, channels, label_count)
    print("\n" + "=" * 100)
    print(f"[QAT] dataset={dataset}, arch={arch}, ckpt={path}")

    model = build_model(args, num_layers=layers, channels=channels, label_count=label_count)
    load_model_weights(model, state)
    model = prepare_model_for_qat(model, args).to(device)

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

    quantized_model = quantize_qat_model(model)
    quantized_test_m = evaluate(quantized_model, test_list_loader, torch.device("cpu"))

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
        "quantized_test_tau_list_acc": quantized_test_m.acc,
        "quantized_test_tau_list_f1": quantized_test_m.f1,
        "backend": args.backend,
        "model_family": args.model_family,
        "frontend": args.frontend if args.model_family == "standard" else "projected_bandpass",
        "quantize_frontend": args.quantize_frontend,
    }
    qat_path, quantized_path = save_qat_artifacts(
        output_dir=Path(args.output_dir),
        stem=stem,
        source_checkpoint=path,
        qat_model=model.cpu(),
        quantized_model=quantized_model,
        row=artifact_row,
        args=args,
    )
    artifact_row["qat_prepared_checkpoint"] = str(qat_path.resolve())
    artifact_row["quantized_checkpoint"] = str(quantized_path.resolve())
    artifact_row["qat_prepared_size_bytes"] = qat_path.stat().st_size
    artifact_row["quantized_size_bytes"] = quantized_path.stat().st_size

    grid_rows = run_scene_grid(
        args,
        model=quantized_model,
        dataset=dataset,
        arch=arch,
        layers=layers,
        channels=channels,
        expected_param_count=param_count,
        source_checkpoint=path,
        quantized_checkpoint=quantized_path,
    )
    for row in grid_rows:
        row.update(
            {
                "best_epoch": best_epoch,
                "best_valid_acc": best_acc,
                "backend": args.backend,
                "model_family": args.model_family,
                "frontend": args.frontend if args.model_family == "standard" else "projected_bandpass",
                "quantize_frontend": args.quantize_frontend,
            }
        )
    return artifact_row, grid_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INT8 QAT fine-tuning for noise-scene DSCNN checkpoints.")
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
    parser.add_argument(
        "--quantize_frontend",
        action=argparse.BooleanOptionalAction,
        default=QUANTIZE_FRONTEND,
        help=(
            "Apply INT8 fake-quant QAT to frontend tensors: waveform, pre-emphasis output, "
            "frontend output, and flattened backbone input. The DSCNN backbone is still "
            "converted to PyTorch INT8 quantized ops."
        ),
    )

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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
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
    print(f"[INFO] device={device}, backend={args.backend}, model_family={args.model_family}")

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
    print(f"[DONE] qat_models={len(train_rows)}, grid_rows={len(grid_rows)}")


if __name__ == "__main__":
    main()
