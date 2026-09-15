from __future__ import annotations

import argparse
import copy
import csv
import json
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
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

try:
    from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, disable_observer
except ImportError:
    from torch.quantization import FakeQuantize, MovingAverageMinMaxObserver, disable_observer

from dscnn_kws.configs import CLASS_LIST
from dscnn_kws.frontend.int8_streaming_mfcc_frontend import Int8StreamingMFCCFrontend
from dscnn_kws.quantization.qat_int8_noise_snr_scene import (
    TAU_SCENES,
    build_loader,
    choose_backend,
    count_usable_noise_files,
)
from dscnn_kws.streaming.streaming_crnn import StreamingKWSModel, parse_cnn_channels
from dscnn_kws.utils import apply_pre_emphasis


# =============================================================================
# Code-level configuration
# =============================================================================
# Edit this block on the Linux server, or override values with CLI flags:
#   python dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_noise_snr_scene.py

INPUT_DIR = "./dscnn_kws/runs/streaming_kt7/streaming_crnn_noise_snr_scene_sweep_kt7_best_models"
PATTERN = "*.pt"
CHECKPOINTS = None
OUTPUT_DIR = "./dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_models"
TRAIN_RESULTS_CSV = "./dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_train_results.csv"
GRID_RESULTS_CSV = "./dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_grid_results.csv"
LIMIT = 0

ROOT = "/root/kws/dscnn_kws/dscnn_kws/data"
DATASET = None
BATCH = 256
NUM_WORKERS = 8
GPU = 1
SEED = 42

QAT_EPOCHS = 10
LR = 5e-5
WEIGHT_DECAY = 1e-6
FREEZE_BN_AFTER_EPOCH = 3
DISABLE_OBSERVER_AFTER_EPOCH = 5
BACKEND = "fbgemm"

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

SAMPLE_RATE = 16000
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
CNN_CHANNELS = None
KERNEL_TIME = None
KERNEL_FREQ = None
GRU_HIDDEN = None
GRU_LAYERS = None
DROPOUT = 0.2
PRE_EMPHASIS = True
PRE_EMPHASIS_COEFF = 0.97
MFCC_CENTER = False
STREAMING_MFCC = True
MEL_FILTER_SHAPE = "triangular"

# QAT and final INT8 frontend both use natural-log PWL MFCC to match the
# hardware-friendly StreamingMFCC path.
LOG_APPROX_MODE = "pwl"
LOG_PWL_NUM_SEGMENTS = 8
LOG_PWL_STRATEGY = "uniform_logx"
LOG_PWL_GAMMA = 1.0
LOG_OFFSET = 1e-6
LOG_INPUT_CLAMP_MIN = 1e-12

INT8_MFCC_LOG_APPROX_MODE = "pwl"
INT8_MFCC_LOG_PWL_NUM_SEGMENTS = 8
INT8_MFCC_LOG_PWL_STRATEGY = "uniform_logx"
INT8_MFCC_LOG_PWL_GAMMA = 1.0
INT8_MFCC_LOG_OFFSET = 1e-6
INT8_MFCC_LOG_INPUT_CLAMP_MIN = 1e-12
INT8_MFCC_COEFF_BITS = 8
INT8_MFCC_REQUANTIZE_POWER = False
INT8_MFCC_REQUANTIZE_MEL = False

CALIBRATION_SPLIT = "validation"
CALIBRATION_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_valid.txt"]
CALIBRATION_NOISE_PROB = 1.0
CALIBRATION_SNR_MIN_DB = -5.0
CALIBRATION_SNR_MAX_DB = 20.0
CALIBRATION_BATCHES = 0


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    precision: float
    recall: float
    f1: float
    num_samples: int


ARCH_RE = re.compile(r"(?P<arch>C[A-Za-z0-9_]+_H(?P<hidden>\d+))", re.IGNORECASE)
CHANNELS_RE = re.compile(r"_ch(?P<channels>[0-9-]+)_gru(?P<hidden>\d+)", re.IGNORECASE)
DATASET_RE = re.compile(r"(?P<dataset>.+?)_C[A-Za-z0-9_]+_H\d+_ch", re.IGNORECASE)


def make_int8_fake_quant() -> FakeQuantize:
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )()


def _sanitize_key(name: str) -> str:
    return name.replace(".", "__")


class QATStreamingCRNNBackbone(nn.Module):
    """CRNN backbone with explicit INT8 fake-quant for Conv, GRU, and FC."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        block_count = len(backbone.cnn)
        self.block_input_fq = nn.ModuleList(make_int8_fake_quant() for _ in range(block_count))
        self.dw_weight_fq = nn.ModuleList(make_int8_fake_quant() for _ in range(block_count))
        self.dw_output_fq = nn.ModuleList(make_int8_fake_quant() for _ in range(block_count))
        self.pw_weight_fq = nn.ModuleList(make_int8_fake_quant() for _ in range(block_count))
        self.pw_output_fq = nn.ModuleList(make_int8_fake_quant() for _ in range(block_count))
        self.pool_output_fq = make_int8_fake_quant()
        self.gru_input_fq = make_int8_fake_quant()
        self.gru_gate_fq = make_int8_fake_quant()
        self.gru_hidden_fq = make_int8_fake_quant()
        self.fc_input_fq = make_int8_fake_quant()
        self.fc_weight_fq = make_int8_fake_quant()
        self.fc_output_fq = make_int8_fake_quant()
        self.gru_weight_fq = nn.ModuleDict(
            {
                _sanitize_key(name): make_int8_fake_quant()
                for name, _param in backbone.gru.named_parameters()
                if "weight" in name
            }
        )

    def _conv_block(self, block: nn.Module, idx: int, x: torch.Tensor) -> torch.Tensor:
        x = self.block_input_fq[idx](x)
        x = block._pad_sequence(x)
        dw_w = self.dw_weight_fq[idx](block.depthwise.weight)
        x = F.conv2d(
            x,
            dw_w,
            bias=block.depthwise.bias,
            stride=block.depthwise.stride,
            padding=block.depthwise.padding,
            dilation=block.depthwise.dilation,
            groups=block.depthwise.groups,
        )
        x = F.relu(block.bn_depthwise(x))
        x = self.dw_output_fq[idx](x)
        pw_w = self.pw_weight_fq[idx](block.pointwise.weight)
        x = F.conv2d(x, pw_w, bias=block.pointwise.bias, stride=1, padding=0)
        x = F.relu(block.bn_pointwise(x))
        return self.pw_output_fq[idx](x)

    def _gru_weight(self, name: str, value: torch.Tensor) -> torch.Tensor:
        key = _sanitize_key(name)
        if key not in self.gru_weight_fq:
            return value
        return self.gru_weight_fq[key](value)

    def _gru_forward(self, sequence: torch.Tensor) -> torch.Tensor:
        gru = self.backbone.gru
        current = self.gru_input_fq(sequence)
        batch_size = current.size(0)
        for layer in range(gru.num_layers):
            hx = current.new_zeros(batch_size, gru.hidden_size)
            w_ih = self._gru_weight(f"weight_ih_l{layer}", getattr(gru, f"weight_ih_l{layer}"))
            w_hh = self._gru_weight(f"weight_hh_l{layer}", getattr(gru, f"weight_hh_l{layer}"))
            b_ih = getattr(gru, f"bias_ih_l{layer}", None)
            b_hh = getattr(gru, f"bias_hh_l{layer}", None)
            outputs: list[torch.Tensor] = []
            for t in range(current.size(1)):
                x_t = current[:, t, :]
                gi = F.linear(x_t, w_ih, b_ih)
                gh = F.linear(hx, w_hh, b_hh)
                i_r, i_z, i_n = gi.chunk(3, dim=1)
                h_r, h_z, h_n = gh.chunk(3, dim=1)
                r = torch.sigmoid(self.gru_gate_fq(i_r + h_r))
                z = torch.sigmoid(self.gru_gate_fq(i_z + h_z))
                n = torch.tanh(self.gru_gate_fq(i_n + r * h_n))
                hx = (1.0 - z) * n + z * hx
                hx = self.gru_hidden_fq(hx)
                outputs.append(hx)
            current = torch.stack(outputs, dim=1)
            if layer + 1 < gru.num_layers and gru.dropout > 0 and self.training:
                current = F.dropout(current, p=gru.dropout, training=True)
        return current

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"Expected features [B, T, F], got {tuple(features.shape)}")
        x = features.unsqueeze(1)
        for idx, block in enumerate(self.backbone.cnn):
            x = self._conv_block(block, idx, x)
        sequence = self.pool_output_fq(x.mean(dim=3).transpose(1, 2))
        output = self._gru_forward(sequence)
        last = self.fc_input_fq(output[:, -1, :])
        last = self.backbone.dropout(last)
        logits = F.linear(last, self.fc_weight_fq(self.backbone.fc.weight), self.backbone.fc.bias)
        return self.fc_output_fq(logits)


class QATStreamingMFCCCRNNModel(nn.Module):
    """StreamingMFCC + CRNN with INT8 fake-quant points from waveform to logits."""

    def __init__(self, model: StreamingKWSModel):
        super().__init__()
        self.feature_extractor = model.feature_extractor
        self.dct_coeff = int(model.dct_coeff)
        self.pre_emphasis = bool(model.pre_emphasis)
        self.pre_emphasis_coeff = float(model.pre_emphasis_coeff)
        self.waveform_fq = make_int8_fake_quant()
        self.pre_emphasis_fq = make_int8_fake_quant()
        self.frontend_output_fq = make_int8_fake_quant()
        self.backbone_input_fq = make_int8_fake_quant()
        self.qat_backbone = QATStreamingCRNNBackbone(model.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.waveform_fq(x)
        if self.pre_emphasis:
            x = apply_pre_emphasis(x, self.pre_emphasis_coeff)
            x = self.pre_emphasis_fq(x)
        features = self.feature_extractor(x)
        features = self.frontend_output_fq(features)
        features = features[:, : self.dct_coeff, :].transpose(1, 2)
        features = self.backbone_input_fq(features)
        return self.qat_backbone(features)


def _activation_scale(x: torch.Tensor, eps: float = 1e-12) -> float:
    return max(float(x.detach().abs().max().cpu().item()), eps) / 127.0


def _quant_dequant(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.clamp(torch.round(x / scale), -128, 127) * scale


def _quantize_int8_tensor(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = _activation_scale(x)
    q = torch.clamp(torch.round(x.detach().cpu() / scale), -128, 127).to(torch.int8)
    return q, scale


class Int8ReferenceCRNNBackbone(nn.Module):
    """INT8 reference CRNN backbone with calibrated activation scales.

    Weight tensors are quantized to int8 every forward pass and immediately
    dequantized for PyTorch functional kernels. This keeps the numerics tied to
    INT8 grids while staying runnable on CPU for scene-grid evaluation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        scales: dict[str, float] | None = None,
        observer_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(backbone).cpu().eval()
        self.scales = dict(scales or {})
        self.observer_enabled = bool(observer_enabled)

    @classmethod
    def from_scale_json(cls, backbone: nn.Module, path: str | Path) -> "Int8ReferenceCRNNBackbone":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(backbone=backbone, scales=payload["activation_scales"], observer_enabled=False)

    def export_scale_json(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "backbone": "Int8ReferenceCRNNBackbone",
            "activation_scales": self.scales,
        }
        if extra:
            payload["extra"] = extra
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def export_int8_weight_state(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, value in self.backbone.state_dict().items():
            if not torch.is_floating_point(value) or "weight" not in name:
                continue
            q, scale = _quantize_int8_tensor(value)
            out[name] = {"int8": q, "scale": scale, "shape": list(value.shape)}
        return out

    def _qdq(self, x: torch.Tensor, stage: str) -> torch.Tensor:
        if self.observer_enabled or stage not in self.scales:
            scale = _activation_scale(x)
            self.scales[stage] = max(self.scales.get(stage, 0.0), scale)
        return _quant_dequant(x, self.scales[stage])

    def _qdq_weight(self, x: torch.Tensor, stage: str) -> torch.Tensor:
        scale = _activation_scale(x)
        self.scales[f"weight.{stage}"] = scale
        return _quant_dequant(x, scale)

    def _conv_block(self, block: nn.Module, idx: int, x: torch.Tensor) -> torch.Tensor:
        x = self._qdq(x, f"cnn.{idx}.input")
        x = block._pad_sequence(x)
        dw_w = self._qdq_weight(block.depthwise.weight, f"cnn.{idx}.depthwise")
        x = F.conv2d(
            x,
            dw_w,
            bias=block.depthwise.bias,
            stride=block.depthwise.stride,
            padding=block.depthwise.padding,
            dilation=block.depthwise.dilation,
            groups=block.depthwise.groups,
        )
        x = self._qdq(F.relu(block.bn_depthwise(x)), f"cnn.{idx}.depthwise_relu")
        pw_w = self._qdq_weight(block.pointwise.weight, f"cnn.{idx}.pointwise")
        x = F.conv2d(x, pw_w, bias=block.pointwise.bias, stride=1, padding=0)
        return self._qdq(F.relu(block.bn_pointwise(x)), f"cnn.{idx}.pointwise_relu")

    def _gru_forward(self, sequence: torch.Tensor) -> torch.Tensor:
        gru = self.backbone.gru
        current = self._qdq(sequence, "gru.input")
        batch_size = current.size(0)
        for layer in range(gru.num_layers):
            hx = current.new_zeros(batch_size, gru.hidden_size)
            w_ih = self._qdq_weight(getattr(gru, f"weight_ih_l{layer}"), f"gru.weight_ih_l{layer}")
            w_hh = self._qdq_weight(getattr(gru, f"weight_hh_l{layer}"), f"gru.weight_hh_l{layer}")
            b_ih = getattr(gru, f"bias_ih_l{layer}", None)
            b_hh = getattr(gru, f"bias_hh_l{layer}", None)
            outputs: list[torch.Tensor] = []
            for t in range(current.size(1)):
                x_t = self._qdq(current[:, t, :], f"gru.layer{layer}.x")
                gi = F.linear(x_t, w_ih, b_ih)
                gh = F.linear(hx, w_hh, b_hh)
                i_r, i_z, i_n = gi.chunk(3, dim=1)
                h_r, h_z, h_n = gh.chunk(3, dim=1)
                r = torch.sigmoid(self._qdq(i_r + h_r, f"gru.layer{layer}.reset_gate"))
                z = torch.sigmoid(self._qdq(i_z + h_z, f"gru.layer{layer}.update_gate"))
                n = torch.tanh(self._qdq(i_n + r * h_n, f"gru.layer{layer}.new_gate"))
                hx = self._qdq((1.0 - z) * n + z * hx, f"gru.layer{layer}.hidden")
                outputs.append(hx)
            current = torch.stack(outputs, dim=1)
        return current

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self._qdq(features, "backbone.input").unsqueeze(1)
        for idx, block in enumerate(self.backbone.cnn):
            x = self._conv_block(block, idx, x)
        sequence = self._qdq(x.mean(dim=3).transpose(1, 2), "cnn.freq_pool")
        output = self._gru_forward(sequence)
        last = self._qdq(output[:, -1, :], "fc.input")
        fc_w = self._qdq_weight(self.backbone.fc.weight, "fc.weight")
        logits = F.linear(last, fc_w, self.backbone.fc.bias)
        return self._qdq(logits, "logits")


class Int8StreamingMFCCInt8CRNNModel(nn.Module):
    def __init__(self, frontend: Int8StreamingMFCCFrontend, backbone: Int8ReferenceCRNNBackbone, dct_coeff: int):
        super().__init__()
        self.feature_extractor = frontend
        self.backbone = backbone
        self.dct_coeff = int(dct_coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = features[:, : self.dct_coeff, :].transpose(1, 2)
        return self.backbone(features)


def infer_dataset_from_name(path: Path) -> str | None:
    match = DATASET_RE.search(path.name)
    return match.group("dataset") if match else None


def infer_arch_from_name(path: Path) -> tuple[str | None, str | None, int | None]:
    channel_match = CHANNELS_RE.search(path.name)
    if channel_match:
        channels = channel_match.group("channels").replace("-", ",")
        return None, channels, int(channel_match.group("hidden"))
    arch_match = ARCH_RE.search(path.name)
    if arch_match:
        return arch_match.group("arch"), None, int(arch_match.group("hidden"))
    return None, None, None


def infer_model_shape_from_state_dict(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    channels: list[int] = []
    idx = 0
    while f"backbone.cnn.{idx}.pointwise.weight" in state:
        channels.append(int(state[f"backbone.cnn.{idx}.pointwise.weight"].shape[0]))
        idx += 1
    if not channels:
        raise ValueError("Cannot infer CRNN cnn_channels from state_dict")

    depthwise0 = state.get("backbone.cnn.0.depthwise.weight")
    if depthwise0 is None:
        raise ValueError("Cannot infer CRNN kernel size from state_dict")
    kernel_time = int(depthwise0.shape[2])
    kernel_freq = int(depthwise0.shape[3])

    weight_hh = state.get("backbone.gru.weight_hh_l0")
    if weight_hh is None:
        raise ValueError("Cannot infer GRU hidden size from state_dict")
    gru_hidden = int(weight_hh.shape[1])

    gru_layers = 0
    while f"backbone.gru.weight_ih_l{gru_layers}" in state:
        gru_layers += 1

    fc_weight = state.get("backbone.fc.weight")
    label_count = int(fc_weight.shape[0]) if fc_weight is not None else len(CLASS_LIST)
    return {
        "cnn_channels": tuple(channels),
        "kernel_time": kernel_time,
        "kernel_freq": kernel_freq,
        "gru_hidden": gru_hidden,
        "gru_layers": max(1, gru_layers),
        "label_count": label_count,
    }


def safe_stem(path: Path) -> str:
    raw = path.parent.name if path.name == "best.pt" else path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


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


def build_source_model(args: argparse.Namespace, shape: dict[str, Any]) -> StreamingKWSModel:
    cnn_channels = parse_cnn_channels(args.cnn_channels) if args.cnn_channels else shape["cnn_channels"]
    return StreamingKWSModel(
        sample_rate=args.sample_rate,
        label_count=shape["label_count"],
        frontend="mfcc",
        dct_coeff=args.dct_coeff,
        window_size_ms=args.window_size_ms,
        window_stride_ms=args.window_stride_ms,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        cnn_channels=tuple(cnn_channels),
        kernel_time=args.kernel_time or shape["kernel_time"],
        kernel_freq=args.kernel_freq or shape["kernel_freq"],
        gru_hidden=args.gru_hidden or shape["gru_hidden"],
        gru_layers=args.gru_layers or shape["gru_layers"],
        dropout=args.dropout,
        mfcc_center=args.mfcc_center,
        streaming_mfcc=args.streaming_mfcc,
        mel_filter_shape=args.mel_filter_shape,
        log_approx_mode=args.log_approx_mode,
        log_pwl_num_segments=args.log_pwl_num_segments,
        log_pwl_strategy=args.log_pwl_strategy,
        log_pwl_gamma=args.log_pwl_gamma,
        log_offset=args.log_offset,
        log_input_clamp_min=args.log_input_clamp_min,
    )


def load_source_weights(model: StreamingKWSModel, state: dict[str, torch.Tensor]) -> str:
    try:
        model.load_state_dict(state, strict=True)
        return "full_model"
    except RuntimeError as exc:
        backbone_state = {
            key[len("backbone.") :]: value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        if not backbone_state:
            raise exc
        model.backbone.load_state_dict(backbone_state, strict=True)
        print("[WARN] full model state_dict did not match; loaded CRNN backbone weights only.")
        return "backbone_only_frontend_rebuilt"


def build_int8_streaming_mfcc_frontend(args: argparse.Namespace, observer_enabled: bool) -> Int8StreamingMFCCFrontend:
    n_fft = int(args.sample_rate * args.window_size_ms / 1000)
    hop_length = int(args.sample_rate * args.window_stride_ms / 1000)
    return Int8StreamingMFCCFrontend(
        sample_rate=args.sample_rate,
        n_mfcc=40,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=40,
        f_min=20.0,
        f_max=float(args.sample_rate / 2),
        dct_norm="ortho",
        mel_filter_shape=args.mel_filter_shape,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        mel_log_mode="natural_log",
        log_offset=args.int8_mfcc_log_offset,
        log_approx_mode=args.int8_mfcc_log_approx_mode,
        log_pwl_num_segments=args.int8_mfcc_log_pwl_num_segments,
        log_pwl_strategy=args.int8_mfcc_log_pwl_strategy,
        log_pwl_gamma=args.int8_mfcc_log_pwl_gamma,
        log_input_clamp_min=args.int8_mfcc_log_input_clamp_min,
        coeff_bits=args.int8_mfcc_coeff_bits,
        requantize_power=args.int8_mfcc_requantize_power,
        requantize_mel=args.int8_mfcc_requantize_mel,
        flush_tail=True,
        observer_enabled=observer_enabled,
    )


def build_train_loader(args: argparse.Namespace, dataset: str):
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


def build_valid_loader(args: argparse.Namespace, dataset: str):
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


def build_test_list_loader(args: argparse.Namespace, dataset: str):
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


def build_calibration_loader(args: argparse.Namespace, dataset: str):
    return build_loader(
        args,
        dataset=dataset,
        split=args.calibration_split,
        is_training=False,
        noise_roots=args.calibration_noise_roots,
        noise_prob=args.calibration_noise_prob,
        snr_min_db=args.calibration_snr_min_db,
        snr_max_db=args.calibration_snr_max_db,
        deterministic_noise=True,
        random_seed=args.seed + 300000,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> EpochMetrics:
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


def train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device) -> EpochMetrics:
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    preds_all: list[int] = []
    labels_all: list[int] = []
    for waveform, labels in tqdm(loader, desc="qat-crnn-train", leave=False):
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


@torch.no_grad()
def calibrate_int8_model(
    args: argparse.Namespace,
    *,
    dataset: str,
    float_backbone: nn.Module,
    frontend_scale_json: Path,
    backbone_scale_json: Path,
) -> tuple[Int8StreamingMFCCFrontend, Int8ReferenceCRNNBackbone, int]:
    frontend = build_int8_streaming_mfcc_frontend(args, observer_enabled=True).cpu().eval()
    backbone = Int8ReferenceCRNNBackbone(float_backbone, observer_enabled=True).cpu().eval()
    model = Int8StreamingMFCCInt8CRNNModel(frontend, backbone, args.dct_coeff).cpu().eval()
    loader = build_calibration_loader(args, dataset)
    batches = 0
    for waveform, _labels in tqdm(loader, desc="calibrate-int8-streaming-crnn", leave=False):
        model(waveform.cpu())
        batches += 1
        if args.calibration_batches > 0 and batches >= args.calibration_batches:
            break

    frontend_scale_json.parent.mkdir(parents=True, exist_ok=True)
    frontend.export_scale_json(
        frontend_scale_json,
        extra={
            "dataset": dataset,
            "calibration_split": args.calibration_split,
            "calibration_noise_roots": args.calibration_noise_roots,
            "calibration_noise_prob": args.calibration_noise_prob,
            "calibration_snr_min_db": args.calibration_snr_min_db,
            "calibration_snr_max_db": args.calibration_snr_max_db,
            "calibration_batches": batches,
        },
    )
    backbone.export_scale_json(
        backbone_scale_json,
        extra={"dataset": dataset, "calibration_batches": batches},
    )
    frontend = Int8StreamingMFCCFrontend.from_scale_json(frontend_scale_json).cpu().eval()
    backbone = Int8ReferenceCRNNBackbone.from_scale_json(float_backbone, backbone_scale_json).cpu().eval()
    return frontend, backbone, batches


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


def run_scene_grid(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    dataset: str,
    arch: str,
    source_checkpoint: Path,
    inference_checkpoint: Path,
    frontend_scale_json: Path,
    backbone_scale_json: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = torch.device("cpu")
    for scene_idx, scene in enumerate(args.scene_names):
        scene_root = Path(args.scene_test_root) / scene
        usable = count_usable_noise_files([str(scene_root)])
        if usable <= 0:
            print(f"[WARN] scene skipped, no usable wavs: {scene_root}")
            continue
        for snr_idx, snr_db in enumerate(args.test_snrs):
            seed = 500000 + scene_idx * 10007 + snr_idx * 101
            loader = build_loader(
                args,
                dataset=dataset,
                split="test",
                is_training=False,
                noise_roots=[str(scene_root)],
                noise_prob=1.0,
                snr_min_db=snr_db,
                snr_max_db=snr_db,
                deterministic_noise=True,
                random_seed=seed,
            )
            metrics = evaluate(model, loader, device)
            row = {
                "dataset": dataset,
                "arch": arch,
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
                "inference_checkpoint": str(inference_checkpoint),
                "frontend_scale_json": str(frontend_scale_json),
                "backbone_scale_json": str(backbone_scale_json),
            }
            rows.append(row)
            print(f"[GRID] {dataset} | {arch} | {scene:<18} | snr={snr_db:>5} | acc={metrics.acc:.4f} | f1={metrics.f1:.4f}")
    return rows


def save_artifacts(
    *,
    output_dir: Path,
    stem: str,
    source_checkpoint: Path,
    qat_model: QATStreamingMFCCCRNNModel,
    inference_model: Int8StreamingMFCCInt8CRNNModel,
    frontend_scale_json: Path,
    backbone_scale_json: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qat_path = output_dir / f"{stem}_streaming_crnn_qat_prepared_best.pt"
    inference_path = output_dir / f"{stem}_int8_streaming_mfcc_int8_crnn.pt"
    common = {
        "source_checkpoint": str(source_checkpoint),
        "metadata": row,
        "qat": {
            "format": "explicit fake-quant int8 StreamingMFCC + CRNN",
            "epochs": args.qat_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "freeze_bn_after_epoch": args.freeze_bn_after_epoch,
            "disable_observer_after_epoch": args.disable_observer_after_epoch,
            "backend": args.backend,
        },
        "frontend_scale_json": str(frontend_scale_json),
        "backbone_scale_json": str(backbone_scale_json),
    }
    torch.save({**common, "state_dict": qat_model.state_dict(), "format": "prepared_qat_float_fakequant"}, qat_path)
    torch.save(
        {
            **common,
            "state_dict": inference_model.state_dict(),
            "int8_weight_state": inference_model.backbone.export_int8_weight_state(),
            "format": "int8_streaming_mfcc_plus_int8_reference_crnn",
        },
        inference_path,
    )
    return qat_path, inference_path


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
    shape = infer_model_shape_from_state_dict(state)
    _arch_name, named_channels, named_hidden = infer_arch_from_name(path)
    if args.cnn_channels is None and named_channels is not None:
        shape["cnn_channels"] = tuple(parse_cnn_channels(named_channels))
    if args.gru_hidden is None and named_hidden is not None:
        shape["gru_hidden"] = named_hidden
    dataset = args.dataset or infer_dataset_from_name(path)
    if not dataset:
        raise ValueError(f"Cannot infer dataset from {path}; pass --dataset.")

    arch = f"C{'-'.join(str(c) for c in shape['cnn_channels'])}_H{shape['gru_hidden']}"
    print("\n" + "=" * 100)
    print(f"[QAT-INT8-STREAMING-CRNN] dataset={dataset}, arch={arch}, ckpt={path}")

    source_model = build_source_model(args, shape)
    load_scope = load_source_weights(source_model, state)
    qat_model = QATStreamingMFCCCRNNModel(source_model).to(device)

    train_loader = build_train_loader(args, dataset)
    valid_loader = build_valid_loader(args, dataset)
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.qat_epochs),
        eta_min=args.lr * 0.05,
    )

    best_acc = -1.0
    best_epoch = 0
    best_state = copy_state_dict_cpu(qat_model)
    train_loss = train_acc = valid_loss = valid_acc = None
    for epoch in range(1, args.qat_epochs + 1):
        if args.freeze_bn_after_epoch > 0 and epoch == args.freeze_bn_after_epoch + 1:
            for module in qat_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            print(f"[INFO] froze BN stats after epoch {args.freeze_bn_after_epoch}")
        if args.disable_observer_after_epoch > 0 and epoch == args.disable_observer_after_epoch + 1:
            qat_model.apply(disable_observer)
            print(f"[INFO] disabled fake-quant observers after epoch {args.disable_observer_after_epoch}")

        train_m = train_one_epoch(qat_model, train_loader, optimizer, device)
        valid_m = evaluate(qat_model, valid_loader, device)
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
            best_state = copy_state_dict_cpu(qat_model)

    qat_model.load_state_dict(best_state, strict=True)
    qat_model.eval()
    test_loader = build_test_list_loader(args, dataset)
    qat_test_m = evaluate(qat_model, test_loader, device)

    stem = safe_stem(path)
    output_dir = Path(args.output_dir)
    frontend_scale_json = output_dir / f"{stem}_int8_streaming_mfcc_scales.json"
    backbone_scale_json = output_dir / f"{stem}_int8_crnn_activation_scales.json"
    trained_backbone = qat_model.qat_backbone.backbone.cpu().eval()
    int8_frontend, int8_backbone, calibration_batches = calibrate_int8_model(
        args,
        dataset=dataset,
        float_backbone=trained_backbone,
        frontend_scale_json=frontend_scale_json,
        backbone_scale_json=backbone_scale_json,
    )
    inference_model = Int8StreamingMFCCInt8CRNNModel(
        frontend=int8_frontend,
        backbone=int8_backbone,
        dct_coeff=args.dct_coeff,
    ).cpu().eval()
    int8_test_m = evaluate(inference_model, test_loader, torch.device("cpu"))
    print(
        "[TEST] "
        f"qat_tau_list_acc={qat_test_m.acc:.4f} f1={qat_test_m.f1:.4f} | "
        f"int8_streaming_mfcc_int8_crnn_tau_list_acc={int8_test_m.acc:.4f} f1={int8_test_m.f1:.4f}"
    )

    train_row = {
        "dataset": dataset,
        "arch": arch,
        "cnn_channels": ",".join(str(c) for c in shape["cnn_channels"]),
        "kernel_time": args.kernel_time or shape["kernel_time"],
        "kernel_freq": args.kernel_freq or shape["kernel_freq"],
        "gru_hidden": args.gru_hidden or shape["gru_hidden"],
        "gru_layers": args.gru_layers or shape["gru_layers"],
        "source_checkpoint": str(path),
        "source_load_scope": load_scope,
        "best_epoch": best_epoch,
        "best_valid_acc": best_acc,
        "last_train_loss": train_loss,
        "last_train_acc": train_acc,
        "last_valid_loss": valid_loss,
        "last_valid_acc": valid_acc,
        "qat_test_tau_list_acc": qat_test_m.acc,
        "qat_test_tau_list_f1": qat_test_m.f1,
        "int8_test_tau_list_acc": int8_test_m.acc,
        "int8_test_tau_list_f1": int8_test_m.f1,
        "frontend": "int8_streaming_mfcc",
        "backbone": "int8_reference_crnn",
        "calibration_batches": calibration_batches,
        "frontend_scale_json": str(frontend_scale_json.resolve()),
        "backbone_scale_json": str(backbone_scale_json.resolve()),
    }
    qat_path, inference_path = save_artifacts(
        output_dir=output_dir,
        stem=stem,
        source_checkpoint=path,
        qat_model=qat_model.cpu(),
        inference_model=inference_model,
        frontend_scale_json=frontend_scale_json,
        backbone_scale_json=backbone_scale_json,
        row=train_row,
        args=args,
    )
    train_row["qat_checkpoint"] = str(qat_path.resolve())
    train_row["inference_checkpoint"] = str(inference_path.resolve())
    train_row["qat_checkpoint_size_bytes"] = qat_path.stat().st_size
    train_row["inference_checkpoint_size_bytes"] = inference_path.stat().st_size

    grid_rows = run_scene_grid(
        args,
        model=inference_model,
        dataset=dataset,
        arch=arch,
        source_checkpoint=path,
        inference_checkpoint=inference_path,
        frontend_scale_json=frontend_scale_json,
        backbone_scale_json=backbone_scale_json,
    )
    for row in grid_rows:
        row.update(
            {
                "cnn_channels": train_row["cnn_channels"],
                "kernel_time": train_row["kernel_time"],
                "kernel_freq": train_row["kernel_freq"],
                "gru_hidden": train_row["gru_hidden"],
                "best_epoch": best_epoch,
                "best_valid_acc": best_acc,
                "calibration_batches": calibration_batches,
            }
        )
    return train_row, grid_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT INT8 StreamingMFCC + CRNN, then evaluate TAU scene x SNR grid.")
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
    parser.add_argument("--freeze_bn_after_epoch", type=int, default=FREEZE_BN_AFTER_EPOCH)
    parser.add_argument("--disable_observer_after_epoch", type=int, default=DISABLE_OBSERVER_AFTER_EPOCH)
    parser.add_argument("--backend", default=BACKEND)
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
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--dct_coeff", type=int, default=DCT_COEFF)
    parser.add_argument("--window_size_ms", type=int, default=WINDOW_SIZE_MS)
    parser.add_argument("--window_stride_ms", type=int, default=WINDOW_STRIDE_MS)
    parser.add_argument("--cnn_channels", default=CNN_CHANNELS)
    parser.add_argument("--kernel_time", type=int, default=KERNEL_TIME)
    parser.add_argument("--kernel_freq", type=int, default=KERNEL_FREQ)
    parser.add_argument("--gru_hidden", type=int, default=GRU_HIDDEN)
    parser.add_argument("--gru_layers", type=int, default=GRU_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=PRE_EMPHASIS)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=PRE_EMPHASIS_COEFF)
    parser.add_argument("--mfcc_center", action=argparse.BooleanOptionalAction, default=MFCC_CENTER)
    parser.add_argument("--streaming_mfcc", action=argparse.BooleanOptionalAction, default=STREAMING_MFCC)
    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default=MEL_FILTER_SHAPE)
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default=LOG_APPROX_MODE)
    parser.add_argument("--log_pwl_num_segments", type=int, default=LOG_PWL_NUM_SEGMENTS)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default=LOG_PWL_STRATEGY)
    parser.add_argument("--log_pwl_gamma", type=float, default=LOG_PWL_GAMMA)
    parser.add_argument("--log_offset", type=float, default=LOG_OFFSET)
    parser.add_argument("--log_input_clamp_min", type=float, default=LOG_INPUT_CLAMP_MIN)
    parser.add_argument("--int8_mfcc_log_approx_mode", choices=["exact", "pwl"], default=INT8_MFCC_LOG_APPROX_MODE)
    parser.add_argument("--int8_mfcc_log_pwl_num_segments", type=int, default=INT8_MFCC_LOG_PWL_NUM_SEGMENTS)
    parser.add_argument("--int8_mfcc_log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default=INT8_MFCC_LOG_PWL_STRATEGY)
    parser.add_argument("--int8_mfcc_log_pwl_gamma", type=float, default=INT8_MFCC_LOG_PWL_GAMMA)
    parser.add_argument("--int8_mfcc_log_offset", type=float, default=INT8_MFCC_LOG_OFFSET)
    parser.add_argument("--int8_mfcc_log_input_clamp_min", type=float, default=INT8_MFCC_LOG_INPUT_CLAMP_MIN)
    parser.add_argument("--int8_mfcc_coeff_bits", type=int, default=INT8_MFCC_COEFF_BITS)
    parser.add_argument("--int8_mfcc_requantize_power", action=argparse.BooleanOptionalAction, default=INT8_MFCC_REQUANTIZE_POWER)
    parser.add_argument("--int8_mfcc_requantize_mel", action=argparse.BooleanOptionalAction, default=INT8_MFCC_REQUANTIZE_MEL)
    parser.add_argument("--calibration_split", choices=["train", "validation", "test"], default=CALIBRATION_SPLIT)
    parser.add_argument("--calibration_noise_roots", nargs="+", default=CALIBRATION_NOISE_ROOTS)
    parser.add_argument("--calibration_noise_prob", type=float, default=CALIBRATION_NOISE_PROB)
    parser.add_argument("--calibration_snr_min_db", type=float, default=CALIBRATION_SNR_MIN_DB)
    parser.add_argument("--calibration_snr_max_db", type=float, default=CALIBRATION_SNR_MAX_DB)
    parser.add_argument("--calibration_batches", type=int, default=CALIBRATION_BATCHES)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.backend = choose_backend(args.backend)
    if args.dct_coeff <= 0:
        raise ValueError("--dct_coeff must be positive")
    if args.window_size_ms != args.window_stride_ms:
        print("[WARN] non-equal window/hop is supported, but current CRNN experiments used 32 ms / 32 ms.")
    for name, roots in [
        ("train_noise_roots", args.train_noise_roots),
        ("valid_noise_roots", args.valid_noise_roots),
        ("test_noise_roots", args.test_noise_roots),
        ("calibration_noise_roots", args.calibration_noise_roots),
    ]:
        usable = count_usable_noise_files(roots)
        print(f"[INFO] {name}={roots}, usable_noise_files={usable}")
        if usable <= 0 and name != "calibration_noise_roots":
            raise FileNotFoundError(f"No usable wav files found for {name}={roots}")
    if args.calibration_snr_min_db > args.calibration_snr_max_db:
        raise ValueError("--calibration_snr_min_db must be <= --calibration_snr_max_db")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.gpu > 0 and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}, backend={args.backend}")

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
    print(f"[DONE] models={len(train_rows)}, grid_rows={len(grid_rows)}")


if __name__ == "__main__":
    main()
