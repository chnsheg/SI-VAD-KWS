from __future__ import annotations

import argparse
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

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
from dscnn_kws.data.dataset import SpeechCommandDataset
from dscnn_kws.model import DSCNN
from dscnn_kws.model.dscnn import calculate_time_steps
from dscnn_kws.train import MFCCDSCNN


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


@dataclass(frozen=True)
class QuantProfile:
    name: str
    input_dir: Path
    output_dir: Path
    results_csv: Path
    eval_mode: str


@dataclass(frozen=True)
class TensorQuantSpec:
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    dtype: str


class SymmetricInt8Quantizer:
    def __init__(self, eps: float = 1e-12):
        self.qmin = -128
        self.qmax = 127
        self.zero_point = 0
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "int8_symmetric"

    def scale_from_amax(self, amax: float) -> float:
        amax = max(float(amax), self.eps)
        return amax / float(self.qmax)

    def spec_from_tensor(self, tensor: torch.Tensor) -> TensorQuantSpec:
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            return TensorQuantSpec(1.0, self.zero_point, self.qmin, self.qmax, "int8")
        amax = float(tensor.detach().abs().max().cpu().item())
        return TensorQuantSpec(self.scale_from_amax(amax), self.zero_point, self.qmin, self.qmax, "int8")

    def quantize_int(self, tensor: torch.Tensor, spec: TensorQuantSpec) -> torch.Tensor:
        q = torch.round(tensor.detach().cpu() / spec.scale) + spec.zero_point
        return torch.clamp(q, spec.qmin, spec.qmax).to(torch.int8)

    def quantize_dequantize(self, tensor: torch.Tensor, spec: TensorQuantSpec) -> torch.Tensor:
        if not torch.is_floating_point(tensor):
            return tensor
        q = torch.round(tensor / spec.scale) + spec.zero_point
        q = torch.clamp(q, spec.qmin, spec.qmax)
        return (q - spec.zero_point) * spec.scale

    def activation_spec(self, amax: float) -> TensorQuantSpec:
        return TensorQuantSpec(self.scale_from_amax(amax), self.zero_point, self.qmin, self.qmax, "int8")


class ActivationCalibrator(nn.Module):
    def __init__(self, model: nn.Module, quantize_frontend: bool):
        super().__init__()
        self.model = model
        self.quantize_frontend = bool(quantize_frontend)
        self.input_amax = 0.0
        self.output_amax = 0.0
        self.module_amax: dict[str, float] = {}
        self._handles: list[Any] = []
        self._register_hooks()

    def _should_hook(self, module_name: str, module: nn.Module) -> bool:
        if module is self.model:
            return False
        if any(module.children()):
            return False
        if not self.quantize_frontend and (
            module_name.startswith("feature_extractor")
            or module_name.startswith("freq_mask")
            or module_name.startswith("time_mask")
        ):
            return False
        return isinstance(
            module,
            (
                nn.Conv1d,
                nn.Conv2d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.ReLU,
                nn.AdaptiveAvgPool1d,
                nn.AdaptiveAvgPool2d,
                nn.Dropout,
                nn.Linear,
            ),
        )

    def _tensor_amax(self, value: Any) -> float:
        if torch.is_tensor(value) and torch.is_floating_point(value):
            return float(value.detach().abs().max().cpu().item())
        if isinstance(value, (tuple, list)):
            return max([self._tensor_amax(item) for item in value] or [0.0])
        return 0.0

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if self._should_hook(name, module):
                self.module_amax.setdefault(name, 0.0)

                def hook(_module, _inputs, output, module_name=name):
                    self.module_amax[module_name] = max(self.module_amax[module_name], self._tensor_amax(output))

                self._handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_amax = max(self.input_amax, self._tensor_amax(x))
        out = self.model(x)
        self.output_amax = max(self.output_amax, self._tensor_amax(out))
        return out


class Int8FakeQuantModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        quantizer: SymmetricInt8Quantizer,
        activation_specs: dict[str, TensorQuantSpec],
        quantize_frontend: bool,
    ):
        super().__init__()
        self.model = model
        self.quantizer = quantizer
        self.activation_specs = activation_specs
        self.quantize_frontend = bool(quantize_frontend)
        self._handles: list[Any] = []
        self._register_output_hooks()

    def _should_hook(self, module_name: str, module: nn.Module) -> bool:
        if module is self.model:
            return False
        if any(module.children()):
            return False
        if not self.quantize_frontend and (
            module_name.startswith("feature_extractor")
            or module_name.startswith("freq_mask")
            or module_name.startswith("time_mask")
        ):
            return False
        return module_name in self.activation_specs

    def _quantize_output(self, output: Any, spec: TensorQuantSpec) -> Any:
        if torch.is_tensor(output):
            return self.quantizer.quantize_dequantize(output, spec)
        if isinstance(output, tuple):
            return tuple(self._quantize_output(item, spec) for item in output)
        if isinstance(output, list):
            return [self._quantize_output(item, spec) for item in output]
        return output

    def _register_output_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if self._should_hook(name, module):
                spec = self.activation_specs[name]
                self._handles.append(
                    module.register_forward_hook(lambda _m, _inp, out, s=spec: self._quantize_output(out, s))
                )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer.quantize_dequantize(x, self.activation_specs["__input__"])
        out = self.model(x)
        return self.quantizer.quantize_dequantize(out, self.activation_specs["__output__"])


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


def build_float_model(
    args: argparse.Namespace,
    *,
    num_layers: int,
    channels: int,
    label_count: int,
) -> MFCCDSCNN:
    time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
    input_dim = time_steps * args.dct_coeff
    backbone = DSCNN(
        input_dim=input_dim,
        label_count=label_count,
        model_size_info=make_model_size_info(num_layers, channels),
        dct_coeff=args.dct_coeff,
    )

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


def load_model_weights(model: MFCCDSCNN, state: dict[str, torch.Tensor]) -> None:
    if any(key.startswith("backbone.") for key in state):
        model.load_state_dict(state, strict=True)
    else:
        model.backbone.load_state_dict(state, strict=True)


def quantize_state_dict(
    state: dict[str, torch.Tensor],
    quantizer: SymmetricInt8Quantizer,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, TensorQuantSpec]]:
    qfloat_state = {}
    qint_state = {}
    specs = {}
    for key, value in state.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            spec = quantizer.spec_from_tensor(value)
            specs[key] = spec
            qint_state[key] = quantizer.quantize_int(value, spec)
            qfloat_state[key] = quantizer.quantize_dequantize(value, spec)
        elif torch.is_tensor(value):
            qfloat_state[key] = value.detach().cpu()
            qint_state[key] = value.detach().cpu()
    return qfloat_state, qint_state, specs


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
    noise_aug: bool,
    noise_roots: list[str] | None,
    snr_db: float | None,
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
        is_training=False,
        class_list=CLASS_LIST,
        class_encoding=CLASS_ENCODING,
        sample_rate=args.sample_rate,
        noise_aug=noise_aug,
        noise_roots=noise_roots,
        noise_prob=args.noise_prob if noise_aug else 0.0,
        noise_snr_min_db=snr_db if snr_db is not None else 0.0,
        noise_snr_max_db=snr_db if snr_db is not None else 0.0,
        deterministic_noise=True,
        random_seed=random_seed,
        allow_online_resample=True,
        strict_sample_rate=False,
    )
    workers = args.num_workers if split != "test" else max(0, args.num_workers // 2)
    return DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def calibrate_activations(
    model: nn.Module,
    quantizer: SymmetricInt8Quantizer,
    loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[dict[str, TensorQuantSpec], int]:
    calibrator = ActivationCalibrator(model, quantize_frontend=args.quantize_frontend)
    calibrator.eval()
    batches = 0
    for waveform, _labels in loader:
        calibrator(waveform)
        batches += 1
        if args.calibration_batches > 0 and batches >= args.calibration_batches:
            break

    activation_specs = {
        "__input__": quantizer.activation_spec(calibrator.input_amax),
        "__output__": quantizer.activation_spec(calibrator.output_amax),
    }
    for name, amax in calibrator.module_amax.items():
        activation_specs[name] = quantizer.activation_spec(amax)
    calibrator.close()
    return activation_specs, batches


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for waveform, labels in loader:
        logits = model(waveform)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item())
        total += int(labels.numel())
        correct += int((preds == labels).sum().item())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return {
        "loss": total_loss / max(1, len(loader)),
        "acc": correct / max(1, total),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "num_samples": total,
    }


def run_clean_eval(model: nn.Module, args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    loader = build_loader(
        args,
        dataset=dataset,
        split="test",
        noise_aug=False,
        noise_roots=None,
        snr_db=None,
        random_seed=args.seed + 200000,
    )
    metrics = evaluate(model, loader)
    return {
        "eval_kind": "clean_test",
        "split": "test",
        "scene": "",
        "snr_db": "",
        "noise_roots": "",
        **metrics,
    }


def run_noise_validation_and_test(model: nn.Module, args: argparse.Namespace, dataset: str) -> list[dict[str, Any]]:
    rows = []
    eval_specs = [
        ("noise_validation_tau_valid_list", "validation", args.valid_noise_roots, args.seed + 100000),
        ("noise_test_tau_test_list", "test", args.test_noise_roots, args.seed + 200000),
    ]
    for eval_kind, split, noise_roots, random_seed in eval_specs:
        usable = count_usable_noise_files(noise_roots)
        if usable <= 0:
            print(f"[WARN] no usable noise files for {eval_kind}: {noise_roots}")
            continue
        loader = build_loader(
            args,
            dataset=dataset,
            split=split,
            noise_aug=True,
            noise_roots=noise_roots,
            snr_db=args.valid_snr_db,
            random_seed=random_seed,
        )
        metrics = evaluate(model, loader)
        rows.append(
            {
                "eval_kind": eval_kind,
                "split": split,
                "scene": "",
                "snr_db": args.valid_snr_db,
                "noise_roots": ";".join(noise_roots),
                "usable_noise_files": usable,
                **metrics,
            }
        )
    return rows


def run_noise_scene_grid(model: nn.Module, args: argparse.Namespace, dataset: str) -> list[dict[str, Any]]:
    rows = []
    for scene_idx, scene in enumerate(args.scene_names):
        scene_root = str(Path(args.scene_test_root) / scene)
        noise_roots = [scene_root]
        usable = count_usable_noise_files(noise_roots)
        if usable <= 0:
            print(f"[WARN] scene skipped, no usable wavs: {scene_root}")
            continue
        for snr_idx, snr_db in enumerate(args.test_snrs):
            seed = 500000 + scene_idx * 10007 + snr_idx * 101
            loader = build_loader(
                args,
                dataset=dataset,
                split="test",
                noise_aug=True,
                noise_roots=noise_roots,
                snr_db=snr_db,
                random_seed=seed,
            )
            metrics = evaluate(model, loader)
            rows.append(
                {
                    "eval_kind": "noise_scene_snr_test",
                    "split": "test",
                    "scene": scene,
                    "snr_db": snr_db,
                    "noise_roots": scene_root,
                    "usable_noise_files": usable,
                    "random_seed": seed,
                    **metrics,
                }
            )
            print(
                f"[GRID] {dataset} | scene={scene:<18} | snr={snr_db:>5} dB | "
                f"acc={metrics['acc']:.4f} | f1={metrics['f1']:.4f}"
            )
    return rows


def save_int8_checkpoint(
    path: Path,
    *,
    source_checkpoint: Path,
    float_q_state: dict[str, torch.Tensor],
    int_q_state: dict[str, torch.Tensor],
    weight_specs: dict[str, TensorQuantSpec],
    activation_specs: dict[str, TensorQuantSpec],
    row: dict[str, Any],
    calibration_batches: int,
    args: argparse.Namespace,
) -> int:
    def spec_to_dict(spec: TensorQuantSpec) -> dict[str, Any]:
        return {
            "scale": spec.scale,
            "zero_point": spec.zero_point,
            "qmin": spec.qmin,
            "qmax": spec.qmax,
            "dtype": spec.dtype,
        }

    payload = {
        "state_dict_qfloat": float_q_state,
        "state_dict_qint": int_q_state,
        "weight_specs": {key: spec_to_dict(spec) for key, spec in weight_specs.items()},
        "activation_specs": {key: spec_to_dict(spec) for key, spec in activation_specs.items()},
        "source_checkpoint": str(source_checkpoint),
        "quantization": {
            "format": "int8_symmetric",
            "method": "post_training_fake_quantization",
            "weights": "per-tensor symmetric int8, scale=max_abs/127",
            "activations": "per-module-output symmetric int8 calibrated on validation",
            "activation_simulation": "round/clamp/dequantize forward hooks",
            "calibration_batches": calibration_batches,
            "quantize_frontend": args.quantize_frontend,
        },
        "metadata": row,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path.stat().st_size


def build_calibration_loader(
    args: argparse.Namespace,
    *,
    profile: QuantProfile,
    dataset: str,
) -> DataLoader:
    if profile.eval_mode == "clean":
        return build_loader(
            args,
            dataset=dataset,
            split="validation",
            noise_aug=False,
            noise_roots=None,
            snr_db=None,
            random_seed=args.seed + 100000,
        )
    if profile.eval_mode == "noise_snr_scene":
        return build_loader(
            args,
            dataset=dataset,
            split="validation",
            noise_aug=True,
            noise_roots=args.valid_noise_roots,
            snr_db=args.valid_snr_db,
            random_seed=args.seed + 100000,
        )
    raise ValueError(profile.eval_mode)


def run_one(
    path: Path,
    profile: QuantProfile,
    args: argparse.Namespace,
    quantizer: SymmetricInt8Quantizer,
) -> list[dict[str, Any]]:
    print(f"[INT8] {profile.name}: {path}")
    state = load_state_dict(path)
    inferred_layers, inferred_channels, label_count = infer_arch_from_state_dict(state)
    named_layers, named_channels = infer_arch_from_name(path)
    num_layers = args.layers or named_layers or inferred_layers
    channels = args.channels or named_channels or inferred_channels
    dataset = args.dataset or infer_dataset_from_name(path)
    if not dataset:
        raise ValueError(f"Cannot infer dataset name from {path}; pass --dataset for single-dataset runs")

    float_q_state, int_q_state, weight_specs = quantize_state_dict(state, quantizer)

    model = build_float_model(args, num_layers=num_layers, channels=channels, label_count=label_count)
    load_model_weights(model, float_q_state)
    model.cpu()
    model.eval()

    calib_loader = build_calibration_loader(args, profile=profile, dataset=dataset)
    activation_specs, calibration_batches = calibrate_activations(model, quantizer, calib_loader, args)
    qmodel = Int8FakeQuantModel(
        model,
        quantizer=quantizer,
        activation_specs=activation_specs,
        quantize_frontend=args.quantize_frontend,
    )

    if profile.eval_mode == "clean":
        eval_rows = [run_clean_eval(qmodel, args, dataset)]
    elif profile.eval_mode == "noise_snr_scene":
        eval_rows = run_noise_validation_and_test(qmodel, args, dataset)
        eval_rows.extend(run_noise_scene_grid(qmodel, args, dataset))
    else:
        raise ValueError(profile.eval_mode)

    out_path = profile.output_dir / f"{safe_stem(path)}_int8.pt"
    common = {
        "profile": profile.name,
        "dataset": dataset,
        "checkpoint": str(path),
        "quantized_checkpoint": str(out_path.resolve()),
        "quant_format": quantizer.name,
        "weight_quant": "per_tensor_symmetric",
        "activation_quant": "per_module_output_symmetric",
        "calibration_batches": calibration_batches,
        "layers": num_layers,
        "channels": channels,
        "expected_params": expected_params(num_layers, channels, len(CLASS_LIST)),
        "label_count": label_count,
        "sample_rate": args.sample_rate,
        "dct_coeff": args.dct_coeff,
        "window_size_ms": args.window_size_ms,
        "window_stride_ms": args.window_stride_ms,
        "quantize_frontend": args.quantize_frontend,
    }
    rows = [{**common, **row} for row in eval_rows]

    first_row = rows[0] if rows else common
    size = save_int8_checkpoint(
        out_path,
        source_checkpoint=path,
        float_q_state=float_q_state,
        int_q_state=int_q_state,
        weight_specs=weight_specs,
        activation_specs=activation_specs,
        row=first_row,
        calibration_batches=calibration_batches,
        args=args,
    )
    for row in rows:
        row["quantized_size_bytes"] = size
    qmodel.close()
    print(f"[OK] saved={out_path} | rows={len(rows)} | size={size} bytes")
    return rows


def discover_checkpoints(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        print(f"[WARN] input dir does not exist, skip: {input_dir}")
        return []
    return sorted(p.resolve() for p in input_dir.glob(pattern) if p.is_file())


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
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
    print(f"[INFO] results saved to: {path.resolve()}")


def build_profiles(args: argparse.Namespace) -> list[QuantProfile]:
    profiles = [
        QuantProfile(
            name="clean",
            input_dir=Path(args.clean_input_dir),
            output_dir=Path(args.clean_output_dir),
            results_csv=Path(args.clean_results_csv),
            eval_mode="clean",
        ),
        QuantProfile(
            name="noise_snr_scene",
            input_dir=Path(args.noise_input_dir),
            output_dir=Path(args.noise_output_dir),
            results_csv=Path(args.noise_results_csv),
            eval_mode="noise_snr_scene",
        ),
    ]
    if args.mode == "both":
        return profiles
    return [p for p in profiles if p.name == args.mode]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Symmetric int8 fake quantization for clean and SNR-scene DSCNN best models."
    )
    parser.add_argument("--mode", choices=["clean", "noise_snr_scene", "both"], default="both")
    parser.add_argument(
        "--clean_input_dir",
        default="/root/kws/dscnn_kws/dscnn_kws/runs/sweep_best_models",
    )
    parser.add_argument(
        "--noise_input_dir",
        default="/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models",
    )
    parser.add_argument("--clean_output_dir", default="./dscnn_kws/quantization/int8_sweep_best_models")
    parser.add_argument("--noise_output_dir", default="./dscnn_kws/quantization/int8_snr_scene_arch_sweep_best_models")
    parser.add_argument("--clean_results_csv", default="./dscnn_kws/quantization/int8_clean_results.csv")
    parser.add_argument("--noise_results_csv", default="./dscnn_kws/quantization/int8_noise_snr_scene_results.csv")
    parser.add_argument("--pattern", default="*.pt")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: process at most N checkpoints per profile")

    parser.add_argument("--root", default="/root/kws/dscnn_kws/dscnn_kws/data")
    parser.add_argument("--dataset", default=None, help="Override dataset inference; useful for a single checkpoint")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--calibration_batches", type=int, default=32, help="0 means use full validation split")
    parser.add_argument(
        "--quantize_frontend",
        action="store_true",
        default=False,
        help="Also fake-quantize selected frontend leaf module outputs. Default keeps frontend mostly float.",
    )

    parser.add_argument("--valid_noise_roots", nargs="+", default=["./dscnn_kws/noise/lists/tau_valid.txt"])
    parser.add_argument("--test_noise_roots", nargs="+", default=["./dscnn_kws/noise/lists/tau_test.txt"])
    parser.add_argument("--noise_prob", type=float, default=1.0)
    parser.add_argument("--valid_snr_db", type=float, default=5.0)
    parser.add_argument("--scene_test_root", default="./dscnn_kws/noise/tau")
    parser.add_argument("--scene_names", nargs="+", default=TAU_SCENES)
    parser.add_argument("--test_snrs", nargs="+", type=float, default=[20.0, 10.0, 5.0, 0.0, -5.0])

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--dct_coeff", type=int, default=10)
    parser.add_argument("--window_size_ms", type=int, default=32)
    parser.add_argument("--window_stride_ms", type=int, default=32)
    parser.add_argument("--layers", type=int, default=None, help="Override inferred layer count")
    parser.add_argument("--channels", type=int, default=None, help="Override inferred channel count")

    parser.add_argument("--frontend", choices=["mfcc", "bandpass"], default="mfcc")
    parser.add_argument("--mfcc_impl", choices=["torchaudio", "torch"], default="torchaudio")
    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default="triangular")
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=0.97)
    parser.add_argument("--bandpass_n_bands", type=int, default=16)
    parser.add_argument("--bandpass_f_min", type=float, default=200.0)
    parser.add_argument("--bandpass_f_max", type=float, default=4000.0)
    parser.add_argument("--bandpass_spacing", choices=["log", "linear"], default="log")
    parser.add_argument("--bandpass_kernel_size", type=int, default=63)
    parser.add_argument("--bandpass_phase_count", type=int, default=1)
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default="exact")
    parser.add_argument("--log_pwl_num_segments", type=int, default=6)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default="uniform_logx")
    parser.add_argument("--log_pwl_gamma", type=float, default=1.0)
    parser.add_argument("--log_offset", type=float, default=1e-6)
    parser.add_argument("--log_input_clamp_min", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantizer = SymmetricInt8Quantizer()
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
    print(
        "[INFO] quant_format=int8_symmetric, qrange=[-128, 127], "
        f"calibration_batches={args.calibration_batches}"
    )

    profiles = build_profiles(args)
    for profile in profiles:
        checkpoints = discover_checkpoints(profile.input_dir, args.pattern)
        if args.limit > 0:
            checkpoints = checkpoints[: args.limit]
        print(f"[INFO] profile={profile.name}, checkpoints={len(checkpoints)}, input_dir={profile.input_dir}")
        rows = []
        for ckpt in checkpoints:
            try:
                rows.extend(run_one(ckpt, profile, args, quantizer))
            except Exception as exc:
                print(f"[ERROR] failed: {ckpt}: {exc}")
                if len(checkpoints) == 1:
                    raise
        write_csv(rows, profile.results_csv)
        print(f"[DONE] profile={profile.name}, checkpoints_done={len(checkpoints)}, result_rows={len(rows)}")


if __name__ == "__main__":
    main()
