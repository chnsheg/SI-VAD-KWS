from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from dscnn_kws.frontend import BitAccuracyMFCCFrontend
from dscnn_kws.quantization.qat_int8_noise_snr_scene import (
    SCENE_NAMES,
    TAU_SCENES,
    TEST_SNRS,
    VALID_NOISE_PROB,
    VALID_SNR_DB,
    build_loader,
    build_standard_model,
    build_test_list_loader,
    build_train_loader,
    build_valid_loader,
    choose_backend,
    copy_state_dict_cpu,
    count_usable_noise_files,
    disable_observer,
    evaluate,
    expected_params,
    freeze_qat_bn_stats,
    infer_arch_from_name,
    infer_arch_from_state_dict,
    infer_dataset_from_name,
    load_model_weights,
    load_state_dict,
    load_state_dict_to_model,
    prepare_model_for_qat,
    quantize_qat_model,
    safe_stem,
    train_one_epoch,
)


# =============================================================================
# Code-level configuration
# =============================================================================
# You can edit this block directly on the Linux server, then run:
#   python dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_noise_snr_scene.py
#
# Command-line flags are also supported and override these defaults.

# Input / output.
INPUT_DIR = "/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models"
PATTERN = "*.pt"
CHECKPOINTS = None
OUTPUT_DIR = "./dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_models"
TRAIN_RESULTS_CSV = "./dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_train_results.csv"
GRID_RESULTS_CSV = "./dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_grid_results.csv"
LIMIT = 0

# Data / runtime.
ROOT = "/root/kws/dscnn_kws/dscnn_kws/data"
DATASET = None
BATCH = 256
NUM_WORKERS = 8
GPU = 1
SEED = 42

# QAT for the DSCNN backbone. The frontend path still receives fake-quant
# constraints during QAT, then is replaced by BitAccuracyMFCCFrontend for final eval.
QAT_EPOCHS = 10
LR = 5e-5
WEIGHT_DECAY = 1e-6
BACKEND = "fbgemm"
FREEZE_BN_AFTER_EPOCH = 3
DISABLE_OBSERVER_AFTER_EPOCH = 5
QUANTIZE_FRONTEND_DURING_QAT = True

# Noise settings, aligned with sweep_fixed_dscnn_noise_snr_scene_acc.py.
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

# Frontend/model shape. The QAT stage now uses TorchMFCC + PWL so the training
# frontend is aligned with the final integer-aware frontend.
SAMPLE_RATE = 16000
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
LAYERS = None
CHANNELS = None
FRONTEND = "mfcc"
MFCC_IMPL = "torch"
MEL_FILTER_SHAPE = "rectangular"
PRE_EMPHASIS = True
PRE_EMPHASIS_COEFF = 0.97
SPEC_AUG = False
SPEC_AUG_FREQ_MASK_PARAM = 1
SPEC_AUG_TIME_MASK_PARAM = 1
SPEC_AUG_NUM_FREQ_MASKS = 1
SPEC_AUG_NUM_TIME_MASKS = 1

# BitAccuracyMFCCFrontend baseline, aligned with
# README_MFCC_BitAccuracy_BASELINE.md.
BIT_MFCC_LOG_OFFSET = 1e-6
BIT_MFCC_LOG_INPUT_CLAMP_MIN = 1e-12
BIT_MFCC_MEL_DRAIN_SHIFT = 14
BIT_MFCC_MFCC_OUTPUT_SCALE = None

# Frontend scale calibration. The default uses validation speech with TAU valid
# noise over the full SNR range so the fixed scales can cover the scene grid.
CALIBRATION_SPLIT = "validation"
CALIBRATION_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_valid.txt"]
CALIBRATION_NOISE_PROB = 1.0
CALIBRATION_SNR_MIN_DB = -5.0
CALIBRATION_SNR_MAX_DB = 20.0
CALIBRATION_BATCHES = 0  # 0 means all batches.

# Unused standard-model bandpass placeholders required by build_standard_model.
BANDPASS_N_BANDS = 10
BANDPASS_F_MIN = 80.0
BANDPASS_F_MAX = 6000.0
BANDPASS_SPACING = "log"
BANDPASS_KERNEL_SIZE = 255
BANDPASS_PHASE_COUNT = 4

# Float TorchMFCC frontend log settings used during QAT. Keep these aligned
# with the final bit-accuracy frontend when possible.
LOG_APPROX_MODE = "pwl"
LOG_PWL_NUM_SEGMENTS = 8
LOG_PWL_STRATEGY = "uniform_logx"
LOG_PWL_GAMMA = 1.0
LOG_OFFSET = 1e-6
LOG_INPUT_CLAMP_MIN = 1e-12


class BitAccuracyMFCCQuantizedBackboneModel(nn.Module):
    """Connect the bit-accuracy MFCC frontend to a converted INT8 backbone."""

    def __init__(self, frontend: BitAccuracyMFCCFrontend, quantized_backbone: nn.Module, dct_coeff: int):
        super().__init__()
        self.feature_extractor = frontend
        self.backbone = quantized_backbone
        self.dct_coeff = int(dct_coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = features[:, : self.dct_coeff, :]
        features = features.permute(0, 2, 1).reshape(features.size(0), -1)
        return self.backbone(features)


def load_source_weights_for_qat(model: nn.Module, state: dict[str, torch.Tensor]) -> str:
    """Load a source checkpoint while allowing the frontend implementation to change.

    Original sweep checkpoints may contain torchaudio frontend buffers. When the
    QAT stage is switched to TorchMFCC + PWL, those frontend buffers no longer
    match, but the DSCNN backbone weights are still valid and should be loaded.
    """

    try:
        load_model_weights(model, state)
        return "full_model"
    except RuntimeError as exc:
        if not hasattr(model, "backbone"):
            raise
        if any(key.startswith("backbone.") for key in state):
            backbone_state = {
                key[len("backbone.") :]: value
                for key, value in state.items()
                if key.startswith("backbone.")
            }
        else:
            backbone_state = state

        try:
            model.backbone.load_state_dict(backbone_state, strict=True)
        except Exception:
            raise exc

        print("[WARN] full model state_dict did not match the current frontend; loaded backbone weights only.")
        return "backbone_only_frontend_rebuilt"


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        paths = [Path(p).resolve() for p in args.checkpoints]
    else:
        paths = sorted(Path(args.input_dir).glob(args.pattern))
        paths = [p.resolve() for p in paths if p.is_file()]
    if args.limit > 0:
        paths = paths[: args.limit]
    return paths


def build_bit_accuracy_mfcc_frontend(args: argparse.Namespace, *, observer_enabled: bool) -> BitAccuracyMFCCFrontend:
    n_fft = int(args.sample_rate * args.window_size_ms / 1000)
    hop_length = int(args.sample_rate * args.window_stride_ms / 1000)
    config = {
        "mel_drain_shift": args.bit_mfcc_mel_drain_shift,
        "mfcc_output_scale": args.bit_mfcc_mfcc_output_scale,
    }
    return BitAccuracyMFCCFrontend(
        sample_rate=args.sample_rate,
        n_mfcc=40,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=40,
        f_min=20.0,
        f_max=float(args.sample_rate / 2),
        center=True,
        dct_norm="ortho",
        mel_filter_shape=args.mel_filter_shape,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        log_offset=args.bit_mfcc_log_offset,
        log_input_clamp_min=args.bit_mfcc_log_input_clamp_min,
        config=config,
        observer_enabled=observer_enabled,
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
def calibrate_bit_accuracy_mfcc_frontend(
    args: argparse.Namespace,
    *,
    dataset: str,
    config_json: Path,
    device: torch.device,
) -> tuple[BitAccuracyMFCCFrontend, int]:
    frontend = build_bit_accuracy_mfcc_frontend(args, observer_enabled=True).to(device)
    frontend.eval()
    loader = build_calibration_loader(args, dataset)

    batches = 0
    for waveform, _labels in tqdm(loader, desc="calibrate-bit-accuracy-mfcc", leave=False):
        frontend(waveform.to(device))
        batches += 1
        if args.calibration_batches > 0 and batches >= args.calibration_batches:
            break

    config_json.parent.mkdir(parents=True, exist_ok=True)
    frontend.export_config_json(
        config_json,
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
    frontend = BitAccuracyMFCCFrontend.from_config_json(config_json).cpu().eval()
    return frontend, batches


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
    bit_mfcc_config_json: Path,
    inference_checkpoint: Path,
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
                "bit_mfcc_config_json": str(bit_mfcc_config_json),
                "inference_checkpoint": str(inference_checkpoint),
            }
            rows.append(row)
            print(
                f"[GRID] {dataset} | {arch} | {scene:<18} | "
                f"snr={snr_db:>5} | acc={metrics.acc:.4f} | f1={metrics.f1:.4f}"
            )
    return rows


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


def save_artifacts(
    *,
    output_dir: Path,
    stem: str,
    source_checkpoint: Path,
    qat_model: nn.Module,
    quantized_qat_model: nn.Module,
    inference_model: nn.Module,
    bit_mfcc_config_json: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qat_path = output_dir / f"{stem}_qat_prepared_best.pt"
    quantized_backbone_path = output_dir / f"{stem}_qat_int8_backbone.pt"
    inference_path = output_dir / f"{stem}_bit_accuracy_mfcc_int8_backbone.pt"
    common = {
        "source_checkpoint": str(source_checkpoint),
        "metadata": row,
        "bit_mfcc_config_json": str(bit_mfcc_config_json),
        "qat": {
            "backend": args.backend,
            "epochs": args.qat_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "freeze_bn_after_epoch": args.freeze_bn_after_epoch,
            "disable_observer_after_epoch": args.disable_observer_after_epoch,
            "quantize_frontend_during_qat": args.quantize_frontend,
        },
        "frontend_quantization": {
            "format": "bit-accuracy mfcc baseline reference",
            "baseline": "README_MFCC_BitAccuracy_BASELINE.md",
            "mel_drain_shift": args.bit_mfcc_mel_drain_shift,
            "mfcc_output_scale": args.bit_mfcc_mfcc_output_scale,
            "config_json": str(bit_mfcc_config_json),
        },
    }
    torch.save({**common, "state_dict": qat_model.state_dict(), "format": "prepared_qat_float"}, qat_path)
    torch.save(
        {**common, "state_dict": quantized_qat_model.state_dict(), "format": "converted_int8_backbone"},
        quantized_backbone_path,
    )
    torch.save(
        {
            **common,
            "state_dict": inference_model.state_dict(),
            "format": "bit_accuracy_mfcc_frontend_plus_converted_int8_backbone",
        },
        inference_path,
    )
    return qat_path, quantized_backbone_path, inference_path


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
    print(f"[QAT+INT8-MFCC] dataset={dataset}, arch={arch}, ckpt={path}")

    model = build_standard_model(args, num_layers=layers, channels=channels, label_count=label_count)
    load_scope = load_source_weights_for_qat(model, state)
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

    quantized_qat_model = quantize_qat_model(model)
    quantized_test_m = evaluate(quantized_qat_model, test_list_loader, torch.device("cpu"))

    stem = safe_stem(path)
    output_dir = Path(args.output_dir)
    config_json = output_dir / f"{stem}_bit_accuracy_mfcc_config.json"
    bit_frontend, calibration_batches = calibrate_bit_accuracy_mfcc_frontend(
        args,
        dataset=dataset,
        config_json=config_json,
        device=torch.device("cpu"),
    )

    inference_model = BitAccuracyMFCCQuantizedBackboneModel(
        frontend=bit_frontend,
        quantized_backbone=quantized_qat_model.backbone,
        dct_coeff=args.dct_coeff,
    ).cpu()
    inference_model.eval()
    bit_mfcc_test_m = evaluate(inference_model, test_list_loader, torch.device("cpu"))
    print(
        "[TEST] "
        f"qat_tau_list_acc={qat_test_m.acc:.4f} f1={qat_test_m.f1:.4f} | "
        f"int8_backbone_tau_list_acc={quantized_test_m.acc:.4f} f1={quantized_test_m.f1:.4f} | "
        f"bit_mfcc_int8_backbone_tau_list_acc={bit_mfcc_test_m.acc:.4f} f1={bit_mfcc_test_m.f1:.4f}"
    )

    artifact_row = {
        "dataset": dataset,
        "arch": arch,
        "layers": layers,
        "channels": channels,
        "expected_params": param_count,
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
        "quantized_backbone_test_tau_list_acc": quantized_test_m.acc,
        "quantized_backbone_test_tau_list_f1": quantized_test_m.f1,
        "bit_mfcc_int8_backbone_test_tau_list_acc": bit_mfcc_test_m.acc,
        "bit_mfcc_int8_backbone_test_tau_list_f1": bit_mfcc_test_m.f1,
        "backend": args.backend,
        "frontend": "bit_accuracy_mfcc",
        "bit_mfcc_config_json": str(config_json.resolve()),
        "bit_mfcc_calibration_batches": calibration_batches,
        "bit_mfcc_mel_filter_shape": args.mel_filter_shape,
        "bit_mfcc_mel_drain_shift": args.bit_mfcc_mel_drain_shift,
        "bit_mfcc_mfcc_output_scale": args.bit_mfcc_mfcc_output_scale,
    }

    qat_path, quantized_backbone_path, inference_path = save_artifacts(
        output_dir=output_dir,
        stem=stem,
        source_checkpoint=path,
        qat_model=model.cpu(),
        quantized_qat_model=quantized_qat_model,
        inference_model=inference_model,
        bit_mfcc_config_json=config_json,
        row=artifact_row,
        args=args,
    )
    artifact_row["qat_prepared_checkpoint"] = str(qat_path.resolve())
    artifact_row["quantized_backbone_checkpoint"] = str(quantized_backbone_path.resolve())
    artifact_row["inference_checkpoint"] = str(inference_path.resolve())
    artifact_row["qat_prepared_size_bytes"] = qat_path.stat().st_size
    artifact_row["quantized_backbone_size_bytes"] = quantized_backbone_path.stat().st_size
    artifact_row["inference_size_bytes"] = inference_path.stat().st_size

    grid_rows = run_scene_grid(
        args,
        model=inference_model,
        dataset=dataset,
        arch=arch,
        layers=layers,
        channels=channels,
        expected_param_count=param_count,
        source_checkpoint=path,
        bit_mfcc_config_json=config_json,
        inference_checkpoint=inference_path,
    )
    for row in grid_rows:
        row.update(
            {
                "best_epoch": best_epoch,
                "best_valid_acc": best_acc,
                "backend": args.backend,
                "bit_mfcc_calibration_batches": calibration_batches,
                "bit_mfcc_mel_filter_shape": args.mel_filter_shape,
                "bit_mfcc_mel_drain_shift": args.bit_mfcc_mel_drain_shift,
                "bit_mfcc_mfcc_output_scale": args.bit_mfcc_mfcc_output_scale,
            }
        )
    return artifact_row, grid_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QAT DSCNN backbone, then evaluate the bit-accuracy MFCC baseline frontend on TAU scenes."
    )
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
    parser.add_argument("--quantize_frontend", action=argparse.BooleanOptionalAction, default=QUANTIZE_FRONTEND_DURING_QAT)

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
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--frontend", choices=["mfcc"], default=FRONTEND)
    parser.add_argument("--mfcc_impl", choices=["torchaudio", "torch"], default=MFCC_IMPL)
    parser.add_argument(
        "--mel_filter_shape",
        choices=["rectangular"],
        default=MEL_FILTER_SHAPE,
        help="Bit-accuracy baseline uses rectangular Mel filters only for now.",
    )
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=PRE_EMPHASIS)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=PRE_EMPHASIS_COEFF)
    parser.add_argument("--spec_aug", action=argparse.BooleanOptionalAction, default=SPEC_AUG)
    parser.add_argument("--spec_aug_freq_mask_param", type=int, default=SPEC_AUG_FREQ_MASK_PARAM)
    parser.add_argument("--spec_aug_time_mask_param", type=int, default=SPEC_AUG_TIME_MASK_PARAM)
    parser.add_argument("--spec_aug_num_freq_masks", type=int, default=SPEC_AUG_NUM_FREQ_MASKS)
    parser.add_argument("--spec_aug_num_time_masks", type=int, default=SPEC_AUG_NUM_TIME_MASKS)

    parser.add_argument("--bit_mfcc_log_offset", type=float, default=BIT_MFCC_LOG_OFFSET)
    parser.add_argument("--bit_mfcc_log_input_clamp_min", type=float, default=BIT_MFCC_LOG_INPUT_CLAMP_MIN)
    parser.add_argument("--bit_mfcc_mel_drain_shift", type=int, default=BIT_MFCC_MEL_DRAIN_SHIFT)
    parser.add_argument("--bit_mfcc_mfcc_output_scale", type=float, default=BIT_MFCC_MFCC_OUTPUT_SCALE)

    parser.add_argument("--calibration_split", choices=["train", "validation", "test"], default=CALIBRATION_SPLIT)
    parser.add_argument("--calibration_noise_roots", nargs="+", default=CALIBRATION_NOISE_ROOTS)
    parser.add_argument("--calibration_noise_prob", type=float, default=CALIBRATION_NOISE_PROB)
    parser.add_argument("--calibration_snr_min_db", type=float, default=CALIBRATION_SNR_MIN_DB)
    parser.add_argument("--calibration_snr_max_db", type=float, default=CALIBRATION_SNR_MAX_DB)
    parser.add_argument("--calibration_batches", type=int, default=CALIBRATION_BATCHES)

    parser.add_argument("--bandpass_n_bands", type=int, default=BANDPASS_N_BANDS)
    parser.add_argument("--bandpass_f_min", type=float, default=BANDPASS_F_MIN)
    parser.add_argument("--bandpass_f_max", type=float, default=BANDPASS_F_MAX)
    parser.add_argument("--bandpass_spacing", choices=["log", "linear"], default=BANDPASS_SPACING)
    parser.add_argument("--bandpass_kernel_size", type=int, default=BANDPASS_KERNEL_SIZE)
    parser.add_argument("--bandpass_phase_count", type=int, default=BANDPASS_PHASE_COUNT)
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default=LOG_APPROX_MODE)
    parser.add_argument("--log_pwl_num_segments", type=int, default=LOG_PWL_NUM_SEGMENTS)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default=LOG_PWL_STRATEGY)
    parser.add_argument("--log_pwl_gamma", type=float, default=LOG_PWL_GAMMA)
    parser.add_argument("--log_offset", type=float, default=LOG_OFFSET)
    parser.add_argument("--log_input_clamp_min", type=float, default=LOG_INPUT_CLAMP_MIN)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frontend != "mfcc":
        raise ValueError("This script is only for the MFCC frontend.")
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
    args.backend = choose_backend(args.backend)
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
