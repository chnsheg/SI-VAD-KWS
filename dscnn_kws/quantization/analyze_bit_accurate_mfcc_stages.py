from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm

from dscnn_kws.frontend import BitAccurateMFCCFakeQuantFrontend, make_bit_accurate_mfcc_config
from dscnn_kws.quantization.bit_accurate_mfcc_bitwidth_utils import (
    load_stage_bit_overrides,
    stage_bit_overrides_json,
)
from dscnn_kws.quantization.qat_bit_accurate_mfcc_accuracy_first import (
    CALIBRATION_NOISE_PROB,
    CALIBRATION_SNR_MAX_DB,
    CALIBRATION_SNR_MIN_DB,
    DATASET,
    LOG_INPUT_CLAMP_MIN,
    LOG_OFFSET,
    LOG_PWL_GAMMA,
    LOG_PWL_NUM_SEGMENTS,
    LOG_PWL_STRATEGY,
    PRE_EMPHASIS,
    PRE_EMPHASIS_COEFF,
    ROOT,
    SAMPLE_RATE,
    WINDOW_SIZE_MS,
    WINDOW_STRIDE_MS,
)
from dscnn_kws.quantization.qat_int8_noise_snr_scene import build_loader, count_usable_noise_files, save_csv


OUTPUT_DIR = "./dscnn_kws/quantization/bit_accurate_mfcc_experiments"
DATASETS = None
BATCH = 128
NUM_WORKERS = 4
GPU = 0
SEED = 42
SPLIT = "validation"
NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_valid.txt"]
NOISE_PROB = CALIBRATION_NOISE_PROB
SNR_MIN_DB = CALIBRATION_SNR_MIN_DB
SNR_MAX_DB = CALIBRATION_SNR_MAX_DB
BATCHES = 8
CONSTRAINT_PROFILE = "upper_bound_s8_mfcc"
REFERENCE_PROFILE = "float_pwl_rectangular"
STAGE_BIT_OVERRIDES = None
STAGE_BIT_OVERRIDES_JSON = None


def _flatten_float(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        x = torch.view_as_real(x)
    return x.detach().to(torch.float32).flatten()


def _corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
    a = _flatten_float(a)
    b = _flatten_float(b)
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((torch.sum(a * b) / denom).cpu().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = _flatten_float(a)
    b = _flatten_float(b)
    if a.numel() != b.numel():
        return float("nan")
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((torch.dot(a, b) / denom).cpu().item())


def compare_tensors(dataset: str, stage: str, ref: torch.Tensor, cand: torch.Tensor) -> dict[str, Any]:
    ref_f = _flatten_float(ref)
    cand_f = _flatten_float(cand)
    if ref_f.numel() != cand_f.numel():
        return {
            "dataset": dataset,
            "stage": stage,
            "status": "shape_mismatch",
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
        }
    diff = cand_f - ref_f
    return {
        "dataset": dataset,
        "stage": stage,
        "status": "ok",
        "numel": int(diff.numel()),
        "mae": float(diff.abs().mean().cpu().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).cpu().item()),
        "max_abs_error": float(diff.abs().max().cpu().item()),
        "mean_error": float(diff.mean().cpu().item()),
        "reference_mean": float(ref_f.mean().cpu().item()),
        "candidate_mean": float(cand_f.mean().cpu().item()),
        "reference_std": float(ref_f.std(unbiased=False).cpu().item()),
        "candidate_std": float(cand_f.std(unbiased=False).cpu().item()),
        "cosine": _cosine(ref, cand),
        "pearson": _corrcoef(ref, cand),
    }


def build_frontend(args: argparse.Namespace, profile: str, *, quantized: bool, device: torch.device) -> BitAccurateMFCCFakeQuantFrontend:
    n_fft = int(args.sample_rate * args.window_size_ms / 1000)
    hop_length = int(args.sample_rate * args.window_stride_ms / 1000)
    cfg = make_bit_accurate_mfcc_config(
        constraint_profile=profile,
        sample_rate=args.sample_rate,
        n_mfcc=40,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=40,
        f_min=20.0,
        f_max=float(args.sample_rate / 2),
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        log_pwl_num_segments=args.log_pwl_num_segments,
        log_pwl_strategy=args.log_pwl_strategy,
        log_pwl_gamma=args.log_pwl_gamma,
        log_offset=args.log_offset,
        log_input_clamp_min=args.log_input_clamp_min,
        stage_bit_overrides=args.stage_bit_overrides_normalized if quantized else None,
    )
    if not quantized:
        for spec in cfg.stage_quant.values():
            spec.enabled = False
    frontend = BitAccurateMFCCFakeQuantFrontend(config=cfg, observer_enabled=quantized).to(device)
    frontend.eval()
    return frontend


@torch.no_grad()
def analyze_dataset(args: argparse.Namespace, dataset: str, device: torch.device) -> list[dict[str, Any]]:
    ref = build_frontend(args, "upper_bound_s8_mfcc", quantized=False, device=device)
    cand = build_frontend(args, args.constraint_profile, quantized=True, device=device)
    loader = build_loader(
        args,
        dataset=dataset,
        split=args.split,
        is_training=False,
        noise_roots=args.noise_roots,
        noise_prob=args.noise_prob,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        deterministic_noise=True,
        random_seed=args.seed + 700000,
    )

    rows: list[dict[str, Any]] = []
    batches = 0
    for waveform, _labels in tqdm(loader, desc=f"stage-align:{dataset}", leave=False):
        waveform = waveform.to(device)
        ref_stages = ref.forward_stages(waveform)
        cand_stages = cand.forward_stages(waveform)
        for stage in sorted(set(ref_stages) & set(cand_stages)):
            rows.append(compare_tensors(dataset, stage, ref_stages[stage], cand_stages[stage]))
        batches += 1
        if args.batches > 0 and batches >= args.batches:
            break

    spec_path = Path(args.output_dir) / f"{dataset}_{args.constraint_profile}_stage_analysis_spec.json"
    cand.export_spec_json(
        spec_path,
        extra={
            "dataset": dataset,
            "split": args.split,
            "noise_roots": args.noise_roots,
            "noise_prob": args.noise_prob,
            "snr_min_db": args.snr_min_db,
            "snr_max_db": args.snr_max_db,
            "batches": batches,
            "stage_bit_overrides": stage_bit_overrides_json(args.stage_bit_overrides_normalized),
        },
    )
    stats_path = Path(args.output_dir) / f"{dataset}_{args.constraint_profile}_stage_stats.json"
    stats_path.write_text(json.dumps(cand.stage_stats, indent=2), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-level analysis for the new bit-accurate MFCC frontend.")
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--gpu", type=int, default=GPU)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split", choices=["train", "validation", "test"], default=SPLIT)
    parser.add_argument("--noise_roots", nargs="+", default=NOISE_ROOTS)
    parser.add_argument("--noise_prob", type=float, default=NOISE_PROB)
    parser.add_argument("--snr_min_db", type=float, default=SNR_MIN_DB)
    parser.add_argument("--snr_max_db", type=float, default=SNR_MAX_DB)
    parser.add_argument("--batches", type=int, default=BATCHES)
    parser.add_argument(
        "--constraint_profile",
        choices=[
            "upper_bound_s8_mfcc",
            "frontend_fakequant",
            "wide_bit_accurate",
            "hardware_baseline",
            "hardware_coeff_baseline",
        ],
        default=CONSTRAINT_PROFILE,
    )
    parser.add_argument("--reference_profile", default=REFERENCE_PROFILE)
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--window_size_ms", type=int, default=WINDOW_SIZE_MS)
    parser.add_argument("--window_stride_ms", type=int, default=WINDOW_STRIDE_MS)
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=PRE_EMPHASIS)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=PRE_EMPHASIS_COEFF)
    parser.add_argument("--log_pwl_num_segments", type=int, default=LOG_PWL_NUM_SEGMENTS)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default=LOG_PWL_STRATEGY)
    parser.add_argument("--log_pwl_gamma", type=float, default=LOG_PWL_GAMMA)
    parser.add_argument("--log_offset", type=float, default=LOG_OFFSET)
    parser.add_argument("--log_input_clamp_min", type=float, default=LOG_INPUT_CLAMP_MIN)
    parser.add_argument("--stage_bit_overrides", nargs="*", default=STAGE_BIT_OVERRIDES)
    parser.add_argument("--stage_bit_overrides_json", default=STAGE_BIT_OVERRIDES_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.stage_bit_overrides_normalized = load_stage_bit_overrides(
        override_items=args.stage_bit_overrides,
        override_json=args.stage_bit_overrides_json,
    )
    if args.stage_bit_overrides_normalized:
        print(f"[INFO] stage_bit_overrides={stage_bit_overrides_json(args.stage_bit_overrides_normalized)}")
    usable = count_usable_noise_files(args.noise_roots)
    print(f"[INFO] noise_roots={args.noise_roots}, usable_noise_files={usable}")
    if usable <= 0:
        raise FileNotFoundError(f"No usable wav files found for noise_roots={args.noise_roots}")
    datasets = args.datasets or ([args.dataset] if args.dataset else [])
    if not datasets:
        raise ValueError("Pass --dataset or --datasets.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.gpu > 0 and torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        all_rows.extend(analyze_dataset(args, dataset, device))
    out_csv = Path(args.output_dir) / f"{args.constraint_profile}_feature_alignment_summary.csv"
    save_csv(all_rows, out_csv)
    print(f"[DONE] rows={len(all_rows)}")


if __name__ == "__main__":
    main()
