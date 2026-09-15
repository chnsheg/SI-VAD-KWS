from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm

from dscnn_kws.frontend import BitAccurateMFCCFakeQuantFrontend, BitAccurateMFCCFrontend
from dscnn_kws.quantization.qat_bit_accurate_mfcc_accuracy_first import (
    BATCH,
    CALIBRATION_NOISE_PROB,
    CALIBRATION_SNR_MAX_DB,
    CALIBRATION_SNR_MIN_DB,
    DATASET,
    NUM_WORKERS,
    ROOT,
    SAMPLE_RATE,
    SEED,
)
from dscnn_kws.quantization.qat_int8_noise_snr_scene import build_loader, count_usable_noise_files, save_csv


OUTPUT_DIR = "./dscnn_kws/quantization/bit_accurate_mfcc_experiments"
DATASETS = None
SPLIT = "validation"
NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_valid.txt"]
NOISE_PROB = CALIBRATION_NOISE_PROB
SNR_MIN_DB = CALIBRATION_SNR_MIN_DB
SNR_MAX_DB = CALIBRATION_SNR_MAX_DB
BATCHES = 0
GPU = 0


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(torch.float32).flatten()


def feature_metrics(dataset: str, fake: torch.Tensor, hard: torch.Tensor) -> dict[str, Any]:
    a = flatten(fake)
    b = flatten(hard)
    diff = b - a
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = float((torch.dot(a, b) / denom).cpu().item()) if float(denom.item()) != 0.0 else float("nan")
    return {
        "dataset": dataset,
        "numel": int(diff.numel()),
        "mae": float(diff.abs().mean().cpu().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).cpu().item()),
        "max_abs_error": float(diff.abs().max().cpu().item()),
        "mean_error": float(diff.mean().cpu().item()),
        "fake_mean": float(a.mean().cpu().item()),
        "hard_mean": float(b.mean().cpu().item()),
        "fake_std": float(a.std(unbiased=False).cpu().item()),
        "hard_std": float(b.std(unbiased=False).cpu().item()),
        "cosine": cosine,
    }


@torch.no_grad()
def evaluate_dataset(args: argparse.Namespace, dataset: str, device: torch.device) -> list[dict[str, Any]]:
    fake = BitAccurateMFCCFakeQuantFrontend.from_spec_json(args.spec, observer_enabled=False).to(device).eval()
    hard = BitAccurateMFCCFrontend.from_spec_json(args.spec, observer_enabled=False).to(device).eval()
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
        random_seed=args.seed + 900000,
    )
    rows: list[dict[str, Any]] = []
    batches = 0
    for waveform, _labels in tqdm(loader, desc=f"integer-parity:{dataset}", leave=False):
        waveform = waveform.to(device)
        fake_out = fake(waveform)
        hard_out = hard(waveform)
        row = feature_metrics(dataset, fake_out, hard_out)
        row["batch_index"] = batches
        rows.append(row)
        batches += 1
        if args.batches > 0 and batches >= args.batches:
            break
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = ["mae", "rmse", "max_abs_error", "mean_error", "cosine"]
    summary: dict[str, Any] = {"batches": len(rows)}
    for key in keys:
        vals = torch.tensor([float(row[key]) for row in rows], dtype=torch.float32)
        summary[f"{key}_mean"] = float(vals.mean().item())
        summary[f"{key}_max"] = float(vals.max().item())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature parity check between fake-quant and hard bit-accurate MFCC frontends.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--split", choices=["train", "validation", "test"], default=SPLIT)
    parser.add_argument("--noise_roots", nargs="+", default=NOISE_ROOTS)
    parser.add_argument("--noise_prob", type=float, default=NOISE_PROB)
    parser.add_argument("--snr_min_db", type=float, default=SNR_MIN_DB)
    parser.add_argument("--snr_max_db", type=float, default=SNR_MAX_DB)
    parser.add_argument("--batches", type=int, default=BATCHES)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--gpu", type=int, default=GPU)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usable = count_usable_noise_files(args.noise_roots)
    print(f"[INFO] noise_roots={args.noise_roots}, usable_noise_files={usable}")
    if usable <= 0:
        raise FileNotFoundError(f"No usable wav files found for noise_roots={args.noise_roots}")
    datasets = args.datasets or ([args.dataset] if args.dataset else [])
    if not datasets:
        raise ValueError("Pass --dataset or --datasets.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.gpu > 0 and torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        rows.extend(evaluate_dataset(args, dataset, device))
    csv_path = out_dir / "integer_parity_summary.csv"
    save_csv(rows, csv_path)
    summary = summarize(rows)
    summary_path = out_dir / "integer_parity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] rows={len(rows)}, summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()

