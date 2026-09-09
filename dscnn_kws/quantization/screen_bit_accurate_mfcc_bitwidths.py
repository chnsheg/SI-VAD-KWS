from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm

from dscnn_kws.frontend import BitAccurateMFCCFakeQuantFrontend, make_bit_accurate_mfcc_config
from dscnn_kws.quantization.bit_accurate_mfcc_bitwidth_utils import (
    OUTPUT_ROOT,
    STAGE_BIT_SWEEP_VALUES,
    build_pairwise_screening_rows,
    build_single_stage_screening_rows,
    parse_overrides_json_cell,
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


OUTPUT_DIR = str(Path(OUTPUT_ROOT) / "feature_screening")
DATASETS = ["mobvoi_hi_xiaowen_binary_hardneg", "mobvoi_nihao_wenwen_binary_hardneg"]
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
CONSTRAINT_PROFILE = "hardware_coeff_baseline"
REFERENCE_PROFILE = "hardware_coeff_baseline"


STAGE_TO_QUANT_KEY = {
    "pcm_or_scaled_sample": "pcm",
    "preemphasis_or_sample_scale": "preemphasis",
    "windowed": "windowed",
    "fft_complex": "fft_data",
    "power": "power",
    "rectangular_mel": "mel",
    "pwl_log_mel": "log_mel",
    "dct_acc": "dct",
    "mfcc_prequant": "dct",
    "mfcc_int8": "mfcc",
    "hann_coeff": "hann_coeff",
    "twiddle_coeff": "twiddle_coeff",
    "dct_coeff": "dct_coeff",
}


def _flatten_float(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        x = torch.view_as_real(x)
    return x.detach().to(torch.float32).flatten()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = _flatten_float(a)
    b = _flatten_float(b)
    if a.numel() != b.numel():
        return float("nan")
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((torch.dot(a, b) / denom).cpu().item())


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def compare_tensors(
    *,
    candidate: dict[str, Any],
    dataset: str,
    measured_stage: str,
    ref: torch.Tensor,
    cand: torch.Tensor,
    cand_stage_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ref_f = _flatten_float(ref)
    cand_f = _flatten_float(cand)
    row = {
        "candidate_id": candidate["candidate_id"],
        "candidate_type": candidate["candidate_type"],
        "target_fields": candidate["target_fields"],
        "target_stages": candidate["target_stages"],
        "stage_bit_overrides": candidate["stage_bit_overrides"],
        "dataset": dataset,
        "measured_stage": measured_stage,
        "status": "ok",
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
    }
    if ref_f.numel() != cand_f.numel():
        row["status"] = "shape_mismatch"
        return row

    diff = cand_f - ref_f
    rmse = torch.sqrt(torch.mean(diff * diff))
    ref_rms = torch.sqrt(torch.mean(ref_f * ref_f))
    row.update(
        {
            "numel": int(diff.numel()),
            "mae": float(diff.abs().mean().cpu().item()),
            "rmse": float(rmse.cpu().item()),
            "relative_rmse": float((rmse / torch.clamp(ref_rms, min=1e-12)).cpu().item()),
            "max_abs_error": float(diff.abs().max().cpu().item()),
            "mean_error": float(diff.mean().cpu().item()),
            "reference_mean": float(ref_f.mean().cpu().item()),
            "candidate_mean": float(cand_f.mean().cpu().item()),
            "reference_std": float(ref_f.std(unbiased=False).cpu().item()),
            "candidate_std": float(cand_f.std(unbiased=False).cpu().item()),
            "reference_min": float(ref_f.min().cpu().item()),
            "reference_max": float(ref_f.max().cpu().item()),
            "candidate_min": float(cand_f.min().cpu().item()),
            "candidate_max": float(cand_f.max().cpu().item()),
            "cosine": _cosine(ref, cand),
            "pearson": _pearson(ref, cand),
        }
    )
    if cand_stage_stats:
        row.update(
            {
                "stage_stat_min": cand_stage_stats.get("min"),
                "stage_stat_max": cand_stage_stats.get("max"),
                "stage_stat_mean": cand_stage_stats.get("mean"),
                "stage_stat_std": cand_stage_stats.get("std"),
                "zero_ratio": cand_stage_stats.get("zero_ratio"),
                "saturation_ratio": cand_stage_stats.get("saturation_ratio"),
            }
        )
    return row


def build_frontend(
    args: argparse.Namespace,
    *,
    profile: str,
    stage_bit_overrides: dict[str, int] | None,
    device: torch.device,
) -> BitAccurateMFCCFakeQuantFrontend:
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
        stage_bit_overrides=stage_bit_overrides,
    )
    frontend = BitAccurateMFCCFakeQuantFrontend(config=cfg, observer_enabled=True).to(device)
    frontend.eval()
    return frontend


def coefficient_rows(
    *,
    candidate: dict[str, Any],
    ref: BitAccurateMFCCFakeQuantFrontend,
    cand: BitAccurateMFCCFakeQuantFrontend,
) -> list[dict[str, Any]]:
    refs = {
        "hann_coeff": ref.hann_coeff_qdq,
        "twiddle_coeff": torch.cat([ref.dft_cos_qdq.reshape(-1), ref.dft_sin_qdq.reshape(-1)]),
        "dct_coeff": ref.dct_mat_qdq,
    }
    cands = {
        "hann_coeff": cand.hann_coeff_qdq,
        "twiddle_coeff": torch.cat([cand.dft_cos_qdq.reshape(-1), cand.dft_sin_qdq.reshape(-1)]),
        "dct_coeff": cand.dct_mat_qdq,
    }
    rows: list[dict[str, Any]] = []
    for stage, ref_tensor in refs.items():
        rows.append(
            compare_tensors(
                candidate=candidate,
                dataset="coefficients",
                measured_stage=stage,
                ref=ref_tensor,
                cand=cands[stage],
                cand_stage_stats=None,
            )
        )
    return rows


def build_screening_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = build_single_stage_screening_rows()
    if args.include_pairwise:
        rows.extend(build_pairwise_screening_rows())
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
    print(f"[INFO] saved: {path.resolve()}")


@torch.no_grad()
def run_candidate(args: argparse.Namespace, candidate: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    overrides = parse_overrides_json_cell(candidate["stage_bit_overrides"])
    ref = build_frontend(args, profile=args.reference_profile, stage_bit_overrides=None, device=device)
    cand = build_frontend(args, profile=args.constraint_profile, stage_bit_overrides=overrides, device=device)
    rows = coefficient_rows(candidate=candidate, ref=ref, cand=cand)

    for dataset in args.datasets:
        ref.reset_observers()
        cand.reset_observers()
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
            random_seed=args.seed + 710000,
        )
        batches = 0
        desc = f"feature:{candidate['candidate_id']}:{dataset}"
        for waveform, _labels in tqdm(loader, desc=desc, leave=False):
            waveform = waveform.to(device)
            ref_stages = ref.forward_stages(waveform)
            cand_stages = cand.forward_stages(waveform)
            for stage in sorted(set(ref_stages) & set(cand_stages)):
                quant_key = STAGE_TO_QUANT_KEY.get(stage)
                rows.append(
                    compare_tensors(
                        candidate=candidate,
                        dataset=dataset,
                        measured_stage=stage,
                        ref=ref_stages[stage],
                        cand=cand_stages[stage],
                        cand_stage_stats=cand.stage_stats.get(quant_key, {}) if quant_key else {},
                    )
                )
            batches += 1
            if args.batches > 0 and batches >= args.batches:
                break
    return rows


def summarize_stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[(row["candidate_id"], row["measured_stage"])].append(row)

    summary: list[dict[str, Any]] = []
    numeric_fields = [
        "mae",
        "rmse",
        "relative_rmse",
        "max_abs_error",
        "mean_error",
        "reference_mean",
        "candidate_mean",
        "reference_std",
        "candidate_std",
        "cosine",
        "pearson",
        "zero_ratio",
        "saturation_ratio",
    ]
    for (_candidate_id, _stage), group in grouped.items():
        first = group[0]
        out = {
            "candidate_id": first["candidate_id"],
            "candidate_type": first["candidate_type"],
            "target_fields": first["target_fields"],
            "target_stages": first["target_stages"],
            "stage_bit_overrides": first["stage_bit_overrides"],
            "measured_stage": first["measured_stage"],
            "num_rows": len(group),
        }
        for field in numeric_fields:
            out[field] = _mean([_safe_float(row.get(field)) for row in group])
        summary.append(out)
    return summary


def _compression_score(stage_bit_overrides: str) -> int:
    overrides = json.loads(stage_bit_overrides)
    score = 0
    for field, bits in overrides.items():
        values = STAGE_BIT_SWEEP_VALUES.get(field)
        if values:
            score += int(values[0]) - int(bits)
    return score


def candidate_status_rows(args: argparse.Namespace, stage_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_summary:
        by_candidate[row["candidate_id"]].append(row)

    status_rows: list[dict[str, Any]] = []
    for candidate_id, group in by_candidate.items():
        first = group[0]
        mfcc_rows = [row for row in group if row["measured_stage"] == "mfcc_int8"]
        key_rows = mfcc_rows or group
        mfcc_cosine = min((_safe_float(row.get("cosine")) for row in key_rows), default=float("nan"))
        mfcc_relative_rmse = max((_safe_float(row.get("relative_rmse")) for row in key_rows), default=float("nan"))
        downstream_relative_rmse = max((_safe_float(row.get("relative_rmse")) for row in group), default=float("nan"))
        max_saturation = max((_safe_float(row.get("saturation_ratio")) for row in group), default=float("nan"))

        if mfcc_cosine >= args.green_mfcc_cosine and mfcc_relative_rmse <= args.green_mfcc_relative_rmse:
            feature_status = "green"
        elif mfcc_cosine >= args.yellow_mfcc_cosine and mfcc_relative_rmse <= args.yellow_mfcc_relative_rmse:
            feature_status = "yellow"
        else:
            feature_status = "red"

        status_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": first["candidate_type"],
                "target_fields": first["target_fields"],
                "target_stages": first["target_stages"],
                "stage_bit_overrides": first["stage_bit_overrides"],
                "compression_score": _compression_score(first["stage_bit_overrides"]),
                "feature_status": feature_status,
                "mfcc_cosine": mfcc_cosine,
                "mfcc_relative_rmse": mfcc_relative_rmse,
                "downstream_max_relative_rmse": downstream_relative_rmse,
                "max_saturation_ratio": max_saturation,
                "notes": first.get("notes", ""),
            }
        )
    return sorted(
        status_rows,
        key=lambda row: (
            {"green": 0, "yellow": 1, "red": 2}.get(row["feature_status"], 3),
            row["candidate_type"],
            row["target_fields"],
            -int(row["compression_score"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large feature screening for bit-accurate MFCC stage bit widths.")
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
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
    parser.add_argument("--constraint_profile", default=CONSTRAINT_PROFILE)
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
    parser.add_argument("--include_pairwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--green_mfcc_cosine", type=float, default=0.999)
    parser.add_argument("--yellow_mfcc_cosine", type=float, default=0.995)
    parser.add_argument("--green_mfcc_relative_rmse", type=float, default=0.02)
    parser.add_argument("--yellow_mfcc_relative_rmse", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usable = count_usable_noise_files(args.noise_roots)
    print(f"[INFO] noise_roots={args.noise_roots}, usable_noise_files={usable}")
    if usable <= 0:
        raise FileNotFoundError(f"No usable wav files found for noise_roots={args.noise_roots}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_rows = build_screening_plan(args)
    plan_csv = out_dir / "bitwidth_feature_screening_plan.csv"
    write_csv_rows(plan_csv, plan_rows)
    if args.plan_only:
        print("[DONE] plan only")
        return

    device = torch.device("cuda" if args.gpu > 0 and torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, Any]] = []
    for candidate in plan_rows:
        all_rows.extend(run_candidate(args, candidate, device))
        save_csv(all_rows, out_dir / "stage_feature_loss_raw.csv")

    stage_summary = summarize_stage_rows(all_rows)
    save_csv(stage_summary, out_dir / "stage_feature_loss_summary.csv")
    status = candidate_status_rows(args, stage_summary)
    save_csv(status, out_dir / "feature_candidate_status.csv")
    print(f"[DONE] candidates={len(plan_rows)}, raw_rows={len(all_rows)}, status_rows={len(status)}")


if __name__ == "__main__":
    main()
