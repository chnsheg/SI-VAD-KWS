from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dscnn_kws.quantization.bit_accurate_mfcc_bitwidth_utils import (
    OUTPUT_ROOT,
    STAGE_BIT_SWEEP_VALUES,
)


QAT_PLAN_CSV = str(Path(OUTPUT_ROOT) / "single_stage_qat" / "full_qat_candidate_plan.csv")
QAT_ROOT = str(Path(OUTPUT_ROOT) / "single_stage_qat")
OUTPUT_DIR = str(Path(OUTPUT_ROOT) / "combined_candidates")
BASELINE_GRID_CSV = "./dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_grid_results.csv"
BASELINE_TRAIN_CSV = "./dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_train_results.csv"
QAT_SCRIPT = "dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py"
INPUT_DIR = "/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models"
PATTERN = "*L5_C64*.pt"
ARCH = "L5_C64"
QAT_EPOCHS = 10


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
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


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def _grid_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("arch", "")),
        str(row.get("scene", "")),
        str(row.get("snr_db", "")),
    )


def compare_grid(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    low_snr_max_db: float,
) -> dict[str, float]:
    baseline_by_key = {_grid_key(row): row for row in baseline_rows}
    candidate_by_key = {_grid_key(row): row for row in candidate_rows}
    common_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    if not common_keys:
        return {
            "common_grid_rows": 0,
            "avg_acc": float("nan"),
            "baseline_avg_acc": float("nan"),
            "avg_acc_drop_pp": float("nan"),
            "low_snr_acc": float("nan"),
            "baseline_low_snr_acc": float("nan"),
            "low_snr_acc_drop_pp": float("nan"),
        }

    base_acc = [_float(baseline_by_key[key].get("acc")) for key in common_keys]
    cand_acc = [_float(candidate_by_key[key].get("acc")) for key in common_keys]
    low_keys = [key for key in common_keys if _float(key[3]) <= low_snr_max_db]
    base_low = [_float(baseline_by_key[key].get("acc")) for key in low_keys]
    cand_low = [_float(candidate_by_key[key].get("acc")) for key in low_keys]

    baseline_avg = _mean(base_acc)
    candidate_avg = _mean(cand_acc)
    baseline_low = _mean(base_low)
    candidate_low = _mean(cand_low)
    return {
        "common_grid_rows": float(len(common_keys)),
        "avg_acc": candidate_avg,
        "baseline_avg_acc": baseline_avg,
        "avg_acc_drop_pp": (baseline_avg - candidate_avg) * 100.0,
        "low_snr_acc": candidate_low,
        "baseline_low_snr_acc": baseline_low,
        "low_snr_acc_drop_pp": (baseline_low - candidate_low) * 100.0,
    }


def fake_hard_gap_pp(train_rows: list[dict[str, Any]]) -> float:
    gaps: list[float] = []
    for row in train_rows:
        fake = _float(row.get("quantized_backbone_fakequant_frontend_test_tau_list_acc"))
        hard = _float(row.get("bit_accurate_mfcc_int8_backbone_test_tau_list_acc"))
        if math.isfinite(fake) and math.isfinite(hard):
            gaps.append(abs(fake - hard) * 100.0)
    return _mean(gaps)


def pass_status(args: argparse.Namespace, row: dict[str, Any]) -> str:
    avg_drop = _float(row.get("avg_acc_drop_pp"))
    low_drop = _float(row.get("low_snr_acc_drop_pp"))
    gap = _float(row.get("fake_hard_gap_pp"))
    if avg_drop <= args.pass_avg_drop_pp and low_drop <= args.pass_low_snr_drop_pp and gap <= args.pass_fake_hard_gap_pp:
        return "pass"
    if avg_drop <= args.borderline_avg_drop_pp and low_drop <= args.borderline_low_snr_drop_pp:
        return "borderline"
    return "fail"


def summarize_qat(args: argparse.Namespace) -> list[dict[str, Any]]:
    plan_rows = read_csv(args.qat_plan_csv)
    baseline_grid = read_csv(args.baseline_grid_csv)
    decisions: list[dict[str, Any]] = []
    for plan in plan_rows:
        candidate_id = plan["candidate_id"]
        candidate_dir = Path(args.qat_root) / candidate_id
        grid_rows = read_csv(candidate_dir / "grid_results.csv")
        train_rows = read_csv(candidate_dir / "train_results.csv")
        if not grid_rows:
            decisions.append({**plan, "qat_status": "missing_grid_results"})
            continue
        metrics = compare_grid(
            baseline_rows=baseline_grid,
            candidate_rows=grid_rows,
            low_snr_max_db=args.low_snr_max_db,
        )
        row = {
            **plan,
            **metrics,
            "fake_hard_gap_pp": fake_hard_gap_pp(train_rows),
            "train_results_csv": str((candidate_dir / "train_results.csv").resolve()),
            "grid_results_csv": str((candidate_dir / "grid_results.csv").resolve()),
        }
        row["qat_status"] = pass_status(args, row)
        decisions.append(row)
    return decisions


def _lowest_bits_from_decisions(
    decisions: list[dict[str, Any]],
    *,
    accepted_statuses: set[str],
) -> dict[str, int]:
    selected: dict[str, int] = {field: values[0] for field, values in STAGE_BIT_SWEEP_VALUES.items()}
    for field in STAGE_BIT_SWEEP_VALUES:
        candidates: list[tuple[int, dict[str, Any]]] = []
        for row in decisions:
            if row.get("candidate_type") != "single_stage":
                continue
            if row.get("target_fields") != field:
                continue
            if row.get("qat_status") not in accepted_statuses:
                continue
            overrides = json.loads(row["stage_bit_overrides"])
            if field in overrides:
                candidates.append((int(overrides[field]), row))
        if candidates:
            selected[field] = min(bits for bits, _row in candidates)
    return selected


def write_strategy_json(path: Path, *, name: str, stage_bits: dict[str, int], accepted_statuses: list[str]) -> None:
    payload = {
        "strategy": name,
        "source": "bit_accurate_mfcc_bitwidth_sweep",
        "accepted_single_stage_statuses": accepted_statuses,
        "stage_bit_overrides": stage_bits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] saved: {path.resolve()}")


def parse_arch(arch: str) -> tuple[str, str]:
    text = arch.upper()
    if not text.startswith("L") or "_C" not in text:
        raise ValueError(f"Unsupported arch format: {arch}")
    layers, channels = text[1:].split("_C", 1)
    return layers, channels


def combined_command(args: argparse.Namespace, *, strategy: str, override_json: Path) -> list[str]:
    layers, channels = parse_arch(args.arch)
    out_dir = Path(args.output_dir) / strategy
    return [
        sys.executable,
        args.qat_script,
        "--constraint_profile",
        "hardware_coeff_baseline",
        "--stage_bit_overrides_json",
        str(override_json),
        "--bitwidth_sweep_id",
        strategy,
        "--output_dir",
        str(out_dir / "models"),
        "--train_results_csv",
        str(out_dir / "train_results.csv"),
        "--grid_results_csv",
        str(out_dir / "grid_results.csv"),
        "--layers",
        layers,
        "--channels",
        channels,
        "--input_dir",
        args.input_dir,
        "--pattern",
        args.pattern,
        "--qat_epochs",
        str(args.qat_epochs),
    ]


def write_combined_commands(args: argparse.Namespace, strategy_paths: dict[str, Path]) -> None:
    commands = [" ".join(combined_command(args, strategy=name, override_json=path)) for name, path in strategy_paths.items()]
    path = Path(args.output_dir) / "combined_qat_commands.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"[INFO] saved: {path.resolve()}")


def write_strategy_summary(path: Path, strategies: dict[str, dict[str, int]]) -> None:
    rows: list[dict[str, Any]] = []
    for name, stage_bits in strategies.items():
        rows.append({"strategy": name, **stage_bits})
    write_csv(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize bit-accurate MFCC bit-width QAT results and emit combined configs.")
    parser.add_argument("--qat_plan_csv", default=QAT_PLAN_CSV)
    parser.add_argument("--qat_root", default=QAT_ROOT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--baseline_grid_csv", default=BASELINE_GRID_CSV)
    parser.add_argument("--baseline_train_csv", default=BASELINE_TRAIN_CSV)
    parser.add_argument("--qat_script", default=QAT_SCRIPT)
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--pattern", default=PATTERN)
    parser.add_argument("--arch", default=ARCH)
    parser.add_argument("--qat_epochs", type=int, default=QAT_EPOCHS)
    parser.add_argument("--low_snr_max_db", type=float, default=0.0)
    parser.add_argument("--pass_avg_drop_pp", type=float, default=0.3)
    parser.add_argument("--pass_low_snr_drop_pp", type=float, default=1.0)
    parser.add_argument("--pass_fake_hard_gap_pp", type=float, default=0.2)
    parser.add_argument("--borderline_avg_drop_pp", type=float, default=0.5)
    parser.add_argument("--borderline_low_snr_drop_pp", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = summarize_qat(args)
    decisions_csv = out_dir / "single_stage_qat_decision.csv"
    write_csv(decisions_csv, decisions)

    accuracy_first = _lowest_bits_from_decisions(decisions, accepted_statuses={"pass"})
    aggressive = _lowest_bits_from_decisions(decisions, accepted_statuses={"pass", "borderline"})

    acc_path = out_dir / "accuracy_first_recommended.json"
    agg_path = out_dir / "aggressive_candidate.json"
    write_strategy_json(acc_path, name="accuracy_first_recommended", stage_bits=accuracy_first, accepted_statuses=["pass"])
    write_strategy_json(agg_path, name="aggressive_candidate", stage_bits=aggressive, accepted_statuses=["pass", "borderline"])
    write_strategy_summary(out_dir / "final_bitwidth_strategy_summary.csv", {
        "accuracy_first_recommended": accuracy_first,
        "aggressive_candidate": aggressive,
    })
    write_combined_commands(args, {
        "accuracy_first_recommended": acc_path,
        "aggressive_candidate": agg_path,
    })
    print("[DONE] summary complete. Chinese final report generation is intentionally not implemented.")


if __name__ == "__main__":
    main()
