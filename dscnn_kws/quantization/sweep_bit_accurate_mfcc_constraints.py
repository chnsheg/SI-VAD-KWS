from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_DIR = "./dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/constraint_sweeps"
QAT_SCRIPT = "dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py"
BASE_PROFILE = "upper_bound_s8_mfcc"
SWEEP = ["constraint_profile"]
ARCH = "L5_C64"
DATASETS = ["mobvoi_hi_xiaowen_binary_hardneg", "mobvoi_nihao_wenwen_binary_hardneg"]
RUN = False


def build_sweep_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles = [
        "upper_bound_s8_mfcc",
        "frontend_fakequant",
        "wide_bit_accurate",
        "hardware_baseline",
        "hardware_coeff_baseline",
    ]
    for idx, profile in enumerate(profiles):
        rows.append(
            {
                "sweep_id": idx,
                "arch": args.arch,
                "datasets": " ".join(args.datasets),
                "constraint_profile": profile,
                "output_dir": str(Path(args.output_dir) / profile),
                "notes": _profile_notes(profile),
            }
        )
    return rows


def _profile_notes(profile: str) -> str:
    if profile == "upper_bound_s8_mfcc":
        return "rectangular Mel + PWL log + S8 MFCC output; intermediate quant disabled"
    if profile == "frontend_fakequant":
        return "enable PWL input, log_mel, DCT, and MFCC fake quant"
    if profile == "wide_bit_accurate":
        return "same enabled stages as frontend_fakequant; keeps wide default widths"
    if profile == "hardware_baseline":
        return "enable all baseline stage quantizers"
    if profile == "hardware_coeff_baseline":
        return "enable baseline stage quantizers plus U16 Hann, S18 windowed, S16 DFT/twiddle, S20 FFT data, and S8 DCT coeff"
    return ""


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
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
    print(f"[INFO] sweep csv saved to: {path.resolve()}")


def command_for_row(args: argparse.Namespace, row: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        args.qat_script,
        "--constraint_profile",
        row["constraint_profile"],
        "--output_dir",
        row["output_dir"],
        "--train_results_csv",
        str(Path(row["output_dir"]) / "train_results.csv"),
        "--grid_results_csv",
        str(Path(row["output_dir"]) / "grid_results.csv"),
        "--layers",
        args.layers,
        "--channels",
        args.channels,
    ]
    if args.input_dir:
        cmd += ["--input_dir", args.input_dir]
    if args.pattern:
        cmd += ["--pattern", args.pattern]
    if args.checkpoints:
        cmd += ["--checkpoints", *args.checkpoints]
    if args.root:
        cmd += ["--root", args.root]
    if args.qat_epochs is not None:
        cmd += ["--qat_epochs", str(args.qat_epochs)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def parse_arch(arch: str) -> tuple[str, str]:
    text = arch.upper()
    if not text.startswith("L") or "_C" not in text:
        raise ValueError(f"Unsupported arch format: {arch}")
    layers, channels = text[1:].split("_C", 1)
    return layers, channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run progressive constraint sweeps for bit-accurate MFCC QAT.")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--qat_script", default=QAT_SCRIPT)
    parser.add_argument("--base_profile", default=BASE_PROFILE)
    parser.add_argument("--sweep", nargs="+", default=SWEEP)
    parser.add_argument("--arch", default=ARCH)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=RUN)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--qat_epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers, channels = parse_arch(args.arch)
    args.layers = layers
    args.channels = channels
    rows = build_sweep_rows(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = out_dir / "constraint_sweep_plan.csv"
    write_csv(rows, sweep_csv)

    commands_path = out_dir / "constraint_sweep_commands.txt"
    commands = [" ".join(command_for_row(args, row)) for row in rows]
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"[INFO] commands saved to: {commands_path.resolve()}")

    if not args.run:
        print("[DONE] dry run only. Pass --run to execute the QAT commands.")
        return

    for row in rows:
        cmd = command_for_row(args, row)
        print("[RUN] " + " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
