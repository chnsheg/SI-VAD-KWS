from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dscnn_kws.quantization.bit_accurate_mfcc_bitwidth_utils import OUTPUT_ROOT


FEATURE_STATUS_CSV = str(Path(OUTPUT_ROOT) / "feature_screening" / "feature_candidate_status.csv")
OUTPUT_DIR = str(Path(OUTPUT_ROOT) / "single_stage_qat")
QAT_SCRIPT = "dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py"
INPUT_DIR = "/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models"
PATTERN = "*L5_C64*.pt"
ARCH = "L5_C64"
QAT_EPOCHS = 10
MAX_CONFIGS = 50
MAX_SINGLE_STAGE_CONFIGS = 40
MAX_PAIRWISE_CONFIGS = 10
PER_STAGE_GREEN = 2
PER_STAGE_YELLOW = 1


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
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


def parse_arch(arch: str) -> tuple[str, str]:
    text = arch.upper()
    if not text.startswith("L") or "_C" not in text:
        raise ValueError(f"Unsupported arch format: {arch}")
    layers, channels = text[1:].split("_C", 1)
    return layers, channels


def _score(row: dict[str, Any]) -> int:
    try:
        return int(float(row.get("compression_score", 0)))
    except Exception:
        return 0


def _status_priority(row: dict[str, Any]) -> int:
    return {"green": 0, "yellow": 1}.get(str(row.get("feature_status", "")), 9)


def select_single_stage(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("candidate_type") != "single_stage":
            continue
        if row.get("feature_status") not in {"green", "yellow"}:
            continue
        groups.setdefault(row["target_fields"], []).append(row)

    selected: list[dict[str, Any]] = []
    for target in sorted(groups):
        group = groups[target]
        green = sorted([row for row in group if row["feature_status"] == "green"], key=_score, reverse=True)
        yellow = sorted([row for row in group if row["feature_status"] == "yellow"], key=_score, reverse=True)
        picked = green[: args.per_stage_green] + yellow[: args.per_stage_yellow]
        if not picked and yellow:
            picked = yellow[:1]
        for row in picked:
            row = dict(row)
            row["selection_reason"] = f"{target}: {row['feature_status']} feature screening candidate"
            selected.append(row)

    selected = sorted(selected, key=lambda row: (row["target_fields"], _status_priority(row), -_score(row)))
    return selected[: args.max_single_stage_configs]


def select_pairwise(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = [
        dict(row)
        for row in rows
        if row.get("candidate_type") == "pairwise" and row.get("feature_status") in {"green", "yellow"}
    ]
    candidates = sorted(candidates, key=lambda row: (_status_priority(row), -_score(row), row["candidate_id"]))
    selected = candidates[: args.max_pairwise_configs]
    for row in selected:
        row["selection_reason"] = f"{row['target_fields']}: selected pairwise {row['feature_status']} candidate"
    return selected


def select_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_csv(args.feature_status_csv)
    selected = select_single_stage(rows, args) + select_pairwise(rows, args)
    selected = selected[: args.max_configs]
    for idx, row in enumerate(selected):
        row["qat_plan_id"] = idx
    return selected


def command_for_row(args: argparse.Namespace, row: dict[str, Any], override_json: Path) -> list[str]:
    layers, channels = parse_arch(args.arch)
    candidate_dir = Path(args.output_dir) / row["candidate_id"]
    cmd = [
        sys.executable,
        args.qat_script,
        "--constraint_profile",
        "hardware_coeff_baseline",
        "--stage_bit_overrides_json",
        str(override_json),
        "--bitwidth_sweep_id",
        row["candidate_id"],
        "--output_dir",
        str(candidate_dir / "models"),
        "--train_results_csv",
        str(candidate_dir / "train_results.csv"),
        "--grid_results_csv",
        str(candidate_dir / "grid_results.csv"),
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
    if args.root:
        cmd += ["--root", args.root]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def write_override_json(path: Path, row: dict[str, Any]) -> None:
    payload = {
        "candidate_id": row["candidate_id"],
        "candidate_type": row["candidate_type"],
        "target_fields": row["target_fields"],
        "feature_status": row["feature_status"],
        "stage_bit_overrides": json.loads(row["stage_bit_overrides"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_commands(args: argparse.Namespace, selected: list[dict[str, Any]]) -> Path:
    out_dir = Path(args.output_dir)
    commands: list[str] = []
    for row in selected:
        override_json = out_dir / "overrides" / f"{row['candidate_id']}.json"
        write_override_json(override_json, row)
        commands.append(" ".join(command_for_row(args, row, override_json)))

    commands_path = out_dir / "single_stage_qat_commands.txt"
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"[INFO] saved: {commands_path.resolve()}")
    return commands_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan full QAT candidates from bit-accurate MFCC feature screening.")
    parser.add_argument("--feature_status_csv", default=FEATURE_STATUS_CSV)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--qat_script", default=QAT_SCRIPT)
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--pattern", default=PATTERN)
    parser.add_argument("--root", default=None)
    parser.add_argument("--arch", default=ARCH)
    parser.add_argument("--qat_epochs", type=int, default=QAT_EPOCHS)
    parser.add_argument("--max_configs", type=int, default=MAX_CONFIGS)
    parser.add_argument("--max_single_stage_configs", type=int, default=MAX_SINGLE_STAGE_CONFIGS)
    parser.add_argument("--max_pairwise_configs", type=int, default=MAX_PAIRWISE_CONFIGS)
    parser.add_argument("--per_stage_green", type=int, default=PER_STAGE_GREEN)
    parser.add_argument("--per_stage_yellow", type=int, default=PER_STAGE_YELLOW)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = select_candidates(args)
    out_dir = Path(args.output_dir)
    write_csv(out_dir / "full_qat_candidate_plan.csv", selected)
    commands_path = write_commands(args, selected)
    print(f"[DONE] selected={len(selected)}, max_configs={args.max_configs}")
    if not args.run:
        print("[DONE] dry run only. Pass --run to execute commands.")
        return
    for line in commands_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        print(f"[RUN] {line}")
        subprocess.run(line.split(), check=True)


if __name__ == "__main__":
    main()
