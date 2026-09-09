from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

'''
python -m dscnn_kws.frontend.run_log_pwl_gamma_scan_seg4 --dataset speech_commands_v0.02_sr8k --sample_rate 8000 --epochs 50 --seed 42 --gammas 0.5 0.6 0.8 1.0 1.2 1.5 2.0 2.5 --extra_train_args "--num_workers 8"
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Scan gamma for 4-segment PWL log approximation")
    parser.add_argument("--python", default="python", type=str)
    parser.add_argument("--dataset", default="speech_commands_v0.02_sr8k", type=str)
    parser.add_argument("--sample_rate", default=8000, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5])
    parser.add_argument("--log_offset", default=1e-6, type=float)
    parser.add_argument("--extra_train_args", default="", type=str)
    parser.add_argument("--save_csv", default="dscnn_kws/frontend/artifacts/log_pwl_gamma_scan_seg4.csv", type=str)
    return parser.parse_args()


def _parse_test_metrics(stdout: str) -> dict[str, float]:
    m = re.search(
        r"\[TEST\]\s+loss=(?P<loss>[0-9.]+)\s+acc=(?P<acc>[0-9.]+)\s+precision=(?P<precision>[0-9.]+)\s+recall=(?P<recall>[0-9.]+)\s+f1=(?P<f1>[0-9.]+)",
        stdout,
    )
    if not m:
        raise RuntimeError("Failed to parse [TEST] metrics from training output")
    return {k: float(v) for k, v in m.groupdict().items()}


def _run(cmd: list[str]) -> str:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def main():
    args = parse_args()

    rows: list[dict[str, float | int]] = []
    for gamma in args.gammas:
        cmd = [
            args.python,
            "-m",
            "dscnn_kws.train",
            "--dataset",
            args.dataset,
            "--sample_rate",
            str(args.sample_rate),
            "--mfcc_impl",
            "torch",
            "--log_approx_mode",
            "pwl",
            "--log_pwl_num_segments",
            "4",
            "--log_pwl_strategy",
            "powerlaw",
            "--log_pwl_gamma",
            str(gamma),
            "--log_offset",
            str(args.log_offset),
            "--seed",
            str(args.seed),
            "--epoch",
            str(args.epochs),
        ]
        if args.extra_train_args.strip():
            cmd.extend(args.extra_train_args.strip().split())

        metrics = _parse_test_metrics(_run(cmd))
        rows.append({"gamma": float(gamma), **metrics})

    rows.sort(key=lambda r: float(r["acc"]), reverse=True)

    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["gamma", "loss", "acc", "precision", "recall", "f1"]
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved gamma scan results: {save_path}")
    if rows:
        best = rows[0]
        print(f"[INFO] Best gamma={best['gamma']}, acc={best['acc']:.4f}, f1={best['f1']:.4f}")


if __name__ == "__main__":
    main()
