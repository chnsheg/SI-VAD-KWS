from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run log-PWL experiment grid and summarize results")
    parser.add_argument("--python", default="python", type=str)
    parser.add_argument("--dataset", default="speech_commands_v0.02_sr8k", type=str)
    parser.add_argument("--sample_rate", default=8000, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--segments", nargs="+", type=int, default=[2, 4, 6, 8, 12])
    parser.add_argument("--strategies", nargs="+", type=str, default=["uniform_logx", "quantile"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--extra_train_args", default="", type=str)
    parser.add_argument("--save_csv", default="dscnn_kws/frontend/artifacts/log_pwl_grid_results.csv", type=str)
    return parser.parse_args()


def _parse_test_metrics(stdout: str) -> dict[str, float]:
    # line format: [TEST] loss=... acc=... precision=... recall=... f1=...
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
    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []

    # Baseline: exact log (torch mfcc)
    for seed in args.seeds:
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
            "exact",
            "--seed",
            str(seed),
            "--epoch",
            str(args.epochs),
        ]
        if args.extra_train_args.strip():
            cmd.extend(args.extra_train_args.strip().split())
        metrics = _parse_test_metrics(_run(cmd))
        rows.append(
            {
                "mode": "exact",
                "strategy": "-",
                "segments": 0,
                "seed": seed,
                **metrics,
            }
        )

    for strategy in args.strategies:
        for seg in args.segments:
            for seed in args.seeds:
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
                    "--log_pwl_strategy",
                    strategy,
                    "--log_pwl_num_segments",
                    str(seg),
                    "--seed",
                    str(seed),
                    "--epoch",
                    str(args.epochs),
                ]
                if args.extra_train_args.strip():
                    cmd.extend(args.extra_train_args.strip().split())
                metrics = _parse_test_metrics(_run(cmd))
                rows.append(
                    {
                        "mode": "pwl",
                        "strategy": strategy,
                        "segments": seg,
                        "seed": seed,
                        **metrics,
                    }
                )

    fieldnames = ["mode", "strategy", "segments", "seed", "loss", "acc", "precision", "recall", "f1"]
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved grid results: {save_path}")


if __name__ == "__main__":
    main()
