from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = [
    "mobvoi_hi_xiaowen_binary_hardneg",
    "mobvoi_nihao_wenwen_binary_hardneg",
]

CRNN_CONFIGS = [
    ("C24x5_H64", "24,24,24,24,24", 64),
    ("C32x5_H48", "32,32,32,32,32", 48),
    ("C24x5_H48", "24,24,24,24,24", 48),
    ("C16x5_H64", "16,16,16,16,16", 64),
    ("C32_32_48_H48", "32,32,48", 48),
]

NUM_CLASSES = 2
SAMPLE_RATE = 16000
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
KERNEL_TIME = 5
KERNEL_FREQ = 3

RESULTS_DIR = Path("dscnn_kws") / "streaming" / "results"
RUNS_DIR = Path("dscnn_kws") / "runs" / "streaming"


def parse_cnn_channels(raw: str) -> list[int]:
    values = [int(item) for item in raw.replace(",", " ").split() if item.strip()]
    if not values:
        raise ValueError("cnn_channels must contain at least one integer")
    return values


def expected_params(cnn_channels: str, gru_hidden: int, labels: int = NUM_CLASSES) -> int:
    channels = parse_cnn_channels(cnn_channels)
    kernel = KERNEL_TIME * KERNEL_FREQ
    total = 0
    prev = 1
    for out in channels:
        total += prev * kernel
        total += prev * out
        total += 2 * prev + 2 * out
        prev = out
    total += 3 * gru_hidden * channels[-1]
    total += 3 * gru_hidden * gru_hidden
    total += 2 * 3 * gru_hidden
    total += gru_hidden * labels + labels
    return total


def expected_macs_window(cnn_channels: str, gru_hidden: int, labels: int = NUM_CLASSES) -> int:
    channels = parse_cnn_channels(cnn_channels)
    hop = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)
    time_steps = (SAMPLE_RATE + hop - 1) // hop
    freq_bins = DCT_COEFF
    kernel = KERNEL_TIME * KERNEL_FREQ
    total = 0
    prev = 1
    for out in channels:
        total += time_steps * freq_bins * (prev * kernel + prev * out)
        prev = out
    total += time_steps * 3 * (gru_hidden * channels[-1] + gru_hidden * gru_hidden)
    total += gru_hidden * labels
    return total


def run_command(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(1)
        if chunk == "" and proc.poll() is not None:
            break
        if not chunk:
            continue
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    proc.wait()
    return proc.returncode, "".join(chunks)


def parse_metrics(output: str) -> dict[str, float | int | None]:
    printed_params = None
    best_valid_acc = None
    test_loss = None
    test_acc = None
    precision = None
    recall = None
    f1 = None

    m = re.search(r"params=(\d+)", output)
    if m:
        printed_params = int(m.group(1))

    valid_accs = [float(x) for x in re.findall(r"valid_loss\s+[0-9.]+\s+acc\s+([0-9.]+)", output)]
    if valid_accs:
        best_valid_acc = max(valid_accs)

    m = re.search(
        r"\[TEST\]\s+loss=([0-9.]+)\s+acc=([0-9.]+)\s+precision=([0-9.]+)\s+recall=([0-9.]+)\s+f1=([0-9.]+)",
        output,
    )
    if m:
        test_loss = float(m.group(1))
        test_acc = float(m.group(2))
        precision = float(m.group(3))
        recall = float(m.group(4))
        f1 = float(m.group(5))

    return {
        "printed_params": printed_params,
        "best_valid_acc": best_valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def copy_best_model(output: str, args: argparse.Namespace, dataset: str, arch: str, cnn_channels: str, gru_hidden: int):
    m = re.search(r"\[INFO\]\s+save_dir=(.+)", output)
    if not m:
        return None, None

    save_dir = Path(m.group(1).strip())
    best_path = save_dir / "best.pt"
    if not best_path.exists():
        return str(save_dir), None

    best_dir = RUNS_DIR / f"{args.out_prefix}_best_models"
    best_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
    safe_channels = cnn_channels.replace(",", "-")
    out_path = best_dir / (
        f"{safe_dataset}_{arch}_ch{safe_channels}_gru{gru_hidden}_"
        f"params{expected_params(cnn_channels, gru_hidden)}_clean_best.pt"
    )
    shutil.copy2(best_path, out_path)
    return str(save_dir), str(out_path)


def train_one(args: argparse.Namespace, dataset: str, arch: str, cnn_channels: str, gru_hidden: int) -> dict:
    save_root = RUNS_DIR / f"{args.out_prefix}_train_runs"
    cmd = [
        sys.executable,
        "-m",
        "dscnn_kws.streaming.train_streaming_crnn",
        "--root",
        args.root,
        "--dataset",
        dataset,
        "--epoch",
        str(args.epoch),
        "--batch",
        str(args.batch),
        "--gpu",
        str(args.gpu),
        "--num_workers",
        str(args.num_workers),
        "--sample_rate",
        str(SAMPLE_RATE),
        "--frontend",
        "mfcc",
        "--streaming_mfcc",
        "--dct_coeff",
        str(DCT_COEFF),
        "--window_size_ms",
        str(WINDOW_SIZE_MS),
        "--window_stride_ms",
        str(WINDOW_STRIDE_MS),
        "--cnn_channels",
        cnn_channels,
        "--kernel_time",
        str(KERNEL_TIME),
        "--kernel_freq",
        str(KERNEL_FREQ),
        "--gru_hidden",
        str(gru_hidden),
        "--allow_online_resample",
        "--no-verify_sample_rate",
        "--no-noise_aug",
        "--no-eval_noise_aug",
        "--save_root",
        str(save_root),
    ]

    print("\n" + "=" * 100)
    print(f"[TRAIN] clean | dataset={dataset} | arch={arch} | channels={cnn_channels} | gru={gru_hidden}")
    print("[CMD]", " ".join(cmd))
    returncode, output = run_command(cmd)

    metrics = parse_metrics(output)
    train_save_dir, best_model = copy_best_model(output, args, dataset, arch, cnn_channels, gru_hidden)
    if best_model:
        print(f"[INFO] best model copied to: {best_model}")
    else:
        print("[WARN] best model was not copied; save_dir/best.pt was not found")

    macs_window = expected_macs_window(cnn_channels, gru_hidden)
    return {
        "dataset": dataset,
        "arch": arch,
        "cnn_channels": cnn_channels,
        "gru_hidden": gru_hidden,
        "expected_params": expected_params(cnn_channels, gru_hidden),
        "expected_macs_window": macs_window,
        "expected_macs_per_frame": macs_window / 32.0,
        **metrics,
        "returncode": returncode,
        "train_save_dir": train_save_dir,
        "best_model_saved_as": best_model,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved to: {path.resolve()}")


def save_results(args: argparse.Namespace, rows: list[dict]) -> None:
    fieldnames = [
        "dataset",
        "arch",
        "cnn_channels",
        "gru_hidden",
        "expected_params",
        "expected_macs_window",
        "expected_macs_per_frame",
        "printed_params",
        "best_valid_acc",
        "test_loss",
        "test_acc",
        "precision",
        "recall",
        "f1",
        "returncode",
        "train_save_dir",
        "best_model_saved_as",
    ]
    write_csv(RESULTS_DIR / f"{args.out_prefix}_results.csv", rows, fieldnames)


def summarize(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("[SUMMARY] clean streaming CRNN sweep")
    valid = [row for row in rows if row.get("test_acc") is not None]
    valid.sort(key=lambda row: (row["dataset"], -float(row["test_acc"])))
    for row in valid:
        print(
            f"{row['dataset']} | {row['arch']:<16} | "
            f"params={row['expected_params']} | test_acc={row['test_acc']:.4f} | f1={row['f1']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep streaming CRNN configs in clean training/testing")
    parser.add_argument("--root", default="./dscnn_kws/data")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--batch", default=256, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--out_prefix", default="streaming_crnn_clean_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for arch, cnn_channels, gru_hidden in CRNN_CONFIGS:
        for dataset in args.datasets:
            row = train_one(args, dataset, arch, cnn_channels, gru_hidden)
            rows.append(row)
            save_results(args, rows)
            summarize(rows)

    save_results(args, rows)
    summarize(rows)


if __name__ == "__main__":
    main()
