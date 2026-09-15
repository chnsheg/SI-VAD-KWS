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

DEFAULT_TRAIN_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_train.txt"]
DEFAULT_VALID_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_valid.txt"]
DEFAULT_TEST_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_test.txt"]

TAU_SCENES = [
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram",
]

TEST_SNRS = [20.0, 10.0, 5.0, 0.0, -5.0]

CRNN_CONFIGS = [
    # ("C24x5_H64", "24,24,24,24,24", 64),
    ("C32x5_H48", "32,32,32,32,32", 48),
    ("C24x5_H48", "24,24,24,24,24", 48),
    # ("C16x5_H64", "16,16,16,16,16", 64),
    ("C32_32_48_H48", "32,32,48", 48),
    # Accuracy-oriented candidates kept under 30k params and 4M MACs per 1s window.
    ("C32x5_H64", "32,32,32,32,32", 64),
    ("C36x5_H64", "36,36,36,36,36", 64),
    ("C40x5_H48", "40,40,40,40,40", 48),
    # ("C40x5_H56", "40,40,40,40,40", 56),
    ("C32x6_H64", "32,32,32,32,32,32", 64),
    ("C34x6_H64", "34,34,34,34,34,34", 64),
    ("C32_40_48_48_H60", "32,40,48,48", 60),
    # ("C24_32_48_48_56_H48", "24,32,48,48,56", 48),
    ("C40_40_48_48_H56", "40,40,48,48", 56),
]

NUM_CLASSES = 2
SAMPLE_RATE = 16000
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
KERNEL_TIME = 7
KERNEL_FREQ = 3
MAX_EXPECTED_PARAMS = 30_000
MAX_EXPECTED_MACS_WINDOW = 4_000_000

TRAIN_NOISE_PROB = 0.8
TRAIN_SNR_MIN_DB = -5.0
TRAIN_SNR_MAX_DB = 20.0
VALID_NOISE_PROB = 1.0
VALID_SNR_DB = 5.0

RESULTS_DIR = Path("dscnn_kws") / "streaming" / "results_kt7"
RUNS_DIR = Path("dscnn_kws") / "runs" / "streaming_kt7"


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


def validate_crnn_configs() -> None:
    too_large = []
    for arch, cnn_channels, gru_hidden in CRNN_CONFIGS:
        params = expected_params(cnn_channels, gru_hidden)
        macs_window = expected_macs_window(cnn_channels, gru_hidden)
        if params > MAX_EXPECTED_PARAMS or macs_window > MAX_EXPECTED_MACS_WINDOW:
            too_large.append(
                f"{arch}: params={params}, macs_window={macs_window}"
            )
    if too_large:
        details = "\n".join(too_large)
        raise ValueError(
            "CRNN_CONFIGS contains configs outside the expected limits "
            f"(params <= {MAX_EXPECTED_PARAMS}, 1s MACs <= {MAX_EXPECTED_MACS_WINDOW}):\n"
            f"{details}"
        )


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
        "test_loss_on_tau_test_5db": test_loss,
        "test_acc_on_tau_test_5db": test_acc,
        "precision_on_tau_test_5db": precision,
        "recall_on_tau_test_5db": recall,
        "f1_on_tau_test_5db": f1,
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
        f"params{expected_params(cnn_channels, gru_hidden)}_noise_best.pt"
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
        "--noise_aug",
        "--eval_noise_aug",
        "--train_noise_roots",
        *args.train_noise_roots,
        "--valid_noise_roots",
        *args.valid_noise_roots,
        "--test_noise_roots",
        *args.test_noise_roots,
        "--noise_aug_prob",
        str(TRAIN_NOISE_PROB),
        "--noise_snr_min_db",
        str(TRAIN_SNR_MIN_DB),
        "--noise_snr_max_db",
        str(TRAIN_SNR_MAX_DB),
        "--eval_noise_aug_prob",
        str(VALID_NOISE_PROB),
        "--eval_noise_snr_min_db",
        str(VALID_SNR_DB),
        "--eval_noise_snr_max_db",
        str(VALID_SNR_DB),
        "--save_root",
        str(save_root),
    ]

    print("\n" + "=" * 100)
    print(f"[TRAIN] noise | dataset={dataset} | arch={arch} | channels={cnn_channels} | gru={gru_hidden}")
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


def run_grid_eval(args: argparse.Namespace, train_row: dict) -> list[dict]:
    ckpt = train_row.get("best_model_saved_as")
    if not ckpt:
        print("[WARN] grid skipped because best checkpoint is missing")
        return []

    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(train_row["dataset"]))
    out_csv = RESULTS_DIR / (
        f"{args.out_prefix}_{safe_dataset}_{train_row['arch']}_grid_results.csv"
    )
    cmd = [
        sys.executable,
        "-m",
        "dscnn_kws.streaming.eval_streaming_crnn_snr_scene",
        "--root",
        args.root,
        "--dataset",
        str(train_row["dataset"]),
        "--ckpt",
        str(ckpt),
        "--batch",
        str(args.batch),
        "--gpu",
        str(args.gpu),
        "--num_workers",
        str(max(0, args.num_workers // 2)),
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
        str(train_row["cnn_channels"]),
        "--kernel_time",
        str(KERNEL_TIME),
        "--kernel_freq",
        str(KERNEL_FREQ),
        "--gru_hidden",
        str(train_row["gru_hidden"]),
        "--allow_online_resample",
        "--no-strict_sample_rate",
        "--scene_test_root",
        args.scene_test_root,
        "--scene_names",
        *args.scene_names,
        "--test_snrs",
        *[str(snr) for snr in args.test_snrs],
        "--out_csv",
        str(out_csv),
    ]
    print("\n" + "=" * 100)
    print(f"[GRID] dataset={train_row['dataset']} | arch={train_row['arch']}")
    print("[CMD]", " ".join(cmd))
    returncode, _ = run_command(cmd)
    if returncode != 0:
        print(f"[WARN] grid eval returned non-zero code: {returncode}")
        return []
    if not out_csv.exists():
        print(f"[WARN] grid csv missing: {out_csv}")
        return []

    rows: list[dict] = []
    with out_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row.update(
                {
                    "arch": train_row["arch"],
                    "expected_params": train_row["expected_params"],
                    "expected_macs_window": train_row["expected_macs_window"],
                    "expected_macs_per_frame": train_row["expected_macs_per_frame"],
                }
            )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved to: {path.resolve()}")


def save_train_results(args: argparse.Namespace, rows: list[dict]) -> None:
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
        "test_loss_on_tau_test_5db",
        "test_acc_on_tau_test_5db",
        "precision_on_tau_test_5db",
        "recall_on_tau_test_5db",
        "f1_on_tau_test_5db",
        "returncode",
        "train_save_dir",
        "best_model_saved_as",
    ]
    write_csv(RESULTS_DIR / f"{args.out_prefix}_train_results.csv", rows, fieldnames)


def save_grid_results(args: argparse.Namespace, rows: list[dict]) -> None:
    fieldnames = [
        "dataset",
        "arch",
        "cnn_channels",
        "gru_hidden",
        "expected_params",
        "expected_macs_window",
        "expected_macs_per_frame",
        "frontend",
        "streaming_mfcc",
        "ckpt",
        "scene",
        "snr_db",
        "scene_noise_root",
        "usable_noise_files",
        "acc",
        "precision",
        "recall",
        "f1",
        "num_samples",
    ]
    write_csv(RESULTS_DIR / f"{args.out_prefix}_grid_results.csv", rows, fieldnames)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def save_summary_results(args: argparse.Namespace, grid_rows: list[dict]) -> None:
    if not grid_rows:
        return

    scene_groups: dict[tuple, list[dict]] = {}
    arch_groups: dict[tuple, list[dict]] = {}
    for row in grid_rows:
        scene_key = (
            row["dataset"],
            row["arch"],
            row["cnn_channels"],
            row["gru_hidden"],
            row["expected_params"],
            row["scene"],
        )
        arch_key = (
            row["dataset"],
            row["arch"],
            row["cnn_channels"],
            row["gru_hidden"],
            row["expected_params"],
        )
        scene_groups.setdefault(scene_key, []).append(row)
        arch_groups.setdefault(arch_key, []).append(row)

    scene_summary = []
    for key, items in scene_groups.items():
        dataset, arch, cnn_channels, gru_hidden, params, scene = key
        accs = [float(item["acc"]) for item in items]
        f1s = [float(item["f1"]) for item in items]
        scene_summary.append(
            {
                "dataset": dataset,
                "arch": arch,
                "cnn_channels": cnn_channels,
                "gru_hidden": gru_hidden,
                "expected_params": params,
                "scene": scene,
                "num_snr_points": len(items),
                "mean_acc": mean(accs),
                "min_acc": min(accs),
                "max_acc": max(accs),
                "mean_f1": mean(f1s),
                "min_f1": min(f1s),
                "max_f1": max(f1s),
                "snrs": " ".join(str(item["snr_db"]) for item in sorted(items, key=lambda x: float(x["snr_db"]), reverse=True)),
            }
        )

    arch_summary = []
    for key, items in arch_groups.items():
        dataset, arch, cnn_channels, gru_hidden, params = key
        accs = [float(item["acc"]) for item in items]
        f1s = [float(item["f1"]) for item in items]
        arch_summary.append(
            {
                "dataset": dataset,
                "arch": arch,
                "cnn_channels": cnn_channels,
                "gru_hidden": gru_hidden,
                "expected_params": params,
                "num_eval_points": len(items),
                "mean_acc": mean(accs),
                "min_acc": min(accs),
                "max_acc": max(accs),
                "mean_f1": mean(f1s),
                "min_f1": min(f1s),
                "max_f1": max(f1s),
            }
        )

    scene_summary.sort(key=lambda x: (x["dataset"], int(x["expected_params"]), x["scene"]))
    arch_summary.sort(key=lambda x: (x["dataset"], int(x["expected_params"])))

    scene_fields = [
        "dataset",
        "arch",
        "cnn_channels",
        "gru_hidden",
        "expected_params",
        "scene",
        "num_snr_points",
        "mean_acc",
        "min_acc",
        "max_acc",
        "mean_f1",
        "min_f1",
        "max_f1",
        "snrs",
    ]
    arch_fields = [
        "dataset",
        "arch",
        "cnn_channels",
        "gru_hidden",
        "expected_params",
        "num_eval_points",
        "mean_acc",
        "min_acc",
        "max_acc",
        "mean_f1",
        "min_f1",
        "max_f1",
    ]
    write_csv(RESULTS_DIR / f"{args.out_prefix}_scene_summary.csv", scene_summary, scene_fields)
    write_csv(RESULTS_DIR / f"{args.out_prefix}_arch_summary.csv", arch_summary, arch_fields)


def validate_noise_roots(name: str, roots: list[str]) -> None:
    missing = [root for root in roots if not Path(root).exists()]
    if missing:
        raise FileNotFoundError(f"{name} contains missing paths: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep streaming CRNN configs with TAU noise and SNR scene grid")
    parser.add_argument("--root", default="./dscnn_kws/data")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--batch", default=256, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--train_noise_roots", nargs="+", default=DEFAULT_TRAIN_NOISE_ROOTS)
    parser.add_argument("--valid_noise_roots", nargs="+", default=DEFAULT_VALID_NOISE_ROOTS)
    parser.add_argument("--test_noise_roots", nargs="+", default=DEFAULT_TEST_NOISE_ROOTS)
    parser.add_argument("--scene_test_root", default="./dscnn_kws/noise/tau")
    parser.add_argument("--scene_names", nargs="+", default=TAU_SCENES)
    parser.add_argument("--test_snrs", nargs="+", type=float, default=TEST_SNRS)
    parser.add_argument("--out_prefix", default="streaming_crnn_noise_snr_scene_sweep_kt7")
    parser.add_argument("--skip_grid", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    validate_crnn_configs()
    validate_noise_roots("train_noise_roots", args.train_noise_roots)
    validate_noise_roots("valid_noise_roots", args.valid_noise_roots)
    validate_noise_roots("test_noise_roots", args.test_noise_roots)

    train_rows: list[dict] = []
    grid_rows: list[dict] = []
    for arch, cnn_channels, gru_hidden in CRNN_CONFIGS:
        for dataset in args.datasets:
            train_row = train_one(args, dataset, arch, cnn_channels, gru_hidden)
            train_rows.append(train_row)
            save_train_results(args, train_rows)

            if not args.skip_grid and train_row.get("best_model_saved_as"):
                rows = run_grid_eval(args, train_row)
                grid_rows.extend(rows)
                save_grid_results(args, grid_rows)
                save_summary_results(args, grid_rows)

    save_train_results(args, train_rows)
    if grid_rows:
        save_grid_results(args, grid_rows)
        save_summary_results(args, grid_rows)


if __name__ == "__main__":
    main()
