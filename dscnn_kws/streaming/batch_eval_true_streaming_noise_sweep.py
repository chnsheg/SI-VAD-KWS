from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch

from dscnn_kws.streaming.eval_streaming_crnn_true_streaming import (
    build_eval_loader,
    build_model,
    evaluate,
    resolve_chunk_samples,
)


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
    ("C24x5_H64", "24,24,24,24,24", 64),
    ("C32x5_H48", "32,32,32,32,32", 48),
    ("C24x5_H48", "24,24,24,24,24", 48),
    ("C16x5_H64", "16,16,16,16,16", 64),
    ("C32_32_48_H48", "32,32,48", 48),
]

RESULTS_DIR = Path("dscnn_kws") / "streaming" / "results"
RUNS_DIR = Path("dscnn_kws") / "runs" / "streaming"

SAMPLE_RATE = 16000
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
KERNEL_TIME = 5
KERNEL_FREQ = 3
NUM_CLASSES = 2


GRID_FIELDS = [
    "dataset",
    "arch",
    "cnn_channels",
    "gru_hidden",
    "expected_params",
    "expected_macs_window",
    "expected_macs_per_frame",
    "ckpt",
    "split",
    "scene",
    "snr_db",
    "scene_noise_root",
    "usable_noise_files",
    "num_samples",
    "chunk_samples",
    "chunk_ms",
    "flush_tail",
    "frames_min",
    "frames_max",
    "stream_acc",
    "stream_precision",
    "stream_recall",
    "stream_f1",
    "offline_acc",
    "offline_precision",
    "offline_recall",
    "offline_f1",
    "stream_offline_pred_agree",
    "stream_offline_max_abs_logit_diff",
    "stream_offline_mean_abs_logit_diff",
]


def parse_cnn_channels(raw: str) -> list[int]:
    values = [int(item) for item in str(raw).replace(",", " ").split() if item.strip()]
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


def count_usable_noise_files(noise_root: str) -> int:
    root = Path(noise_root)
    if root.is_file():
        count = 0
        base = root.parent
        for line in root.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            path = Path(item)
            if not path.is_absolute():
                path = base / path
            if path.suffix.lower() == ".wav" and path.exists() and path.stat().st_size > 44:
                count += 1
        return count
    if root.is_dir():
        return sum(1 for p in root.rglob("*.wav") if p.stat().st_size > 44)
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_rows_from_best_model_dir(best_model_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not best_model_dir.exists():
        return rows

    for ckpt in sorted(best_model_dir.glob("*_noise_best.pt")):
        name = ckpt.name
        inferred = None
        for arch, channels, gru_hidden in CRNN_CONFIGS:
            marker = f"_{arch}_ch"
            if marker not in name:
                continue
            dataset = name.split(marker, 1)[0]
            m = re.search(r"_ch(?P<channels>[0-9-]+)_gru(?P<gru>\d+)_params(?P<params>\d+)_noise_best\.pt$", name)
            if not m:
                continue
            inferred = {
                "dataset": dataset,
                "arch": arch,
                "cnn_channels": m.group("channels").replace("-", ","),
                "gru_hidden": m.group("gru"),
                "expected_params": m.group("params"),
                "best_model_saved_as": str(ckpt),
            }
            break
        if inferred is not None:
            rows.append(inferred)
    return rows


def load_model_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    train_results_csv = Path(args.train_results_csv)
    if train_results_csv.exists():
        rows = read_csv(train_results_csv)
    else:
        print(f"[WARN] train results CSV not found, scanning best model dir: {args.best_model_dir}")
        rows = infer_rows_from_best_model_dir(Path(args.best_model_dir))

    selected: list[dict[str, str]] = []
    dataset_filter = set(args.datasets or [])
    arch_filter = set(args.archs or [])
    for row in rows:
        ckpt = str(row.get("best_model_saved_as") or row.get("ckpt") or "").strip()
        if not ckpt:
            continue
        if dataset_filter and str(row.get("dataset")) not in dataset_filter:
            continue
        if arch_filter and str(row.get("arch")) not in arch_filter:
            continue
        row = dict(row)
        row["best_model_saved_as"] = ckpt
        row["cnn_channels"] = str(row.get("cnn_channels") or "")
        row["gru_hidden"] = str(row.get("gru_hidden") or "")
        if not row["cnn_channels"] or not row["gru_hidden"]:
            continue
        if not str(row.get("expected_params") or "").strip():
            row["expected_params"] = str(expected_params(row["cnn_channels"], int(row["gru_hidden"])))
        macs = expected_macs_window(row["cnn_channels"], int(row["gru_hidden"]))
        row["expected_macs_window"] = str(row.get("expected_macs_window") or macs)
        row["expected_macs_per_frame"] = str(row.get("expected_macs_per_frame") or macs / 32.0)
        selected.append(row)
    return selected


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"[INFO] CSV saved to: {path.resolve()}")


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def save_grid_results(args: argparse.Namespace, rows: list[dict]) -> None:
    write_csv(RESULTS_DIR / f"{args.out_prefix}_grid_results.csv", rows, GRID_FIELDS)


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
        accs = [float(item["stream_acc"]) for item in items]
        f1s = [float(item["stream_f1"]) for item in items]
        agrees = [
            float(item["stream_offline_pred_agree"])
            for item in items
            if str(item.get("stream_offline_pred_agree", "")).strip()
        ]
        scene_summary.append(
            {
                "dataset": dataset,
                "arch": arch,
                "cnn_channels": cnn_channels,
                "gru_hidden": gru_hidden,
                "expected_params": params,
                "scene": scene,
                "num_snr_points": len(items),
                "mean_stream_acc": mean(accs),
                "min_stream_acc": min(accs),
                "max_stream_acc": max(accs),
                "mean_stream_f1": mean(f1s),
                "min_stream_f1": min(f1s),
                "max_stream_f1": max(f1s),
                "mean_stream_offline_pred_agree": mean(agrees),
                "snrs": " ".join(
                    str(item["snr_db"])
                    for item in sorted(items, key=lambda x: float(x["snr_db"]), reverse=True)
                ),
            }
        )

    arch_summary = []
    for key, items in arch_groups.items():
        dataset, arch, cnn_channels, gru_hidden, params = key
        accs = [float(item["stream_acc"]) for item in items]
        f1s = [float(item["stream_f1"]) for item in items]
        agrees = [
            float(item["stream_offline_pred_agree"])
            for item in items
            if str(item.get("stream_offline_pred_agree", "")).strip()
        ]
        arch_summary.append(
            {
                "dataset": dataset,
                "arch": arch,
                "cnn_channels": cnn_channels,
                "gru_hidden": gru_hidden,
                "expected_params": params,
                "num_eval_points": len(items),
                "mean_stream_acc": mean(accs),
                "min_stream_acc": min(accs),
                "max_stream_acc": max(accs),
                "mean_stream_f1": mean(f1s),
                "min_stream_f1": min(f1s),
                "max_stream_f1": max(f1s),
                "mean_stream_offline_pred_agree": mean(agrees),
            }
        )

    scene_summary.sort(key=lambda x: (x["dataset"], int(float(x["expected_params"])), x["scene"]))
    arch_summary.sort(key=lambda x: (x["dataset"], int(float(x["expected_params"]))))

    scene_fields = [
        "dataset",
        "arch",
        "cnn_channels",
        "gru_hidden",
        "expected_params",
        "scene",
        "num_snr_points",
        "mean_stream_acc",
        "min_stream_acc",
        "max_stream_acc",
        "mean_stream_f1",
        "min_stream_f1",
        "max_stream_f1",
        "mean_stream_offline_pred_agree",
        "snrs",
    ]
    arch_fields = [
        "dataset",
        "arch",
        "cnn_channels",
        "gru_hidden",
        "expected_params",
        "num_eval_points",
        "mean_stream_acc",
        "min_stream_acc",
        "max_stream_acc",
        "mean_stream_f1",
        "min_stream_f1",
        "max_stream_f1",
        "mean_stream_offline_pred_agree",
    ]
    write_csv(RESULTS_DIR / f"{args.out_prefix}_scene_summary.csv", scene_summary, scene_fields)
    write_csv(RESULTS_DIR / f"{args.out_prefix}_arch_summary.csv", arch_summary, arch_fields)


def make_eval_args(
    args: argparse.Namespace,
    model_row: dict[str, str],
    scene_root: str,
    snr_db: float,
    seed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        root=args.root,
        dataset=model_row["dataset"],
        ckpt=model_row["best_model_saved_as"],
        split=args.split,
        batch=args.batch,
        gpu=args.gpu,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_batches=args.max_batches,
        allow_online_resample=args.allow_online_resample,
        strict_sample_rate=args.strict_sample_rate,
        sample_rate=SAMPLE_RATE,
        dct_coeff=DCT_COEFF,
        window_size_ms=WINDOW_SIZE_MS,
        window_stride_ms=WINDOW_STRIDE_MS,
        mfcc_center=False,
        streaming_mfcc=True,
        chunk_samples=args.chunk_samples,
        chunk_ms=args.chunk_ms,
        flush_tail=args.flush_tail,
        cnn_channels=model_row["cnn_channels"],
        kernel_time=KERNEL_TIME,
        kernel_freq=KERNEL_FREQ,
        gru_hidden=int(model_row["gru_hidden"]),
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        mel_filter_shape=args.mel_filter_shape,
        log_approx_mode=args.log_approx_mode,
        log_pwl_num_segments=args.log_pwl_num_segments,
        log_pwl_strategy=args.log_pwl_strategy,
        log_pwl_gamma=args.log_pwl_gamma,
        log_pwl_fit_json=args.log_pwl_fit_json,
        log_offset=args.log_offset,
        log_input_clamp_min=args.log_input_clamp_min,
        eval_noise_aug=True,
        noise_roots=None,
        test_noise_roots=[scene_root],
        eval_noise_aug_prob=1.0,
        eval_noise_snr_min_db=snr_db,
        eval_noise_snr_max_db=snr_db,
        seed=seed,
        compare_offline=args.compare_offline,
        out_csv="",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch true-streaming evaluation for noise-sweep StreamingMFCC+CRNN checkpoints."
    )
    parser.add_argument("--root", default="./dscnn_kws/data")
    parser.add_argument(
        "--train_results_csv",
        default="./dscnn_kws/streaming/results/streaming_crnn_noise_snr_scene_sweep_train_results.csv",
    )
    parser.add_argument(
        "--best_model_dir",
        default=str(RUNS_DIR / "streaming_crnn_noise_snr_scene_sweep_best_models"),
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--archs", nargs="*", default=None)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--batch", default=256, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--allow_online_resample", action="store_true", default=False)
    parser.add_argument("--strict_sample_rate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk_samples", default=None, type=int)
    parser.add_argument("--chunk_ms", default=None, type=float)
    parser.add_argument("--flush_tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene_test_root", default="./dscnn_kws/noise/tau")
    parser.add_argument("--scene_names", nargs="+", default=TAU_SCENES)
    parser.add_argument("--test_snrs", nargs="+", type=float, default=TEST_SNRS)
    parser.add_argument("--seed", default=700000, type=int)
    parser.add_argument("--compare_offline", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--gru_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre_emphasis_coeff", default=0.97, type=float)
    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default="triangular")
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default="exact")
    parser.add_argument("--log_pwl_num_segments", default=6, type=int)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default="uniform_logx")
    parser.add_argument("--log_pwl_gamma", default=1.0, type=float)
    parser.add_argument("--log_pwl_fit_json", default=None, type=str)
    parser.add_argument("--log_offset", default=1e-6, type=float)
    parser.add_argument("--log_input_clamp_min", default=1e-12, type=float)
    parser.add_argument("--out_prefix", default="streaming_crnn_noise_snr_scene_true_streaming")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.gpu > 0 and torch.cuda.is_available() else "cpu")

    model_rows = load_model_rows(args)
    if not model_rows:
        raise FileNotFoundError(
            "No noise-sweep checkpoints found. Check --train_results_csv or --best_model_dir."
        )
    print(f"[INFO] loaded checkpoint rows: {len(model_rows)}")
    print(f"[INFO] device={device}, true_streaming=ON, compare_offline={args.compare_offline}")

    grid_rows: list[dict] = []
    for model_idx, model_row in enumerate(model_rows):
        ckpt = Path(model_row["best_model_saved_as"])
        if not ckpt.exists():
            print(f"[WARN] checkpoint skipped because file is missing: {ckpt}")
            continue

        eval_base_args = make_eval_args(args, model_row, scene_root="", snr_db=0.0, seed=args.seed)
        print("\n" + "=" * 100)
        print(
            f"[MODEL] {model_idx + 1}/{len(model_rows)} | "
            f"dataset={model_row['dataset']} | arch={model_row.get('arch', '')} | "
            f"channels={model_row['cnn_channels']} | gru={model_row['gru_hidden']}"
        )
        print(f"[CKPT] {ckpt}")
        model = build_model(eval_base_args, device)
        chunk_samples = resolve_chunk_samples(eval_base_args, model)

        for scene_idx, scene in enumerate(args.scene_names):
            scene_root = os.path.join(args.scene_test_root, scene)
            usable = count_usable_noise_files(scene_root)
            if usable <= 0:
                print(f"[WARN] scene skipped, no usable wavs: {scene_root}")
                continue

            for snr_idx, snr_db in enumerate(args.test_snrs):
                seed = args.seed + model_idx * 1000003 + scene_idx * 10007 + snr_idx * 101
                eval_args = make_eval_args(args, model_row, scene_root=scene_root, snr_db=snr_db, seed=seed)
                loader = build_eval_loader(eval_args)
                print(
                    f"[TRUE_GRID] dataset={model_row['dataset']} | arch={model_row.get('arch', '')} | "
                    f"scene={scene} | snr={snr_db:g} dB | usable_noise={usable}"
                )
                result = evaluate(model, loader, device, eval_args, chunk_samples)
                row = {
                    "dataset": model_row["dataset"],
                    "arch": model_row.get("arch", ""),
                    "cnn_channels": model_row["cnn_channels"],
                    "gru_hidden": model_row["gru_hidden"],
                    "expected_params": model_row["expected_params"],
                    "expected_macs_window": model_row["expected_macs_window"],
                    "expected_macs_per_frame": model_row["expected_macs_per_frame"],
                    "ckpt": model_row["best_model_saved_as"],
                    "scene": scene,
                    "snr_db": snr_db,
                    "scene_noise_root": scene_root,
                    "usable_noise_files": usable,
                    **result,
                }
                grid_rows.append(row)
                save_grid_results(args, grid_rows)
                save_summary_results(args, grid_rows)
                print(
                    f"[TRUE_RESULT] acc={row['stream_acc']:.4f} f1={row['stream_f1']:.4f} "
                    f"offline_acc={row.get('offline_acc', '')} "
                    f"agree={row.get('stream_offline_pred_agree', '')}"
                )

    save_grid_results(args, grid_rows)
    save_summary_results(args, grid_rows)


if __name__ == "__main__":
    main()
