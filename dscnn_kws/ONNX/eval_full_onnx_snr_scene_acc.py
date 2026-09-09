from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
from dscnn_kws.data.dataset import SpeechCommandDataset


DEFAULT_ONNX_DIR = REPO_ROOT / "dscnn_kws" / "ONNX" / "models_snr_scene_mfcc_friendly"
DEFAULT_ROOT = REPO_ROOT / "dscnn_kws" / "data"
DEFAULT_SCENE_TEST_ROOT = REPO_ROOT / "dscnn_kws" / "noise" / "tau"
DEFAULT_OUT_DIR = REPO_ROOT

DEFAULT_SCENES = [
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

DEFAULT_SNRS = [20.0, 10.0, 5.0, 0.0, -5.0]

MODEL_RE = re.compile(
    r"(?P<dataset>.+?)_"
    r"(?P<arch>L(?P<layers>\d+)_C(?P<channels>\d+))"
    r".*?params(?P<params>\d+)",
    re.IGNORECASE,
)


def discover_onnx_models(args: argparse.Namespace) -> list[Path]:
    if args.onnx_models:
        paths = [Path(p) for p in args.onnx_models]
    else:
        paths = sorted(Path(args.onnx_dir).glob(args.pattern))
    return [p.resolve() for p in paths if p.exists()]


def parse_model_name(path: Path) -> dict[str, Any]:
    match = MODEL_RE.search(path.name)
    if not match:
        if not path.name.endswith(".onnx"):
            raise ValueError(f"Not an ONNX model: {path}")
        return {
            "dataset": None,
            "arch": path.stem,
            "layers": None,
            "channels": None,
            "expected_params": None,
        }
    return {
        "dataset": match.group("dataset"),
        "arch": match.group("arch"),
        "layers": int(match.group("layers")),
        "channels": int(match.group("channels")),
        "expected_params": int(match.group("params")),
    }


def count_usable_noise_files(noise_roots: list[str]) -> int:
    count = 0
    for raw_root in noise_roots:
        root = Path(raw_root)
        if root.is_file():
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
        elif root.is_dir():
            count += sum(1 for p in root.rglob("*.wav") if p.stat().st_size > 44)
    return count


def build_eval_loader(
    *,
    args: argparse.Namespace,
    dataset: str,
    noise_roots: list[str],
    snr_db: float,
    seed: int,
) -> DataLoader:
    data_path = Path(args.root) / dataset
    eval_dataset = SpeechCommandDataset(
        dataset_path=str(data_path),
        json_filename=str(data_path / "test_manifest.json"),
        is_training=False,
        class_list=CLASS_LIST,
        class_encoding=CLASS_ENCODING,
        sample_rate=args.sample_rate,
        noise_aug=True,
        noise_roots=noise_roots,
        noise_prob=1.0,
        noise_snr_min_db=snr_db,
        noise_snr_max_db=snr_db,
        deterministic_noise=True,
        random_seed=seed,
        allow_online_resample=True,
        strict_sample_rate=False,
    )
    return DataLoader(
        eval_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def make_ort_session(path: Path, args: argparse.Namespace):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if args.ort_intra_op_num_threads > 0:
        options.intra_op_num_threads = args.ort_intra_op_num_threads
    if args.ort_inter_op_num_threads > 0:
        options.inter_op_num_threads = args.ort_inter_op_num_threads
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def macro_metrics(labels: list[int], preds: list[int], num_classes: int) -> dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int64)
    preds_np = np.asarray(preds, dtype=np.int64)
    total = int(labels_np.size)
    correct = int(np.sum(labels_np == preds_np))

    precisions = []
    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = int(np.sum((labels_np == cls) & (preds_np == cls)))
        fp = int(np.sum((labels_np != cls) & (preds_np == cls)))
        fn = int(np.sum((labels_np == cls) & (preds_np != cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": correct / max(1, total),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "num_samples": total,
    }


def eval_onnx_acc(session, loader: DataLoader, args: argparse.Namespace) -> dict[str, float]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    all_preds: list[int] = []
    all_labels: list[int] = []

    for waveform, labels in loader:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        waveform_np = waveform.numpy().astype(np.float32, copy=False)
        logits = session.run([output_name], {input_name: waveform_np})[0]
        preds = np.argmax(logits, axis=1).astype(np.int64)
        all_preds.extend(preds.tolist())
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.numpy().astype(np.int64).tolist())
        else:
            all_labels.extend([int(x) for x in labels])

    return macro_metrics(all_labels, all_preds, num_classes=len(CLASS_LIST))


def run_grid_for_model(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    info = parse_model_name(path)
    dataset = args.dataset or info["dataset"]
    if not dataset:
        raise ValueError(
            f"Cannot infer dataset from ONNX filename: {path.name}. "
            "Use --dataset to specify it explicitly."
        )

    session = make_ort_session(path, args)
    rows = []
    for scene_idx, scene in enumerate(args.scene_names):
        scene_root = Path(args.scene_test_root) / scene
        noise_roots = [str(scene_root)]
        usable = count_usable_noise_files(noise_roots)
        if usable <= 0:
            print(f"[WARN] scene skipped, no usable wavs: {scene_root}")
            continue

        for snr_idx, snr_db in enumerate(args.test_snrs):
            loader = build_eval_loader(
                args=args,
                dataset=dataset,
                noise_roots=noise_roots,
                snr_db=float(snr_db),
                seed=args.seed_base + scene_idx * 10007 + snr_idx * 101,
            )
            metrics = eval_onnx_acc(session, loader, args)
            row = {
                "dataset": dataset,
                "arch": info["arch"],
                "layers": info["layers"],
                "channels": info["channels"],
                "expected_params": info["expected_params"],
                "scene": scene,
                "snr_db": float(snr_db),
                "scene_noise_root": str(scene_root),
                "usable_noise_files": usable,
                "acc": metrics["acc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "num_samples": metrics["num_samples"],
                "onnx": str(path),
            }
            rows.append(row)
            print(
                f"[GRID] {dataset} | {info['arch']} | scene={scene:<18} | "
                f"snr={float(snr_db):>5g} dB | acc={metrics['acc']:.4f} | f1={metrics['f1']:.4f}"
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_grid(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scene_groups = {}
    arch_groups = {}
    for row in rows:
        scene_key = (
            row["dataset"],
            row["arch"],
            row["layers"],
            row["channels"],
            row["expected_params"],
            row["scene"],
        )
        arch_key = (
            row["dataset"],
            row["arch"],
            row["layers"],
            row["channels"],
            row["expected_params"],
        )
        scene_groups.setdefault(scene_key, []).append(row)
        arch_groups.setdefault(arch_key, []).append(row)

    scene_summary = []
    for key, items in scene_groups.items():
        dataset, arch, layers, channels, params, scene = key
        accs = [float(item["acc"]) for item in items]
        f1s = [float(item["f1"]) for item in items]
        scene_summary.append(
            {
                "dataset": dataset,
                "arch": arch,
                "layers": layers,
                "channels": channels,
                "expected_params": params,
                "scene": scene,
                "num_snr_points": len(items),
                "mean_acc": mean(accs),
                "min_acc": min(accs),
                "max_acc": max(accs),
                "mean_f1": mean(f1s),
                "min_f1": min(f1s),
                "max_f1": max(f1s),
                "snrs": " ".join(
                    str(item["snr_db"]) for item in sorted(items, key=lambda x: float(x["snr_db"]), reverse=True)
                ),
            }
        )

    arch_summary = []
    for key, items in arch_groups.items():
        dataset, arch, layers, channels, params = key
        accs = [float(item["acc"]) for item in items]
        f1s = [float(item["f1"]) for item in items]
        arch_summary.append(
            {
                "dataset": dataset,
                "arch": arch,
                "layers": layers,
                "channels": channels,
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

    scene_summary.sort(key=lambda x: (x["dataset"], x["expected_params"] or -1, x["scene"]))
    arch_summary.sort(key=lambda x: (x["dataset"], x["expected_params"] or -1))
    return scene_summary, arch_summary


def save_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    grid_path = out_dir / f"{args.out_prefix}_grid_results.csv"
    scene_path = out_dir / f"{args.out_prefix}_scene_summary.csv"
    arch_path = out_dir / f"{args.out_prefix}_arch_summary.csv"

    grid_fields = [
        "dataset",
        "arch",
        "layers",
        "channels",
        "expected_params",
        "scene",
        "snr_db",
        "scene_noise_root",
        "usable_noise_files",
        "acc",
        "precision",
        "recall",
        "f1",
        "num_samples",
        "onnx",
    ]
    write_csv(grid_path, rows, grid_fields)
    print(f"[INFO] Grid CSV saved to: {grid_path.resolve()}")

    if not rows:
        return

    scene_summary, arch_summary = summarize_grid(rows)
    scene_fields = [
        "dataset",
        "arch",
        "layers",
        "channels",
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
        "layers",
        "channels",
        "expected_params",
        "num_eval_points",
        "mean_acc",
        "min_acc",
        "max_acc",
        "mean_f1",
        "min_f1",
        "max_f1",
    ]
    write_csv(scene_path, scene_summary, scene_fields)
    write_csv(arch_path, arch_summary, arch_fields)
    print(f"[INFO] Scene summary CSV saved to: {scene_path.resolve()}")
    print(f"[INFO] Arch summary CSV saved to: {arch_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exported full ONNX KWS models across fixed SNRs and TAU scenes."
    )
    parser.add_argument("--onnx_dir", default=str(DEFAULT_ONNX_DIR))
    parser.add_argument("--pattern", default="*.onnx")
    parser.add_argument("--onnx_models", nargs="*", default=None)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset for all models. By default it is inferred from each ONNX filename.",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--scene_test_root", default=str(DEFAULT_SCENE_TEST_ROOT))
    parser.add_argument("--scene_names", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--test_snrs", nargs="+", type=float, default=DEFAULT_SNRS)
    parser.add_argument("--seed_base", type=int, default=500000)
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out_prefix", default="onnx_snr_scene_mfcc_friendly")
    parser.add_argument(
        "--ort_intra_op_num_threads",
        type=int,
        default=1,
        help="Set ONNX Runtime intra-op threads. Default 1 avoids container CPU affinity warnings.",
    )
    parser.add_argument("--ort_inter_op_num_threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = discover_onnx_models(args)
    if not models:
        raise FileNotFoundError(f"No ONNX model found under {args.onnx_dir}")

    print(f"[INFO] found {len(models)} ONNX model(s)")
    print(f"[INFO] root={Path(args.root).resolve()}")
    print(f"[INFO] scene_test_root={Path(args.scene_test_root).resolve()}")
    print(f"[INFO] scenes={args.scene_names}")
    print(f"[INFO] snrs={args.test_snrs}")

    all_rows: list[dict[str, Any]] = []
    for model_path in models:
        print(f"[EVAL] {model_path}")
        rows = run_grid_for_model(model_path, args)
        all_rows.extend(rows)

    save_outputs(all_rows, args)
    print(f"[DONE] evaluated {len(models)} model(s), {len(all_rows)} grid rows")


if __name__ == "__main__":
    main()
