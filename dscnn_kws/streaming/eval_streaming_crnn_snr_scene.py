from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
from dscnn_kws.data.dataset import SpeechCommandDataset
from dscnn_kws.frontend import load_log_pwl_json
from dscnn_kws.streaming.streaming_crnn import StreamingKWSModel, model_config_dict, parse_cnn_channels
from dscnn_kws.utils import parameter_number


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

DEFAULT_SNRS = [20.0, 10.0, 5.0, 0.0, -5.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate streaming CRNN over TAU scene x SNR grid")
    parser.add_argument("--root", default="./dscnn_kws/data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--batch", default=256, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--allow_online_resample", action="store_true", default=False)
    parser.add_argument("--strict_sample_rate", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--frontend", choices=["mfcc", "bandpass"], default="mfcc")
    parser.add_argument("--dct_coeff", default=10, type=int)
    parser.add_argument("--window_size_ms", default=32, type=int)
    parser.add_argument("--window_stride_ms", default=32, type=int)
    parser.add_argument("--mfcc_center", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming_mfcc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bandpass_n_bands", default=10, type=int)
    parser.add_argument("--bandpass_f_min", default=200.0, type=float)
    parser.add_argument("--bandpass_f_max", default=4000.0, type=float)
    parser.add_argument("--bandpass_spacing", choices=["log", "linear"], default="log")
    parser.add_argument("--bandpass_kernel_size", default=63, type=int)
    parser.add_argument("--bandpass_phase_count", default=1, type=int)

    parser.add_argument("--cnn_channels", default="24,24,24,24,24", type=str)
    parser.add_argument("--kernel_time", default=5, type=int)
    parser.add_argument("--kernel_freq", default=3, type=int)
    parser.add_argument("--gru_hidden", default=64, type=int)
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

    parser.add_argument("--scene_test_root", default="./dscnn_kws/noise/tau", type=str)
    parser.add_argument("--scene_names", nargs="+", default=TAU_SCENES)
    parser.add_argument("--test_snrs", nargs="+", type=float, default=DEFAULT_SNRS)
    parser.add_argument("--seed", default=500000, type=int)
    parser.add_argument(
        "--out_csv",
        default="./dscnn_kws/streaming/results/streaming_crnn_snr_scene_grid_results.csv",
        type=str,
    )
    return parser.parse_args()


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


def load_state_dict(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} is not a state_dict or checkpoint dict")
    return obj


def build_model(args: argparse.Namespace, device: torch.device) -> StreamingKWSModel:
    log_pwl_breakpoints = None
    log_pwl_slopes = None
    log_pwl_intercepts = None
    if args.log_pwl_fit_json:
        cfg = load_log_pwl_json(args.log_pwl_fit_json)
        log_pwl_breakpoints = cfg["breakpoints"]
        log_pwl_slopes = cfg["slopes"]
        log_pwl_intercepts = cfg["intercepts"]

    model = StreamingKWSModel(
        sample_rate=args.sample_rate,
        label_count=len(CLASS_LIST),
        frontend=args.frontend,
        dct_coeff=args.dct_coeff,
        window_size_ms=args.window_size_ms,
        window_stride_ms=args.window_stride_ms,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        cnn_channels=parse_cnn_channels(args.cnn_channels),
        kernel_time=args.kernel_time,
        kernel_freq=args.kernel_freq,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        bandpass_n_bands=args.bandpass_n_bands,
        bandpass_f_min=args.bandpass_f_min,
        bandpass_f_max=args.bandpass_f_max,
        bandpass_spacing=args.bandpass_spacing,
        bandpass_kernel_size=args.bandpass_kernel_size,
        bandpass_phase_count=args.bandpass_phase_count,
        mfcc_center=args.mfcc_center,
        streaming_mfcc=args.streaming_mfcc,
        mel_filter_shape=args.mel_filter_shape,
        log_approx_mode=args.log_approx_mode,
        log_pwl_num_segments=args.log_pwl_num_segments,
        log_pwl_strategy=args.log_pwl_strategy,
        log_pwl_gamma=args.log_pwl_gamma,
        log_pwl_breakpoints=log_pwl_breakpoints,
        log_pwl_slopes=log_pwl_slopes,
        log_pwl_intercepts=log_pwl_intercepts,
        log_offset=args.log_offset,
        log_input_clamp_min=args.log_input_clamp_min,
    ).to(device)
    model.load_state_dict(load_state_dict(args.ckpt, device), strict=True)
    model.eval()
    return model


def build_eval_loader(args: argparse.Namespace, noise_roots: list[str], snr_db: float, seed: int) -> DataLoader:
    data_path = os.path.join(args.root, args.dataset)
    dataset = SpeechCommandDataset(
        dataset_path=data_path,
        json_filename=os.path.join(data_path, "test_manifest.json"),
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
        allow_online_resample=args.allow_online_resample,
        strict_sample_rate=args.strict_sample_rate,
    )
    kwargs = {
        "batch_size": args.batch,
        "shuffle": False,
        "drop_last": False,
        "num_workers": args.num_workers,
        "pin_memory": args.gpu > 0,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = max(2, args.prefetch_factor)
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def eval_acc(model: StreamingKWSModel, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    total = 0
    correct = 0
    preds_all: list[int] = []
    labels_all: list[int] = []

    for waveform, labels in loader:
        waveform = waveform.to(device)
        labels = labels.to(device)
        logits = model(waveform)
        preds = torch.argmax(logits, dim=1)
        total += int(labels.size(0))
        correct += int((preds == labels).sum().item())
        preds_all.extend(preds.cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())

    return {
        "acc": correct / max(1, total),
        "precision": precision_score(labels_all, preds_all, average="macro", zero_division=0),
        "recall": recall_score(labels_all, preds_all, average="macro", zero_division=0),
        "f1": f1_score(labels_all, preds_all, average="macro", zero_division=0),
        "num_samples": total,
    }


def write_csv(path: str, rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "frontend",
        "streaming_mfcc",
        "cnn_channels",
        "gru_hidden",
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
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved to: {out_path.resolve()}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.gpu > 0 and torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    print(f"[INFO] device={device}, params={parameter_number(model)}")
    print(f"[INFO] model_config={model_config_dict(model)}")
    print(f"[INFO] ckpt={args.ckpt}")

    rows: list[dict] = []
    for scene_idx, scene in enumerate(args.scene_names):
        scene_root = os.path.join(args.scene_test_root, scene)
        noise_roots = [scene_root]
        usable = count_usable_noise_files(noise_roots)
        if usable <= 0:
            print(f"[WARN] scene skipped, no usable wavs: {scene_root}")
            continue

        for snr_idx, snr_db in enumerate(args.test_snrs):
            seed = args.seed + scene_idx * 10007 + snr_idx * 101
            loader = build_eval_loader(args, noise_roots=noise_roots, snr_db=snr_db, seed=seed)
            metrics = eval_acc(model, loader, device)
            row = {
                "dataset": args.dataset,
                "frontend": args.frontend,
                "streaming_mfcc": args.streaming_mfcc,
                "cnn_channels": args.cnn_channels,
                "gru_hidden": args.gru_hidden,
                "ckpt": args.ckpt,
                "scene": scene,
                "snr_db": snr_db,
                "scene_noise_root": scene_root,
                "usable_noise_files": usable,
                "acc": metrics["acc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "num_samples": metrics["num_samples"],
            }
            rows.append(row)
            print(
                f"[GRID] {args.dataset} | scene={scene:<18} | snr={snr_db:>5} dB | "
                f"acc={metrics['acc']:.4f} | f1={metrics['f1']:.4f}"
            )

    write_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()

