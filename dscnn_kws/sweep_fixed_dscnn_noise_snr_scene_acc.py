from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = "./dscnn_kws/data"

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

EPOCH = 30
BATCH = 256
SAMPLE_RATE = 16000
GPU = 1
NUM_WORKERS = 8
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
MEL_FILTER_SHAPE = "triangular"
PRE_EMPHASIS = True
PRE_EMPHASIS_COEFF = 0.97
NUM_CLASSES = 2

TRAIN_NOISE_PROB = 0.8
TRAIN_SNR_MIN_DB = -5.0
TRAIN_SNR_MAX_DB = 20.0

VALID_NOISE_PROB = 1.0
VALID_SNR_DB = 5.0

ARCHS = [
    ("L5_C64", 5, 64),
    ("L5_C48", 5, 48),
    ("L5_C32", 5, 32),
    ("L5_C24", 5, 24),
    ("L5_C16", 5, 16),
    ("L4_C16", 4, 16),
    ("L3_C16", 3, 16),
    ("L5_C12", 5, 12),
    ("L4_C12", 4, 12),
    ("L3_C12", 3, 12),
    ("L5_C8", 5, 8),
    ("L4_C8", 4, 8),
    ("L3_C8", 3, 8),
    ("L2_C16", 2, 16),
    ("L2_C12", 2, 12),
    ("L2_C8", 2, 8),
    ("L5_C6", 5, 6),
    ("L4_C6", 4, 6),
    ("L3_C6", 3, 6),
    ("L2_C6", 2, 6),
    ("L1_C8", 1, 8),
    ("L3_C4", 3, 4),
    ("L2_C4", 2, 4),
    ("L1_C6", 1, 6),
    ("L1_C4", 1, 4),
]
ARCH_BY_NAME = {name: (name, layers, channels) for name, layers, channels in ARCHS}

OUT_TRAIN_CSV = Path("snr_scene_arch_sweep_train_results.csv")
OUT_GRID_CSV = Path("snr_scene_arch_sweep_grid_results.csv")
OUT_SCENE_SUMMARY_CSV = Path("snr_scene_arch_sweep_scene_summary.csv")
OUT_ARCH_SUMMARY_CSV = Path("snr_scene_arch_sweep_arch_summary.csv")
BEST_MODEL_DIR = Path("dscnn_kws") / "runs" / "snr_scene_arch_sweep_best_models"
WRITE_OUTPUTS = True


def configure_outputs(out_prefix: str, write_outputs: bool) -> None:
    global OUT_TRAIN_CSV
    global OUT_GRID_CSV
    global OUT_SCENE_SUMMARY_CSV
    global OUT_ARCH_SUMMARY_CSV
    global BEST_MODEL_DIR
    global WRITE_OUTPUTS

    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(out_prefix).strip())
    if not safe_prefix:
        raise ValueError("--out_prefix must not be empty")

    OUT_TRAIN_CSV = Path(f"{safe_prefix}_train_results.csv")
    OUT_GRID_CSV = Path(f"{safe_prefix}_grid_results.csv")
    OUT_SCENE_SUMMARY_CSV = Path(f"{safe_prefix}_scene_summary.csv")
    OUT_ARCH_SUMMARY_CSV = Path(f"{safe_prefix}_arch_summary.csv")
    BEST_MODEL_DIR = Path("dscnn_kws") / "runs" / f"{safe_prefix}_best_models"
    WRITE_OUTPUTS = bool(write_outputs)


def make_model_size_info(num_layers: int, channels: int):
    info = [num_layers]
    info += [channels, 10, 4, 2, 2]
    for _ in range(num_layers - 1):
        info += [channels, 3, 3, 1, 1]
    return info


def expected_params(num_layers: int, channels: int, num_classes: int = 2):
    c = channels
    n = num_layers
    return (n - 1) * c * c + (42 + 13 * (n - 1) + num_classes) * c + num_classes


def calculate_time_steps(sample_rate: int, window_stride_ms: int, audio_duration_ms: int = 1000) -> int:
    stride_samples = int(sample_rate * window_stride_ms / 1000)
    audio_samples = int(sample_rate * audio_duration_ms / 1000)
    if stride_samples <= 0:
        return 1
    return audio_samples // stride_samples + 1


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


def copy_best_model(output: str, dataset: str, arch_name: str, num_layers: int, channels: int, suffix: str = "noise_best"):
    m = re.search(r"\[INFO\]\s+save_dir=(.+)", output)
    if not m:
        return None, None

    save_dir = Path(m.group(1).strip())
    best_path = save_dir / "best.pt"
    if not best_path.exists():
        return str(save_dir), None

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    params = expected_params(num_layers, channels, NUM_CLASSES)
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix.strip()) or "noise_best"
    out_path = BEST_MODEL_DIR / (
        f"{safe_dataset}_{arch_name}_layers{num_layers}_channels{channels}_"
        f"params{params}_{safe_suffix}.pt"
    )
    shutil.copy2(best_path, out_path)
    return str(save_dir), str(out_path)


def train_one(
    args,
    dataset: str,
    arch_name: str,
    num_layers: int,
    channels: int,
    model_size_info: list[int],
):
    cmd = [
        sys.executable,
        "-m",
        "dscnn_kws.train",
        "--root",
        args.root,
        "--dataset",
        dataset,
        "--sample_rate",
        str(args.sample_rate),
        "--gpu",
        str(args.gpu),
        "--num_workers",
        str(args.num_workers),
        "--epoch",
        str(args.epoch),
        "--batch",
        str(args.batch),
        "--dct_coeff",
        str(args.dct_coeff),
        "--window_size_ms",
        str(args.window_size_ms),
        "--window_stride_ms",
        str(args.window_stride_ms),
        "--mel_filter_shape",
        args.mel_filter_shape,
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
        str(args.train_noise_prob),
        "--noise_snr_min_db",
        str(args.train_snr_min_db),
        "--noise_snr_max_db",
        str(args.train_snr_max_db),
        "--eval_noise_aug_prob",
        str(args.valid_noise_prob),
        "--eval_noise_snr_min_db",
        str(args.valid_snr_db),
        "--eval_noise_snr_max_db",
        str(args.valid_snr_db),
        "--model_size_info",
        *[str(x) for x in model_size_info],
        "--pre_emphasis" if args.pre_emphasis else "--no-pre_emphasis",
        "--pre_emphasis_coeff",
        str(args.pre_emphasis_coeff),
    ]

    print("\n" + "=" * 100)
    print(f"[TRAIN] dataset={dataset}, arch={arch_name}")
    print("[CMD]", " ".join(cmd))

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

    chunks = []
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
    output = "".join(chunks)

    test_acc = None
    f1 = None
    valid_accs = [float(x) for x in re.findall(r"valid_loss\s+[0-9.]+\s+acc\s+([0-9.]+)", output)]
    best_valid_acc = max(valid_accs) if valid_accs else None
    m = re.search(
        r"\[TEST\]\s+loss=([0-9.]+)\s+acc=([0-9.]+)\s+precision=([0-9.]+)\s+recall=([0-9.]+)\s+f1=([0-9.]+)",
        output,
    )
    if m:
        test_acc = float(m.group(2))
        f1 = float(m.group(5))

    save_dir, best_model_saved_as = copy_best_model(
        output,
        dataset,
        arch_name,
        num_layers,
        channels,
        suffix=args.best_model_suffix,
    )
    return {
        "dataset": dataset,
        "arch": arch_name,
        "layers": num_layers,
        "channels": channels,
        "expected_params": expected_params(num_layers, channels, NUM_CLASSES),
        "pre_emphasis": args.pre_emphasis,
        "pre_emphasis_coeff": args.pre_emphasis_coeff,
        "best_valid_acc": best_valid_acc,
        "test_acc_on_tau_test_list": test_acc,
        "f1_on_tau_test_list": f1,
        "returncode": proc.returncode,
        "train_save_dir": save_dir,
        "best_model_saved_as": best_model_saved_as,
    }


def load_state_dict(path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(args, model_size_info: list[int], ckpt: str, device):
    from dscnn_kws.configs import CLASS_LIST
    from dscnn_kws.model import DSCNN
    from dscnn_kws.model.dscnn import calculate_time_steps as model_calculate_time_steps
    from dscnn_kws.train import MFCCDSCNN

    time_steps = model_calculate_time_steps(args.sample_rate, args.window_stride_ms)
    input_dim = time_steps * args.dct_coeff
    backbone = DSCNN(
        input_dim=input_dim,
        label_count=len(CLASS_LIST),
        model_size_info=model_size_info,
        dct_coeff=args.dct_coeff,
    )
    model = MFCCDSCNN(
        backbone=backbone,
        frontend="mfcc",
        sample_rate=args.sample_rate,
        dct_coeff=args.dct_coeff,
        window_size_ms=args.window_size_ms,
        window_stride_ms=args.window_stride_ms,
        bandpass_n_bands=16,
        bandpass_f_min=200.0,
        bandpass_f_max=4000.0,
        bandpass_spacing="log",
        bandpass_kernel_size=63,
        bandpass_phase_count=1,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        spec_aug=False,
        spec_aug_freq_mask_param=1,
        spec_aug_time_mask_param=1,
        spec_aug_num_freq_masks=0,
        spec_aug_num_time_masks=0,
        mfcc_impl="torchaudio",
        mel_filter_shape=args.mel_filter_shape,
        log_approx_mode="exact",
        log_pwl_num_segments=6,
        log_pwl_strategy="uniform_logx",
        log_pwl_gamma=1.0,
        log_pwl_breakpoints=None,
        log_pwl_slopes=None,
        log_pwl_intercepts=None,
        log_offset=1e-6,
        log_input_clamp_min=1e-12,
    ).to(device)
    model.load_state_dict(load_state_dict(ckpt, device))
    model.eval()
    return model


def build_eval_loader(args, dataset: str, noise_roots: list[str], snr_db: float, seed: int):
    from torch.utils.data import DataLoader

    from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
    from dscnn_kws.data.dataset import SpeechCommandDataset

    data_path = os.path.join(args.root, dataset)
    eval_dataset = SpeechCommandDataset(
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
        allow_online_resample=True,
        strict_sample_rate=False,
    )

    return DataLoader(
        eval_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=args.gpu > 0,
    )


def eval_acc(model, loader, device) -> dict:
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score

    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for waveform, labels in loader:
            waveform = waveform.to(device)
            labels = labels.to(device)
            logits = model(waveform)
            preds = torch.argmax(logits, dim=1)
            total += int(labels.size(0))
            correct += int((preds == labels).sum().item())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return {
        "acc": correct / max(1, total),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "num_samples": total,
    }


def run_snr_scene_grid(
    args,
    dataset: str,
    arch_name: str,
    num_layers: int,
    channels: int,
    model_size_info: list[int],
    ckpt: str,
) -> list[dict]:
    import torch

    device = torch.device("cuda" if args.gpu > 0 and torch.cuda.is_available() else "cpu")
    model = build_model(args, model_size_info, ckpt, device)
    rows = []

    for scene_idx, scene in enumerate(args.scene_names):
        scene_root = os.path.join(args.scene_test_root, scene)
        noise_roots = [scene_root]
        usable = count_usable_noise_files(noise_roots)
        if usable <= 0:
            print(f"[WARN] scene skipped, no usable wavs: {scene_root}")
            continue
        for snr_idx, snr_db in enumerate(args.test_snrs):
            loader = build_eval_loader(
                args=args,
                dataset=dataset,
                noise_roots=noise_roots,
                snr_db=snr_db,
                seed=500000 + scene_idx * 10007 + snr_idx * 101,
            )
            metrics = eval_acc(model, loader, device)
            row = {
                "dataset": dataset,
                "arch": arch_name,
                "layers": num_layers,
                "channels": channels,
                "expected_params": expected_params(num_layers, channels, NUM_CLASSES),
                "pre_emphasis": args.pre_emphasis,
                "pre_emphasis_coeff": args.pre_emphasis_coeff,
                "scene": scene,
                "snr_db": snr_db,
                "scene_noise_root": scene_root,
                "usable_noise_files": usable,
                "acc": metrics["acc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "num_samples": metrics["num_samples"],
                "ckpt": ckpt,
            }
            rows.append(row)
            print(
                f"[GRID] {dataset} | {arch_name} | scene={scene:<18} | "
                f"snr={snr_db:>5} dB | acc={metrics['acc']:.4f} | f1={metrics['f1']:.4f}"
            )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_train_results(rows: list[dict]):
    if not WRITE_OUTPUTS:
        print("[INFO] CSV output disabled; train results were not written.")
        return

    fieldnames = [
        "dataset",
        "arch",
        "layers",
        "channels",
        "expected_params",
        "pre_emphasis",
        "pre_emphasis_coeff",
        "best_valid_acc",
        "test_acc_on_tau_test_list",
        "f1_on_tau_test_list",
        "returncode",
        "train_save_dir",
        "best_model_saved_as",
    ]
    write_csv(OUT_TRAIN_CSV, rows, fieldnames)
    print(f"[INFO] Train CSV saved to: {OUT_TRAIN_CSV.resolve()}")


def save_grid_results(rows: list[dict]):
    if not WRITE_OUTPUTS:
        print("[INFO] CSV output disabled; grid results were not written.")
        return

    fieldnames = [
        "dataset",
        "arch",
        "layers",
        "channels",
        "expected_params",
        "pre_emphasis",
        "pre_emphasis_coeff",
        "scene",
        "snr_db",
        "scene_noise_root",
        "usable_noise_files",
        "acc",
        "precision",
        "recall",
        "f1",
        "num_samples",
        "ckpt",
    ]
    write_csv(OUT_GRID_CSV, rows, fieldnames)
    print(f"[INFO] Grid CSV saved to: {OUT_GRID_CSV.resolve()}")


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_scene_grid(rows: list[dict]) -> tuple[list[dict], list[dict]]:
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
                "snrs": " ".join(str(item["snr_db"]) for item in sorted(items, key=lambda x: float(x["snr_db"]), reverse=True)),
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

    scene_summary.sort(key=lambda x: (x["dataset"], x["expected_params"], x["scene"]))
    arch_summary.sort(key=lambda x: (x["dataset"], x["expected_params"]))
    return scene_summary, arch_summary


def save_summary_results(grid_rows: list[dict]):
    if not WRITE_OUTPUTS:
        print("[INFO] CSV output disabled; summary results were not written.")
        return

    if not grid_rows:
        return

    scene_summary, arch_summary = summarize_scene_grid(grid_rows)
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
    write_csv(OUT_SCENE_SUMMARY_CSV, scene_summary, scene_fields)
    write_csv(OUT_ARCH_SUMMARY_CSV, arch_summary, arch_fields)
    print(f"[INFO] Scene summary CSV saved to: {OUT_SCENE_SUMMARY_CSV.resolve()}")
    print(f"[INFO] Arch summary CSV saved to: {OUT_ARCH_SUMMARY_CSV.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep DSCNN sizes, then test ACC across fixed SNRs and TAU scenes."
    )
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)

    parser.add_argument(
        "--arch_names",
        nargs="+",
        default=None,
        help="Architecture names to sweep. Default: all built-in ARCHS. Example: --arch_names L5_C64 L3_C16",
    )
    parser.add_argument(
        "--single_arch",
        action="store_true",
        help="Use --arch_name/--num_layers/--channels or --model_size_info instead of sweeping built-in ARCHS.",
    )
    parser.add_argument("--arch_name", default="L5_C64")
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--model_size_info", nargs="+", type=int, default=None)

    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--gpu", type=int, default=GPU)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--dct_coeff", type=int, default=DCT_COEFF)
    parser.add_argument("--window_size_ms", type=int, default=WINDOW_SIZE_MS)
    parser.add_argument("--window_stride_ms", type=int, default=WINDOW_STRIDE_MS)
    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default=MEL_FILTER_SHAPE)
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=PRE_EMPHASIS)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=PRE_EMPHASIS_COEFF)

    parser.add_argument("--train_noise_roots", nargs="+", default=DEFAULT_TRAIN_NOISE_ROOTS)
    parser.add_argument("--valid_noise_roots", nargs="+", default=DEFAULT_VALID_NOISE_ROOTS)
    parser.add_argument("--test_noise_roots", nargs="+", default=DEFAULT_TEST_NOISE_ROOTS)
    parser.add_argument("--train_noise_prob", type=float, default=TRAIN_NOISE_PROB)
    parser.add_argument("--train_snr_min_db", type=float, default=TRAIN_SNR_MIN_DB)
    parser.add_argument("--train_snr_max_db", type=float, default=TRAIN_SNR_MAX_DB)
    parser.add_argument("--valid_noise_prob", type=float, default=VALID_NOISE_PROB)
    parser.add_argument("--valid_snr_db", type=float, default=VALID_SNR_DB)

    parser.add_argument("--scene_test_root", default="./dscnn_kws/noise/tau")
    parser.add_argument("--scene_names", nargs="+", default=TAU_SCENES)
    parser.add_argument("--test_snrs", nargs="+", type=float, default=[20.0, 10.0, 5.0, 0.0, -5.0])
    parser.add_argument("--skip_train", action="store_true", help="Use --ckpt instead of training.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint used when --skip_train is set.")
    parser.add_argument(
        "--out_prefix",
        default="snr_scene_arch_sweep",
        help="Prefix for CSV outputs and copied best-model directory.",
    )
    parser.add_argument("--train_csv", default=None, help="Explicit path for train CSV output.")
    parser.add_argument("--grid_csv", default=None, help="Explicit path for SNR/scene grid CSV output.")
    parser.add_argument("--scene_summary_csv", default=None, help="Explicit path for scene summary CSV output.")
    parser.add_argument("--arch_summary_csv", default=None, help="Explicit path for architecture summary CSV output.")
    parser.add_argument("--best_model_dir", default=None, help="Explicit directory for copied best checkpoints.")
    parser.add_argument("--best_model_suffix", default="noise_best", help="Suffix used in copied best checkpoint names.")
    parser.add_argument(
        "--write_outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write CSV output files. Use --no-write_outputs to disable CSV writes.",
    )
    return parser.parse_args()


def validate_noise_roots(name: str, roots: list[str]):
    count = count_usable_noise_files(roots)
    print(f"[INFO] {name}={roots}")
    print(f"[INFO] {name}_usable_noise_files={count}")
    if count <= 0:
        raise FileNotFoundError(f"No usable wav files found for {name}={roots}")


def selected_archs(args) -> list[tuple[str, int, int, list[int]]]:
    if args.single_arch or args.model_size_info is not None:
        msi = args.model_size_info or make_model_size_info(args.num_layers, args.channels)
        return [(args.arch_name, args.num_layers, args.channels, msi)]

    if args.arch_names:
        unknown = [name for name in args.arch_names if name not in ARCH_BY_NAME]
        if unknown:
            raise ValueError(f"Unknown arch_names={unknown}. Available: {sorted(ARCH_BY_NAME)}")
        archs = [ARCH_BY_NAME[name] for name in args.arch_names]
    else:
        archs = ARCHS

    return [
        (arch_name, num_layers, channels, make_model_size_info(num_layers, channels))
        for arch_name, num_layers, channels in archs
    ]


def main():
    global OUT_TRAIN_CSV
    global OUT_GRID_CSV
    global OUT_SCENE_SUMMARY_CSV
    global OUT_ARCH_SUMMARY_CSV
    global BEST_MODEL_DIR

    args = parse_args()
    configure_outputs(args.out_prefix, args.write_outputs)
    if args.train_csv:
        OUT_TRAIN_CSV = Path(args.train_csv)
    if args.grid_csv:
        OUT_GRID_CSV = Path(args.grid_csv)
    if args.scene_summary_csv:
        OUT_SCENE_SUMMARY_CSV = Path(args.scene_summary_csv)
    if args.arch_summary_csv:
        OUT_ARCH_SUMMARY_CSV = Path(args.arch_summary_csv)
    if args.best_model_dir:
        BEST_MODEL_DIR = Path(args.best_model_dir)
    arch_specs = selected_archs(args)

    print(f"[INFO] out_prefix={args.out_prefix}")
    print(f"[INFO] write_outputs={'ON' if WRITE_OUTPUTS else 'OFF'}")
    print(f"[INFO] best_model_dir={BEST_MODEL_DIR}")
    if WRITE_OUTPUTS:
        print(f"[INFO] train_csv={OUT_TRAIN_CSV}")
        print(f"[INFO] grid_csv={OUT_GRID_CSV}")
        print(f"[INFO] scene_summary_csv={OUT_SCENE_SUMMARY_CSV}")
        print(f"[INFO] arch_summary_csv={OUT_ARCH_SUMMARY_CSV}")

    validate_noise_roots("train_noise_roots", args.train_noise_roots)
    validate_noise_roots("valid_noise_roots", args.valid_noise_roots)
    validate_noise_roots("test_noise_roots", args.test_noise_roots)

    train_rows = []
    grid_rows = []
    if args.skip_train and (len(arch_specs) > 1 or len(args.datasets) > 1):
        raise ValueError("--skip_train currently supports one dataset and one architecture. Use --single_arch and one --datasets item.")

    for arch_name, num_layers, channels, model_size_info in arch_specs:
        for dataset in args.datasets:
            if args.skip_train:
                if not args.ckpt:
                    raise ValueError("--skip_train requires --ckpt")
                ckpt = args.ckpt
                train_row = {
                    "dataset": dataset,
                    "arch": arch_name,
                    "layers": num_layers,
                    "channels": channels,
                    "expected_params": expected_params(num_layers, channels, NUM_CLASSES),
                    "pre_emphasis": args.pre_emphasis,
                    "pre_emphasis_coeff": args.pre_emphasis_coeff,
                    "best_valid_acc": None,
                    "test_acc_on_tau_test_list": None,
                    "f1_on_tau_test_list": None,
                    "returncode": None,
                    "train_save_dir": None,
                    "best_model_saved_as": ckpt,
                }
            else:
                train_row = train_one(args, dataset, arch_name, num_layers, channels, model_size_info)
                ckpt = train_row["best_model_saved_as"]
                train_rows.append(train_row)
                save_train_results(train_rows)

            if not ckpt:
                print(f"[WARN] skip grid, no checkpoint for dataset={dataset}, arch={arch_name}")
                continue
            rows = run_snr_scene_grid(args, dataset, arch_name, num_layers, channels, model_size_info, ckpt)
            grid_rows.extend(rows)
            save_grid_results(grid_rows)
            save_summary_results(grid_rows)

    if train_rows:
        save_train_results(train_rows)
    if grid_rows:
        save_grid_results(grid_rows)
        save_summary_results(grid_rows)


if __name__ == "__main__":
    main()
