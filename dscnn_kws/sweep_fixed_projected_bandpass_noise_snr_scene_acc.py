from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dscnn_kws.sweep_fixed_bandpass_noise_snr_scene_acc as base


# =============================================================================
# Code-level configuration
# =============================================================================
# Normally you only need to edit this block, then run:
#   python dscnn_kws/sweep_fixed_projected_bandpass_noise_snr_scene_acc.py
#
# Command-line flags are still supported and override these defaults.

# Data and output.
ROOT = base.ROOT
DATASETS = [
    "mobvoi_hi_xiaowen_binary_hardneg",
    "mobvoi_nihao_wenwen_binary_hardneg",
]
OUT_PREFIX = "projected_bandpass40_to10_pwl_snr_scene_arch_sweep"
WRITE_OUTPUTS = True

# Architecture. Default is sweep mode.
# Keep SINGLE_ARCH=False and ARCH_NAMES=None to sweep all local built-in archs.
# Set ARCH_NAMES=["L5_C64", "L5_C48"] to sweep selected archs.
# Set SINGLE_ARCH=True for one DSCNN backbone such as L5_C64.
SINGLE_ARCH = False
ARCH_NAMES = [
    "L5_C64",
    "L5_C48",
    "L5_C32",
    "L5_C24",
    "L5_C16",
]
ARCH_NAME = "L5_C64"
NUM_LAYERS = 5
CHANNELS = 64
MODEL_SIZE_INFO = None

# Local architecture list. This file no longer reuses base.ARCHS.
ARCHS = [
    ("L5_C64", 5, 64),
    ("L5_C48", 5, 48),
    ("L5_C32", 5, 32),
    ("L5_C24", 5, 24),
    ("L5_C16", 5, 16),
]
ARCH_BY_NAME = {name: (name, layers, channels) for name, layers, channels in ARCHS}

# Training runtime.
EPOCH = base.EPOCH
BATCH = base.BATCH
SAMPLE_RATE = base.SAMPLE_RATE
GPU = base.GPU
NUM_WORKERS = base.NUM_WORKERS
WINDOW_SIZE_MS = base.WINDOW_SIZE_MS
WINDOW_STRIDE_MS = base.WINDOW_STRIDE_MS

# Projected bandpass frontend. DSCNN input stays DCT_COEFF x time_steps.
DCT_COEFF = 10
BANDPASS_INTERNAL_BANDS = 40
BANDPASS_F_MIN = 80.0
BANDPASS_F_MAX = 6000.0
BANDPASS_SPACING = "log"
BANDPASS_KERNEL_SIZE = 255
BANDPASS_PHASE_COUNT = 4
PROJECTION_INIT = "dct"  # dct, average, or random
TRAINABLE_PROJECTION = True

# Log/PWL configuration.
LOG_APPROX_MODE = "pwl"
LOG_PWL_NUM_SEGMENTS = 8
LOG_PWL_STRATEGY = base.LOG_PWL_STRATEGY
LOG_PWL_GAMMA = base.LOG_PWL_GAMMA
LOG_OFFSET = base.LOG_OFFSET
LOG_INPUT_CLAMP_MIN = base.LOG_INPUT_CLAMP_MIN

# Noise augmentation and scene/SNR evaluation.
TRAIN_NOISE_ROOTS = base.DEFAULT_TRAIN_NOISE_ROOTS
VALID_NOISE_ROOTS = base.DEFAULT_VALID_NOISE_ROOTS
TEST_NOISE_ROOTS = base.DEFAULT_TEST_NOISE_ROOTS
TRAIN_NOISE_PROB = base.TRAIN_NOISE_PROB
TRAIN_SNR_MIN_DB = base.TRAIN_SNR_MIN_DB
TRAIN_SNR_MAX_DB = base.TRAIN_SNR_MAX_DB
VALID_NOISE_PROB = base.VALID_NOISE_PROB
VALID_SNR_DB = base.VALID_SNR_DB
SCENE_TEST_ROOT = "./dscnn_kws/noise/tau"
SCENE_NAMES = base.TAU_SCENES
TEST_SNRS = [20.0, 10.0, 5.0, 0.0, -5.0]

# Evaluation-only mode. Set SKIP_TRAIN=True and CKPT to a checkpoint path.
SKIP_TRAIN = False
CKPT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep DSCNN with projected bandpass frontend: internal bands -> learnable projection -> 10 dims."
    )
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--arch_names", nargs="+", default=ARCH_NAMES)
    parser.add_argument("--single_arch", action=argparse.BooleanOptionalAction, default=SINGLE_ARCH)
    parser.add_argument("--arch_name", default=ARCH_NAME)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--model_size_info", nargs="+", type=int, default=MODEL_SIZE_INFO)

    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--gpu", type=int, default=GPU)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--dct_coeff", type=int, default=DCT_COEFF)
    parser.add_argument("--window_size_ms", type=int, default=WINDOW_SIZE_MS)
    parser.add_argument("--window_stride_ms", type=int, default=WINDOW_STRIDE_MS)

    parser.add_argument("--bandpass_internal_bands", type=int, default=BANDPASS_INTERNAL_BANDS)
    parser.add_argument("--bandpass_f_min", type=float, default=BANDPASS_F_MIN)
    parser.add_argument("--bandpass_f_max", type=float, default=BANDPASS_F_MAX)
    parser.add_argument("--bandpass_spacing", choices=["log", "linear"], default=BANDPASS_SPACING)
    parser.add_argument("--bandpass_kernel_size", type=int, default=BANDPASS_KERNEL_SIZE)
    parser.add_argument("--bandpass_phase_count", type=int, default=BANDPASS_PHASE_COUNT)
    parser.add_argument("--projection_init", choices=["dct", "average", "random"], default=PROJECTION_INIT)
    parser.add_argument(
        "--trainable_projection",
        action=argparse.BooleanOptionalAction,
        default=TRAINABLE_PROJECTION,
    )
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default=LOG_APPROX_MODE)
    parser.add_argument("--log_pwl_num_segments", type=int, default=LOG_PWL_NUM_SEGMENTS)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default=LOG_PWL_STRATEGY)
    parser.add_argument("--log_pwl_gamma", type=float, default=LOG_PWL_GAMMA)
    parser.add_argument("--log_offset", type=float, default=LOG_OFFSET)
    parser.add_argument("--log_input_clamp_min", type=float, default=LOG_INPUT_CLAMP_MIN)

    parser.add_argument("--train_noise_roots", nargs="+", default=TRAIN_NOISE_ROOTS)
    parser.add_argument("--valid_noise_roots", nargs="+", default=VALID_NOISE_ROOTS)
    parser.add_argument("--test_noise_roots", nargs="+", default=TEST_NOISE_ROOTS)
    parser.add_argument("--train_noise_prob", type=float, default=TRAIN_NOISE_PROB)
    parser.add_argument("--train_snr_min_db", type=float, default=TRAIN_SNR_MIN_DB)
    parser.add_argument("--train_snr_max_db", type=float, default=TRAIN_SNR_MAX_DB)
    parser.add_argument("--valid_noise_prob", type=float, default=VALID_NOISE_PROB)
    parser.add_argument("--valid_snr_db", type=float, default=VALID_SNR_DB)

    parser.add_argument("--scene_test_root", default=SCENE_TEST_ROOT)
    parser.add_argument("--scene_names", nargs="+", default=SCENE_NAMES)
    parser.add_argument("--test_snrs", nargs="+", type=float, default=TEST_SNRS)
    parser.add_argument("--skip_train", action=argparse.BooleanOptionalAction, default=SKIP_TRAIN)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--out_prefix", default=OUT_PREFIX)
    parser.add_argument("--write_outputs", action=argparse.BooleanOptionalAction, default=WRITE_OUTPUTS)
    return parser.parse_args()


def make_model_size_info(num_layers: int, channels: int) -> list[int]:
    info = [num_layers]
    info += [channels, 10, 4, 2, 2]
    for _ in range(num_layers - 1):
        info += [channels, 3, 3, 1, 1]
    return info


def selected_archs(args) -> list[tuple[str, int, int, list[int]]]:
    if args.single_arch or args.model_size_info is not None:
        model_size_info = args.model_size_info or make_model_size_info(args.num_layers, args.channels)
        return [(args.arch_name, args.num_layers, args.channels, model_size_info)]

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


def copy_best_model(args, output: str, dataset: str, arch_name: str, num_layers: int, channels: int):
    m = re.search(r"\[INFO\]\s+save_dir=(.+)", output)
    if not m:
        return None, None

    save_dir = Path(m.group(1).strip())
    best_path = save_dir / "best.pt"
    if not best_path.exists():
        return str(save_dir), None

    base.BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    params = base.expected_params(num_layers, channels, base.NUM_CLASSES)
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
    out_path = base.BEST_MODEL_DIR / (
        f"{safe_dataset}_{arch_name}_layers{num_layers}_channels{channels}_"
        f"params{params}_projected_bandpass{args.bandpass_internal_bands}_to{args.dct_coeff}_"
        f"{args.log_approx_mode}_noise_best.pt"
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
        "dscnn_kws.train_projected_bandpass",
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
        "--bandpass_internal_bands",
        str(args.bandpass_internal_bands),
        "--bandpass_f_min",
        str(args.bandpass_f_min),
        "--bandpass_f_max",
        str(args.bandpass_f_max),
        "--bandpass_spacing",
        args.bandpass_spacing,
        "--bandpass_kernel_size",
        str(args.bandpass_kernel_size),
        "--bandpass_phase_count",
        str(args.bandpass_phase_count),
        "--projection_init",
        args.projection_init,
        "--log_approx_mode",
        args.log_approx_mode,
        "--log_pwl_num_segments",
        str(args.log_pwl_num_segments),
        "--log_pwl_strategy",
        args.log_pwl_strategy,
        "--log_pwl_gamma",
        str(args.log_pwl_gamma),
        "--log_offset",
        str(args.log_offset),
        "--log_input_clamp_min",
        str(args.log_input_clamp_min),
        "--window_size_ms",
        str(args.window_size_ms),
        "--window_stride_ms",
        str(args.window_stride_ms),
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
    ]
    if args.trainable_projection:
        cmd.append("--trainable_projection")
    else:
        cmd.append("--no-trainable_projection")

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

    save_dir, best_model_saved_as = copy_best_model(args, output, dataset, arch_name, num_layers, channels)
    return {
        "dataset": dataset,
        "arch": arch_name,
        "layers": num_layers,
        "channels": channels,
        "expected_params": base.expected_params(num_layers, channels, base.NUM_CLASSES),
        "best_valid_acc": best_valid_acc,
        "test_acc_on_tau_test_list": test_acc,
        "f1_on_tau_test_list": f1,
        "returncode": proc.returncode,
        "train_save_dir": save_dir,
        "best_model_saved_as": best_model_saved_as,
    }


def build_model(args, model_size_info: list[int], ckpt: str, device):
    from dscnn_kws.configs import CLASS_LIST
    from dscnn_kws.model import DSCNN
    from dscnn_kws.model.dscnn import calculate_time_steps
    from dscnn_kws.train_projected_bandpass import ProjectedBandpassDSCNN

    time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
    input_dim = time_steps * args.dct_coeff
    backbone = DSCNN(
        input_dim=input_dim,
        label_count=len(CLASS_LIST),
        model_size_info=model_size_info,
        dct_coeff=args.dct_coeff,
    )
    model = ProjectedBandpassDSCNN(
        backbone=backbone,
        sample_rate=args.sample_rate,
        dct_coeff=args.dct_coeff,
        window_size_ms=args.window_size_ms,
        window_stride_ms=args.window_stride_ms,
        bandpass_internal_bands=args.bandpass_internal_bands,
        bandpass_f_min=args.bandpass_f_min,
        bandpass_f_max=args.bandpass_f_max,
        bandpass_spacing=args.bandpass_spacing,
        bandpass_kernel_size=args.bandpass_kernel_size,
        bandpass_phase_count=args.bandpass_phase_count,
        projection_init=args.projection_init,
        trainable_projection=args.trainable_projection,
        pre_emphasis=True,
        pre_emphasis_coeff=0.97,
        spec_aug=False,
        spec_aug_freq_mask_param=1,
        spec_aug_time_mask_param=1,
        spec_aug_num_freq_masks=0,
        spec_aug_num_time_masks=0,
        log_approx_mode=args.log_approx_mode,
        log_pwl_num_segments=args.log_pwl_num_segments,
        log_pwl_strategy=args.log_pwl_strategy,
        log_pwl_gamma=args.log_pwl_gamma,
        log_pwl_breakpoints=None,
        log_pwl_slopes=None,
        log_pwl_intercepts=None,
        log_offset=args.log_offset,
        log_input_clamp_min=args.log_input_clamp_min,
    ).to(device)
    model.load_state_dict(base.load_state_dict(ckpt, device))
    model.eval()
    return model


def run_snr_scene_grid(
    args,
    dataset: str,
    arch_name: str,
    num_layers: int,
    channels: int,
    model_size_info: list[int],
    ckpt: str,
) -> list[dict]:
    from dscnn_kws.utils import prepare_device

    device, _ = prepare_device(args.gpu)
    model = build_model(args, model_size_info, ckpt, device)
    rows = []
    for scene_idx, scene in enumerate(args.scene_names):
        scene_root = os.path.join(args.scene_test_root, scene)
        noise_roots = [scene_root]
        usable_noise_files = base.count_usable_noise_files(noise_roots)
        if usable_noise_files <= 0:
            print(f"[WARN] skip scene={scene}: no usable wav files under {scene_root}")
            continue

        for snr_idx, snr_db in enumerate(args.test_snrs):
            seed = 500000 + scene_idx * 10007 + snr_idx * 101
            loader = base.build_eval_loader(args, dataset, noise_roots, snr_db, seed)
            metrics = base.eval_acc(model, loader, device)
            row = {
                "dataset": dataset,
                "arch": arch_name,
                "layers": num_layers,
                "channels": channels,
                "expected_params": base.expected_params(num_layers, channels, base.NUM_CLASSES),
                "scene": scene,
                "snr_db": snr_db,
                "scene_noise_root": scene_root,
                "usable_noise_files": usable_noise_files,
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


def main():
    args = parse_args()
    if args.bandpass_internal_bands < args.dct_coeff:
        raise ValueError(
            f"--bandpass_internal_bands must be >= --dct_coeff, got "
            f"{args.bandpass_internal_bands} < {args.dct_coeff}"
        )

    base.configure_outputs(args.out_prefix, args.write_outputs)
    arch_specs = selected_archs(args)

    print("[INFO] frontend=projected_bandpass")
    print(
        "[INFO] projected_bandpass="
        f"internal_bands={args.bandpass_internal_bands}, "
        f"output_bands={args.dct_coeff}, "
        f"f_min={args.bandpass_f_min}, "
        f"f_max={args.bandpass_f_max}, "
        f"spacing={args.bandpass_spacing}, "
        f"kernel_size={args.bandpass_kernel_size}, "
        f"phase_count={args.bandpass_phase_count}, "
        f"projection_init={args.projection_init}, "
        f"trainable_projection={args.trainable_projection}"
    )
    print(f"[INFO] log_approx_mode={args.log_approx_mode}")
    if args.log_approx_mode == "pwl":
        print(
            f"[INFO] log_pwl_num_segments={args.log_pwl_num_segments}, "
            f"strategy={args.log_pwl_strategy}, gamma={args.log_pwl_gamma}"
        )
    if args.write_outputs:
        print(f"[INFO] train_csv={base.OUT_TRAIN_CSV}")
        print(f"[INFO] grid_csv={base.OUT_GRID_CSV}")
        print(f"[INFO] scene_summary_csv={base.OUT_SCENE_SUMMARY_CSV}")
        print(f"[INFO] arch_summary_csv={base.OUT_ARCH_SUMMARY_CSV}")
        print(f"[INFO] best_model_dir={base.BEST_MODEL_DIR}")

    base.validate_noise_roots("train_noise_roots", args.train_noise_roots)
    base.validate_noise_roots("valid_noise_roots", args.valid_noise_roots)
    base.validate_noise_roots("test_noise_roots", args.test_noise_roots)

    if args.skip_train and (len(arch_specs) > 1 or len(args.datasets) > 1):
        raise ValueError("--skip_train currently supports one dataset and one architecture.")

    train_rows = []
    grid_rows = []
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
                    "expected_params": base.expected_params(num_layers, channels, base.NUM_CLASSES),
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
            base.save_train_results(train_rows)
            if ckpt:
                rows = run_snr_scene_grid(args, dataset, arch_name, num_layers, channels, model_size_info, ckpt)
                grid_rows.extend(rows)
                base.save_grid_results(grid_rows)
                base.save_summary_results(grid_rows)

    if not train_rows:
        base.save_train_results(train_rows)
    if not grid_rows:
        base.save_grid_results(grid_rows)
        base.save_summary_results(grid_rows)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
