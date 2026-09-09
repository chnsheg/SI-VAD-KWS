from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
from torch import optim

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
from dscnn_kws.data import build_dataloaders
from dscnn_kws.engine import Trainer
from dscnn_kws.frontend import load_log_pwl_json
from dscnn_kws.streaming.streaming_crnn import StreamingKWSModel, model_config_dict, parse_cnn_channels
from dscnn_kws.utils import parameter_number, prepare_device, set_random_seed, verify_dataset_sample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train streaming CRNN-GRU KWS prototype")

    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch", default=256, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--non_deterministic", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--root", default="./dataset", type=str)
    parser.add_argument("--dataset", default="speech_commands_v0.02_sr8k", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--noise_aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_noise_aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise_roots", nargs="*", default=None)
    parser.add_argument("--train_noise_roots", nargs="*", default=None)
    parser.add_argument("--valid_noise_roots", nargs="*", default=None)
    parser.add_argument("--test_noise_roots", nargs="*", default=None)
    parser.add_argument("--noise_aug_prob", default=0.8, type=float)
    parser.add_argument("--noise_snr_min_db", default=-5.0, type=float)
    parser.add_argument("--noise_snr_max_db", default=20.0, type=float)
    parser.add_argument("--eval_noise_aug_prob", default=None, type=float)
    parser.add_argument("--eval_noise_snr_min_db", default=None, type=float)
    parser.add_argument("--eval_noise_snr_max_db", default=None, type=float)
    parser.add_argument("--allow_online_resample", action="store_true", default=False)
    parser.add_argument("--strict_sample_rate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify_sample_rate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify_sample_per_split", default=80, type=int)

    # Keep frontend/input defaults aligned with the previous L5_C64 DSCNN sweep.
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

    # parser.add_argument("--cnn_channels", default="64,64,64,64,64", type=str)
    parser.add_argument("--cnn_channels", default="24,24,24,24,24", type=str)
    parser.add_argument("--kernel_time", default=5, type=int)
    parser.add_argument("--kernel_freq", default=3, type=int)
    parser.add_argument("--gru_hidden", default=64, type=int)
    parser.add_argument("--gru_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)

    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre_emphasis_coeff", default=0.97, type=float)
    parser.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--scheduler", choices=["cos", "step"], default="cos")
    parser.add_argument("--t0", default=10, type=int)
    parser.add_argument("--t_mult", default=1, type=int)
    parser.add_argument("--eta_min", default=None, type=float)
    parser.add_argument("--step_size", default=20, type=int)
    parser.add_argument("--gamma", default=0.2, type=float)
    parser.add_argument("--label_smoothing", default=0.0, type=float)

    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default="triangular")
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default="exact")
    parser.add_argument("--log_pwl_num_segments", default=6, type=int)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default="uniform_logx")
    parser.add_argument("--log_pwl_gamma", default=1.0, type=float)
    parser.add_argument("--log_pwl_fit_json", default=None, type=str)
    parser.add_argument("--log_offset", default=1e-6, type=float)
    parser.add_argument("--log_input_clamp_min", default=1e-12, type=float)

    parser.add_argument("--save_root", default="dscnn_kws/streaming/runs", type=str)
    return parser.parse_args()


def build_optimizer_scheduler(args: argparse.Namespace, model: torch.nn.Module):
    if args.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cos":
        eta_min = args.eta_min if args.eta_min is not None else args.lr * 0.01
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.t0,
            T_mult=args.t_mult,
            eta_min=eta_min,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return optimizer, scheduler


def main() -> None:
    args = parse_args()
    if args.frontend == "bandpass" and args.dct_coeff != args.bandpass_n_bands:
        raise ValueError("For bandpass frontend, --dct_coeff must equal --bandpass_n_bands")

    log_pwl_breakpoints = None
    log_pwl_slopes = None
    log_pwl_intercepts = None
    if args.log_pwl_fit_json:
        cfg = load_log_pwl_json(args.log_pwl_fit_json)
        log_pwl_breakpoints = cfg["breakpoints"]
        log_pwl_slopes = cfg["slopes"]
        log_pwl_intercepts = cfg["intercepts"]

    set_random_seed(args.seed, deterministic=not args.non_deterministic)
    device, _ = prepare_device(args.gpu)

    data_path = os.path.join(args.root, args.dataset)
    if args.verify_sample_rate:
        verify_dataset_sample_rate(
            data_path=data_path,
            expected_sample_rate=args.sample_rate,
            sample_per_split=max(1, args.verify_sample_per_split),
            random_seed=args.seed,
        )

    train_loader, valid_loader, test_loader = build_dataloaders(data_path, CLASS_LIST, CLASS_ENCODING, args)
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

    optimizer, scheduler = build_optimizer_scheduler(args, model)
    run_name = (
        f"streaming_crnn_{args.frontend}_{args.dataset}_"
        f"ch{args.cnn_channels.replace(',', '-')}_gru{args.gru_hidden}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_dir = os.path.join(args.save_root, run_name)

    print(f"[INFO] data_path={data_path}")
    print(f"[INFO] device={device}, params={parameter_number(model)}")
    print(f"[INFO] save_dir={save_dir}")
    print(f"[INFO] model_config={model_config_dict(model)}")
    print(
        f"[INFO] frontend={args.frontend}, "
        f"streaming_mfcc={args.streaming_mfcc}, mfcc_center={args.mfcc_center}"
    )
    print(f"[INFO] cnn_channels={parse_cnn_channels(args.cnn_channels)}, gru_hidden={args.gru_hidden}")
    print(
        f"[INFO] noise_aug={'ON' if args.noise_aug else 'OFF'}, "
        f"eval_noise_aug={'ON' if args.eval_noise_aug else 'OFF'}, "
        f"snr=[{args.noise_snr_min_db}, {args.noise_snr_max_db}] dB"
    )

    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        save_dir=save_dir,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
