from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
from dscnn_kws.data import build_dataloaders
from dscnn_kws.frontend import create_mel_filterbank
from dscnn_kws.frontend.pwl_fit_utils import fit_piecewise_linear_log_from_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Fit piecewise-linear approximation for log() in MFCC")
    parser.add_argument("--root", default="./dataset", type=str)
    parser.add_argument("--dataset", default="speech_commands_v0.02_sr8k", type=str)
    parser.add_argument("--sample_rate", default=8000, type=int)
    parser.add_argument("--window_size_ms", default=32, type=int)
    parser.add_argument("--window_stride_ms", default=32, type=int)
    parser.add_argument("--batch", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--prefetch_factor", default=2, type=int)
    parser.add_argument("--noise_aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow_online_resample", action="store_true", default=False)
    parser.add_argument("--strict_sample_rate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", default=0, type=int, help="only used to satisfy dataloader pin_memory config")

    parser.add_argument("--max_batches", default=120, type=int)
    parser.add_argument("--max_points", default=300000, type=int)
    parser.add_argument("--num_segments", default=6, type=int)
    parser.add_argument("--strategy", choices=["uniform_logx", "quantile", "powerlaw"], default="uniform_logx")
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--log_offset", default=1e-6, type=float)
    parser.add_argument("--sample_seed", default=42, type=int)
    parser.add_argument("--save_json", default="dscnn_kws/frontend/artifacts/log_pwl_fit.json", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)

    data_path = str(Path(args.root) / args.dataset)
    train_loader, _, _ = build_dataloaders(data_path, CLASS_LIST, CLASS_ENCODING, args)

    n_fft = int(args.sample_rate * args.window_size_ms / 1000)
    hop_length = int(args.sample_rate * args.window_stride_ms / 1000)
    mel_fb = create_mel_filterbank(
        sample_rate=args.sample_rate,
        n_fft=n_fft,
        n_mels=40,
        f_min=20,
        f_max=int(args.sample_rate / 2),
        filter_shape="triangular",
    )
    window = torch.hann_window(n_fft)

    samples = []
    total_points = 0
    for bi, (waveform, _) in enumerate(train_loader):
        if bi >= args.max_batches or total_points >= args.max_points:
            break
        x = waveform.squeeze(1)
        stft = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        power_spec = stft.real.pow(2) + stft.imag.pow(2)
        mel_spec = torch.matmul(mel_fb, power_spec)
        xvals = (mel_spec + args.log_offset).reshape(-1)

        if xvals.numel() > 0:
            need = max(0, args.max_points - total_points)
            if need <= 0:
                break
            if xvals.numel() > need:
                idx = torch.randperm(xvals.numel())[:need]
                xvals = xvals[idx]
            samples.append(xvals)
            total_points += xvals.numel()

    if not samples:
        raise RuntimeError("No mel samples collected for fitting")

    x_samples = torch.cat(samples, dim=0)
    fit = fit_piecewise_linear_log_from_samples(
        x_samples=x_samples,
        num_segments=args.num_segments,
        strategy=args.strategy,
        gamma=args.gamma,
    )
    fit.update(
        {
            "sample_rate": args.sample_rate,
            "window_size_ms": args.window_size_ms,
            "window_stride_ms": args.window_stride_ms,
            "n_mels": 40,
            "f_min": 20,
            "f_max": int(args.sample_rate / 2),
            "log_offset": args.log_offset,
            "sample_points": int(x_samples.numel()),
        }
    )

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(fit, f, ensure_ascii=False, indent=2)

    print(f"[INFO] saved={save_path}")
    print(f"[INFO] strategy={fit['strategy']}, segments={fit['num_segments']}")
    print(f"[INFO] fit_mae={fit['fit_mae']:.6f}, fit_max_ae={fit['fit_max_ae']:.6f}")


if __name__ == "__main__":
    main()
