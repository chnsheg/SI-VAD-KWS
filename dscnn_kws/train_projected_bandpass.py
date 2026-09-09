from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torchaudio.transforms import FrequencyMasking, TimeMasking

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST, DEFAULT_MODEL_SIZE_INFO
from dscnn_kws.data import build_dataloaders
from dscnn_kws.engine import Trainer
from dscnn_kws.frontend import load_log_pwl_json
from dscnn_kws.frontend.projected_bandpass_torch import ProjectedBandpass
from dscnn_kws.model import DSCNN
from dscnn_kws.model.dscnn import calculate_time_steps
from dscnn_kws.utils import apply_pre_emphasis, parameter_number, prepare_device, set_random_seed, verify_dataset_sample_rate


class ProjectedBandpassDSCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        sample_rate: int,
        dct_coeff: int,
        window_size_ms: int,
        window_stride_ms: int,
        bandpass_internal_bands: int,
        bandpass_f_min: float,
        bandpass_f_max: float,
        bandpass_spacing: str,
        bandpass_kernel_size: int,
        bandpass_phase_count: int,
        projection_init: str,
        trainable_projection: bool,
        pre_emphasis: bool,
        pre_emphasis_coeff: float,
        spec_aug: bool,
        spec_aug_freq_mask_param: int,
        spec_aug_time_mask_param: int,
        spec_aug_num_freq_masks: int,
        spec_aug_num_time_masks: int,
        log_approx_mode: str,
        log_pwl_num_segments: int,
        log_pwl_strategy: str,
        log_pwl_gamma: float,
        log_pwl_breakpoints: list[float] | None,
        log_pwl_slopes: list[float] | None,
        log_pwl_intercepts: list[float] | None,
        log_offset: float,
        log_input_clamp_min: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.dct_coeff = dct_coeff
        self.pre_emphasis = pre_emphasis
        self.pre_emphasis_coeff = pre_emphasis_coeff
        self.spec_aug = spec_aug

        n_fft = int(sample_rate * window_size_ms / 1000)
        hop_length = int(sample_rate * window_stride_ms / 1000)
        self.feature_extractor = ProjectedBandpass(
            sample_rate=sample_rate,
            internal_bands=bandpass_internal_bands,
            output_bands=dct_coeff,
            frame_length=n_fft,
            frame_hop=hop_length,
            f_min=bandpass_f_min,
            f_max=bandpass_f_max,
            spacing=bandpass_spacing,
            kernel_size=bandpass_kernel_size,
            phase_count=bandpass_phase_count,
            log_approx_mode=log_approx_mode,
            log_pwl_num_segments=log_pwl_num_segments,
            log_pwl_strategy=log_pwl_strategy,
            log_pwl_gamma=log_pwl_gamma,
            log_pwl_breakpoints=log_pwl_breakpoints,
            log_pwl_slopes=log_pwl_slopes,
            log_pwl_intercepts=log_pwl_intercepts,
            log_offset=log_offset,
            log_input_clamp_min=log_input_clamp_min,
            projection_init=projection_init,
            trainable_projection=trainable_projection,
        )
        self.freq_mask = FrequencyMasking(freq_mask_param=max(1, spec_aug_freq_mask_param))
        self.time_mask = TimeMasking(time_mask_param=max(1, spec_aug_time_mask_param))
        self.spec_aug_num_freq_masks = max(0, spec_aug_num_freq_masks)
        self.spec_aug_num_time_masks = max(0, spec_aug_num_time_masks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if self.pre_emphasis:
            x = apply_pre_emphasis(x, self.pre_emphasis_coeff)
        features = self.feature_extractor(x)
        if self.training and self.spec_aug:
            for _ in range(self.spec_aug_num_freq_masks):
                features = self.freq_mask(features)
            for _ in range(self.spec_aug_num_time_masks):
                features = self.time_mask(features)
        features = features[:, : self.dct_coeff, :]
        features = features.permute(0, 2, 1).reshape(features.size(0), -1)
        return self.backbone(features)


def parse_args():
    parser = argparse.ArgumentParser(description="DSCNN KWS training with projected bandpass frontend")
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

    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--dct_coeff", default=10, type=int)
    parser.add_argument("--window_size_ms", default=32, type=int)
    parser.add_argument("--window_stride_ms", default=32, type=int)
    parser.add_argument("--model_size_info", nargs="+", type=int, default=DEFAULT_MODEL_SIZE_INFO)
    parser.add_argument("--bandpass_internal_bands", default=40, type=int)
    parser.add_argument("--bandpass_f_min", default=80.0, type=float)
    parser.add_argument("--bandpass_f_max", default=6000.0, type=float)
    parser.add_argument("--bandpass_spacing", choices=["log", "linear"], default="log")
    parser.add_argument("--bandpass_kernel_size", default=255, type=int)
    parser.add_argument("--bandpass_phase_count", default=4, type=int)
    parser.add_argument("--projection_init", choices=["dct", "average", "random"], default="dct")
    parser.add_argument("--trainable_projection", action=argparse.BooleanOptionalAction, default=True)

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
    parser.add_argument("--spec_aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spec_aug_freq_mask_param", default=1, type=int)
    parser.add_argument("--spec_aug_time_mask_param", default=1, type=int)
    parser.add_argument("--spec_aug_num_freq_masks", default=1, type=int)
    parser.add_argument("--spec_aug_num_time_masks", default=1, type=int)
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default="pwl")
    parser.add_argument("--log_pwl_num_segments", default=8, type=int)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default="uniform_logx")
    parser.add_argument("--log_pwl_gamma", default=1.0, type=float)
    parser.add_argument("--log_pwl_fit_json", default=None, type=str)
    parser.add_argument("--log_offset", default=1e-6, type=float)
    parser.add_argument("--log_input_clamp_min", default=1e-12, type=float)
    return parser.parse_args()


def build_optimizer_scheduler(args, model: nn.Module):
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.weight_decay,
        )

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


def main():
    args = parse_args()
    if args.bandpass_internal_bands < args.dct_coeff:
        raise ValueError(
            f"bandpass_internal_bands must be >= dct_coeff, got "
            f"{args.bandpass_internal_bands} < {args.dct_coeff}"
        )

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

    time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
    input_dim = time_steps * args.dct_coeff
    backbone = DSCNN(
        input_dim=input_dim,
        label_count=len(CLASS_LIST),
        model_size_info=args.model_size_info,
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
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        spec_aug=args.spec_aug,
        spec_aug_freq_mask_param=args.spec_aug_freq_mask_param,
        spec_aug_time_mask_param=args.spec_aug_time_mask_param,
        spec_aug_num_freq_masks=args.spec_aug_num_freq_masks,
        spec_aug_num_time_masks=args.spec_aug_num_time_masks,
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
        f"projected_bandpass_{args.dataset}_ib{args.bandpass_internal_bands}_"
        f"to{args.dct_coeff}_lr{args.lr}_ep{args.epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_dir = os.path.join("dscnn_kws", "runs", run_name)

    print(f"[INFO] data_path={data_path}")
    print(f"[INFO] device={device}, params={parameter_number(model)}")
    print(f"[INFO] save_dir={save_dir}")
    print(f"[INFO] frontend=projected_bandpass")
    print(
        f"[INFO] projected_bandpass: internal_bands={args.bandpass_internal_bands}, "
        f"output_bands={args.dct_coeff}, f_min={args.bandpass_f_min}, f_max={args.bandpass_f_max}, "
        f"spacing={args.bandpass_spacing}, kernel_size={args.bandpass_kernel_size}, "
        f"phase_count={args.bandpass_phase_count}, projection_init={args.projection_init}, "
        f"trainable_projection={args.trainable_projection}"
    )
    print(f"[INFO] log_approx_mode={args.log_approx_mode}")
    if args.log_approx_mode == "pwl":
        print(
            f"[INFO] log_pwl_num_segments={args.log_pwl_num_segments}, "
            f"strategy={args.log_pwl_strategy}, gamma={args.log_pwl_gamma}"
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
