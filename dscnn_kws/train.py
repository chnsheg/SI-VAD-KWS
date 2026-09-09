from __future__ import annotations

import argparse
import math
import os
import traceback
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torchaudio.functional import create_dct
from torchaudio.transforms import FrequencyMasking, MFCC, MelSpectrogram, TimeMasking

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST, DEFAULT_MODEL_SIZE_INFO
from dscnn_kws.data import build_confirmation_pair_loader, build_dataloaders
from dscnn_kws.engine import Trainer
from dscnn_kws.frontend import TorchBandpass, TorchMFCC, load_log_pwl_json
from dscnn_kws.model import CepstralTCN, DSCNN, LSTM, MFCCLSTM
from dscnn_kws.model.dscnn import calculate_time_steps
from dscnn_kws.utils import apply_pre_emphasis, parameter_number, prepare_device, set_random_seed, verify_dataset_sample_rate
from dscnn_kws.utils import barrier, destroy_distributed, init_distributed, is_rank_zero
from dscnn_kws.utils.training_artifacts import FailureArtifactReporter, TrainingArtifactWriter


class WarmupCosineScheduler:
    """A checkpointable per-update linear warmup followed by cosine decay."""

    def __init__(self, optimizer, *, warmup_steps: int, total_steps: int, eta_min: float):
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        if total_steps <= warmup_steps:
            raise ValueError("total_steps must exceed warmup_steps")
        if eta_min < 0:
            raise ValueError("eta_min must be non-negative")
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.eta_min = float(eta_min)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = 0
        self._set_lrs(self.last_step)

    def _lr(self, base_lr: float, step: int) -> float:
        if step < self.warmup_steps:
            return base_lr * float(step + 1) / float(self.warmup_steps)
        progress = min(1.0, float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps))
        return self.eta_min + (base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _set_lrs(self, step: int) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._lr(base_lr, step)

    def step(self) -> None:
        self.last_step += 1
        self._set_lrs(self.last_step)

    def state_dict(self) -> dict[str, object]:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "eta_min": self.eta_min,
            "base_lrs": list(self.base_lrs),
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.last_step = int(state_dict["last_step"])
        self._set_lrs(self.last_step)


class BatchInvariantMFCC(nn.Module):
    """Per-item torchaudio-compatible dB MFCC without batch-dependent top_db."""

    def __init__(self, sample_rate: int, n_mfcc: int, melkwargs: dict[str, object]) -> None:
        super().__init__()
        melkwargs = dict(melkwargs)
        n_mels = int(melkwargs.get("n_mels", 128))
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, **melkwargs)
        self.register_buffer("dct_mat", create_dct(n_mfcc, n_mels, norm="ortho"))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spectrogram(waveform)
        db = 10.0 * torch.log10(mel.clamp_min(1e-10))
        db = torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - 80.0)
        return torch.matmul(db.transpose(-1, -2), self.dct_mat).transpose(-1, -2)


class MFCCDSCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        frontend: str,
        sample_rate: int,
        dct_coeff: int,
        window_size_ms: int,
        window_stride_ms: int,
        bandpass_n_bands: int,
        bandpass_f_min: float,
        bandpass_f_max: float,
        bandpass_spacing: str,
        bandpass_kernel_size: int,
        bandpass_phase_count: int,
        pre_emphasis: bool,
        pre_emphasis_coeff: float,
        spec_aug: bool,
        spec_aug_freq_mask_param: int,
        spec_aug_time_mask_param: int,
        spec_aug_num_freq_masks: int,
        spec_aug_num_time_masks: int,
        mfcc_impl: str,
        mel_filter_shape: str,
        log_approx_mode: str,
        log_pwl_num_segments: int,
        log_pwl_strategy: str,
        log_pwl_gamma: float,
        log_pwl_breakpoints: list[float] | None,
        log_pwl_slopes: list[float] | None,
        log_pwl_intercepts: list[float] | None,
        log_offset: float,
        log_input_clamp_min: float,
        amp_backbone: bool = False,
        mfcc_scale: str = "torchaudio_db",
        mfcc_c0_cmn: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.frontend = frontend
        self.dct_coeff = dct_coeff
        self.pre_emphasis = pre_emphasis
        self.pre_emphasis_coeff = pre_emphasis_coeff
        self.spec_aug = spec_aug
        self.mfcc_impl = mfcc_impl
        if mfcc_scale not in {"natural_log", "torchaudio_db"}:
            raise ValueError("mfcc_scale must be 'natural_log' or 'torchaudio_db'")
        self.mfcc_scale = mfcc_scale
        self.mfcc_c0_cmn = bool(mfcc_c0_cmn)
        if self.mfcc_c0_cmn and frontend != "mfcc":
            raise ValueError("mfcc_c0_cmn requires frontend='mfcc'")
        self.mel_filter_shape = mel_filter_shape
        self.amp_backbone = amp_backbone

        n_fft = int(sample_rate * window_size_ms / 1000)
        hop_length = int(sample_rate * window_stride_ms / 1000)
        if frontend == "mfcc":
            if mfcc_impl == "torchaudio":
                melkwargs = {
                        "n_fft": n_fft,
                        "win_length": n_fft,
                        "hop_length": hop_length,
                        "n_mels": 40,
                        "f_min": 20,
                        "f_max": int(sample_rate / 2),
                        "window_fn": torch.hann_window,
                        "center": True,
                    }
                if mfcc_scale == "torchaudio_db":
                    self.feature_extractor = BatchInvariantMFCC(sample_rate, 40, melkwargs)
                else:
                    self.feature_extractor = MFCC(
                        sample_rate=sample_rate,
                        n_mfcc=40,
                        log_mels=True,
                        melkwargs=melkwargs,
                    )
            else:
                self.feature_extractor = TorchMFCC(
                    sample_rate=sample_rate,
                    n_mfcc=40,
                    n_fft=n_fft,
                    win_length=n_fft,
                    hop_length=hop_length,
                    n_mels=40,
                    f_min=20,
                    f_max=int(sample_rate / 2),
                    center=True,
                    dct_norm="ortho",
                    mel_filter_shape=mel_filter_shape,
                    log_approx_mode=log_approx_mode,
                    log_pwl_num_segments=log_pwl_num_segments,
                    log_pwl_strategy=log_pwl_strategy,
                    log_pwl_gamma=log_pwl_gamma,
                    log_pwl_breakpoints=log_pwl_breakpoints,
                    log_pwl_slopes=log_pwl_slopes,
                    log_pwl_intercepts=log_pwl_intercepts,
                    log_offset=log_offset,
                    log_input_clamp_min=log_input_clamp_min,
                )
        else:
            self.feature_extractor = TorchBandpass(
                sample_rate=sample_rate,
                n_bands=bandpass_n_bands,
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
            )
        self.freq_mask = FrequencyMasking(freq_mask_param=max(1, spec_aug_freq_mask_param))
        self.time_mask = TimeMasking(time_mask_param=max(1, spec_aug_time_mask_param))
        self.spec_aug_num_freq_masks = max(0, spec_aug_num_freq_masks)
        self.spec_aug_num_time_masks = max(0, spec_aug_num_time_masks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.float()
        if self.pre_emphasis:
            x = apply_pre_emphasis(x, self.pre_emphasis_coeff)
        # torchaudio.MFCC uses AmplitudeToDB(top_db=80) by default.  When
        # called with [B, T], its 3-D mel tensor is interpreted as one
        # ``(channel, frequency, time)`` block and the top-dB reference is
        # shared across the whole batch.  Consequently a sample's features
        # (and score) depend on which other utterances happen to be in the
        # batch; realtime inference (B=1) then disagrees with training.
        # Keep the historical dB transform while making the leading batch
        # dimension explicit as a non-reduced axis: [B, 1, T] ->
        # [B, 1, n_mels, frames].  amplitude_to_DB reduces only the final
        # three dimensions, yielding an independent cutoff per sample.
        mfcc = self.feature_extractor(x.unsqueeze(1) if self.mfcc_impl == "torchaudio" else x).float()
        if mfcc.dim() == 4 and mfcc.size(1) == 1:
            mfcc = mfcc.squeeze(1)
        if bool(getattr(self, "mfcc_c0_cmn", False)):
            c0 = mfcc[:, :1, :]
            mfcc = torch.cat((c0 - c0.mean(dim=-1, keepdim=True), mfcc[:, 1:, :]), dim=1)
        if self.training and self.spec_aug:
            for _ in range(self.spec_aug_num_freq_masks):
                mfcc = self.freq_mask(mfcc)
            for _ in range(self.spec_aug_num_time_masks):
                mfcc = self.time_mask(mfcc)
        mfcc = mfcc[:, : self.dct_coeff, :]
        mfcc = mfcc.permute(0, 2, 1).reshape(mfcc.size(0), -1)
        if self.amp_backbone and mfcc.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.backbone(mfcc)
            return logits.float()
        return self.backbone(mfcc)


def parse_args():
    parser = argparse.ArgumentParser(description="DSCNN-only KWS training")
    parser.add_argument("--model", choices=["dscnn", "cepstral_tcn", "lstm"], default="dscnn")
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--save_dir", default=None, type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument(
        "--temporal_init_from_global",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "将旧 global-pooling DSCNN checkpoint 迁移为 temporal head 的初始化；"
            "仅复制网络权重并从 epoch 1/新优化器开始训练"
        ),
    )
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--live_telemetry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch", default=256, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument("--ddp_static_graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ddp_gradient_as_bucket_view", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp_backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trainable_tail_blocks", default=0, type=int)
    parser.add_argument(
        "--freeze_tail_batch_norm_stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep BatchNorm running statistics fixed in trainable DSCNN tail blocks while "
            "leaving their affine weight and bias trainable"
        ),
    )
    parser.add_argument("--max_train_steps", default=None, type=int)
    parser.add_argument("--offline_augmented_dataset", action="store_true", default=False)
    parser.add_argument("--online_window_jitter_ms", default=None, type=int)
    parser.add_argument("--train_manifest", default="", type=str)
    parser.add_argument(
        "--pair_train_manifest",
        default="",
        type=str,
        help="显式连续双窗训练 JSONL；必须与 --pair_objective 一起使用",
    )
    parser.add_argument("--pair_hop_ms", default=96.0, type=float)
    parser.add_argument("--pair_batch", default=None, type=int)
    parser.add_argument(
        "--pair_positive_per_batch",
        default=0,
        type=int,
        help="每个 rank 的 pair batch 固定正例数；0 保持历史 shuffle，r14 显式启用",
    )
    parser.add_argument("--pair_num_workers", default=None, type=int)
    parser.add_argument(
        "--pair_interval",
        default=1,
        type=int,
        help="每 N 个基础分类 step 计算一次相邻双窗辅助损失",
    )
    parser.add_argument("--validation_manifest", default="", type=str)
    parser.add_argument("--test_manifest", default="", type=str)
    parser.add_argument("--packed_train_index", default="", type=str)
    parser.add_argument("--packed_block_records", default=16_384, type=int)
    parser.add_argument("--mixture_base_pack", default="", type=str)
    parser.add_argument("--mixture_raw_anchor_pack", default="", type=str)
    parser.add_argument("--mixture_hard_negative_pack", default="", type=str)
    parser.add_argument("--mixture-v3-role", dest="mixture_v3_role", action="append", default=[])
    parser.add_argument("--mixture-v3-quota", dest="mixture_v3_quota", action="append", default=[])
    parser.add_argument(
        "--mixture-v3-allow-replacement-role",
        dest="mixture_v3_allow_replacement_role",
        action="append",
        default=[],
    )
    parser.add_argument("--mixture_steps_per_epoch", default=0, type=int)
    parser.add_argument("--skip_test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min_positive_recall", default=0.0, type=float)
    parser.add_argument("--early_stopping_min_epoch", default=0, type=int)
    parser.add_argument("--early_stopping_patience", default=0, type=int)
    parser.add_argument("--candidate-checkpoint-every", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--non_deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="与旧工程一致：默认启用非确定性 cudnn 以获得更接近历史训练行为",
    )
    parser.add_argument("--root", default="./dataset", type=str)
    parser.add_argument("--dataset", default="speech_commands_v0.02_sr8k", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--packed_loader_workers", default=1, type=int)
    parser.add_argument("--packed_prefetch_factor", default=1, type=int)
    parser.add_argument("--noise_aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_noise_aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise_roots", nargs="*", default=None)
    parser.add_argument("--train_noise_roots", nargs="*", default=None)
    parser.add_argument("--valid_noise_roots", nargs="*", default=None)
    parser.add_argument("--test_noise_roots", nargs="*", default=None)
    parser.add_argument("--noise_aug_prob", default=0.8, type=float)
    parser.add_argument("--noise_snr_min_db", default=-5.0, type=float)
    parser.add_argument("--noise_snr_max_db", default=20.0, type=float)
    parser.add_argument(
        "--online-augment-profile",
        dest="online_augment_profile",
        choices=["none", "nihao_wenwen_v1"],
        default="none",
        help="Enable role-aware online waveform mixing without materializing augmented WAV files",
    )
    parser.add_argument(
        "--online-noise-domain",
        dest="online_noise_domain",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Noise WAV, directory, or path-list grouped as a sampling domain; repeatable",
    )
    parser.add_argument(
        "--online-noise-domain-weight",
        dest="online_noise_domain_weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
    )
    parser.add_argument(
        "--online-positive-snr-band",
        dest="online_positive_snr_band",
        action="append",
        default=[],
        metavar="LOW:HIGH:WEIGHT",
    )
    parser.add_argument(
        "--online-negative-snr-band",
        dest="online_negative_snr_band",
        action="append",
        default=[],
        metavar="LOW:HIGH:WEIGHT",
    )
    parser.add_argument(
        "--online-hard-negative-snr-band",
        dest="online_hard_negative_snr_band",
        action="append",
        default=[],
        metavar="LOW:HIGH:WEIGHT",
    )
    parser.add_argument(
        "--online-role-quota",
        dest="online_role_quota",
        action="append",
        default=[],
        metavar="ROLE=COUNT",
        help="Exact per-rank batch counts; defaults to 50%% positive and 20%% hard negative when available",
    )
    parser.add_argument(
        "--online-steps-per-epoch",
        dest="online_steps_per_epoch",
        default=0,
        type=int,
        help="Role-balanced updates per epoch; 0 uses ceil(dataset_size / batch_size)",
    )
    parser.add_argument("--eval_noise_aug_prob", default=None, type=float)
    parser.add_argument("--eval_noise_snr_min_db", default=None, type=float)
    parser.add_argument("--eval_noise_snr_max_db", default=None, type=float)
    parser.add_argument("--allow_online_resample", action="store_true", default=False)
    parser.add_argument("--strict_sample_rate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify_sample_rate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify_sample_per_split", default=80, type=int)

    parser.add_argument("--sample_rate", default=8000, type=int)
    parser.add_argument("--frontend", choices=["mfcc", "bandpass"], default="mfcc")
    parser.add_argument("--dct_coeff", default=13, type=int)
    parser.add_argument("--window_size_ms", default=32, type=int)
    parser.add_argument("--window_stride_ms", default=32, type=int)
    parser.add_argument("--model_size_info", nargs="+", type=int, default=DEFAULT_MODEL_SIZE_INFO)
    parser.add_argument("--pooling", choices=["global", "temporal"], default="global")
    parser.add_argument("--temporal_bins", default=4, type=int)
    parser.add_argument("--tcn_channels", default=68, type=int)
    parser.add_argument("--tcn_blocks", default=4, type=int)
    parser.add_argument("--tcn_kernel_size", default=3, type=int)
    parser.add_argument("--tcn_dilations", nargs="+", default=[1, 1, 2, 2], type=int)
    parser.add_argument("--tcn_temporal_bins", default=8, type=int)
    parser.add_argument("--bandpass_n_bands", default=16, type=int)
    parser.add_argument("--bandpass_f_min", default=200.0, type=float)
    parser.add_argument("--bandpass_f_max", default=4000.0, type=float)
    parser.add_argument("--bandpass_spacing", choices=["log", "linear"], default="log")
    parser.add_argument("--bandpass_kernel_size", default=63, type=int)
    parser.add_argument("--bandpass_phase_count", default=1, type=int)

    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre_emphasis_coeff", default=0.97, type=float)

    parser.add_argument("--opt", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--scheduler", choices=["cos", "step"], default="cos")
    parser.add_argument("--t0", default=10, type=int)
    parser.add_argument("--t_mult", default=1, type=int)
    parser.add_argument("--eta_min", default=None, type=float)
    parser.add_argument("--step_size", default=20, type=int)
    parser.add_argument("--gamma", default=0.2, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument(
        "--pair_objective",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="训练显式相邻两窗 [B,2,C,T]，使损失对齐部署端连续两帧确认规则",
    )
    parser.add_argument("--pair_runtime_threshold", default=0.8, type=float)
    parser.add_argument("--pair_softmin_temperature", default=0.25, type=float)
    parser.add_argument("--pair_positive_margin", default=0.0, type=float)
    parser.add_argument("--pair_negative_margin", default=0.0, type=float)
    parser.add_argument("--pair_frame_ce_weight", default=1.0, type=float)
    parser.add_argument("--pair_positive_weight", default=1.0, type=float)
    parser.add_argument("--pair_positive_hinge_tail_fraction", default=1.0, type=float)
    parser.add_argument("--pair_negative_weight", default=1.0, type=float)
    parser.add_argument("--pair_negative_cvar_fraction", default=0.1, type=float)
    parser.add_argument("--pair_negative_source_cvar_fraction", default=1.0, type=float)
    parser.add_argument("--pair_negative_frame_target_probability", default=0.5, type=float)
    parser.add_argument("--pair_negative_frame_weight", default=0.0, type=float)
    parser.add_argument("--pair_tail_ranking_weight", default=0.0, type=float)
    parser.add_argument("--pair_tail_ranking_margin", default=0.0, type=float)
    parser.add_argument("--pair_positive_tail_fraction", default=0.1, type=float)
    parser.add_argument("--pair_objective_weight", default=1.0, type=float)
    parser.add_argument("--spec_aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spec_aug_freq_mask_param", default=1, type=int)
    parser.add_argument("--spec_aug_time_mask_param", default=1, type=int)
    parser.add_argument("--spec_aug_num_freq_masks", default=1, type=int)
    parser.add_argument("--spec_aug_num_time_masks", default=1, type=int)
    parser.add_argument("--mfcc_impl", choices=["torchaudio", "torch"], default="torchaudio")
    parser.add_argument("--mfcc_scale", choices=["natural_log", "torchaudio_db"], default="torchaudio_db")
    parser.add_argument("--mfcc_c0_cmn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default="triangular")
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default="exact")
    parser.add_argument("--log_pwl_num_segments", default=6, type=int)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default="uniform_logx")
    parser.add_argument("--log_pwl_gamma", default=1.0, type=float)
    parser.add_argument("--log_pwl_fit_json", default=None, type=str)
    parser.add_argument("--log_offset", default=1e-6, type=float)
    parser.add_argument("--log_input_clamp_min", default=1e-12, type=float)
    return parser.parse_args()


def normalize_offline_data_args(args):
    if bool(getattr(args, "offline_augmented_dataset", False)):
        args.noise_aug = False
        args.eval_noise_aug = False
        setattr(args, "online_window_jitter_ms", None)
    return args


def wrap_distributed_model(
    model: nn.Module,
    *,
    local_rank: int,
    ddp_static_graph: bool = False,
    ddp_gradient_as_bucket_view: bool = False,
) -> nn.Module:
    """Use optional DDP performance switches when the installed torch supports them."""
    kwargs: dict[str, object] = {"device_ids": [local_rank]}
    if ddp_static_graph:
        kwargs["static_graph"] = True
    if ddp_gradient_as_bucket_view:
        kwargs["gradient_as_bucket_view"] = True
    try:
        return torch.nn.parallel.DistributedDataParallel(model, **kwargs)
    except TypeError:
        if len(kwargs) == 1:
            raise
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


def build_optimizer_scheduler(args, model: nn.Module):
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("optimizer has no trainable parameters")
    if args.opt == "adam":
        optimizer = optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(
            trainable_parameters,
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.9),
            nesterov=True,
            weight_decay=args.weight_decay,
        )

    warmup_steps = int(getattr(args, "warmup_steps", 0))
    if warmup_steps > 0:
        if args.scheduler != "cos":
            raise ValueError("warmup_steps requires the cosine scheduler")
        eta_min_arg = getattr(args, "eta_min", None)
        eta_min = eta_min_arg if eta_min_arg is not None else args.lr * 0.01
        total_steps = int(getattr(args, "scheduler_total_steps", max(1, args.epoch)))
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=eta_min,
        )
        setattr(args, "scheduler_step_per_batch", True)
    elif args.scheduler == "cos":
        eta_min_arg = getattr(args, "eta_min", None)
        eta_min = eta_min_arg if eta_min_arg is not None else args.lr * 0.01
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(getattr(args, "epoch", getattr(args, "t0", 1)))),
            eta_min=eta_min,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(args, "step_size", 20),
            gamma=getattr(args, "gamma", 0.2),
        )
    if warmup_steps == 0:
        setattr(args, "scheduler_step_per_batch", False)
    return optimizer, scheduler


def _active_dscnn_classifier(backbone: nn.Module) -> tuple[str, nn.Module]:
    classifier_name = "temporal_fc" if getattr(backbone, "pooling", "global") == "temporal" else "final_fc"
    classifier = getattr(backbone, classifier_name, None)
    if classifier is None:
        raise ValueError(f"freeze_backbone requires a DSCNN backbone.{classifier_name} classifier")
    return classifier_name, classifier


def configure_frozen_dscnn_trainable_tail(model: nn.Module, *, tail_blocks: int = 0) -> tuple[str, ...]:
    """Freeze a DSCNN except for its active classifier and an optional convolutional tail."""
    if tail_blocks < 0:
        raise ValueError("tail_blocks must be non-negative")
    backbone = getattr(model, "backbone", None)
    conv_layers = getattr(backbone, "conv_layers", None)
    if backbone is None or conv_layers is None:
        raise ValueError("freeze_backbone requires a DSCNN backbone with conv_layers and a classifier")
    classifier_name, classifier = _active_dscnn_classifier(backbone)
    if tail_blocks > len(conv_layers):
        raise ValueError(f"tail_blocks={tail_blocks} exceeds DSCNN depth={len(conv_layers)}")

    tail_start = len(conv_layers) - tail_blocks
    classifier_prefix = f"backbone.{classifier_name}."
    trainable_prefixes = [classifier_prefix]
    trainable_prefixes.extend(f"backbone.conv_layers.{index}." for index in range(tail_start, len(conv_layers)))
    trainable_names = tuple(
        name for name, _ in model.named_parameters() if any(name.startswith(prefix) for prefix in trainable_prefixes)
    )
    classifier_parameter_names = {f"{classifier_prefix}{name}" for name, _ in classifier.named_parameters()}
    if not trainable_names or not classifier_parameter_names.issubset(trainable_names):
        raise ValueError(f"freeze_backbone requires a DSCNN backbone.{classifier_name} classifier")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable_names)
    return trainable_names


def freeze_backbone_except_final_fc(model: nn.Module) -> tuple[str, ...]:
    """Freeze DSCNN feature layers while retaining its active binary classifier."""
    return configure_frozen_dscnn_trainable_tail(model, tail_blocks=0)


def set_frozen_backbone_eval(
    model: nn.Module,
    *,
    freeze_tail_batch_norm_stats: bool = False,
) -> None:
    """Keep frozen DSCNN modules in eval mode while training the configured tail."""
    backbone = getattr(model, "backbone", None)
    conv_layers = getattr(backbone, "conv_layers", None)
    if backbone is None or conv_layers is None:
        raise ValueError("freeze_backbone requires a DSCNN backbone with conv_layers and a classifier")
    _, classifier = _active_dscnn_classifier(backbone)
    backbone.eval()
    for layer in conv_layers:
        if any(parameter.requires_grad for parameter in layer.parameters()):
            layer.train()
            if freeze_tail_batch_norm_stats:
                for module in layer.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
    classifier.train()


def configure_scheduler_total_steps(args, train_loader) -> None:
    """Bind per-update schedules to the actual per-rank loader length."""
    warmup_steps = int(getattr(args, "warmup_steps", 0))
    if warmup_steps <= 0:
        return
    total_steps = int(len(train_loader)) * int(args.epoch)
    if total_steps <= warmup_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be smaller than total optimizer updates ({total_steps})"
        )
    args.scheduler_total_steps = total_steps


def fit_and_teardown(
    trainer: Trainer,
    artifact_writer: TrainingArtifactWriter | None,
    failure_reporter: FailureArtifactReporter | None = None,
):
    return _run_with_failure_reporting(trainer.fit, lambda: artifact_writer, failure_reporter)


def _run_with_failure_reporting(
    run,
    artifact_writer_getter,
    failure_reporter: FailureArtifactReporter | None = None,
    destroy_on_exit: bool = True,
):
    try:
        return run()
    except BaseException as error:
        artifact_writer = artifact_writer_getter()
        is_interruption = isinstance(error, KeyboardInterrupt)
        try:
            if is_rank_zero() and artifact_writer is not None:
                if is_interruption:
                    artifact_writer.write_terminal_status(
                        "interrupted",
                        {
                            "rank": 0,
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )
                else:
                    artifact_writer.write_failure(error, traceback.format_exc())
            elif failure_reporter is not None:
                if is_interruption:
                    failure_reporter.write_interrupted(error, traceback.format_exc())
                else:
                    failure_reporter.write_failure(error, traceback.format_exc())
        except BaseException:
            pass
        raise
    finally:
        if destroy_on_exit:
            # A failed peer may never reach a barrier; torchrun terminates remaining workers.
            destroy_distributed()


def _bootstrap_failure_save_dir(args) -> str:
    return args.save_dir or os.path.join("dscnn_kws", "runs", "bootstrap-failures")


def _report_bootstrap_failure(
    reporter: FailureArtifactReporter | None,
    save_dir: str,
    rank: int,
    error: BaseException,
) -> None:
    if reporter is None:
        try:
            reporter = FailureArtifactReporter(save_dir, rank=rank)
        except BaseException:
            return
    try:
        if isinstance(error, KeyboardInterrupt):
            reporter.write_interrupted(error, traceback.format_exc())
        else:
            reporter.write_failure(error, traceback.format_exc())
    except BaseException:
        pass


def _resolve_run_artifacts(args) -> tuple[str, str]:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.distributed:
        timestamp_payload = [timestamp if is_rank_zero() else None]
        torch.distributed.broadcast_object_list(timestamp_payload, src=0)
        timestamp = timestamp_payload[0]
    run_name = args.run_name or f"{args.model}_{args.dataset}_lr{args.lr}_ep{args.epoch}_{timestamp}"
    args.run_name = run_name
    return run_name, args.save_dir or os.path.join("dscnn_kws", "runs", run_name)


def sample_rate_verification_manifests(args) -> dict[str, str]:
    """Return only the splits that this training invocation is allowed to read."""
    manifests = {
        "train": getattr(args, "train_manifest", ""),
        "validation": getattr(args, "validation_manifest", ""),
    }
    if not bool(getattr(args, "skip_test", False)):
        manifests["test"] = getattr(args, "test_manifest", "")
    return manifests


def main():
    args = parse_args()
    normalize_offline_data_args(args)
    if getattr(args, "max_train_steps", None) is not None and args.max_train_steps < 1:
        raise ValueError("--max_train_steps must be positive")
    if bool(getattr(args, "temporal_init_from_global", False)) and not getattr(args, "resume", None):
        raise ValueError("--temporal_init_from_global requires --resume pointing to a global-pooling checkpoint")
    if bool(getattr(args, "temporal_init_from_global", False)) and args.model != "dscnn":
        raise ValueError("--temporal_init_from_global is supported only for DSCNN")
    if not 0.0 <= getattr(args, "min_positive_recall", 0.0) <= 1.0:
        raise ValueError("--min_positive_recall must be in [0, 1]")
    if getattr(args, "early_stopping_min_epoch", 0) < 0:
        raise ValueError("--early_stopping_min_epoch must be non-negative")
    if getattr(args, "early_stopping_patience", 0) < 0:
        raise ValueError("--early_stopping_patience must be non-negative")
    if getattr(args, "candidate_checkpoint_every", 0) < 0:
        raise ValueError("--candidate-checkpoint-every must be non-negative")
    freeze_backbone = bool(getattr(args, "freeze_backbone", False))
    trainable_tail_blocks = int(getattr(args, "trainable_tail_blocks", 0))
    if freeze_backbone and args.model != "dscnn":
        raise ValueError("--freeze_backbone is supported only for DSCNN")
    if trainable_tail_blocks < 0:
        raise ValueError("--trainable_tail_blocks must be non-negative")
    if trainable_tail_blocks and not freeze_backbone:
        raise ValueError("--trainable_tail_blocks requires --freeze_backbone")
    freeze_tail_batch_norm_stats = bool(getattr(args, "freeze_tail_batch_norm_stats", False))
    if freeze_tail_batch_norm_stats and not freeze_backbone:
        raise ValueError("--freeze_tail_batch_norm_stats requires --freeze_backbone")
    if freeze_tail_batch_norm_stats and trainable_tail_blocks == 0:
        raise ValueError("--freeze_tail_batch_norm_stats requires --trainable_tail_blocks greater than zero")
    if args.frontend == "bandpass" and args.dct_coeff != args.bandpass_n_bands:
        raise ValueError(
            f"For bandpass frontend, dct_coeff must equal bandpass_n_bands. "
            f"Got dct_coeff={args.dct_coeff}, bandpass_n_bands={args.bandpass_n_bands}"
        )

    log_pwl_breakpoints = None
    log_pwl_slopes = None
    log_pwl_intercepts = None
    if args.log_pwl_fit_json:
        cfg = load_log_pwl_json(args.log_pwl_fit_json)
        log_pwl_breakpoints = cfg["breakpoints"]
        log_pwl_slopes = cfg["slopes"]
        log_pwl_intercepts = cfg["intercepts"]

    distributed_context = init_distributed(args.distributed, args.local_rank)
    bootstrap_save_dir = _bootstrap_failure_save_dir(args)
    bootstrap_reporter = None
    run_started = False
    try:
        args.rank = distributed_context.rank
        args.world_size = distributed_context.world_size
        run_name, save_dir = _resolve_run_artifacts(args)
        failure_reporter = FailureArtifactReporter(save_dir, rank=args.rank)
        artifact_writer = None

        def initialize_and_fit():
            nonlocal artifact_writer
            if is_rank_zero():
                artifact_writer = TrainingArtifactWriter(save_dir, argv=vars(args), run_variant=run_name)
            barrier()

            set_random_seed(args.seed + distributed_context.rank, deterministic=not args.non_deterministic)
            device = distributed_context.device
            if not args.distributed:
                device, _ = prepare_device(args.gpu)

            data_path = os.path.join(args.root, args.dataset)
            if args.verify_sample_rate:
                manifest_paths = sample_rate_verification_manifests(args)
                verify_dataset_sample_rate(
                    data_path=data_path,
                    expected_sample_rate=args.sample_rate,
                    sample_per_split=max(1, args.verify_sample_per_split),
                    random_seed=args.seed,
                    manifest_paths=manifest_paths if any(manifest_paths.values()) else None,
                    include_test=not bool(getattr(args, "skip_test", False)),
                )

            train_loader, valid_loader, test_loader = build_dataloaders(data_path, CLASS_LIST, CLASS_ENCODING, args)
            pair_steps_per_epoch = math.ceil(len(train_loader) / int(getattr(args, "pair_interval", 1)))
            pair_train_loader = build_confirmation_pair_loader(
                CLASS_ENCODING,
                args,
                steps_per_epoch=pair_steps_per_epoch,
            )
            configure_scheduler_total_steps(args, train_loader)

            time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
            input_dim = time_steps * args.dct_coeff

            if args.model == "dscnn":
                backbone = DSCNN(
                    input_dim=input_dim,
                    label_count=len(CLASS_LIST),
                    model_size_info=args.model_size_info,
                    dct_coeff=args.dct_coeff,
                    pooling=args.pooling,
                    temporal_bins=args.temporal_bins,
                )
                wrapper_class = MFCCDSCNN
            elif args.model == "cepstral_tcn":
                backbone = CepstralTCN(
                    input_dim=input_dim,
                    label_count=len(CLASS_LIST),
                    dct_coeff=args.dct_coeff,
                    channels=args.tcn_channels,
                    num_blocks=args.tcn_blocks,
                    kernel_size=args.tcn_kernel_size,
                    dilations=args.tcn_dilations,
                    temporal_bins=args.tcn_temporal_bins,
                )
                wrapper_class = MFCCDSCNN
            else:
                backbone = LSTM(
                    input_dim=input_dim, label_count=len(CLASS_LIST), dct_coeff=args.dct_coeff
                )
                wrapper_class = MFCCLSTM

            wrapper_kwargs = {
                "backbone": backbone,
                "frontend": args.frontend,
                "sample_rate": args.sample_rate,
                "dct_coeff": args.dct_coeff,
                "window_size_ms": args.window_size_ms,
                "window_stride_ms": args.window_stride_ms,
                "bandpass_n_bands": args.bandpass_n_bands,
                "bandpass_f_min": args.bandpass_f_min,
                "bandpass_f_max": args.bandpass_f_max,
                "bandpass_spacing": args.bandpass_spacing,
                "bandpass_kernel_size": args.bandpass_kernel_size,
                "bandpass_phase_count": args.bandpass_phase_count,
                "pre_emphasis": args.pre_emphasis,
                "pre_emphasis_coeff": args.pre_emphasis_coeff,
                "spec_aug": args.spec_aug,
                "spec_aug_freq_mask_param": args.spec_aug_freq_mask_param,
                "spec_aug_time_mask_param": args.spec_aug_time_mask_param,
                "spec_aug_num_freq_masks": args.spec_aug_num_freq_masks,
                "spec_aug_num_time_masks": args.spec_aug_num_time_masks,
                "mfcc_impl": args.mfcc_impl,
                "mfcc_scale": args.mfcc_scale,
                "mel_filter_shape": args.mel_filter_shape,
                "log_approx_mode": args.log_approx_mode,
                "log_pwl_num_segments": args.log_pwl_num_segments,
                "log_pwl_strategy": args.log_pwl_strategy,
                "log_pwl_gamma": args.log_pwl_gamma,
                "log_pwl_breakpoints": log_pwl_breakpoints,
                "log_pwl_slopes": log_pwl_slopes,
                "log_pwl_intercepts": log_pwl_intercepts,
                "log_offset": args.log_offset,
                "log_input_clamp_min": args.log_input_clamp_min,
            }
            if args.model in {"dscnn", "cepstral_tcn"}:
                wrapper_kwargs["amp_backbone"] = bool(getattr(args, "amp_backbone", False))
                wrapper_kwargs["mfcc_c0_cmn"] = bool(getattr(args, "mfcc_c0_cmn", False))
            model = wrapper_class(**wrapper_kwargs).to(device)
            frozen_parameter_names: tuple[str, ...] = ()
            if freeze_backbone:
                frozen_parameter_names = configure_frozen_dscnn_trainable_tail(
                    model, tail_blocks=trainable_tail_blocks
                )
            if args.distributed:
                model = wrap_distributed_model(
                    model,
                    local_rank=distributed_context.local_rank,
                    ddp_static_graph=bool(getattr(args, "ddp_static_graph", False)),
                    ddp_gradient_as_bucket_view=bool(getattr(args, "ddp_gradient_as_bucket_view", False)),
                )

            optimizer, scheduler = build_optimizer_scheduler(args, model)
            eval_noise_aug_prob = args.eval_noise_aug_prob if args.eval_noise_aug_prob is not None else args.noise_aug_prob
            eval_noise_snr_min_db = (
                args.eval_noise_snr_min_db if args.eval_noise_snr_min_db is not None else args.noise_snr_min_db
            )
            eval_noise_snr_max_db = (
                args.eval_noise_snr_max_db if args.eval_noise_snr_max_db is not None else args.noise_snr_max_db
            )
            if is_rank_zero():
                print(f"[INFO] data_path={data_path}")
                print(f"[INFO] device={device}, params={parameter_number(model.module if args.distributed else model)}")
                print(f"[INFO] save_dir={save_dir}")
                print(f"[INFO] spec_aug={'ON' if args.spec_aug else 'OFF'}")
                print(
                    f"[INFO] noise_aug={'ON' if args.noise_aug else 'OFF'}, "
                    f"eval_noise_aug={'ON' if args.eval_noise_aug else 'OFF'}, "
                    f"prob={args.noise_aug_prob}, snr=[{args.noise_snr_min_db}, {args.noise_snr_max_db}] dB"
                )
                print(
                    f"[INFO] eval_noise_config: prob={eval_noise_aug_prob}, "
                    f"snr=[{eval_noise_snr_min_db}, {eval_noise_snr_max_db}] dB"
                )
                if args.noise_roots or args.train_noise_roots or args.valid_noise_roots or args.test_noise_roots:
                    print(f"[INFO] noise_roots={args.noise_roots}")
                    print(f"[INFO] train_noise_roots={args.train_noise_roots}")
                    print(f"[INFO] valid_noise_roots={args.valid_noise_roots}")
                    print(f"[INFO] test_noise_roots={args.test_noise_roots}")
                print(f"[INFO] frontend={args.frontend}")
                if frozen_parameter_names:
                    print(
                        f"[INFO] freeze_backbone=ON, trainable_tail_blocks={trainable_tail_blocks}, "
                        f"freeze_tail_batch_norm_stats={freeze_tail_batch_norm_stats}, "
                        f"trainable_parameters={list(frozen_parameter_names)}"
                    )
                print(f"[INFO] mfcc_impl={args.mfcc_impl}")
                print(f"[INFO] mel_filter_shape={args.mel_filter_shape}")
                print(f"[INFO] log_approx_mode={args.log_approx_mode}")
            if args.frontend == "bandpass" and is_rank_zero():
                print(
                    f"[INFO] bandpass: n_bands={args.bandpass_n_bands}, "
                    f"f_min={args.bandpass_f_min}, f_max={args.bandpass_f_max}, "
                    f"spacing={args.bandpass_spacing}, kernel_size={args.bandpass_kernel_size}, "
                    f"phase_count={args.bandpass_phase_count}, "
                    f"window_stride_ms={args.window_stride_ms}, stride_samples={int(args.sample_rate * args.window_stride_ms / 1000)}"
                )
                expected_time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
                print(f"[INFO] bandpass expected_time_steps={expected_time_steps}")
            if args.log_approx_mode == "pwl" and is_rank_zero():
                if args.log_pwl_fit_json:
                    print(f"[INFO] log_pwl_fit_json={args.log_pwl_fit_json}")
                else:
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
                artifact_writer=artifact_writer,
                pair_train_loader=pair_train_loader,
            )
            return trainer.fit()

        run_started = True
        return _run_with_failure_reporting(
            initialize_and_fit,
            lambda: artifact_writer,
            failure_reporter,
            destroy_on_exit=False,
        )
    except BaseException as error:
        if not run_started:
            _report_bootstrap_failure(
                bootstrap_reporter,
                bootstrap_save_dir,
                distributed_context.rank,
                error,
            )
        raise
    finally:
        # Distributed setup succeeded above, so every subsequent exit destroys its process group.
        destroy_distributed()


if __name__ == "__main__":
    main()
