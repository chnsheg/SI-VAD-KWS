from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST, DEFAULT_MODEL_SIZE_INFO
from dscnn_kws.data.dataset import SpeechCommandDataset
from dscnn_kws.model import CepstralTCN, DSCNN
from dscnn_kws.model.dscnn import calculate_time_steps
from dscnn_kws.train import MFCCDSCNN


def load_state_dict(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            state_dict = payload.get(key)
            if isinstance(state_dict, dict):
                return state_dict
    return payload


def build_eval_loader(data_path, split, args, manifest_path=None):
    if split == "validation":
        manifest = manifest_path or os.path.join(data_path, "validation_manifest.json")
        noise_roots = args.valid_noise_roots or args.noise_roots
        random_seed = args.seed + 100000
    elif split == "test":
        manifest = manifest_path or os.path.join(data_path, "test_manifest.json")
        noise_roots = args.test_noise_roots or args.noise_roots
        random_seed = args.seed + 200000
    else:
        raise ValueError(split)

    dataset = SpeechCommandDataset(
        dataset_path=data_path,
        json_filename=manifest,
        is_training=False,
        class_list=CLASS_LIST,
        class_encoding=CLASS_ENCODING,
        sample_rate=args.sample_rate,
        noise_aug=args.eval_noise_aug,
        noise_roots=noise_roots,
        noise_prob=args.noise_aug_prob,
        noise_snr_min_db=args.noise_snr_min_db,
        noise_snr_max_db=args.noise_snr_max_db,
        deterministic_noise=True,
        random_seed=random_seed,
        allow_online_resample=True,
        strict_sample_rate=False,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.gpu > 0,
    )


def build_model(args, device):
    time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
    input_dim = time_steps * args.dct_coeff

    architecture = getattr(args, "model", "dscnn")
    if architecture == "dscnn":
        backbone = DSCNN(
            input_dim=input_dim,
            label_count=len(CLASS_LIST),
            model_size_info=args.model_size_info,
            dct_coeff=args.dct_coeff,
            pooling=getattr(args, "pooling", "global"),
            temporal_bins=int(getattr(args, "temporal_bins", 4)),
        )
    elif architecture == "cepstral_tcn":
        backbone = CepstralTCN(
            input_dim=input_dim,
            label_count=len(CLASS_LIST),
            dct_coeff=args.dct_coeff,
            channels=int(getattr(args, "tcn_channels", 68)),
            num_blocks=int(getattr(args, "tcn_blocks", 4)),
            kernel_size=int(getattr(args, "tcn_kernel_size", 3)),
            dilations=getattr(args, "tcn_dilations", [1, 1, 2, 2]),
            temporal_bins=int(getattr(args, "tcn_temporal_bins", 8)),
        )
    else:
        raise ValueError(f"Unsupported evaluation model: {architecture}")

    model = MFCCDSCNN(
        backbone=backbone,
        frontend=args.frontend,
        sample_rate=args.sample_rate,
        dct_coeff=args.dct_coeff,
        window_size_ms=args.window_size_ms,
        window_stride_ms=args.window_stride_ms,
        bandpass_n_bands=args.bandpass_n_bands,
        bandpass_f_min=args.bandpass_f_min,
        bandpass_f_max=args.bandpass_f_max,
        bandpass_spacing=args.bandpass_spacing,
        bandpass_kernel_size=args.bandpass_kernel_size,
        bandpass_phase_count=args.bandpass_phase_count,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coeff=args.pre_emphasis_coeff,
        spec_aug=False,
        spec_aug_freq_mask_param=1,
        spec_aug_time_mask_param=1,
        spec_aug_num_freq_masks=0,
        spec_aug_num_time_masks=0,
        mfcc_impl=args.mfcc_impl,
        mfcc_scale=getattr(args, "mfcc_scale", "torchaudio_db"),
        mfcc_c0_cmn=bool(getattr(args, "mfcc_c0_cmn", False)),
        mel_filter_shape=args.mel_filter_shape,
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

    model.load_state_dict(load_state_dict(args.ckpt, device))
    model.eval()
    return model


@torch.no_grad()
def collect_scores(model, loader, device, positive_index):
    scores = []
    labels = []

    for waveform, y in loader:
        waveform = waveform.to(device)
        logits = model(waveform)
        prob = torch.softmax(logits, dim=1)[:, positive_index]

        scores.extend(prob.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())

    return np.asarray(scores), np.asarray(labels)


def choose_threshold_for_target_fah(valid_pos, valid_neg, target_fah, window_sec):
    neg_hours = len(valid_neg) * window_sec / 3600.0
    max_fp = int(np.floor(target_fah * neg_hours + 1e-12))

    sorted_neg = np.sort(valid_neg)[::-1]

    if max_fp <= 0:
        # 要求 validation 上 0 次 false alarm
        threshold = np.nextafter(sorted_neg[0], np.inf)
    elif max_fp >= len(sorted_neg):
        threshold = -np.inf
    else:
        high = sorted_neg[max_fp - 1]
        low = sorted_neg[max_fp]

        if high > low:
            threshold = (high + low) / 2.0
        else:
            threshold = np.nextafter(high, np.inf)

    return float(threshold)


def eval_at_threshold(pos_scores, neg_scores, threshold, window_sec):
    tp = int(np.sum(pos_scores >= threshold))
    fn = int(np.sum(pos_scores < threshold))
    fp = int(np.sum(neg_scores >= threshold))
    tn = int(np.sum(neg_scores < threshold))

    neg_hours = len(neg_scores) * window_sec / 3600.0

    frr = fn / max(1, len(pos_scores))
    fah = fp / max(1e-12, neg_hours)
    acc = (tp + tn) / max(1, len(pos_scores) + len(neg_scores))

    return {
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "frr": frr,
        "fah": fah,
        "acc": acc,
        "neg_hours": neg_hours,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default="./dscnn_kws/data")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--validation_manifest", default="")
    parser.add_argument("--test_manifest", default="")

    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--eval_noise_aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise_roots", nargs="*", default=None)
    parser.add_argument("--valid_noise_roots", nargs="*", default=None)
    parser.add_argument("--test_noise_roots", nargs="*", default=None)
    parser.add_argument("--noise_aug_prob", default=0.8, type=float)
    parser.add_argument("--noise_snr_min_db", default=-5.0, type=float)
    parser.add_argument("--noise_snr_max_db", default=20.0, type=float)

    parser.add_argument("--target_fah", nargs="+", type=float, default=[1.0])
    parser.add_argument("--window_sec", default=1.0, type=float)

    parser.add_argument("--frontend", choices=["mfcc", "bandpass"], default="mfcc")
    parser.add_argument("--dct_coeff", default=13, type=int)
    parser.add_argument("--window_size_ms", default=32, type=int)
    parser.add_argument("--window_stride_ms", default=32, type=int)
    parser.add_argument("--model", choices=["dscnn", "cepstral_tcn"], default="dscnn")
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

    parser.add_argument("--mfcc_impl", choices=["torchaudio", "torch"], default="torchaudio")
    parser.add_argument("--mfcc_scale", choices=["natural_log", "torchaudio_db"], default="torchaudio_db")
    parser.add_argument("--mfcc_c0_cmn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mel_filter_shape", choices=["triangular", "rectangular"], default="triangular")
    parser.add_argument("--log_approx_mode", choices=["exact", "pwl"], default="exact")
    parser.add_argument("--log_pwl_num_segments", default=6, type=int)
    parser.add_argument("--log_pwl_strategy", choices=["uniform_logx", "quantile", "powerlaw"], default="uniform_logx")
    parser.add_argument("--log_pwl_gamma", default=1.0, type=float)
    parser.add_argument("--log_offset", default=1e-6, type=float)
    parser.add_argument("--log_input_clamp_min", default=1e-12, type=float)

    args = parser.parse_args()

    if "positive" not in CLASS_ENCODING or "negative" not in CLASS_ENCODING:
        raise ValueError(f"当前 CLASS_LIST={CLASS_LIST}，需要包含 positive 和 negative")

    positive_index = CLASS_ENCODING["positive"]
    negative_index = CLASS_ENCODING["negative"]

    device = torch.device("cuda:0" if args.gpu > 0 and torch.cuda.is_available() else "cpu")

    data_path = os.path.join(args.root, args.dataset)

    print(f"[INFO] data_path={data_path}")
    print(f"[INFO] ckpt={args.ckpt}")
    print(f"[INFO] device={device}")
    print(f"[INFO] positive_index={positive_index}, negative_index={negative_index}")

    valid_loader = build_eval_loader(data_path, "validation", args, manifest_path=args.validation_manifest or None)
    test_loader = build_eval_loader(data_path, "test", args, manifest_path=args.test_manifest or None)

    model = build_model(args, device)

    print("[INFO] collecting validation scores...")
    valid_scores, valid_labels = collect_scores(model, valid_loader, device, positive_index)

    print("[INFO] collecting test scores...")
    test_scores, test_labels = collect_scores(model, test_loader, device, positive_index)

    valid_pos = valid_scores[valid_labels == positive_index]
    valid_neg = valid_scores[valid_labels == negative_index]
    test_pos = test_scores[test_labels == positive_index]
    test_neg = test_scores[test_labels == negative_index]

    print(f"[INFO] valid positive={len(valid_pos)}, negative={len(valid_neg)}, negative_hours={len(valid_neg) * args.window_sec / 3600:.3f}")
    print(f"[INFO] test  positive={len(test_pos)}, negative={len(test_neg)}, negative_hours={len(test_neg) * args.window_sec / 3600:.3f}")

    print(f"[INFO] valid FAH resolution: {3600 / max(1, len(valid_neg) * args.window_sec):.6f}")
    print(f"[INFO] test  FAH resolution: {3600 / max(1, len(test_neg) * args.window_sec):.6f}")

    for target in args.target_fah:
        threshold = choose_threshold_for_target_fah(
            valid_pos=valid_pos,
            valid_neg=valid_neg,
            target_fah=target,
            window_sec=args.window_sec,
        )

        valid_m = eval_at_threshold(valid_pos, valid_neg, threshold, args.window_sec)
        test_m = eval_at_threshold(test_pos, test_neg, threshold, args.window_sec)

        print("\n" + "=" * 80)
        print(f"[TARGET] FAH <= {target}")
        print(f"[THRESHOLD] selected on validation = {threshold:.8f}")

        print(
            f"[VALID] FAH={valid_m['fah']:.6f}, FRR={valid_m['frr']:.6f}, ACC={valid_m['acc']:.6f}, "
            f"TP={valid_m['tp']}, FN={valid_m['fn']}, FP={valid_m['fp']}, TN={valid_m['tn']}"
        )

        print(
            f"[TEST ] FAH={test_m['fah']:.6f}, FRR={test_m['frr']:.6f}, ACC={test_m['acc']:.6f}, "
            f"TP={test_m['tp']}, FN={test_m['fn']}, FP={test_m['fp']}, TN={test_m['tn']}"
        )


if __name__ == "__main__":
    main()
