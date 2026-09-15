from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
from dscnn_kws.data.dataset import SpeechCommandDataset
from dscnn_kws.frontend import StreamingMFCC, load_log_pwl_json
from dscnn_kws.streaming.streaming_crnn import StreamingKWSModel, model_config_dict, parse_cnn_channels
from dscnn_kws.utils import parameter_number


CSV_FIELDNAMES = [
    "split",
    "dataset",
    "ckpt",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate StreamingMFCC + CRNN with true chunk-by-chunk inference on dataset waveforms."
    )
    parser.add_argument("--root", default="./dscnn_kws/data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--batch", default=256, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--allow_online_resample", action="store_true", default=False)
    parser.add_argument("--strict_sample_rate", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--dct_coeff", default=10, type=int)
    parser.add_argument("--window_size_ms", default=32, type=int)
    parser.add_argument("--window_stride_ms", default=32, type=int)
    parser.add_argument("--mfcc_center", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming_mfcc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk_samples", default=None, type=int)
    parser.add_argument("--chunk_ms", default=None, type=float)
    parser.add_argument("--flush_tail", action=argparse.BooleanOptionalAction, default=True)

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

    parser.add_argument("--eval_noise_aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise_roots", nargs="*", default=None)
    parser.add_argument("--test_noise_roots", nargs="*", default=None)
    parser.add_argument("--eval_noise_aug_prob", default=1.0, type=float)
    parser.add_argument("--eval_noise_snr_min_db", default=5.0, type=float)
    parser.add_argument("--eval_noise_snr_max_db", default=5.0, type=float)
    parser.add_argument("--seed", default=500000, type=int)

    parser.add_argument("--compare_offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out_csv",
        default="./dscnn_kws/streaming/results/streaming_crnn_true_streaming_eval.csv",
        type=str,
    )
    return parser.parse_args()


def load_state_dict(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
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
        frontend="mfcc",
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


def split_manifest_name(split: str) -> str:
    if split == "train":
        return "train_manifest.json"
    if split == "valid":
        return "validation_manifest.json"
    return "test_manifest.json"


def build_eval_loader(args: argparse.Namespace) -> DataLoader:
    data_path = os.path.join(args.root, args.dataset)
    noise_roots = args.test_noise_roots or args.noise_roots
    dataset = SpeechCommandDataset(
        dataset_path=data_path,
        json_filename=os.path.join(data_path, split_manifest_name(args.split)),
        is_training=False,
        class_list=CLASS_LIST,
        class_encoding=CLASS_ENCODING,
        sample_rate=args.sample_rate,
        noise_aug=args.eval_noise_aug,
        noise_roots=noise_roots,
        noise_prob=args.eval_noise_aug_prob,
        noise_snr_min_db=args.eval_noise_snr_min_db,
        noise_snr_max_db=args.eval_noise_snr_max_db,
        deterministic_noise=True,
        random_seed=args.seed,
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


def resolve_chunk_samples(args: argparse.Namespace, model: StreamingKWSModel) -> int:
    if args.chunk_samples is not None and args.chunk_ms is not None:
        raise ValueError("Use only one of --chunk_samples and --chunk_ms")
    if args.chunk_ms is not None:
        chunk_samples = int(round(args.sample_rate * float(args.chunk_ms) / 1000.0))
    elif args.chunk_samples is not None:
        chunk_samples = int(args.chunk_samples)
    else:
        chunk_samples = int(model.feature_extractor.hop_length)
    if chunk_samples <= 0:
        raise ValueError("chunk size must be positive")
    return chunk_samples


def stream_pre_emphasis_chunk(
    chunk: torch.Tensor,
    prev_sample: torch.Tensor | None,
    coeff: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if chunk.numel() == 0:
        return chunk, prev_sample
    if coeff <= 0:
        return chunk, chunk[:, -1].detach()

    y = chunk.clone()
    if chunk.size(1) > 1:
        y[:, 1:] = chunk[:, 1:] - coeff * chunk[:, :-1]
    if prev_sample is None:
        y[:, 0] = chunk[:, 0]
    else:
        y[:, 0] = chunk[:, 0] - coeff * prev_sample.to(device=chunk.device, dtype=chunk.dtype)
    return y, chunk[:, -1].detach()


@torch.no_grad()
def true_streaming_logits(
    model: StreamingKWSModel,
    waveform: torch.Tensor,
    chunk_samples: int,
    flush_tail: bool,
) -> tuple[torch.Tensor, int]:
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform [B, T] or [B, 1, T], got {tuple(waveform.shape)}")
    if not isinstance(model.feature_extractor, StreamingMFCC):
        raise ValueError("True audio streaming requires --streaming_mfcc so the frontend is StreamingMFCC")

    batch_size = waveform.size(0)
    mfcc_state = model.feature_extractor.init_stream_state(
        batch_size=batch_size,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    crnn_state = model.backbone.init_stream_state()
    prev_sample: torch.Tensor | None = None
    last_logits: torch.Tensor | None = None
    frame_count = 0

    for start in range(0, waveform.size(1), chunk_samples):
        chunk = waveform[:, start : start + chunk_samples]
        if model.pre_emphasis:
            chunk, prev_sample = stream_pre_emphasis_chunk(chunk, prev_sample, model.pre_emphasis_coeff)
        mfcc, mfcc_state = model.feature_extractor.forward_stream_chunk(chunk, state=mfcc_state, flush=False)
        if mfcc.size(2) == 0:
            continue

        frames = mfcc[:, : model.dct_coeff, :].transpose(1, 2)
        for t in range(frames.size(1)):
            last_logits, crnn_state = model.backbone.forward_stream_frame(frames[:, t, :], crnn_state)
            frame_count += 1

    if flush_tail:
        mfcc, mfcc_state = model.feature_extractor.flush_stream(mfcc_state)
        if mfcc.size(2) > 0:
            frames = mfcc[:, : model.dct_coeff, :].transpose(1, 2)
            for t in range(frames.size(1)):
                last_logits, crnn_state = model.backbone.forward_stream_frame(frames[:, t, :], crnn_state)
                frame_count += 1

    if last_logits is None:
        raise RuntimeError("No MFCC frame was produced; check chunk size and waveform length")
    return last_logits, frame_count


def metrics_from_labels(labels: list[int], preds: list[int]) -> dict[str, float]:
    correct = sum(int(a == b) for a, b in zip(labels, preds))
    total = len(labels)
    return {
        "acc": correct / max(1, total),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(
    model: StreamingKWSModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    chunk_samples: int,
) -> dict[str, float | int | str]:
    stream_preds: list[int] = []
    labels_all: list[int] = []
    offline_preds: list[int] = []
    max_abs_diff = 0.0
    mean_abs_diff_sum = 0.0
    compare_samples = 0
    min_frames: int | None = None
    max_frames = 0

    iterator = tqdm(loader, desc="true-stream-eval", leave=False)
    for batch_idx, (waveform, labels) in enumerate(iterator):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        waveform = waveform.to(device)
        labels = labels.to(device)

        stream_logits, frame_count = true_streaming_logits(
            model,
            waveform=waveform,
            chunk_samples=chunk_samples,
            flush_tail=args.flush_tail,
        )
        stream_batch_preds = torch.argmax(stream_logits, dim=1)
        stream_preds.extend(stream_batch_preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

        min_frames = frame_count if min_frames is None else min(min_frames, frame_count)
        max_frames = max(max_frames, frame_count)

        if args.compare_offline:
            offline_logits = model(waveform)
            offline_batch_preds = torch.argmax(offline_logits, dim=1)
            offline_preds.extend(offline_batch_preds.cpu().tolist())

            diff = (stream_logits - offline_logits).abs()
            max_abs_diff = max(max_abs_diff, float(diff.max().item()))
            mean_abs_diff_sum += float(diff.mean().item()) * int(labels.size(0))
            compare_samples += int(labels.size(0))

    stream_metrics = metrics_from_labels(labels_all, stream_preds)
    result: dict[str, float | int | str] = {
        "split": args.split,
        "dataset": args.dataset,
        "ckpt": args.ckpt,
        "num_samples": len(labels_all),
        "chunk_samples": chunk_samples,
        "chunk_ms": chunk_samples * 1000.0 / args.sample_rate,
        "flush_tail": str(args.flush_tail),
        "frames_min": min_frames or 0,
        "frames_max": max_frames,
        "stream_acc": stream_metrics["acc"],
        "stream_precision": stream_metrics["precision"],
        "stream_recall": stream_metrics["recall"],
        "stream_f1": stream_metrics["f1"],
    }

    if args.compare_offline and offline_preds:
        offline_metrics = metrics_from_labels(labels_all, offline_preds)
        agree = sum(int(a == b) for a, b in zip(stream_preds, offline_preds)) / max(1, len(offline_preds))
        result.update(
            {
                "offline_acc": offline_metrics["acc"],
                "offline_precision": offline_metrics["precision"],
                "offline_recall": offline_metrics["recall"],
                "offline_f1": offline_metrics["f1"],
                "stream_offline_pred_agree": agree,
                "stream_offline_max_abs_logit_diff": max_abs_diff,
                "stream_offline_mean_abs_logit_diff": mean_abs_diff_sum / max(1, compare_samples),
            }
        )
    return result


def write_csv(path: str, row: dict[str, float | int | str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    full_row = {name: row.get(name, "") for name in CSV_FIELDNAMES}
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(full_row)
    print(f"[INFO] CSV appended to: {out_path.resolve()}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.gpu > 0 and torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    chunk_samples = resolve_chunk_samples(args, model)
    loader = build_eval_loader(args)

    print(f"[INFO] device={device}, params={parameter_number(model)}")
    print(f"[INFO] model_config={model_config_dict(model)}")
    print(f"[INFO] ckpt={args.ckpt}")
    print(
        f"[INFO] true_streaming=ON, split={args.split}, "
        f"chunk_samples={chunk_samples}, chunk_ms={chunk_samples * 1000.0 / args.sample_rate:.3f}, "
        f"flush_tail={args.flush_tail}"
    )
    print(
        f"[INFO] eval_noise_aug={'ON' if args.eval_noise_aug else 'OFF'}, "
        f"snr=[{args.eval_noise_snr_min_db}, {args.eval_noise_snr_max_db}] dB"
    )

    row = evaluate(model, loader, device, args, chunk_samples)
    write_csv(args.out_csv, row)

    print(
        "[TRUE_STREAM_TEST] "
        f"acc={row['stream_acc']:.4f} "
        f"precision={row['stream_precision']:.4f} "
        f"recall={row['stream_recall']:.4f} "
        f"f1={row['stream_f1']:.4f} "
        f"frames=[{row['frames_min']},{row['frames_max']}]"
    )
    if args.compare_offline:
        print(
            "[OFFLINE_COMPARE] "
            f"offline_acc={row.get('offline_acc', 0.0):.4f} "
            f"pred_agree={row.get('stream_offline_pred_agree', 0.0):.6f} "
            f"max_abs_logit_diff={row.get('stream_offline_max_abs_logit_diff', 0.0):.6g} "
            f"mean_abs_logit_diff={row.get('stream_offline_mean_abs_logit_diff', 0.0):.6g}"
        )


if __name__ == "__main__":
    main()
