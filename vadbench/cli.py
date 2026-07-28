from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import torch
import yaml

from vadbench.algorithms.base import VADAlgorithm, load_pickled_algorithm
from vadbench.algorithms.neural import TorchFrameAlgorithm
from vadbench.algorithms.registry import create_algorithm, list_algorithms
from vadbench.audio import load_audio
from vadbench.data.aishell4_realneg_vad import Aishell4RealnegVADConfig, prepare_aishell4_realneg_vad
from vadbench.data.ava_speech import AvaSpeechConfig, prepare_ava_speech
from vadbench.data.ms_snsd_vad import MS_SNSD_EVENT_CLASSES, MSSNSDVADConfig, prepare_ms_snsd_vad
from vadbench.data.yesno_synth import YesNoSynthConfig, prepare_yesno_synth
from vadbench.features import align_length, log_mel_spectrogram, mfcc_features
from vadbench.frame_prediction import FramePrediction
from vadbench.manifest import ManifestRecord, read_manifest, validate_manifest, write_manifest
from vadbench.metrics import aggregate_metrics, ava_paper_metrics, binary_frame_metrics, choose_best_threshold, roc_auc, threshold_for_fpr


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vadbench", description="VAD research benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-algorithms")
    list_parser.set_defaults(func=cmd_list_algorithms)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_sub = prepare_parser.add_subparsers(dest="dataset", required=True)
    yesno = prepare_sub.add_parser("yesno-synth")
    yesno.add_argument("--download", action="store_true")
    yesno.add_argument("--raw-root", default="data/raw")
    yesno.add_argument("--out", default="data/yesno_synth")
    yesno.add_argument("--sample-rate", type=int, default=16000)
    yesno.add_argument("--seed", type=int, default=7)
    yesno.add_argument("--train-samples", type=int, default=64)
    yesno.add_argument("--val-samples", type=int, default=16)
    yesno.add_argument("--test-samples", type=int, default=16)
    yesno.add_argument("--duration-min", type=float, default=6.0)
    yesno.add_argument("--duration-max", type=float, default=10.0)
    yesno.set_defaults(func=cmd_prepare_yesno)

    ms_snsd = prepare_sub.add_parser("ms-snsd-vad")
    ms_snsd.add_argument("--download", action="store_true")
    ms_snsd.add_argument("--raw-root", default="data/raw/ms-snsd")
    ms_snsd.add_argument("--out", default="data/ms_snsd_vad_v2")
    ms_snsd.add_argument("--protocol", default="v2", choices=["v1", "v2"])
    ms_snsd.add_argument("--sample-rate", type=int, default=16000)
    ms_snsd.add_argument("--frame-ms", type=float, default=25.0)
    ms_snsd.add_argument("--hop-ms", type=float, default=10.0)
    ms_snsd.add_argument("--n-mels", type=int, default=64)
    ms_snsd.add_argument("--clip-sec", type=float, default=10.0)
    ms_snsd.add_argument("--total-hours", type=float, default=20.0)
    ms_snsd.add_argument("--snr-levels", default="0,5,10,15,20")
    ms_snsd.add_argument("--seed", type=int, default=7)
    ms_snsd.add_argument("--no-precompute-features", action="store_true")
    ms_snsd.add_argument("--target-speech-ratio", type=float, default=0.45)
    ms_snsd.add_argument("--speech-ratio-min", type=float, default=0.35)
    ms_snsd.add_argument("--speech-ratio-max", type=float, default=0.55)
    ms_snsd.add_argument("--hard-negative-ratio", type=float, default=0.25)
    ms_snsd.add_argument("--silence-ratio", type=float, default=0.15)
    ms_snsd.add_argument("--low-noise-ratio", type=float, default=0.15)
    ms_snsd.add_argument("--exclude-noise-keywords", default="Babble,AirportAnnouncement,AirportAnnouncements,Neighbor")
    ms_snsd.set_defaults(func=cmd_prepare_ms_snsd)

    aishell4 = prepare_sub.add_parser("aishell4-realneg-vad")
    aishell4.add_argument("--aishell4-root", default="data/raw/aishell4")
    aishell4.add_argument("--fsd50k-root", default="data/raw/fsd50k")
    aishell4.add_argument("--out", default="data/aishell4_realneg_vad")
    aishell4.add_argument("--download-aishell4", action="store_true")
    aishell4.add_argument("--download-fsd50k", action="store_true")
    aishell4.add_argument("--sample-rate", type=int, default=16000)
    aishell4.add_argument("--frame-ms", type=float, default=25.0)
    aishell4.add_argument("--hop-ms", type=float, default=10.0)
    aishell4.add_argument("--n-mels", type=int, default=64)
    aishell4.add_argument("--chunk-sec", type=float, default=10.0)
    aishell4.add_argument("--negative-ratio", type=float, default=0.35)
    aishell4.add_argument("--mic-channel", type=int, default=0)
    aishell4.add_argument("--seed", type=int, default=7)
    aishell4.add_argument("--max-hours", type=float, default=None)
    aishell4.add_argument("--target-hours", type=float, default=None)
    aishell4.add_argument("--aishell4-hours", type=float, default=None)
    aishell4.add_argument("--fsd50k-hours", type=float, default=None)
    aishell4.add_argument("--aishell4-subsets", default="train_L,train_M,train_S")
    aishell4.add_argument("--aishell4-room-ratios", default="L:0.25,M:0.45,S:0.30")
    aishell4.add_argument("--use-official-aishell4-test", action="store_true")
    aishell4.add_argument("--fsd50k-train-source", default="dev")
    aishell4.add_argument("--fsd50k-test-source", default="eval")
    aishell4.add_argument("--no-precompute-features", action="store_true")
    aishell4.add_argument(
        "--fsd50k-exclude-keywords",
        default="speech,conversation,narration,singing,vocal,human voice,child speech,children speaking,crowd,choir,chant,babbling",
    )
    aishell4.set_defaults(func=cmd_prepare_aishell4_realneg)

    ava = prepare_sub.add_parser("ava-speech")
    ava.add_argument("--download-labels", action="store_true")
    ava.add_argument("--label-csv", default=None)
    ava.add_argument("--cache-root", default=r"C:\vadbench_cache\ava_speech")
    ava.add_argument("--manifest-out", default="manifests/ava_speech_manifest.jsonl")
    ava.add_argument("--video-id-file", default=None)
    ava.add_argument("--media-root", "--local-media-root", dest="media_root", default=None)
    ava.add_argument("--use-yt-dlp", action="store_true")
    ava.add_argument("--cookies", default=None)
    ava.add_argument("--cookies-from-browser", default=None)
    ava.add_argument("--failed-video-file", default=None)
    ava.add_argument("--retry-failed-only", action="store_true")
    ava.add_argument("--max-videos", type=int, default=None)
    ava.add_argument("--split-seed", type=int, default=7)
    ava.add_argument("--sample-rate", type=int, default=16000)
    ava.add_argument("--frame-ms", type=float, default=25.0)
    ava.add_argument("--hop-ms", type=float, default=10.0)
    ava.add_argument("--chunk-sec", type=float, default=30.0)
    ava.add_argument("--n-mels", type=int, default=64)
    ava.add_argument("--no-extract-audio", action="store_true")
    ava.add_argument("--no-precompute-features", action="store_true")
    ava.add_argument("--no-keep-wav", action="store_true")
    ava.set_defaults(func=cmd_prepare_ava)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", required=True)
    eval_parser.set_defaults(func=cmd_eval)

    experiment = subparsers.add_parser("experiment")
    experiment_sub = experiment.add_subparsers(dest="experiment_name", required=True)
    latency = experiment_sub.add_parser("causal-crnn-latency-sweep")
    latency.add_argument("--raw-root", default="data/raw/ms-snsd")
    latency.add_argument("--out-root", default="data/ms_snsd_vad_v2_latency_sweep")
    latency.add_argument("--run-root", default="runs/causal_crnn_latency_sweep_ms_snsd_v2")
    latency.add_argument("--total-hours", type=float, default=20.0)
    latency.add_argument("--clip-sec", type=float, default=10.0)
    latency.add_argument("--sample-rate", type=int, default=16000)
    latency.add_argument("--n-mels", type=int, default=64)
    latency.add_argument("--snr-levels", default="0,5,10,15,20")
    latency.add_argument("--seed", type=int, default=7)
    latency.add_argument("--epochs", type=int, default=40)
    latency.add_argument("--batch-size", type=int, default=16)
    latency.add_argument("--accuracy-floor", type=float, default=0.90)
    latency.add_argument("--smoke", action="store_true")
    latency.set_defaults(func=cmd_experiment_causal_crnn_latency_sweep)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--config", required=True)
    infer.add_argument("--wav", required=True)
    infer.add_argument("--out", required=True)
    infer.set_defaults(func=cmd_infer)

    pseudo = subparsers.add_parser("pseudo-label")
    pseudo.add_argument("--config", required=True)
    pseudo.add_argument("--out", required=True)
    pseudo.add_argument("--split", default=None)
    pseudo.set_defaults(func=cmd_pseudo_label)

    return parser


def cmd_list_algorithms(args: argparse.Namespace) -> int:
    for name in list_algorithms():
        print(name)
    return 0


def cmd_prepare_yesno(args: argparse.Namespace) -> int:
    config = YesNoSynthConfig(
        raw_root=Path(args.raw_root),
        out_dir=Path(args.out),
        download=bool(args.download),
        sample_rate=int(args.sample_rate),
        seed=int(args.seed),
        train_samples=int(args.train_samples),
        val_samples=int(args.val_samples),
        test_samples=int(args.test_samples),
        duration_min_sec=float(args.duration_min),
        duration_max_sec=float(args.duration_max),
    )
    manifest = prepare_yesno_synth(config)
    print(f"Wrote manifest: {manifest}")
    return 0


def cmd_prepare_ms_snsd(args: argparse.Namespace) -> int:
    snr_levels = [float(item.strip()) for item in str(args.snr_levels).split(",") if item.strip()]
    exclude_noise_keywords = [item.strip() for item in str(args.exclude_noise_keywords).split(",") if item.strip()]
    config = MSSNSDVADConfig(
        raw_root=Path(args.raw_root),
        out_dir=Path(args.out),
        download=bool(args.download),
        protocol=str(args.protocol),
        sample_rate=int(args.sample_rate),
        frame_ms=float(args.frame_ms),
        hop_ms=float(args.hop_ms),
        n_mels=int(args.n_mels),
        clip_sec=float(args.clip_sec),
        total_hours=float(args.total_hours),
        snr_levels=snr_levels,
        seed=int(args.seed),
        precompute_features=not bool(args.no_precompute_features),
        target_speech_ratio=float(args.target_speech_ratio),
        speech_ratio_min=float(args.speech_ratio_min),
        speech_ratio_max=float(args.speech_ratio_max),
        hard_negative_ratio=float(args.hard_negative_ratio),
        silence_ratio=float(args.silence_ratio),
        low_noise_ratio=float(args.low_noise_ratio),
        exclude_noise_keywords=exclude_noise_keywords,
    )
    manifest = prepare_ms_snsd_vad(config)
    print(f"Wrote manifest: {manifest}")
    return 0


def cmd_prepare_aishell4_realneg(args: argparse.Namespace) -> int:
    exclude_keywords = [item.strip() for item in str(args.fsd50k_exclude_keywords).split(",") if item.strip()]
    aishell4_subsets = [item.strip() for item in str(args.aishell4_subsets).split(",") if item.strip()]
    room_ratios = _parse_room_ratios(str(args.aishell4_room_ratios))
    config = Aishell4RealnegVADConfig(
        aishell4_root=Path(args.aishell4_root),
        fsd50k_root=Path(args.fsd50k_root),
        out_dir=Path(args.out),
        download_aishell4=bool(args.download_aishell4),
        download_fsd50k=bool(args.download_fsd50k),
        sample_rate=int(args.sample_rate),
        frame_ms=float(args.frame_ms),
        hop_ms=float(args.hop_ms),
        n_mels=int(args.n_mels),
        chunk_sec=float(args.chunk_sec),
        negative_ratio=float(args.negative_ratio),
        mic_channel=int(args.mic_channel),
        seed=int(args.seed),
        max_hours=float(args.max_hours) if args.max_hours is not None else None,
        target_hours=float(args.target_hours) if args.target_hours is not None else None,
        aishell4_hours=float(args.aishell4_hours) if args.aishell4_hours is not None else None,
        fsd50k_hours=float(args.fsd50k_hours) if args.fsd50k_hours is not None else None,
        aishell4_subsets=aishell4_subsets,
        aishell4_room_ratios=room_ratios,
        use_official_aishell4_test=bool(args.use_official_aishell4_test),
        fsd50k_train_source=str(args.fsd50k_train_source),
        fsd50k_test_source=str(args.fsd50k_test_source),
        precompute_features=not bool(args.no_precompute_features),
        fsd50k_exclude_keywords=exclude_keywords,
    )
    manifest = prepare_aishell4_realneg_vad(config)
    print(f"Wrote manifest: {manifest}")
    return 0


def _parse_room_ratios(value: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Invalid room ratio item: {item!r}; expected e.g. L:0.25")
        key, raw = item.split(":", 1)
        ratios[key.strip().upper()] = float(raw.strip())
    return ratios


def cmd_prepare_ava(args: argparse.Namespace) -> int:
    config = AvaSpeechConfig(
        cache_root=Path(args.cache_root),
        manifest_out=Path(args.manifest_out),
        label_csv=Path(args.label_csv) if args.label_csv else None,
        download_labels=bool(args.download_labels),
        video_id_file=Path(args.video_id_file) if args.video_id_file else None,
        media_root=Path(args.media_root) if args.media_root else None,
        use_yt_dlp=bool(args.use_yt_dlp),
        cookies=Path(args.cookies) if args.cookies else None,
        cookies_from_browser=args.cookies_from_browser,
        failed_video_file=Path(args.failed_video_file) if args.failed_video_file else None,
        retry_failed_only=bool(args.retry_failed_only),
        max_videos=args.max_videos,
        split_seed=int(args.split_seed),
        sample_rate=int(args.sample_rate),
        frame_ms=float(args.frame_ms),
        hop_ms=float(args.hop_ms),
        chunk_sec=float(args.chunk_sec),
        n_mels=int(args.n_mels),
        extract_audio=not bool(args.no_extract_audio),
        precompute_features=not bool(args.no_precompute_features),
        keep_wav=not bool(args.no_keep_wav),
    )
    manifest = prepare_ava_speech(config)
    print(f"Wrote manifest: {manifest}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = _load_config(config_path)
    _seed_everything(int(config.get("seed", 7)))
    manifest_path = Path(config["manifest"])
    base_dir = manifest_path.parent
    _check_manifest_lightweight(manifest_path)
    train_records = read_manifest(manifest_path, split="train")
    val_records = read_manifest(manifest_path, split="val")
    run_dir = _prepare_run_dir(config, config_path)
    algorithm = _create_algorithm_from_config(config)
    history = algorithm.fit(train_records, val_records, base_dir=base_dir, run_dir=run_dir, training=config.get("training", {}))

    checkpoint_dir = run_dir / "checkpoints"
    if isinstance(algorithm, TorchFrameAlgorithm):
        checkpoint_path = checkpoint_dir / "best.pt"
        algorithm.save_checkpoint(checkpoint_path)
        model_stats = _model_stats_with_context(algorithm.model_stats(frames=1), config)
        _write_json(run_dir / "model_stats.json", model_stats)
    else:
        checkpoint_path = checkpoint_dir / "model.pkl"
        algorithm.save(checkpoint_path)
        model_stats = {}

    _write_json(run_dir / "train_metrics.json", {"history": history, "checkpoint": str(checkpoint_path), "model_stats": model_stats})
    print(f"Wrote checkpoint: {checkpoint_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = _load_config(config_path)
    manifest_path = Path(config["manifest"])
    base_dir = manifest_path.parent
    _check_manifest_lightweight(manifest_path)
    run_dir = _prepare_run_dir(config, config_path)
    algorithm = _load_or_fit_algorithm(config, manifest_path, base_dir)
    if isinstance(algorithm, TorchFrameAlgorithm):
        _write_json(run_dir / "model_stats.json", _model_stats_with_context(algorithm.model_stats(frames=1), config))

    eval_cfg = config.get("eval", {})
    split = eval_cfg.get("split", "test")
    records = read_manifest(manifest_path, split=split)
    threshold = _resolve_threshold(config, algorithm, manifest_path, base_dir, run_dir=run_dir)
    post = config.get("postprocess", {})
    result = evaluate_records(
        algorithm,
        records,
        base_dir,
        run_dir,
        threshold=threshold,
        eval_cfg=eval_cfg,
        min_speech_ms=float(post.get("min_speech_ms", 60.0)),
        min_silence_ms=float(post.get("min_silence_ms", 100.0)),
    )
    if eval_cfg.get("paper_metrics", False):
        result["paper"] = evaluate_paper_metrics(
            algorithm,
            records,
            base_dir,
            manifest_path,
            run_dir,
            target_fpr=float(eval_cfg.get("target_fpr", 0.315)),
            eval_cfg=eval_cfg,
            output_name="paper_metrics.json",
        )
        result["paper_val_threshold"] = evaluate_paper_metrics(
            algorithm,
            records,
            base_dir,
            manifest_path,
            run_dir,
            target_fpr=float(eval_cfg.get("target_fpr", 0.315)),
            tune_split=eval_cfg.get("tune_split", "val"),
            use_tune_threshold=True,
            eval_cfg=eval_cfg,
            output_name="paper_metrics_val_threshold.json",
        )
    _write_json(run_dir / "metrics.json", result)
    print(f"Wrote metrics: {run_dir / 'metrics.json'}")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


def _check_manifest_lightweight(path: str | Path, samples_per_bucket: int = 2) -> list[ManifestRecord]:
    path = Path(path)
    records = read_manifest(path)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    seen: set[str] = set()
    buckets: dict[tuple[str, str], list[ManifestRecord]] = {}
    base_dir = path.parent
    for record in records:
        if record.id in seen:
            raise ValueError(f"Duplicate manifest id: {record.id}")
        seen.add(record.id)
        if record.sample_rate <= 0 or record.duration_sec <= 0 or record.frame_hop_ms <= 0:
            raise ValueError(f"{record.id}: invalid manifest timing fields")
        key = (record.split, record.source)
        bucket = buckets.setdefault(key, [])
        if len(bucket) < samples_per_bucket:
            bucket.append(record)
    for items in buckets.values():
        for record in items:
            audio_path = record.resolve_audio(base_dir)
            label_path = record.resolve_label(base_dir)
            if not audio_path.exists():
                raise FileNotFoundError(f"{record.id}: missing audio {audio_path}")
            if not label_path.exists():
                raise FileNotFoundError(f"{record.id}: missing label {label_path}")
            labels = np.load(label_path)
            if labels.ndim != 1:
                raise ValueError(f"{record.id}: labels must be 1D")
            class_label_path = record.resolve_class_label(base_dir)
            if class_label_path is not None:
                class_labels = np.load(class_label_path)
                if class_labels.ndim != 1 or len(class_labels) != len(labels):
                    raise ValueError(f"{record.id}: invalid class labels")
            feature_path = record.resolve_feature(base_dir)
            if feature_path is not None:
                features = np.load(feature_path)
                if features.ndim != 2 or features.shape[0] != len(labels):
                    raise ValueError(f"{record.id}: invalid feature shape")
    return records


def cmd_experiment_causal_crnn_latency_sweep(args: argparse.Namespace) -> int:
    snr_levels = [float(item.strip()) for item in str(args.snr_levels).split(",") if item.strip()]
    out_root = Path(args.out_root)
    run_root = Path(args.run_root)
    config_root = run_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    variants = _latency_sweep_variants(smoke=bool(args.smoke))
    if args.smoke:
        args.total_hours = min(float(args.total_hours), 0.01)
        args.epochs = 1
    summary_rows: list[dict[str, object]] = []
    for variant in variants:
        tag = _latency_variant_tag(variant)
        data_dir = out_root / tag
        print(f"[latency-sweep] preparing {tag}", flush=True)
        manifest = prepare_ms_snsd_vad(
            MSSNSDVADConfig(
                raw_root=Path(args.raw_root),
                out_dir=data_dir,
                download=False,
                sample_rate=int(args.sample_rate),
                frame_ms=float(variant["frame_ms"]),
                hop_ms=float(variant["hop_ms"]),
                n_mels=int(args.n_mels),
                clip_sec=float(args.clip_sec),
                total_hours=float(args.total_hours),
                snr_levels=snr_levels,
                seed=int(args.seed),
                precompute_features=True,
            )
        )
        train_config = _latency_train_config(args, variant, manifest, run_root / tag)
        eval_config = _latency_eval_config(args, variant, manifest, run_root / tag)
        train_config_path = config_root / f"{tag}.train.yaml"
        eval_config_path = config_root / f"{tag}.eval.yaml"
        _write_yaml(train_config_path, train_config)
        _write_yaml(eval_config_path, eval_config)
        print(f"[latency-sweep] training {tag}", flush=True)
        cmd_train(argparse.Namespace(config=str(train_config_path)))
        print(f"[latency-sweep] evaluating {tag}", flush=True)
        cmd_eval(argparse.Namespace(config=str(eval_config_path)))
        row = _latency_result_row(tag, variant, eval_config["run_dir"], accuracy_floor=float(args.accuracy_floor))
        summary_rows.append(row)
        _write_latency_summary(run_root, summary_rows, accuracy_floor=float(args.accuracy_floor))

    best = _select_latency_best(summary_rows, accuracy_floor=float(args.accuracy_floor))
    if best is None and not args.smoke:
        print("[latency-sweep] no hop=20ms variant reached target; running hop=15ms fallback", flush=True)
        fallback_args = argparse.Namespace(**vars(args))
        fallback_args.smoke = False
        for variant in _latency_fallback_variants():
            tag = _latency_variant_tag(variant)
            if any(row["tag"] == tag for row in summary_rows):
                continue
            data_dir = out_root / tag
            print(f"[latency-sweep] preparing {tag}", flush=True)
            manifest = prepare_ms_snsd_vad(
                MSSNSDVADConfig(
                    raw_root=Path(args.raw_root),
                    out_dir=data_dir,
                    download=False,
                    sample_rate=int(args.sample_rate),
                    frame_ms=float(variant["frame_ms"]),
                    hop_ms=float(variant["hop_ms"]),
                    n_mels=int(args.n_mels),
                    clip_sec=float(args.clip_sec),
                    total_hours=float(args.total_hours),
                    snr_levels=snr_levels,
                    seed=int(args.seed),
                    precompute_features=True,
                )
            )
            train_config = _latency_train_config(args, variant, manifest, run_root / tag)
            eval_config = _latency_eval_config(args, variant, manifest, run_root / tag)
            train_config_path = config_root / f"{tag}.train.yaml"
            eval_config_path = config_root / f"{tag}.eval.yaml"
            _write_yaml(train_config_path, train_config)
            _write_yaml(eval_config_path, eval_config)
            print(f"[latency-sweep] training {tag}", flush=True)
            cmd_train(argparse.Namespace(config=str(train_config_path)))
            print(f"[latency-sweep] evaluating {tag}", flush=True)
            cmd_eval(argparse.Namespace(config=str(eval_config_path)))
            summary_rows.append(_latency_result_row(tag, variant, eval_config["run_dir"], accuracy_floor=float(args.accuracy_floor)))
            _write_latency_summary(run_root, summary_rows, accuracy_floor=float(args.accuracy_floor))
        best = _select_latency_best(summary_rows, accuracy_floor=float(args.accuracy_floor))

    _write_latency_summary(run_root, summary_rows, accuracy_floor=float(args.accuracy_floor))
    if best is not None:
        _write_yaml(run_root / "best_config.yaml", best)
        print("[latency-sweep] best:")
        print(json.dumps(best, indent=2, sort_keys=True))
    else:
        print(f"[latency-sweep] no variant reached F1 >= {float(args.accuracy_floor):.3f}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    algorithm = _create_algorithm_from_config(config)
    _load_checkpoint_if_present(algorithm, config)
    feature_cfg = config.get("features", {})
    target_sr = int(feature_cfg.get("sample_rate", 16000))
    waveform, sample_rate = load_audio(args.wav, target_sr)
    prediction = algorithm.predict(waveform, sample_rate, source_id=Path(args.wav).stem)
    np.save(out_dir / "prediction.npy", prediction.scores)
    threshold_cfg = config.get("postprocess", {}).get("threshold", algorithm.threshold or 0.5)
    threshold = float(algorithm.threshold if threshold_cfg == "auto" else threshold_cfg)
    segments = prediction.to_segments(
        threshold=threshold,
        min_speech_ms=float(config.get("postprocess", {}).get("min_speech_ms", 60.0)),
        min_silence_ms=float(config.get("postprocess", {}).get("min_silence_ms", 100.0)),
    )
    _write_segments(out_dir / "segments.csv", [(Path(args.wav).stem, segments)])
    print(f"Wrote inference outputs: {out_dir}")
    return 0


def cmd_pseudo_label(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    manifest_path = Path(config["manifest"])
    base_dir = manifest_path.parent
    algorithm = _create_algorithm_from_config(config)
    _load_checkpoint_if_present(algorithm, config)
    records = read_manifest(manifest_path, split=args.split)
    out_dir = Path(args.out)
    label_dir = out_dir / "soft_labels"
    new_records: list[ManifestRecord] = []
    feature_cfg = config.get("features", {})
    target_sr = int(feature_cfg.get("sample_rate", 16000))
    for record in records:
        waveform, sample_rate = load_audio(record.resolve_audio(base_dir), target_sr)
        prediction = algorithm.predict(waveform, sample_rate, source_id=record.id)
        rel_label = Path("soft_labels") / f"{record.id}.npy"
        (out_dir / rel_label).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / rel_label, prediction.scores.astype(np.float32))
        new_records.append(
            ManifestRecord(
                id=record.id,
                audio_path=str(record.resolve_audio(base_dir)),
                label_path=rel_label.as_posix(),
                split=record.split,
                sample_rate=record.sample_rate,
                duration_sec=record.duration_sec,
                frame_hop_ms=record.frame_hop_ms,
                source=f"pseudo:{config['algorithm']}",
            )
        )
    manifest_out = out_dir / "manifest_soft.jsonl"
    write_manifest(new_records, manifest_out)
    print(f"Wrote pseudo-label manifest: {manifest_out}")
    return 0


def evaluate_records(
    algorithm: VADAlgorithm,
    records: list[ManifestRecord],
    base_dir: str | Path,
    run_dir: str | Path,
    threshold: float,
    eval_cfg: dict | None = None,
    min_speech_ms: float = 60.0,
    min_silence_ms: float = 100.0,
) -> dict:
    run_dir = Path(run_dir)
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    per_file: list[dict[str, float | str]] = []
    segment_rows: list[tuple[str, list[dict[str, float]]]] = []
    frame_metrics: list[dict[str, float]] = []
    event_counts = _empty_ms_snsd_event_counts()
    source_counts: dict[str, dict[str, float]] = {}
    target_sr = getattr(algorithm, "sample_rate", None)
    eval_cfg = eval_cfg or {}
    for idx, record in enumerate(records, start=1):
        labels = np.load(record.resolve_label(base_dir)).astype(np.uint8)
        prediction, scores = _cached_prediction(
            algorithm,
            record,
            base_dir,
            target_sr,
            eval_cfg,
            pred_dir,
            expected_len=len(labels),
        )
        metrics = binary_frame_metrics(labels, scores, threshold=threshold)
        _accumulate_ms_snsd_event_metrics(event_counts, record, base_dir, scores, threshold)
        _accumulate_source_metrics(source_counts, record, labels, scores, threshold)
        frame_metrics.append(metrics)
        per_file.append({"id": record.id, **metrics})
        segment_rows.append(
            (
                record.id,
                prediction.to_segments(
                    threshold=threshold,
                    min_speech_ms=min_speech_ms,
                    min_silence_ms=min_silence_ms,
                ),
            )
        )
        if idx == 1 or idx % int(eval_cfg.get("log_every_records", 500) or 500) == 0 or idx == len(records):
            print(f"[eval] processed {idx}/{len(records)} records", flush=True)
    _write_segments(run_dir / "segments.csv", segment_rows)
    _write_json(run_dir / "per_file_metrics.json", per_file)
    aggregate = aggregate_metrics(frame_metrics)
    aggregate["threshold"] = float(threshold)
    result = {"aggregate": aggregate, "per_file": per_file}
    event_metrics = _finalize_ms_snsd_event_metrics(event_counts)
    if event_metrics:
        result["event_metrics"] = event_metrics
    source_metrics = _finalize_source_metrics(source_counts)
    if source_metrics:
        result["source_metrics"] = source_metrics
    return result


def evaluate_paper_metrics(
    algorithm: VADAlgorithm,
    records: list[ManifestRecord],
    base_dir: str | Path,
    manifest_path: Path,
    run_dir: str | Path,
    target_fpr: float = 0.315,
    tune_split: str | None = None,
    use_tune_threshold: bool = False,
    eval_cfg: dict | None = None,
    output_name: str = "paper_metrics.json",
) -> dict[str, float]:
    eval_cfg = eval_cfg or {}
    threshold = (
        _resolve_paper_threshold(algorithm, manifest_path, base_dir, run_dir, target_fpr, tune_split, eval_cfg)
        if use_tune_threshold and tune_split is not None
        else None
    )
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    target_sr = getattr(algorithm, "sample_rate", None)
    pred_dir = Path(run_dir) / "predictions"
    for idx, record in enumerate(records, start=1):
        labels = _record_binary_or_class_labels(record, base_dir)
        _, scores = _cached_prediction(
            algorithm,
            record,
            base_dir,
            target_sr,
            eval_cfg,
            pred_dir,
            expected_len=len(labels),
        )
        labels_all.append(labels)
        scores_all.append(scores)
        if idx == 1 or idx % int(eval_cfg.get("log_every_records", 500) or 500) == 0 or idx == len(records):
            print(f"[paper-eval:{output_name}] processed {idx}/{len(records)} records", flush=True)
    if not labels_all:
        return {}
    y_labels = np.concatenate(labels_all)
    y_scores = np.concatenate(scores_all)
    if int(np.max(y_labels)) > 1:
        metrics = ava_paper_metrics(y_labels, y_scores, target_fpr, threshold)
    else:
        binary_threshold = threshold_for_fpr(y_labels, y_scores, target_fpr) if threshold is None else float(threshold)
        frame_metrics = binary_frame_metrics(y_labels, y_scores, binary_threshold)
        metrics = {
            "target_fpr": float(target_fpr),
            "threshold_at_fpr": float(binary_threshold),
            "fpr": float(frame_metrics["fp"] / max(frame_metrics["fp"] + frame_metrics["tn"], 1.0)),
            "tpr_at_target_fpr": float(frame_metrics["recall"]),
            "tpr_at_fpr_0_315": float(frame_metrics["recall"]) if abs(float(target_fpr) - 0.315) < 1e-9 else float("nan"),
            "tpr_all": float(frame_metrics["recall"]),
            "auroc_all": roc_auc(y_labels, y_scores),
        }
    _write_json(Path(run_dir) / output_name, metrics)
    return metrics


def _resolve_paper_threshold(
    algorithm: VADAlgorithm,
    manifest_path: Path,
    base_dir: str | Path,
    run_dir: str | Path,
    target_fpr: float,
    tune_split: str,
    eval_cfg: dict | None = None,
) -> float:
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    target_sr = getattr(algorithm, "sample_rate", None)
    eval_cfg = eval_cfg or {}
    tune_records = read_manifest(manifest_path, split=tune_split)
    pred_dir = Path(run_dir) / "predictions"
    for idx, record in enumerate(tune_records, start=1):
        labels = _record_binary_or_class_labels(record, base_dir)
        _, scores = _cached_prediction(
            algorithm,
            record,
            base_dir,
            target_sr,
            eval_cfg,
            pred_dir,
            expected_len=len(labels),
        )
        labels_all.append(labels)
        scores_all.append(scores)
        if idx == 1 or idx % int(eval_cfg.get("log_every_records", 500) or 500) == 0 or idx == len(tune_records):
            print(f"[paper-threshold:{tune_split}] processed {idx}/{len(tune_records)} records", flush=True)
    if not labels_all:
        return float(algorithm.threshold if algorithm.threshold is not None else 0.5)
    return threshold_for_fpr(np.concatenate(labels_all), np.concatenate(scores_all), target_fpr)


def _is_ms_snsd_event_record(record: ManifestRecord) -> bool:
    source = (record.source or "").lower()
    label_source = (record.label_source or "").lower()
    return "ms-snsd" in source or "protocol=v" in label_source


def _is_aishell4_realneg_record(record: ManifestRecord) -> bool:
    source = (record.source or "").lower()
    label_source = (record.label_source or "").lower()
    return "aishell4" in source or "fsd50k" in source or "aishell4" in label_source or "fsd50k" in label_source


def _record_binary_or_class_labels(record: ManifestRecord, base_dir: str | Path) -> np.ndarray:
    class_path = record.resolve_class_label(base_dir)
    if class_path is not None and class_path.exists():
        raw_labels = np.load(class_path).astype(np.uint8)
        if _is_ms_snsd_event_record(record):
            return _ms_snsd_event_labels_to_binary(record, raw_labels)
        if _is_aishell4_realneg_record(record):
            return (raw_labels == 1).astype(np.uint8)
        return raw_labels
    return np.load(record.resolve_label(base_dir)).astype(np.uint8)


def _ms_snsd_event_labels_to_binary(record: ManifestRecord, event_labels: np.ndarray) -> np.ndarray:
    if not _is_ms_snsd_event_record(record):
        return (np.asarray(event_labels) > 0).astype(np.uint8)
    labels = np.asarray(event_labels, dtype=np.uint8)
    speech_ids = (
        MS_SNSD_EVENT_CLASSES["clean_speech"],
        MS_SNSD_EVENT_CLASSES["speech_with_noise"],
    )
    return np.isin(labels, speech_ids).astype(np.uint8)


def _empty_ms_snsd_event_counts() -> dict[str, dict[str, float]]:
    return {
        name: {"frames": 0.0, "positive_predictions": 0.0}
        for name in MS_SNSD_EVENT_CLASSES
    }


def _accumulate_ms_snsd_event_metrics(
    counts: dict[str, dict[str, float]],
    record: ManifestRecord,
    base_dir: str | Path,
    scores: np.ndarray,
    threshold: float,
) -> None:
    if not _is_ms_snsd_event_record(record):
        return
    class_path = record.resolve_class_label(base_dir)
    if class_path is None or not class_path.exists():
        return
    event_labels = np.load(class_path).astype(np.uint8)
    length = min(len(event_labels), len(scores))
    if length <= 0:
        return
    event_labels = event_labels[:length]
    predicted_speech = np.asarray(scores[:length], dtype=np.float32) >= float(threshold)
    for name, value in MS_SNSD_EVENT_CLASSES.items():
        mask = event_labels == value
        counts[name]["frames"] += float(np.sum(mask))
        counts[name]["positive_predictions"] += float(np.sum(predicted_speech & mask))


def _finalize_ms_snsd_event_metrics(counts: dict[str, dict[str, float]]) -> dict[str, float]:
    total_frames = sum(item["frames"] for item in counts.values())
    if total_frames <= 0.0:
        return {}
    speech_frames = counts["clean_speech"]["frames"] + counts["speech_with_noise"]["frames"]
    speech_hits = counts["clean_speech"]["positive_predictions"] + counts["speech_with_noise"]["positive_predictions"]
    output = {
        "speech_recall": float(speech_hits / max(speech_frames, 1.0)),
        "silence_false_positive_rate": _event_positive_rate(counts, "silence"),
        "low_background_noise_false_positive_rate": _event_positive_rate(counts, "low_background_noise"),
        "noise_only_false_positive_rate": _event_positive_rate(counts, "noise_only_event"),
        "hard_negative_false_positive_rate": _event_positive_rate(counts, "hard_negative_event"),
        "clean_speech_recall": _event_positive_rate(counts, "clean_speech"),
        "speech_with_noise_recall": _event_positive_rate(counts, "speech_with_noise"),
    }
    for name, item in counts.items():
        output[f"{name}_frames"] = float(item["frames"])
    return output


def _event_positive_rate(counts: dict[str, dict[str, float]], name: str) -> float:
    item = counts[name]
    return float(item["positive_predictions"] / max(item["frames"], 1.0))


def _accumulate_source_metrics(
    counts: dict[str, dict[str, float]],
    record: ManifestRecord,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    source = str(record.source or "unknown")
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    length = min(len(labels), len(scores))
    if length <= 0:
        return
    labels = labels[:length].astype(bool)
    predicted = scores[:length] >= float(threshold)
    item = counts.setdefault(source, {"frames": 0.0, "speech": 0.0, "non_speech": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0})
    item["frames"] += float(length)
    item["speech"] += float(np.sum(labels))
    item["non_speech"] += float(np.sum(~labels))
    item["tp"] += float(np.sum(labels & predicted))
    item["fp"] += float(np.sum(~labels & predicted))
    item["fn"] += float(np.sum(labels & ~predicted))
    item["tn"] += float(np.sum(~labels & ~predicted))


def _finalize_source_metrics(counts: dict[str, dict[str, float]]) -> dict[str, float]:
    if not counts:
        return {}
    out: dict[str, float] = {}
    total_non_speech = sum(item["non_speech"] for item in counts.values())
    total_fp = sum(item["fp"] for item in counts.values())
    for source, item in sorted(counts.items()):
        key = _metric_key(source)
        out[f"{key}_frames"] = float(item["frames"])
        out[f"{key}_speech_recall"] = float(item["tp"] / max(item["speech"], 1.0))
        out[f"{key}_false_positive_rate"] = float(item["fp"] / max(item["non_speech"], 1.0))
    if "AISHELL4" in counts:
        out["aishell4_speech_recall"] = float(counts["AISHELL4"]["tp"] / max(counts["AISHELL4"]["speech"], 1.0))
    if "FSD50K-hard-negative" in counts:
        out["fsd50k_hard_negative_false_positive_rate"] = float(
            counts["FSD50K-hard-negative"]["fp"] / max(counts["FSD50K-hard-negative"]["non_speech"], 1.0)
        )
    out["non_speech_false_positive_rate"] = float(total_fp / max(total_non_speech, 1.0))
    return out


def _metric_key(source: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in source).strip("_")


def _resolve_threshold(config: dict, algorithm: VADAlgorithm, manifest_path: Path, base_dir: Path, run_dir: Path | None = None) -> float:
    post = config.get("postprocess", {})
    threshold_cfg = post.get("threshold", None)
    eval_cfg = config.get("eval", {})
    should_tune = bool(eval_cfg.get("tune_threshold", False)) or threshold_cfg == "auto"
    if should_tune:
        tune_split = eval_cfg.get("tune_split", "val")
        records = read_manifest(manifest_path, split=tune_split)
        labels_all: list[np.ndarray] = []
        scores_all: list[np.ndarray] = []
        target_sr = getattr(algorithm, "sample_rate", None)
        pred_dir = run_dir / "predictions" if run_dir is not None else None
        for idx, record in enumerate(records, start=1):
            labels = np.load(record.resolve_label(base_dir)).astype(np.uint8)
            if pred_dir is not None:
                _, scores = _cached_prediction(
                    algorithm,
                    record,
                    base_dir,
                    target_sr,
                    eval_cfg,
                    pred_dir,
                    expected_len=len(labels),
                )
            else:
                scores = align_length(_predict_record(algorithm, record, base_dir, target_sr, eval_cfg=eval_cfg).scores, len(labels), pad_value=0.0)
            labels_all.append(labels)
            scores_all.append(scores)
            if idx == 1 or idx % int(eval_cfg.get("log_every_records", 500) or 500) == 0 or idx == len(records):
                print(f"[threshold:{tune_split}] processed {idx}/{len(records)} records", flush=True)
        if labels_all:
            threshold, _ = choose_best_threshold(np.concatenate(labels_all), np.concatenate(scores_all))
            algorithm.threshold = threshold
            return float(threshold)
    if threshold_cfg is not None and threshold_cfg != "auto":
        return float(threshold_cfg)
    return float(algorithm.threshold if algorithm.threshold is not None else 0.5)


def _cached_prediction(
    algorithm: VADAlgorithm,
    record: ManifestRecord,
    base_dir: str | Path,
    target_sr: int | None,
    eval_cfg: dict,
    pred_dir: Path,
    expected_len: int,
) -> tuple[FramePrediction, np.ndarray]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{record.id}.npy"
    if pred_path.exists():
        try:
            cached = np.load(pred_path).astype(np.float32)
            if len(cached) == expected_len:
                return FramePrediction(scores=cached, frame_hop_ms=float(record.frame_hop_ms), source_id=record.id), cached
        except Exception:
            pass
    prediction = _predict_record(algorithm, record, base_dir, target_sr, eval_cfg=eval_cfg)
    scores = align_length(prediction.scores, expected_len, pad_value=0.0).astype(np.float32)
    np.save(pred_path, scores)
    return FramePrediction(scores=scores, frame_hop_ms=prediction.frame_hop_ms, source_id=prediction.source_id), scores


def _predict_record(
    algorithm: VADAlgorithm,
    record: ManifestRecord,
    base_dir: str | Path,
    target_sr: int | None,
    eval_cfg: dict | None = None,
):
    eval_cfg = eval_cfg or {}
    feature_path = record.resolve_feature(base_dir)
    if (
        isinstance(algorithm, TorchFrameAlgorithm)
        and getattr(algorithm, "feature_type", "logmel") == "logmel"
        and feature_path is not None
        and feature_path.exists()
    ):
        features = np.load(feature_path).astype(np.float32)
        return _predict_features_with_eval_options(algorithm, features, record.id, eval_cfg)
    waveform, sample_rate = load_audio(record.resolve_audio(base_dir), target_sr)
    if isinstance(algorithm, TorchFrameAlgorithm):
        if getattr(algorithm, "feature_type", "logmel") == "mfcc":
            features = mfcc_features(
                waveform,
                sample_rate,
                algorithm.n_mels,
                algorithm.n_mels,
                algorithm.frame_ms,
                algorithm.hop_ms,
                normalize=True,
            )
        else:
            features = log_mel_spectrogram(
                waveform,
                sample_rate,
                algorithm.n_mels,
                algorithm.frame_ms,
                algorithm.hop_ms,
                normalize=True,
            )
        return _predict_features_with_eval_options(algorithm, features, record.id, eval_cfg)
    prediction = algorithm.predict(waveform, sample_rate, source_id=record.id)
    return _postprocess_prediction_scores(prediction, eval_cfg)


def _predict_features_with_eval_options(
    algorithm: TorchFrameAlgorithm,
    features: np.ndarray,
    source_id: str,
    eval_cfg: dict,
) -> FramePrediction:
    window_sec = eval_cfg.get("sliding_window_sec", None)
    if window_sec is None:
        return _postprocess_prediction_scores(algorithm.predict_features(features, source_id=source_id), eval_cfg)
    window_frames = max(1, int(round(float(window_sec) * 1000.0 / float(algorithm.hop_ms))))
    overlap = float(eval_cfg.get("sliding_overlap", 0.0))
    step_frames = max(1, int(round(window_frames * (1.0 - overlap))))
    if len(features) <= window_frames:
        return _postprocess_prediction_scores(algorithm.predict_features(features, source_id=source_id), eval_cfg)
    score_sum = np.zeros(len(features), dtype=np.float32)
    weight_sum = np.zeros(len(features), dtype=np.float32)
    starts = list(range(0, max(1, len(features) - window_frames + 1), step_frames))
    last_start = max(0, len(features) - window_frames)
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        end = min(len(features), start + window_frames)
        chunk = features[start:end]
        prediction = algorithm.predict_features(chunk, source_id=source_id)
        scores = align_length(prediction.scores, end - start, pad_value=0.0).astype(np.float32)
        score_sum[start:end] += scores
        weight_sum[start:end] += 1.0
    scores = score_sum / np.maximum(weight_sum, 1.0)
    return _postprocess_prediction_scores(FramePrediction(scores=scores, frame_hop_ms=algorithm.hop_ms, source_id=source_id), eval_cfg)


def _postprocess_prediction_scores(prediction: FramePrediction, eval_cfg: dict) -> FramePrediction:
    scores = prediction.scores
    median_ms = float(eval_cfg.get("median_smoothing_ms", 0.0) or 0.0)
    if median_ms > 0.0:
        scores = _median_filter_1d(scores, max(1, int(round(median_ms / prediction.frame_hop_ms))))
    return FramePrediction(scores=scores, frame_hop_ms=prediction.frame_hop_ms, source_id=prediction.source_id)


def _median_filter_1d(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if width <= 1 or len(values) <= 1:
        return values
    if width % 2 == 0:
        width += 1
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    out = np.empty_like(values)
    for idx in range(len(values)):
        out[idx] = float(np.median(padded[idx : idx + width]))
    return out


def _load_or_fit_algorithm(config: dict, manifest_path: Path, base_dir: Path) -> VADAlgorithm:
    model_path = config.get("model_path")
    if model_path:
        return load_pickled_algorithm(model_path)
    algorithm = _create_algorithm_from_config(config)
    if _load_checkpoint_if_present(algorithm, config):
        return algorithm
    if isinstance(algorithm, TorchFrameAlgorithm):
        raise FileNotFoundError("Neural eval requires 'checkpoint' in config")
    train_records = read_manifest(manifest_path, split="train")
    val_records = read_manifest(manifest_path, split=config.get("eval", {}).get("tune_split", "val"))
    algorithm.fit(train_records, val_records, base_dir=base_dir)
    return algorithm


def _load_checkpoint_if_present(algorithm: VADAlgorithm, config: dict) -> bool:
    checkpoint = config.get("checkpoint")
    if not checkpoint:
        return False
    if not isinstance(algorithm, TorchFrameAlgorithm):
        raise ValueError("Only torch algorithms support 'checkpoint'; use 'model_path' for pickled algorithms")
    algorithm.load_checkpoint(checkpoint)
    return True


def _create_algorithm_from_config(config: dict) -> VADAlgorithm:
    features = dict(config.get("features", {}))
    training = dict(config.get("training", {}))
    algorithm_kwargs = dict(config.get("algorithm_kwargs", {}))
    kwargs = {
        "sample_rate": int(features.get("sample_rate", 16000)),
        "frame_ms": float(features.get("frame_ms", 25.0)),
        "hop_ms": float(features.get("hop_ms", 10.0)),
        "frame_hop_ms": float(features.get("hop_ms", 10.0)),
        "n_mels": int(features.get("n_mels", 64)),
        "n_mfcc": int(features.get("n_mfcc", 64)),
        "device": training.get("device", "auto"),
        "training": training,
        "feature_type": features.get("feature_type", "logmel"),
    }
    kwargs.update(algorithm_kwargs)
    return create_algorithm(str(config["algorithm"]), **kwargs)


def _prepare_run_dir(config: dict, config_path: Path) -> Path:
    run_dir = Path(config.get("run_dir", Path("runs") / str(config.get("experiment_name", config["algorithm"]))))
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    return run_dir


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _write_json(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_yaml(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _model_stats_with_context(stats: dict[str, float], config: dict) -> dict[str, float]:
    features = dict(config.get("features", {}))
    training = dict(config.get("training", {}))
    augment = training.get("augment", {}) if isinstance(training.get("augment"), dict) else {}
    eval_cfg = dict(config.get("eval", {}))
    hop_ms = float(features.get("hop_ms", 10.0))
    frame_ms = float(features.get("frame_ms", 25.0))
    context_frames = int(augment.get("segment_frames") or 0)
    if context_frames <= 0 and eval_cfg.get("sliding_window_sec") is not None:
        context_frames = max(1, int(round(float(eval_cfg["sliding_window_sec"]) * 1000.0 / max(hop_ms, 1e-9))))
    if context_frames <= 0:
        context_frames = 1
    macs_per_frame = float(stats.get("macs_per_frame", 0.0))
    out = dict(stats)
    out.update(
        {
            "frame_ms": float(frame_ms),
            "hop_ms": float(hop_ms),
            "context_frames": float(context_frames),
            "context_audio_ms": float(frame_ms + max(context_frames - 1, 0) * hop_ms),
            "feature_frame_rate_hz": float(1000.0 / hop_ms),
            "feature_window_samples_per_sec": float((1000.0 / hop_ms) * frame_ms),
            "streaming_macs_per_second": float(macs_per_frame * 1000.0 / hop_ms),
            "window_rerun_macs_per_second": float(macs_per_frame * context_frames * 1000.0 / hop_ms),
        }
    )
    return out


def _latency_sweep_variants(smoke: bool = False) -> list[dict[str, float | int]]:
    if smoke:
        return [{"frame_ms": 20.0, "hop_ms": 20.0, "context_frames": 13}]
    variants: list[dict[str, float | int]] = []
    for frame_ms in (20.0, 25.0, 32.0):
        for context_frames in (13, 25, 50):
            variants.append({"frame_ms": frame_ms, "hop_ms": 20.0, "context_frames": context_frames})
    return variants


def _latency_fallback_variants() -> list[dict[str, float | int]]:
    return [
        {"frame_ms": 25.0, "hop_ms": 15.0, "context_frames": 17},
        {"frame_ms": 25.0, "hop_ms": 15.0, "context_frames": 34},
        {"frame_ms": 25.0, "hop_ms": 15.0, "context_frames": 67},
    ]


def _latency_variant_tag(variant: dict[str, float | int]) -> str:
    return f"nano_f{int(float(variant['frame_ms']))}_h{int(float(variant['hop_ms']))}_ctx{int(variant['context_frames'])}"


def _latency_train_config(args: argparse.Namespace, variant: dict[str, float | int], manifest: Path, run_dir: Path) -> dict:
    hop_ms = float(variant["hop_ms"])
    context_frames = int(variant["context_frames"])
    max_time_mask = max(1, int(round(context_frames * 0.2)))
    time_shift_frames = max(1, int(round(80.0 / hop_ms)))
    return {
        "experiment_name": f"{_latency_variant_tag(variant)}_train",
        "manifest": manifest.as_posix(),
        "algorithm": "causal_crnn_vad_nano",
        "run_dir": (run_dir / "train").as_posix(),
        "seed": int(args.seed),
        "features": {
            "sample_rate": int(args.sample_rate),
            "frame_ms": float(variant["frame_ms"]),
            "hop_ms": hop_ms,
            "n_mels": int(args.n_mels),
            "n_mfcc": int(args.n_mels),
            "feature_type": "logmel",
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": 0.001,
            "min_lr": 0.00001,
            "weight_decay": 0.00001,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "class_balanced_sampling": True,
            "pos_weight": "auto",
            "early_stop_patience": 8,
            "device": "auto",
            "num_workers": 2,
            "augment": {
                "segment_frames": context_frames,
                "gain_db": 4.0,
                "time_shift_frames": time_shift_frames,
                "noise_std": 0.02,
                "freq_masks": 2,
                "time_masks": 2,
                "max_freq_mask": 8,
                "max_time_mask": max_time_mask,
            },
        },
        "postprocess": {"threshold": 0.5, "min_speech_ms": 80.0, "min_silence_ms": 160.0},
    }


def _latency_eval_config(args: argparse.Namespace, variant: dict[str, float | int], manifest: Path, run_dir: Path) -> dict:
    hop_ms = float(variant["hop_ms"])
    context_frames = int(variant["context_frames"])
    return {
        "experiment_name": f"{_latency_variant_tag(variant)}_eval",
        "manifest": manifest.as_posix(),
        "algorithm": "causal_crnn_vad_nano",
        "checkpoint": (run_dir / "train" / "checkpoints" / "best.pt").as_posix(),
        "run_dir": (run_dir / "eval").as_posix(),
        "features": {
            "sample_rate": int(args.sample_rate),
            "frame_ms": float(variant["frame_ms"]),
            "hop_ms": hop_ms,
            "n_mels": int(args.n_mels),
            "n_mfcc": int(args.n_mels),
            "feature_type": "logmel",
        },
        "eval": {
            "split": "test",
            "tune_split": "val",
            "tune_threshold": True,
            "paper_metrics": True,
            "target_fpr": 0.315,
            "sliding_window_sec": float(context_frames * hop_ms / 1000.0),
            "sliding_overlap": 0.0,
            "median_smoothing_ms": 0.0,
        },
        "training": {"device": "auto", "augment": {"segment_frames": context_frames}},
        "postprocess": {"threshold": "auto", "min_speech_ms": 80.0, "min_silence_ms": 160.0},
    }


def _latency_result_row(tag: str, variant: dict[str, float | int], eval_run_dir: str | Path, accuracy_floor: float) -> dict[str, object]:
    run_dir = Path(eval_run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    stats = json.loads((run_dir / "model_stats.json").read_text(encoding="utf-8"))
    aggregate = metrics.get("aggregate", metrics)
    paper = metrics.get("paper_val_threshold") or metrics.get("paper") or {}
    row = {
        "tag": tag,
        "frame_ms": float(variant["frame_ms"]),
        "hop_ms": float(variant["hop_ms"]),
        "context_frames": int(variant["context_frames"]),
        "context_audio_ms": float(stats["context_audio_ms"]),
        "params": int(stats["params"]),
        "macs_per_frame": float(stats["macs_per_frame"]),
        "streaming_macs_per_second": float(stats["streaming_macs_per_second"]),
        "window_rerun_macs_per_second": float(stats["window_rerun_macs_per_second"]),
        "feature_frame_rate_hz": float(stats["feature_frame_rate_hz"]),
        "feature_window_samples_per_sec": float(stats["feature_window_samples_per_sec"]),
        "f1": float(aggregate["f1"]),
        "precision": float(aggregate["precision"]),
        "recall": float(aggregate["recall"]),
        "accuracy": float(aggregate["accuracy"]),
        "auroc_all": float(paper.get("auroc_all", 0.0)),
        "tpr_at_fpr_0_315": float(paper.get("tpr_at_fpr_0_315", 0.0)),
        "meets_accuracy_floor": bool(float(aggregate["f1"]) >= accuracy_floor),
        "run_dir": run_dir.as_posix(),
    }
    return row


def _select_latency_best(summary_rows: list[dict[str, object]], accuracy_floor: float) -> dict[str, object] | None:
    candidates = [row for row in summary_rows if float(row["f1"]) >= accuracy_floor]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            float(row["streaming_macs_per_second"]),
            int(row["context_frames"]),
            -float(row["auroc_all"]),
            -float(row["tpr_at_fpr_0_315"]),
        ),
    )[0]


def _write_latency_summary(run_root: Path, rows: list[dict[str, object]], accuracy_floor: float) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: (float(row["streaming_macs_per_second"]), int(row["context_frames"]), -float(row["f1"])))
    _write_json(run_root / "summary.json", {"accuracy_floor": accuracy_floor, "rows": sorted_rows, "best": _select_latency_best(rows, accuracy_floor)})
    csv_path = run_root / "summary.csv"
    if not sorted_rows:
        return
    fieldnames = list(sorted_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)


def _write_segments(path: str | Path, segment_rows: list[tuple[str, list[dict[str, float]]]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "start_sec", "end_sec", "score"])
        writer.writeheader()
        for item_id, segments in segment_rows:
            for segment in segments:
                writer.writerow({"id": item_id, **segment})


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
